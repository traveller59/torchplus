import ctypes
import math
import time

import numba
import numpy as np
import torch
from numba import cuda
from torch.autograd import Variable

from torchplus.tools import torch_to_np_type, _torch_string_type_to_class


def _scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    if torch.cuda.is_available():
        return torch_scatter_nd_gpu(indices, updates, shape)
    else:
        ret = torch.zeros(*shape).type_as(updates).cuda()
        ndim = indices.shape[-1]
        output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
        flatted_indices = indices.view(-1, ndim)
        slices = [flatted_indices[:, i] for i in range(ndim)]
        slices += [Ellipsis]
        ret[slices] = updates.view(*output_shape)
        return ret


def _gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


class ScatterNd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, updates, shape):
        ctx.save_for_backward(indices)
        return _scatter_nd(indices, updates, shape)

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_variables[0]
        grad_indices = grad_updates = grad_shape = None
        if ctx.needs_input_grad[1]:
            grad_updates = _gather_nd(grad_output, indices)
        return grad_indices, grad_updates, grad_shape


class GatherNd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, indices):
        shape = torch.LongTensor(list(params.shape))
        ctx.save_for_backward(indices, shape)
        return _gather_nd(params, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, shape = ctx.saved_variables
        shape_list = shape.cpu().numpy().tolist()
        # print(shape_list)
        # shape = torch.Size(*shape_list)
        grad_params = grad_indices = None
        if ctx.needs_input_grad[0]:
            grad_params = _scatter_nd(indices, grad_output, shape_list)
        return grad_params, grad_indices


scatter_nd = ScatterNd.apply

gather_nd = GatherNd.apply


@cuda.jit
def scatter_nd_kernel(indices, updates, out, output_shape_prefix,
                      batch_strides, num_indices, slice_size, slice_dim):
    bix = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    tix = cuda.threadIdx.x
    gdx = cuda.gridDim.x
    for index in range(bix * bdx + tix, num_indices, bdx * gdx):
        i = 0
        out_of_bounds = False
        for dim in range(slice_dim):
            offset = slice_dim * index + dim
            ix_d = indices[offset]
            if ix_d > output_shape_prefix[dim]:
                out_of_bounds = True
            i += ix_d * batch_strides[dim] * slice_size
        if not out_of_bounds:
            for si in range(slice_size):
                cuda.atomic.add(out, i + si, updates[index * slice_size + si])


def tensor_to_devicearray(t, dtype=np.float32):
    dtype_to_size = {
        np.float16: 2,
        np.float32: 4,
        np.float64: 8,
        np.int32: 4,
        np.int64: 8,
    }
    if isinstance(t, Variable):
        ttype = t.data.type()
    else:
        ttype = t.type()
    dtype = torch_to_np_type(ttype)
    type_size = dtype_to_size[dtype]
    ctx = cuda.cudadrv.driver.driver.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()),
                                           t.numel() * type_size)
    return cuda.cudadrv.devicearray.DeviceNDArray(
        t.size(), [i * type_size for i in t.stride()],
        np.dtype(dtype),
        gpu_data=mp,
        stream=torch.cuda.current_stream().cuda_stream)


def tensor_to_devicearray_cpu(t, dtype=np.float32):
    # only for numba cuda sim mode
    dtype_to_size = {
        np.float16: 2,
        np.float32: 4,
        np.float64: 8,
        np.int32: 4,
        np.int64: 8,
    }
    # type_size = dtype_to_size[dtype]
    # ctx = cuda.cudadrv.driver.driver.get_context()
    # mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()),
    #                                        t.numel() * type_size)
    return cuda.to_device(t.cpu().numpy())


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


# @numba.jit# (nopython=True)
def _prepare_inputs(shape, indices):
    indices_shape = list(indices.shape)
    slice_dim = indices_shape[-1] if len(indices_shape) > 1 else 1
    slice_size_big = 1
    for i in range(slice_dim, len(shape)):
        slice_size_big *= shape[i]
    safe_slice_dim = max(1, slice_dim)
    num_updates = indices.size // safe_slice_dim
    return slice_dim, num_updates, slice_size_big


def scatter_nd_gpu(indices, updates, shape, out, stream=0):
    # t = time.time()
    slice_dim, num_updates, slice_size = _prepare_inputs(shape, indices)
    indices_shape = list(indices.shape)
    # slice_dim = int(indices_shape[-1] if len(indices_shape) > 1 else 1)
    # slice_size = int(indices.size)
    # num_updates = int(indices_shape[0])
    slice_num = indices.size
    updates_size = updates.size
    out_size = out.size
    IXDIM = slice_dim
    THREAD_PER_BLOCK = 512
    N = num_updates
    BLOCK_COUNT = div_up(N, THREAD_PER_BLOCK)
    output_shape_prefix_cpu = np.array(shape[:IXDIM], dtype=np.int64)
    output_shape_prefix = cuda.to_device(
        output_shape_prefix_cpu, stream=stream)
    batch_strides = np.zeros([IXDIM], dtype=np.int64)
    for dim in range(IXDIM - 1, -1, -1):
        if dim == IXDIM - 1:
            batch_strides[dim] = 1
        else:
            batch_strides[dim] = batch_strides[
                dim + 1] * output_shape_prefix_cpu[dim + 1]
    batch_strides_dev = cuda.to_device(batch_strides, stream=stream)
    scatter_nd_kernel[BLOCK_COUNT, THREAD_PER_BLOCK, stream](
        indices.reshape(slice_num), updates.reshape(updates_size),
        out.reshape(out_size), output_shape_prefix, batch_strides_dev,
        num_updates, slice_size, IXDIM)


def torch_scatter_nd_gpu(indices, updates, shape):
    # t = time.time()
    # nptype = torch_to_np_type(indices.type())
    fptype = updates.type()
    indices = indices.float()
    updates = updates.float()
    indices_dev = tensor_to_devicearray(indices, np.float32)
    updates_dev = tensor_to_devicearray(updates, np.float32)
    if isinstance(updates, Variable):
        out_ttype = updates.data.type()
    else:
        out_ttype = updates.type()
    out = _torch_string_type_to_class(out_ttype)(*shape).zero_()
    out_dev = tensor_to_devicearray(out, np.float32)
    stream = torch.cuda.current_stream().cuda_stream
    scatter_nd_gpu(indices_dev, updates_dev, shape, out_dev, stream)
    return out.type(fptype)


def torch_scatter_nd_gpu_simulate(indices, updates, shape):
    indices_dev = tensor_to_devicearray_cpu(indices)
    updates_dev = tensor_to_devicearray_cpu(updates)
    if isinstance(updates, Variable):
        out_ttype = updates.data.type()
    else:
        out_ttype = updates.type()
    out = np.zeros(shape).astype(torch_to_np_type(out_ttype))
    out_dev = cuda.to_device(out)
    stream = torch.cuda.current_stream().cuda_stream
    scatter_nd_gpu(indices_dev, updates_dev, shape, out_dev, stream, device_id)
    return torch.from_numpy(out).type(out_ttype)

