import numpy as np
import torch


def _scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
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
