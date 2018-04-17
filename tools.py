import functools
import inspect
import sys
from collections import OrderedDict

import numba
import numpy as np
import torch


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def get_kw_to_default_map(func):
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper


def np_dtype_to_torch(dtype):
    type_map = {
        np.dtype(np.float16): torch.HalfTensor,
        np.dtype(np.float32): torch.FloatTensor,
        np.dtype(np.float64): torch.DoubleTensor,
        np.dtype(np.int32): torch.IntTensor,
        np.dtype(np.int64): torch.LongTensor,
        np.dtype(np.uint8): torch.ByteTensor,
    }
    return type_map[dtype]


def np_dtype_to_np_type(dtype):
    type_map = {
        np.dtype(np.float16): np.float16,
        np.dtype(np.float32): np.float32,
        np.dtype(np.float64): np.float64,
        np.dtype(np.int32): np.int32,
        np.dtype(np.int64): np.int64,
        np.dtype(np.uint8): np.uint8,
    }
    return type_map[dtype]


def np_type_to_torch(dtype, cuda=False):
    type_map = {
        np.float16: torch.HalfTensor,
        np.float32: torch.FloatTensor,
        np.float64: torch.DoubleTensor,
        np.int32: torch.IntTensor,
        np.int64: torch.LongTensor,
        np.uint8: torch.ByteTensor,
    }
    cuda_type_map = {
        np.float16: torch.cuda.HalfTensor,
        np.float32: torch.cuda.FloatTensor,
        np.float64: torch.cuda.DoubleTensor,
        np.int32: torch.cuda.IntTensor,
        np.int64: torch.cuda.LongTensor,
        np.uint8: torch.cuda.ByteTensor,
    }
    if cuda:
        return cuda_type_map[dtype]
    else:
        return type_map[dtype]


def np_type_to_numba(dtype):
    type_map = {
        np.float16: numba.float16,
        np.float32: numba.float32,
        np.float64: numba.float64,
        np.int32: numba.int32,
        np.int64: numba.int64,
        np.uint8: numba.uint8,
    }
    return type_map[dtype]


def torch_to_np_type(ttype):
    type_map = {
        'torch.HalfTensor': np.float16,
        'torch.FloatTensor': np.float32,
        'torch.DoubleTensor': np.float64,
        'torch.IntTensor': np.int32,
        'torch.LongTensor': np.int64,
        'torch.ByteTensor': np.uint8,
        'torch.cuda.HalfTensor': np.float16,
        'torch.cuda.FloatTensor': np.float32,
        'torch.cuda.DoubleTensor': np.float64,
        'torch.cuda.IntTensor': np.int32,
        'torch.cuda.LongTensor': np.int64,
        'torch.cuda.ByteTensor': np.uint8,
    }
    return type_map[ttype]


def _torch_string_type_to_class(ttype):
    type_map = {
        'torch.HalfTensor': torch.HalfTensor,
        'torch.FloatTensor': torch.FloatTensor,
        'torch.DoubleTensor': torch.DoubleTensor,
        'torch.IntTensor': torch.IntTensor,
        'torch.LongTensor': torch.LongTensor,
        'torch.ByteTensor': torch.ByteTensor,
        'torch.cuda.HalfTensor': torch.cuda.HalfTensor,
        'torch.cuda.FloatTensor': torch.cuda.FloatTensor,
        'torch.cuda.DoubleTensor': torch.cuda.DoubleTensor,
        'torch.cuda.IntTensor': torch.cuda.IntTensor,
        'torch.cuda.LongTensor': torch.cuda.LongTensor,
        'torch.cuda.ByteTensor': torch.cuda.ByteTensor,
    }
    return type_map[ttype]


def torch_to_np_dtype(ttype):
    type_map = {
        torch.HalfTensor: np.dtype(np.float16),
        torch.FloatTensor: np.dtype(np.float32),
        torch.DoubleTensor: np.dtype(np.float64),
        torch.IntTensor: np.dtype(np.int32),
        torch.LongTensor: np.dtype(np.int64),
        torch.ByteTensor: np.dtype(np.uint8),
    }
    return type_map[ttype]


def isinf(tensor):
    return tensor == torch.FloatTensor([float('inf')]).type_as(tensor)


def to_tensor(arg):
    if isinstance(arg, np.ndarray):
        return torch.from_numpy(arg).type(np_dtype_to_torch(arg.dtype))
    elif isinstance(arg, (list, tuple)):
        arg = np.array(arg)
        return torch.from_numpy(arg).type(np_dtype_to_torch(arg.dtype))
    else:
        raise ValueError("unsupported arg type.")


def zeros(*sizes, dtype=np.float32, cuda=False):
    torch_tensor_cls = np_type_to_torch(dtype, cuda)
    return torch_tensor_cls(*sizes).zero_()


def get_tensor_class(tensor):
    return _torch_string_type_to_class(tensor.type())
