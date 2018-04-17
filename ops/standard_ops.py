try:
    import numba
    from .array_ops import scatter_nd, gather_nd
except:
    print("WARNING: scatter_nd need numba.")

from .einsum import einsum