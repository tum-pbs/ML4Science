""" Functions related to tensor permutation.
"""
from itertools import permutations
from typing import Union, Optional, Any

import numpy as np

from ..backend import default_backend
from ._shape import concat_shapes_, batch, DimFilter, Shape, SHAPE_TYPES, shape, non_batch, channel, dual
from ._magic_ops import unpack_dim, expand, stack, slice_
from ._tensors import reshaped_tensor, TensorOrTree, Tensor, wrap
from ._ops import unravel_index


def random_permutation(*shape: Union[Shape, Any], dims=non_batch, index_dim=channel('index')) -> Tensor:
    """
    Generate random permutations of the integers between 0 and the size of `shape`.

    When multiple dims are given, the permutation is randomized across all of them and tensor of multi-indices is returned.

    Batch dims result in batches of permutations.

    Args:
        *shape: `Shape` of the result tensor, including `dims` and batches.
        *dims: Sequence dims for an individual permutation. The total `Shape.volume` defines the maximum integer.
            All other dims from `shape` are treated as batch.

    Returns:
        `Tensor`
    """
    assert dims is not batch, f"dims cannot include all batch dims because that violates the batch principle. Specify batch dims by name instead."
    shape = concat_shapes_(*shape)
    assert not shape.dual_rank, f"random_permutation does not support dual dims but got {shape}"
    perm_dims = shape.only(dims)
    batches = shape - perm_dims
    nu = perm_dims.non_uniform_shape
    batches -= nu
    assert nu in shape, f"Non-uniform permutation dims {perm_dims} must be included in the shape but got {shape}"
    b = default_backend()
    result = []
    for idx in nu.meshgrid():
        perm_dims_i = perm_dims.after_gather(idx)
        native = b.random_permutations(batches.volume, perm_dims_i.volume)
        if perm_dims_i.rank == 0:  # cannot add index_dim
            result.append(reshaped_tensor(native, [batches, ()], convert=False))
        else:
            native = b.unravel_index(native, perm_dims_i.sizes)
            result.append(reshaped_tensor(native, [batches, perm_dims_i, index_dim.with_size(perm_dims_i.name_list)], convert=False))
    return stack(result, nu)


def pick_random(value: TensorOrTree, dim: DimFilter, count: Union[int, Shape, None] = 1, weight: Optional[Tensor] = None) -> TensorOrTree:
    """
    Pick one or multiple random entries from `value`.

    Args:
        value: Tensor or tree. When containing multiple tensors, the corresponding entries are picked on all tensors that have `dim`.
            You can pass `range` (the type) to retrieve the picked indices.
        dim: Dimension along which to pick random entries. `Shape` with one dim.
        count: Number of entries to pick. When specified as a `Shape`, lists picked values along `count` instead of `dim`.
        weight: Probability weight of each item along `dim`. Will be normalized to sum to 1.

    Returns:
        `Tensor` or tree equal to `value`.
    """
    v_shape = shape(value)
    dim = v_shape.only(dim)
    if count is None and dim.well_defined:
        count = dim.size
    n = dim.volume if count is None else (count.volume if isinstance(count, SHAPE_TYPES) else count)
    if n == dim.volume and weight is None:
        idx = random_permutation(dim & v_shape.batch & dim.non_uniform_shape, dims=dim)
        idx = unpack_dim(idx, dim, count) if isinstance(count, SHAPE_TYPES) else idx
    else:
        nu_dims = v_shape.non_uniform_shape
        idx_slices = []
        for nui in nu_dims.meshgrid():
            u_dim = dim.after_gather(nui)
            weight_np = weight.numpy([u_dim]) if weight is not None else None
            if u_dim.volume >= n:
                np_idx = np.random.choice(u_dim.volume, size=n, replace=False, p=weight_np / weight_np.sum() if weight is not None else None)
            elif u_dim.volume > 0:
                np_idx = np.arange(n) % u_dim.volume
            else:
                raise ValueError(f"Cannot pick random from empty tensor {u_dim}")
            idx = wrap(np_idx, count if isinstance(count, SHAPE_TYPES) else u_dim.without_sizes())
            # idx = ravel_index()
            idx_slices.append(expand(idx, channel(index=u_dim.name)))
        idx = stack(idx_slices, nu_dims)
    return slice_(value, idx)


def all_permutations(dims: Shape, list_dim=dual('perm'), index_dim: Optional[Shape] = channel('index'), convert=False) -> Tensor:
    """
    Returns a `Tensor` containing all possible permutation indices of `dims` along `list_dim`.

    Args:
        dims: Dims along which elements are permuted.
        list_dim: Single dim along which to list the permutations.
        index_dim: Dim listing vector components for multi-dim permutations. Can be `None` if `dims.rank == 1`.
        convert: Whether to convert the permutations to the default backend. If `False`, the result is backed by NumPy.

    Returns:
        Permutations as a single index `Tensor`.
    """
    np_perms = np.asarray(list(permutations(range(dims.volume))))
    perms = reshaped_tensor(np_perms, [list_dim, dims], convert=convert)
    if index_dim is None:
        assert len(dims) == 1, f"For multi-dim permutations, index_dim must be specified."
        return perms
    return unravel_index(perms, dims, index_dim)
