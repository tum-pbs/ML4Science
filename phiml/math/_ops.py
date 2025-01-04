import functools
import math
import warnings
from numbers import Number
from typing import Tuple, Callable, Any, Union, Optional, Dict, Collection, Sequence, Set

import numpy as np

from ..backend import default_backend, choose_backend, Backend, get_precision, convert as b_convert, BACKENDS, NoBackendFound, ComputeDevice, NUMPY
from ..backend._dtype import DType, combine_types
from .magic import PhiTreeNode
from ._magic_ops import expand, pack_dims, unpack_dim, cast, value_attributes, bool_to_int, tree_map, concat, stack, unstack, rename_dims, slice_, all_attributes, squeeze, ipack
from ._shape import (Shape, EMPTY_SHAPE,
                     spatial, batch, channel, instance, merge_shapes, parse_dim_order, concat_shapes,
                     IncompatibleShapes, DimFilter, non_batch, dual, shape, shape as get_shape, primal, auto, non_spatial, non_dual, resolve_index, concat_shapes_, SHAPE_TYPES,
                     Dim)
from . import extrapolation as e_
from ._tensors import (Tensor, wrap, tensor, broadcastable_native_tensors, Dense, TensorStack,
                       custom_op2, compatible_tensor, variable_attributes, disassemble_tree, assemble_tree,
                       is_scalar, Layout, expand_tensor, TensorOrTree, cached, variable_shape,
                       reshaped_tensor, discard_constant_dims, variable_dim_names, backend_for, preferred_backend_for)
from ._sparse import (CompressedSparseMatrix, dense, SparseCoordinateTensor, get_format, to_format, stored_indices,
                      tensor_like, sparse_dims, same_sparsity_pattern, is_sparse, sparse_dot, sparse_sum, sparse_gather, sparse_max,
                      sparse_min, dense_dims, sparse_mean, stored_values, sparse_matrix_dims, CompactSparseTensor)


def convert(x, backend: Backend = None, use_dlpack=True):
    """
    Convert the native representation of a `Tensor` or `phiml.math.magic.PhiTreeNode` to the native format of `backend`.

    *Warning*: This operation breaks the automatic differentiation chain.

    See Also:
        `phiml.math.backend.convert()`.

    Args:
        x: `Tensor` to convert. If `x` is a `phiml.math.magic.PhiTreeNode`, its variable attributes are converted.
        backend: Target backend. If `None`, uses the current default backend, see `phiml.math.backend.backend()`.

    Returns:
        `Tensor` with native representation belonging to `backend`.
    """
    if isinstance(x, Tensor):
        return x._from_spec_and_natives(x._spec_dict(), [b_convert(n, use_dlpack=False) for n in x._natives()])
    elif isinstance(x, PhiTreeNode):
        return tree_map(convert, x, backend=backend, use_dlpack=use_dlpack)
    else:
        return b_convert(x, backend, use_dlpack=use_dlpack)


def to_device(value, device: ComputeDevice or str, convert=True, use_dlpack=True):
    """
    Allocates the tensors of `value` on `device`.
    If the value already exists on that device, this function may either create a copy of `value` or return `value` directly.

    See Also:
        `to_cpu()`.

    Args:
        value: `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.
        device: Device to allocate value on.
            Either `ComputeDevice` or category `str`, such as `'CPU'` or `'GPU'`.
        convert: Whether to convert tensors that do not belong to the corresponding backend to compatible native tensors.
            If `False`, this function has no effect on numpy tensors.
        use_dlpack: Only if `convert==True`.
            Whether to use the DLPack library to convert from one GPU-enabled backend to another.

    Returns:
        Same type as `value`.
    """
    assert isinstance(device, (ComputeDevice, str)), f"device must be a ComputeDevice or str but got {type(device)}"
    return tree_map(_to_device, value, device=device, convert_to_backend=convert, use_dlpack=use_dlpack)


def _to_device(value: Tensor or Any, device: ComputeDevice or str, convert_to_backend: bool, use_dlpack: bool):
    if isinstance(value, Tensor):
        if not convert and value.backend == NUMPY:
            return value
        natives = [_to_device(n, device, convert_to_backend, use_dlpack) for n in value._natives()]
        return value._with_natives_replaced(natives)
    else:
        old_backend = choose_backend(value)
        if isinstance(device, str):
            device = old_backend.list_devices(device)[0]
        if old_backend != device.backend:
            if convert_to_backend:
                value = b_convert(value, device.backend, use_dlpack=use_dlpack)
            else:
                return value
        return device.backend.allocate_on_device(value, device)


def all_available(*values) -> bool:
    """
    Tests if all tensors contained in the given `values` are currently known and can be read.
    Placeholder tensors used to trace functions for just-in-time compilation or matrix construction are considered not available, even when they hold example values like with PyTorch's JIT.

    Tensors are not available during `jit_compile()`, `jit_compile_linear()` or while using TensorFlow's legacy graph mode.
    
    Tensors are typically available when the backend operates in eager mode and is not currently tracing a function.

    This can be used instead of the native checks

    * PyTorch: `torch._C._get_tracing_state()`
    * TensorFlow: `tf.executing_eagerly()`
    * Jax: `isinstance(x, jax.core.Tracer)`

    Args:
        values: Tensors to check.

    Returns:
        `True` if no value is a placeholder or being traced, `False` otherwise.
    """
    _, tensors = disassemble_tree(values, cache=False)
    return all([t.available for t in tensors])


def seed(seed: int):
    """
    Sets the current seed of all backends and the built-in `random` package.

    Calling this function with a fixed value at the start of an application yields reproducible results
    as long as the same backend is used.

    Args:
        seed: Seed to use.
    """
    for backend in BACKENDS:
        backend.seed(seed)
    import random
    random.seed(0)


def copy(value: Tensor):
    """
    Copies the data buffer and encapsulating `Tensor` object.

    Args:
        value: `Tensor` to be copied.

    Returns:
        Copy of `value`.
    """
    if value._is_tracer:
        warnings.warn("Tracing tensors cannot be copied.", RuntimeWarning)
        return value
    return value._op1(lambda native: choose_backend(native).copy(native))


def native_call(f: Callable, *inputs: Tensor, channels_last=None, channel_dim='vector', spatial_dim=None):
    """
    Calls `f` with the native representations of the `inputs` tensors in standard layout and returns the result as a `Tensor`.

    All inputs are converted to native tensors (including precision cast) depending on `channels_last`:

    * `channels_last=True`: Dimension layout `(total_batch_size, spatial_dims..., total_channel_size)`
    * `channels_last=False`: Dimension layout `(total_batch_size, total_channel_size, spatial_dims...)`

    All batch dimensions are compressed into a single dimension with `total_batch_size = input.shape.batch.volume`.
    The same is done for all channel dimensions.

    Additionally, missing batch and spatial dimensions are added so that all `inputs` have the same batch and spatial shape.

    Args:
        f: Function to be called on native tensors of `inputs`.
            The function output must have the same dimension layout as the inputs, unless overridden by `spatial_dim`,
            and the batch size must be identical.
        *inputs: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
            If `None`, the channels are put in the default position associated with the current backend,
            see `phiml.math.backend.Backend.prefers_channels_last()`.
        channel_dim: Name of the channel dimension of the result.
        spatial_dim: Name of the spatial dimension of the result.

    Returns:
        `Tensor` with batch and spatial dimensions of `inputs`, unless overridden by `spatial_dim`,
        and single channel dimension `channel_dim`.
    """
    if channels_last is None:
        try:
            backend = choose_backend(f)
        except NoBackendFound:
            backend = preferred_backend_for(*inputs)
        channels_last = backend.prefers_channels_last()
    b_dims = merge_shapes(*[i.shape.batch & i.shape.dual for i in inputs])
    s_dims = merge_shapes(*[i.shape.spatial for i in inputs])
    natives = []
    for i in inputs:
        groups = [b_dims, *i.shape.spatial.names, i.shape.channel] if channels_last else [b_dims, i.shape.channel, *i.shape.spatial.names]
        natives.append(i.native(groups, force_expand=False))
    output = f(*natives)
    if isinstance(channel_dim, str):
        channel_dim = channel(channel_dim)
    assert isinstance(channel_dim, SHAPE_TYPES), "channel_dim must be a Shape or str"
    if isinstance(output, (tuple, list)):
        raise NotImplementedError()
    if spatial_dim is None:
        ndim = choose_backend(output).ndims(output)
        if ndim == 1:
            groups = [b_dims]
        elif ndim == 2:
            groups = [b_dims, channel_dim]
        else:
            groups = [b_dims, *s_dims, channel_dim] if channels_last else [b_dims, channel_dim, *s_dims]
    else:
        if isinstance(spatial_dim, str):
            spatial_dim = spatial(spatial_dim)
        assert isinstance(spatial_dim, SHAPE_TYPES), "spatial_dim must be a Shape or str"
        groups = [b_dims, *spatial_dim, channel_dim] if channels_last else [b_dims, channel_dim, *spatial_dim]
    result = reshaped_tensor(output, groups, convert=False)
    if result.shape.get_size(channel_dim.name) == 1 and not channel_dim.item_names[0]:
        result = result.dimension(channel_dim.name)[0]  # remove vector dim if not required
    return result


def print_(obj: Union[Tensor, PhiTreeNode, Number, tuple, list, None] = None, name: str = ""):
    """
    Print a tensor with no more than two spatial dimensions, slicing it along all batch and channel dimensions.
    
    Unlike NumPy's array printing, the dimensions are sorted.
    Elements along the alphabetically first dimension is printed to the right, the second dimension upward.
    Typically, this means x right, y up.

    Args:
        obj: tensor-like
        name: name of the tensor

    Returns:

    """
    def variables(obj) -> dict:
        if hasattr(obj, '__variable_attrs__') or hasattr(obj, '__value_attrs__'):
            return {f".{a}": getattr(obj, a) for a in variable_attributes(obj)}
        elif isinstance(obj, (tuple, list)):
            return {f"[{i}]": item for i, item in enumerate(obj)}
        elif isinstance(obj, dict):
            return obj
        else:
            raise ValueError(f"Not PhiTreeNode: {type(obj)}")

    if name:
        print(" " * 12 + name)
    if obj is None:
        print("None")
    elif isinstance(obj, Tensor):
        print(f"{obj:full}")
    elif isinstance(obj, PhiTreeNode):
        for n, val in variables(obj).items():
            print_(val, name + n)
    else:
        print(f"{wrap(obj):full}")


def slice_off(x, *slices: Dict[str, Union[slice, int, str]]):
    """

    Args:
        x: Any instance of `phiml.math.magic.Shapable`
        *slices:

    Returns:

    """
    if not slices:
        return x
    x_shape = shape(x)
    dims = set().union(*[s.keys() for s in slices])
    dims = x_shape.only(dims).names
    depth = max(len(s) for s in slices)
    if depth == 1:
        if len(dims) == 1:
            d = dims[0]
            if all(all(_edge_slice(x_shape, dim, s) for dim, s in s_dict.items()) for s_dict in slices):  # only edges
                edge_slices = [_edge_slice(x_shape, dim, s) for s_dict in slices for dim, s in s_dict.items()]
                if any(s.start == 0 and s.stop is None for s in edge_slices):  # everything sliced off
                    return x[{d: slice(0, 0)}]
                start_slices = [s for s in edge_slices if s.start == 0]
                end_slices = [s for s in edge_slices if s.stop is None]
                start = max(s.stop for s in start_slices) if start_slices else 0  # at this point, s.stop must be an int
                end = min(s.start for s in end_slices) if end_slices else None
                return x[{d: slice(start, end)}]
            else:
                size = x_shape.get_size(d)
                mask = np.ones(size, dtype=np.bool_)
                for s_dict in slices:
                    s = next(iter(s_dict.values()))
                    if isinstance(s, str):
                        names = x_shape.get_item_names(d)
                        s = [names.index(n.strip()) for n in s.split(',')]
                    mask[s] = 0
                return boolean_mask(x, d, wrap(mask, x_shape[d]))
    unstack_dim = x_shape.only(_preferred_unstack_dim(x, dims))
    x_slices = unstack(x, unstack_dim)
    x_slices_out = []
    for i, x_slice in enumerate(x_slices):
        slices_without_unstack_dim = [{k: v for k, v in s_dict.items() if k != unstack_dim.name} for s_dict in slices if _includes_slice(s_dict, unstack_dim, i)]
        sliced_x_slice = slice_off(x_slice, *slices_without_unstack_dim)
        x_slices_out.append(sliced_x_slice)
    assembled = stack(x_slices_out, unstack_dim)
    slices_for_unstack_dim_only = [s_dict for s_dict in slices if len(s_dict) == 1 and unstack_dim.name in s_dict]
    result = slice_off(assembled, *slices_for_unstack_dim_only)
    return result


def _edge_slice(x_shape: Shape, dim: str, s):
    size = x_shape.get_size(dim)
    if isinstance(s, str):
        s = [n.strip() for n in s.split(',')]
    if isinstance(s, slice):
        reaches_end = s.stop is None or s.stop >= size
        if not s.start:
            return slice(0, None if reaches_end else s.stop)
        elif reaches_end:
            return slice(s.start, None)
    elif isinstance(s, int):
        if s == 0:
            return slice(0, 1)
        elif s == -1 or s == size - 1:
            return slice(size - 1, None)
    elif isinstance(s, (tuple, list)):
        names = x_shape.get_item_names(dim)
        indices = [i if isinstance(i, int) else names.index(i) for i in s]
        if all(n in indices for n in range(max(indices))):
            return slice(0, max(indices) + 1)
        elif all(n in indices for n in range(min(indices), size)):
            return slice(min(indices), None)


def _preferred_unstack_dim(x, dims: Collection[str]) -> str:
    if shape(x).is_non_uniform:
        nu = shape(x).non_uniform
        return nu.non_uniform_shape.names[0]
    else:
        return min(dims, key=lambda d: x.shape.get_size(d))


def _includes_slice(s_dict: dict, dim: Shape, i: int):
    if dim.name not in s_dict:
        return True
    s = s_dict[dim.name]
    if isinstance(s, str):
        s = [n.strip() for n in s.split(',')]
    if isinstance(s, int):
        return s == i
    elif isinstance(s, slice):
        return (s.start or 0) <= i < (s.stop if s.stop is not None else dim.size)
    elif isinstance(s, (tuple, list)):
        names = dim.item_names[0]
        indices = [i if isinstance(i, int) else names.index(i) for i in s]
        return i in indices


def _initialize(uniform_initializer, shapes: Tuple[Shape]) -> Tensor:
    shape = concat_shapes_(*shapes)
    assert shape.well_defined, f"When creating a Tensor, shape needs to have definitive sizes but got {shape}"
    if shape.is_non_uniform:
        stack_dim = shape.non_uniform_shape[0]
        shapes = shape.unstack(stack_dim.name)
        tensors = [_initialize(uniform_initializer, s) for s in shapes]
        return stack_tensors(tensors, stack_dim)
    else:
        return uniform_initializer(shape)


def zeros(*shape: Shape, dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Define a tensor with specified shape with value `0.0` / `0` / `False` everywhere.
    
    This method may not immediately allocate the memory to store the values.

    See Also:
        `zeros_like()`, `ones()`.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: Data type as `DType` object. Defaults to `float` matching the current precision setting.

    Returns:
        `Tensor`
    """
    return _initialize(lambda shape: expand_tensor(Dense(default_backend().zeros((), dtype=DType.as_dtype(dtype)), (), EMPTY_SHAPE, default_backend()), shape), shape)


def zeros_like(obj: Union[Tensor, PhiTreeNode]) -> Union[Tensor, PhiTreeNode]:
    """ Create a `Tensor` containing only `0.0` / `0` / `False` with the same shape and dtype as `obj`. """
    nest, values = disassemble_tree(obj, cache=False, attr_type=value_attributes)
    zeros_ = []
    for val in values:
        val = wrap(val)
        with val.backend:
            zeros_.append(zeros(val.shape, dtype=val.dtype))
    return assemble_tree(nest, zeros_, attr_type=value_attributes)


def ones(*shape: Shape, dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Define a tensor with specified shape with value `1.0`/ `1` / `True` everywhere.
    
    This method may not immediately allocate the memory to store the values.

    See Also:
        `ones_like()`, `zeros()`.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: Data type as `DType` object. Defaults to `float` matching the current precision setting.

    Returns:
        `Tensor`
    """
    return _initialize(lambda shape: expand_tensor(Dense(default_backend().ones((), dtype=DType.as_dtype(dtype)), (), EMPTY_SHAPE, default_backend()), shape), shape)


def ones_like(value: Tensor) -> Tensor:
    """ Create a `Tensor` containing only `1.0` / `1` / `True` with the same shape and dtype as `obj`. """
    return zeros_like(value) + 1


def random_normal(*shape: Shape, dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Creates a `Tensor` with the specified shape, filled with random values sampled from a normal / Gaussian distribution.

    Implementations:

    * NumPy: [`numpy.random.standard_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html)
    * PyTorch: [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html)
    * TensorFlow: [`tf.random.normal`](https://www.tensorflow.org/api_docs/python/tf/random/normal)
    * Jax: [`jax.random.normal`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html)

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: (optional) floating point `DType`. If `None`, a float tensor with the current default precision is created, see `get_precision()`.

    Returns:
        `Tensor`
    """

    def uniform_random_normal(shape):
        backend = choose_backend(*shape.sizes, prefer_default=True)
        native = backend.random_normal(shape.sizes, DType.as_dtype(dtype))
        return Dense(native, shape.names, shape, backend)

    return _initialize(uniform_random_normal, shape)


def random_uniform(*shape: Shape,
                   low: Union[Tensor, float] = 0,
                   high: Union[Tensor, float] = 1,
                   dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Creates a `Tensor` with the specified shape, filled with random values sampled from a uniform distribution.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: (optional) `DType` or `(kind, bits)`.
            The dtype kind must be one of `float`, `int`, `complex`.
            If not specified, a `float` tensor with the current default precision is created, see `get_precision()`.
        low: Minimum value, included.
        high: Maximum value, excluded.
    Returns:
        `Tensor`
    """
    if get_shape(low).volume == 1 and get_shape(high).volume == 1:
        low = low.native() if isinstance(low, Tensor) else low
        high = high.native() if isinstance(high, Tensor) else high
        def uniform_random_uniform(shape):
            backend = choose_backend(low, high, *shape.sizes, prefer_default=True)
            native = backend.random_uniform(shape.sizes, low, high, DType.as_dtype(dtype))
            return Dense(native, shape.names, shape, backend)
        return _initialize(uniform_random_uniform, shape)
    else:
        def uniform_random_uniform(shape):
            backend = choose_backend(*shape.sizes, prefer_default=True)
            native = backend.random_uniform(shape.sizes, 0, 1, DType.as_dtype(dtype))
            return Dense(native, shape.names, shape, backend)
        return _initialize(uniform_random_uniform, shape) * (high - low) + low


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


def swap_axes(x, axes):
    """
    Swap the dimension order of `x`.
    This operation is generally not necessary for `Tensor`s because tensors will be reshaped under the hood or when getting the native/numpy representations.
    It can be used to transpose native tensors.

    Implementations:

    * NumPy: [`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
    * PyTorch: [`x.permute`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute)
    * TensorFlow: [`tf.transpose`](https://www.tensorflow.org/api_docs/python/tf/transpose)
    * Jax: [`jax.numpy.transpose`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html)

    Args:
        x: `Tensor` or native tensor or `phiml.math.magic.Shapable`.
        axes: `tuple` or `list`

    Returns:
        `Tensor` or native tensor, depending on `x`.
    """
    if isinstance(x, Tensor):
        if isinstance(axes, (tuple, list)) and all(isinstance(a, int) for a in axes):
            axes = [x.shape.names[a] for a in axes]
        if x.shape[axes].names == x.shape.only(axes).names:  # order is correct
            return x
        new_shape = x.shape[axes]
        packed = x.__pack_dims__(new_shape, instance('_t_flat'), None)
        return unpack_dim(packed, '_t_flat', new_shape)
    else:
        return choose_backend(x).transpose(x, axes)


def sort(x: Tensor, dim: DimFilter = non_batch) -> Tensor:
    """
    Sort the values of `x` along `dim`.
    In order to sort a flattened array, use `pack_dims` first.

    Args:
        x: `Tensor`
        dim: Dimension to sort. If not present, sorting will be skipped. Defaults to non-batch dim.

    Returns:
        Sorted `Tensor` or `x` if `x` is constant along `dims`.
    """
    x_shape = x.shape
    dim = x_shape.only(dim)
    v_names = variable_dim_names(x)
    if dim.name not in v_names:
        return x  # nothing to do; x is constant along dim
    assert dim.rank == 1, f"Can only sort one dimension at a time. Use pack_dims() to jointly sort over multiple dimensions."
    axis = v_names.index(dim.name)
    x_native = x._native if isinstance(x, Dense) else x.native(x.shape)
    sorted_native = x.backend.sort(x_native, axis=axis)
    if x.shape.get_item_names(dim):
        warnings.warn(f"sort() removes item names along sorted axis '{dim}'. Was {x.shape.get_item_names(dim)}", RuntimeWarning, stacklevel=2)
        x_shape = x_shape.with_dim_size(dim, x_shape.get_size(dim), keep_item_names=False)
    return Dense(sorted_native, v_names, x_shape, x.backend)


def cumulative_sum(x: Tensor, dim: DimFilter, include_0=False, include_sum=True, index_dim: Union[str, Shape, None] = None):
    """
    Performs a cumulative sum of `x` along `dim`.

    Implementations:

    * NumPy: [`cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
    * PyTorch: [`cumsum`](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
    * TensorFlow: [`cumsum`](https://www.tensorflow.org/api_docs/python/tf/math/cumsum)
    * Jax: [`cumsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cumsum.html)

    Args:
        x: `Tensor`
        dim: Dimension along which to sum, as `str` or `Shape`. If multiple dims are passed, `x` the cumulative sum will be computed on the flattened array.
        include_0: If `True`, adds a 0 to the result before the first value.
        include_sum: If `False`, the total sum will be sliced off the result.
        index_dim: If given, adds an index dimension for `dim`.

    Returns:
        `Tensor` with the same shape as `x`.
    """
    dim = x.shape.only(dim, reorder=True)
    assert dim.rank >= 1, f"dim must contain at least one dimension."
    assert dim.rank == 1 or include_0 + include_sum == 1, f"When summing over multiple flattened dims, exaclty one of (include_0, include_sum) must be True but got include_0={include_0}, include_sum={include_sum}"
    broadcast = broadcast_dims(x)
    assert dim.only(broadcast).is_empty, f"Cannot compute cumulative sum along {dim} because input is not uniform along that dimension."
    def uniform_cumulative_sum(x: Tensor, index_dim=index_dim, dim=dim.names):
        dim = x.shape.only(dim, reorder=True)
        native_x = x._reshaped_native([x.shape - dim, dim])
        b = choose_backend(native_x)
        native_result = b.cumsum(native_x, 1)
        if include_0:
            native_result = b.pad(native_result, ((0, 0), (1, 0)))
        if not include_sum:
            native_result = native_result[:, :-1]
        result = reshaped_tensor(native_result, [x.shape - dim, dim + (include_0 + include_sum) - 1])
        if index_dim is not None:
            assert dim.rank == 1, f"multi-dimensional indices not yet supported"
            if isinstance(index_dim, str):
                index_dim = auto(index_dim, channel)
            index_dim = index_dim.with_size(dim.name_list)
            result = expand(result, index_dim)
        return result
    return broadcast_op(uniform_cumulative_sum, [x], broadcast)


def fftfreq(resolution: Shape, dx: Union[Tensor, float] = 1, dtype: DType = None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    Args:
        resolution: Grid resolution measured in cells
        dx: Distance between sampling points in real space.
        dtype: Data type of the returned tensor (Default value = None)

    Returns:
        `Tensor` holding the frequencies of the corresponding values computed by math.fft
    """
    assert resolution.spatial and f"resolution must contain at least one spatial dimension"
    k = meshgrid(**{dim.name: np.fft.fftfreq(int(dim.size)) for dim in resolution.spatial})
    k /= dx
    return to_float(k) if dtype is None else cast(k, dtype)


def meshgrid(dims: Union[Callable, Shape] = spatial, stack_dim=channel('vector'), **dimensions: Union[int, Tensor, tuple, list, Any]) -> Tensor:
    """
    Generate a mesh-grid `Tensor` from keyword dimensions.

    Args:
        **dimensions: Mesh-grid dimensions, mapping names to values.
            Values may be `int`, 1D `Tensor` or 1D native tensor.
        dims: Dimension type of mesh-grid dimensions, one of `spatial`, `channel`, `batch`, `instance`.
        stack_dim: Channel dim along which grids are stacked.
            This is optional for 1D mesh-grids. In that case returns a `Tensor` without a stack dim if `None` or an empty `Shape` is passed.

    Returns:
        Mesh-grid `Tensor` with the dimensions of `dims` / `dimensions` and `stack_dim`.

    Examples:
        >>> math.meshgrid(x=2, y=2)
        (xˢ=2, yˢ=2, vectorᶜ=x,y) 0.500 ± 0.500 (0e+00...1e+00)

        >>> math.meshgrid(x=2, y=(-1, 1))
        (xˢ=2, yˢ=2, vectorᶜ=x,y) 0.250 ± 0.829 (-1e+00...1e+00)

        >>> math.meshgrid(x=2, stack_dim=None)
        (0, 1) along xˢ
    """
    assert 'dim_type' not in dimensions, f"dim_type has been renamed to dims"
    assert not stack_dim or stack_dim.name not in dimensions
    if isinstance(dims, SHAPE_TYPES):
        assert not dimensions, f"When passing a Shape to meshgrid(), no kwargs are allowed"
        dimensions = {d: s for d, s in zip(dims.names, dims.sizes)}
        grid_shape = dims
        dim_values = [tuple(range(s)) for s in dims.sizes]
    else:
        dim_type = dims
        assert callable(dim_type), f"dims must be a Shape or dimension type but got {dims}"
        dim_values = []
        dim_sizes = []
        for dim, spec in dimensions.items():
            if isinstance(spec, int) or (isinstance(spec, Tensor) and spec.rank == 0 and spec.dtype.kind == int):
                dim_values.append(tuple(range(int(spec))))
                dim_sizes.append(spec)
            elif isinstance(spec, Tensor):
                assert spec.rank == 1, f"Only 1D sequences allowed, got {spec} for dimension '{dim}'."
                dim_values.append(spec.native())
                dim_sizes.append(spec.shape.volume)
            else:
                backend = choose_backend(spec)
                shape = backend.staticshape(spec)
                assert len(shape) == 1, "Only 1D sequences allowed, got {spec} for dimension '{dim}'."
                dim_values.append(spec)
                dim_sizes.append(shape[0])
        grid_shape = dim_type(**{dim: size for dim, size in zip(dimensions.keys(), dim_sizes)})
    backend = choose_backend(*dim_values, prefer_default=True)
    indices_list = backend.meshgrid(*dim_values)
    channels = [Dense(t, grid_shape.names, grid_shape, backend) for t in indices_list]
    if not stack_dim:
        assert len(channels) == 1, f"meshgrid with multiple dimension requires a valid stack_dim but got {stack_dim}"
        return channels[0]
    if stack_dim.item_names[0] is None:
        stack_dim = stack_dim.with_size(tuple(dimensions.keys()))
    return stack_tensors(channels, stack_dim)


def linspace(start: Union[float, Tensor, tuple, list], stop: Union[float, Tensor, tuple, list], dim: Shape) -> Tensor:
    """
    Returns `number` evenly spaced numbers between `start` and `stop` along `dim`.

    If `dim` contains multiple dimensions, evenly spaces values along each dimension, then stacks the result along a new channel dimension called `vector`.

    See Also:
        `arange()`, `meshgrid()`.

    Args:
        start: First value, `int` or `Tensor`.
        stop: Last value, `int` or `Tensor`.
        dim: Linspace dimension of integer size.
            The size determines how many values to linearly space between `start` and `stop`.
            The values will be laid out along `dim`.

    Returns:
        `Tensor`

    Examples:
        >>> math.linspace(0, 1, spatial(x=5))
        (0.000, 0.250, 0.500, 0.750, 1.000) along xˢ

        >>> math.linspace(0, (-1, 1), spatial(x=3))
        (0.000, 0.000); (-0.500, 0.500); (-1.000, 1.000) (xˢ=3, vectorᶜ=2)
    """
    assert isinstance(dim, SHAPE_TYPES), f"dim must be a Shape but got {dim}"
    assert dim.is_uniform, f"dim must be uniform but got {dim}"
    start = wrap(start)
    stop = wrap(stop)
    if dim.rank > 1:
        return meshgrid(dim) / (dim - 1) * (stop - start) + start
    if is_scalar(start) and is_scalar(stop):
        backend = preferred_backend_for(start, stop)
        native_linspace = backend.linspace(start.native(), stop.native(), dim.size)
        return Dense(native_linspace, dim.names, dim, backend)
    else:
        from ._functional import map_
        return map_(linspace, start, stop, dim=dim)


def arange(dim: Shape, start_or_stop: Union[int, None] = None, stop: Union[int, None] = None, step=1, backend=None):
    """
    Returns evenly spaced values between `start` and `stop`.
    If only one limit is given, `0` is used for the start.

    See Also:
        `range_tensor()`, `linspace()`, `meshgrid()`.

    Args:
        dim: Dimension name and type as `Shape` object.
            The `size` of `dim` is interpreted as `stop` unless `start_or_stop` is specified.
        start_or_stop: (Optional) `int`. Interpreted as `start` if `stop` is specified as well. Otherwise this is `stop`.
        stop: (Optional) `int`. `stop` value.
        step: Distance between values.
        backend: Backend to use for creating the tensor. If unspecified, uses the current default.

    Returns:
        `Tensor`
    """
    assert dim.primal.rank <= 1, f"dim can have at most one primal dimension"
    if dim.primal.rank == 0:
        assert dim.rank == 1, f"When no primal dimension is specified, dim must have rank 1"
        range_dim = dim
    else:
        range_dim = dim.primal
    if start_or_stop is None:
        assert stop is None, "start_or_stop must be specified when stop is given."
        assert dim.well_defined, "When start_or_stop is not specified, all sizes of dim must be specified."
        start, stop = 0, (dim.primal.size if dim.primal else dim.size)
    elif stop is None:
        start, stop = 0, start_or_stop
    else:
        start = start_or_stop
    start, stop, step = wrap(start), wrap(stop), wrap(step)
    assert range_dim not in start and range_dim not in stop and range_dim not in step, f"range dim {range_dim} must not be present in either start, stop, or step"
    def batched_range(dims: Shape, start: Tensor, stop: Tensor, step: Tensor):
        batches = (dims - range_dim) & start.shape & stop.shape & step.shape
        if batches:
            b0 = batches.non_uniform[0] if batches.is_non_uniform else batches
            ranges = [batched_range(dims.after_gather(i), start[i], stop[i], step[i]) for i in b0.meshgrid()]
            return stack(ranges, b0)
        b = backend or preferred_backend_for(start, stop)
        native = b.range(start.native(), stop.native(), step.native(), DType(int, 32))
        return Dense(native, range_dim.names, range_dim.with_size(len(native)), b)
    return batched_range(dim, start, stop, step)


def range_tensor(*shape: Shape):
    """
    Returns a `Tensor` with given `shape` containing the linear indices of each element.
    For 1D tensors, this equivalent to `arange()` with `step=1`.

    See Also:
        `arange()`, `meshgrid()`.

    Args:
        shape: Tensor shape.

    Returns:
        `Tensor`
    """
    shape = concat_shapes_(*shape)
    data = arange(spatial('range'), 0, shape.volume)
    return unpack_dim(data, 'range', shape)


def stack_tensors(values: Union[tuple, list], dim: Shape):
    if len(values) == 1 and not dim:
        return values[0]
    values = [wrap(v) for v in values]
    values = [squeeze(v, dim) for v in values]
    values = cast_same(*values)
    # --- sparse to dense ---
    if any(isinstance(t, (SparseCoordinateTensor, CompressedSparseMatrix)) for t in values) and not all(isinstance(t, (SparseCoordinateTensor, CompressedSparseMatrix)) for t in values):
        values = [dense(v) for v in values]
    # --- trivial case ---
    if len(values) == 1 and isinstance(values[0], Dense):
        return Dense(values[0]._native, values[0]._names, values[0].shape & dim.with_size(1), values[0]._backend)
    # --- not directly stackable ---
    non_stackable = broadcast_dims(*values)
    if non_stackable or any(is_sparse(v) for v in values):  # stackable sparse would have been handled by __stack__() before calling this function
        return TensorStack(values, dim)
    if any(v._is_tracer for v in values):
        return TensorStack(values, dim)
    broadcast_shape = merge_shapes(*[v.shape for v in values], allow_varying_sizes=True)
    if not broadcast_shape.well_defined:
        return TensorStack(values, dim)
    # --- uniform stack ---
    dim = dim.with_size(len(values))
    native_broadcast_shape = merge_shapes(*[variable_shape(v) for v in values])
    natives = [v._reshaped_native([*native_broadcast_shape]) for v in values]
    names = (native_broadcast_shape & dim).names
    b = choose_backend(*natives)
    native_stacked = b.stack(natives, axis=names.index(dim.name))
    expanded_shape = merge_shapes(dim, *[v.shape for v in values])  # put dim to its group
    return Dense(native_stacked, names, expanded_shape, b)


def concat_tensor(values: Union[tuple, list], dim: str) -> Tensor:
    assert len(values) > 0, "concat() got empty sequence"
    assert isinstance(dim, str), f"dim must be a single-dimension Shape but got '{dim}' of type {type(dim)}"
    if any(v._is_tracer for v in values):
        from ._trace import concat_tracers
        return concat_tracers(values, dim)

    def inner_concat(*values):
        broadcast_shape: Shape = values[0].shape  # merge_shapes(*[t.shape.with_sizes([None] * t.shape.rank) for t in values])
        dim_index = broadcast_shape.index(dim)
        backend = backend_for(*values)
        natives = [v.native(order=broadcast_shape.names) for v in values]
        concatenated = backend.concat(natives, dim_index)
        if all([v.shape.get_item_names(dim) is not None or v.shape.get_size(dim) == 0 for v in values]):
            broadcast_shape = broadcast_shape.with_dim_size(dim, sum([v.shape.get_item_names(dim) or () for v in values], ()))
        else:
            broadcast_shape = broadcast_shape.with_dim_size(dim, sum([v.shape.get_size(dim) for v in values]))
        return Dense(concatenated, broadcast_shape.names, broadcast_shape, backend)

    result = broadcast_op(inner_concat, values)
    return result


def unflatten_unpack(value: Tensor, dim: DimFilter, shapes: Sequence[Shape]):
    dim = value.shape.only(dim)
    required = sum([s.volume for s in shapes])
    assert required == dim.size, f"Unpacked shapes must match {dim} but got {shapes} which total {required}"
    result = []
    i = 0
    for s in shapes:
        end = i + s.volume
        sliced = value[{dim: slice(i, end)}]
        result.append(sliced.__unpack_dim__(dim.name, s))
        i = end
    return result


def pad(value: Tensor, widths: Union[dict, tuple, list], mode: Union['e_.Extrapolation', Tensor, Number, str, dict] = 0, **kwargs) -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation.
    Unlike `Extrapolation.pad()`, this function can handle negative widths which slice off outer values.

    Args:
        value: `Tensor` to be padded
        widths: Number of values to add at the edge of `value`. Negative values can be used to slice off edge values. Must be one of the following:

            * `tuple` containing `(lower: int, upper: int)`. This will pad all non-batch dimensions by `lower` and `upper` at the lower and upper edge, respectively.
            * `dict` mapping `dim: str -> (lower: int, upper: int)`
            * Sequence of slicing `dict`s. This will add all values specified by the slicing dicts and is the inverse operation to `slice_off`. Exactly one value in each slicing dict must be a `slice` object.

        mode: Padding mode used to determine values added from positive `widths`.
            Must be one of the following: `Extrapolation`, `Tensor` or number for constant extrapolation, name of extrapolation as `str`.
        kwargs: Additional padding arguments.
            These are ignored by the standard extrapolations defined in `phiml.math.extrapolation` but can be used to pass additional contextual information to custom extrapolations.

    Returns:
        Padded `Tensor`

    Examples:
        >>> math.pad(math.ones(spatial(x=10, y=10)), {'x': (1, 1), 'y': (2, 1)}, 0)
        (xˢ=12, yˢ=13) 0.641 ± 0.480 (0e+00...1e+00)

        >>> math.pad(math.ones(spatial(x=10, y=10)), {'x': (1, -1)}, 0)
        (xˢ=10, yˢ=10) 0.900 ± 0.300 (0e+00...1e+00)
    """
    mode = e_.as_extrapolation(mode)
    if isinstance(widths, (tuple, list)):
        if len(widths) == 0 or isinstance(widths[0], dict):  # add sliced-off slices
            return _pad_slices(value, widths, mode, **kwargs)
        if len(widths) == 2 and isinstance(widths[0], int) and isinstance(widths[1], int):  # (lower, upper)
            assert non_batch(value).rank == 1, f"Can only pad 1D tensors (excluding batch dims) when widths=(lower, upper) but got {shape(value)} and widths={widths}"
            widths = {non_batch(value).name: widths}
        else:  # ((lo0, up0), (lo1, up1), ...)
            assert len(widths) == non_batch(value), f"Cannot pad tensor with non-batch dims {non_batch(value)} by widths {widths}. Sizes must match."
            warnings.warn("Padding by sequence of (lower, upper) is not recommended. Please use a dict instead.", SyntaxWarning, stacklevel=2)
            widths = {dim: w for dim, w in zip(non_batch(value).names, widths)}
    has_negative_widths = any(w0 < 0 or w1 < 0 for w0, w1 in widths.values())
    has_positive_widths = any(w0 > 0 or w1 > 0 for w0, w1 in widths.values())
    slices = None
    if has_negative_widths:
        slices = {dim: slice(max(0, -w[0]), min(0, w[1]) or None) for dim, w in widths.items()}
        widths = {dim: (max(0, w[0]), max(0, w[1])) for dim, w in widths.items()}
    result_padded = mode.pad(value, widths, **kwargs) if has_positive_widths else value
    result_sliced = result_padded[slices] if has_negative_widths else result_padded
    return result_sliced


def _pad_slices(x: Tensor, pad_slices: Sequence[Dict[str, Any]], mode: 'e_.Extrapolation', **kwargs) -> Tensor:
    def matches_i(pad_slice: Dict[str, int], sel: Dict[str, int]):
        for dim, s in pad_slice.items():
            if dim in sel:
                if s != sel[dim]:
                    return False
        return True

    def to_lower_upper(s: slice, size: int) -> Tuple[int, int]:
        if not s.start:  # pad left
            return s.stop, 0
        else:  # pad right
            return 0, (-s.start if s.start < 0 else size - s.start)

    sel_dims = set.union(*[{dim for dim, s in s_dict.items() if not isinstance(s, slice)} for s_dict in pad_slices])
    sel_dims = x.shape.only(sel_dims)
    pad_slices = [resolve_index(x.shape, ps) for ps in pad_slices]
    padded_slices = []
    for i in sel_dims.meshgrid():
        shape_i = x.shape.after_gather(i)
        pad_slices_i = [{dim: s for dim, s in ps.items() if dim not in i} for ps in pad_slices if matches_i(ps, i)]
        widths_i = [{dim: to_lower_upper(s, shape_i.get_size(dim)) for dim, s in ps.items()} for ps in pad_slices_i]
        sum_widths_i = {dim: np.asarray([0, 0]) for dim in set().union(*widths_i)}
        for width_i in widths_i:
            dim, lu = next(iter(width_i.items()))
            sum_widths_i[dim] += lu
        padded_slices.append(mode[i].pad(x[i], {k: tuple(v) for k, v in sum_widths_i.items()}, **kwargs))
    return stack(padded_slices, sel_dims)
    # for w in widths:
    #     selection = {dim: i for dim, i in w.items() if not isinstance(i, slice)}
    #     value_sel = value[selection]
    #     slice_ = {dim: to_lower_upper(s, value_sel.shape.get_size(dim)) for dim, s in w.items() if isinstance(s, slice)}
    #     assert len(slice_) == 1, f"Each slicing dict must contain one slice() value when padding by slices but got {w}"
    #     pad_dim = next(iter(slice_))
    #     v = mode[selection].pad_values(value_sel, width, pad_dim, **kwargs)


def closest_grid_values(grid: Tensor,
                        coordinates: Tensor,
                        extrap: 'e_.Extrapolation',
                        stack_dim_prefix='closest_',
                        **kwargs):
    """
    Finds the neighboring grid points in all directions and returns their values.
    The result will have 2^d values for each vector in coordinates in d dimensions.

    If `coordinates` does not have a channel dimension with item names, the spatial dims of `grid` will be used.

    Args:
        grid: grid data. The grid is spanned by the spatial dimensions of the tensor
        coordinates: tensor with 1 channel dimension holding vectors pointing to locations in grid index space
        extrap: grid extrapolation
        stack_dim_prefix: For each spatial dimension `dim`, stacks lower and upper closest values along dimension `stack_dim_prefix+dim`.
        kwargs: Additional information for the extrapolation.

    Returns:
        `Tensor` of shape (batch, coord_spatial, grid_spatial=(2, 2,...), grid_channel)
    """
    return broadcast_op(functools.partial(_closest_grid_values, extrap=extrap, stack_dim_prefix=stack_dim_prefix, pad_kwargs=kwargs), [grid, coordinates])


def _closest_grid_values(grid: Tensor,
                         coordinates: Tensor,
                         extrap: 'e_.Extrapolation',
                         stack_dim_prefix: str,
                         pad_kwargs: dict):
    # alternative method: pad array for all 2^d combinations, then stack to simplify gather.
    dim_names = channel(coordinates).item_names[0] or grid.shape.spatial.names
    dims = grid.shape.only(dim_names, reorder=True)
    assert len(dims) == len(dim_names), f"all grid dims {dim_names} must be present on grid but got shape {grid.shape}"
    # --- Pad tensor where transform is not possible ---
    non_copy_pad = {dim: (0 if extrap.is_copy_pad(dim, False) else 1, 0 if extrap.is_copy_pad(dim, True) else 1) for dim in dim_names}
    grid = extrap.pad(grid, non_copy_pad, **pad_kwargs)
    coordinates += wrap([not extrap.is_copy_pad(dim, False) for dim in dim_names], channel('vector'))
    # --- Transform coordiantes ---
    min_coords = to_int32(floor(coordinates))
    max_coords = extrap.transform_coordinates(min_coords + 1, grid.shape)
    min_coords = extrap.transform_coordinates(min_coords, grid.shape)

    def left_right(is_hi_by_axis_left, ax_idx):
        is_hi_by_axis_right = is_hi_by_axis_left | np.array([ax == ax_idx for ax in range(dims.rank)])
        coords_left = where(is_hi_by_axis_left, max_coords, min_coords)
        coords_right = where(is_hi_by_axis_right, max_coords, min_coords)
        if ax_idx == dims.rank - 1:
            values_left = gather(grid, coords_left)
            values_right = gather(grid, coords_right)
        else:
            values_left = left_right(is_hi_by_axis_left, ax_idx + 1)
            values_right = left_right(is_hi_by_axis_right, ax_idx + 1)
        return stack_tensors([values_left, values_right], channel(**{f"{stack_dim_prefix}{dim_names[ax_idx]}": 2}))

    result = left_right(np.array([False] * dims.rank), 0)
    return result


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: Union['e_.Extrapolation', float, str], **kwargs):
    """
    Samples values of `grid` at the locations referenced by `coordinates`.
    Values lying in between sample points are determined via linear interpolation.

    If `coordinates` has a channel dimension, its item names are used to determine the grid dimensions of `grid`.
    Otherwise, the spatial dims of `grid` will be used.

    For values outside the valid bounds of `grid` (`coord < 0 or coord > grid.shape - 1`), `extrap` is used to determine the neighboring grid values.
    If the extrapolation does not support resampling, the grid is padded by one cell layer before resampling.
    In that case, values lying further outside will not be sampled according to the extrapolation.

    Args:
        grid: Grid with at least one spatial dimension and no instance dimensions.
        coordinates: Coordinates with a single channel dimension called `'vector'`.
            The size of the `vector` dimension must match the number of spatial dimensions of `grid`.
        extrap: Extrapolation used to determine the values of `grid` outside its valid bounds.
        kwargs: Additional information for the extrapolation.

    Returns:
        `Tensor` with channel dimensions of `grid`, spatial and instance dimensions of `coordinates` and combined batch dimensions.
    """
    extrap = e_.as_extrapolation(extrap) if extrap is not None else None
    if not channel(coordinates):
        assert spatial(grid).rank == 1, f"grid must have 1 spatial dimension if coordinates does not have a channel dimension"
        coordinates = expand(coordinates, channel(vector=spatial(grid)))
    assert channel(coordinates).rank == 1, f"coordinates must have at most one channel dimension but got {channel(coordinates)}"
    coordinates = rename_dims(coordinates, channel, 'vector')
    result = broadcast_op(functools.partial(_grid_sample, extrap=extrap, pad_kwargs=kwargs), [grid, coordinates])
    return result


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: Union['e_.Extrapolation', None], pad_kwargs: dict):
    """
    Args:
        grid:
        coordinates: has exactly one channel dimension
        extrap:
        pad_kwargs:
    """
    dim_names = channel(coordinates).item_names[0] or grid.shape.spatial.names
    dims = grid.shape.only(dim_names, reorder=True)
    assert len(dims) == len(dim_names), f"all grid dims {dim_names} must be present on grid but got shape {grid.shape}"
    if grid.shape.batch == coordinates.shape.batch or grid.shape.batch.volume == 1 or coordinates.shape.batch.volume == 1:
        # call backend.grid_sample()
        batch_dims = (grid.shape.batch & coordinates.shape.batch).without(dims)
        backend = backend_for(grid, coordinates)
        result = NotImplemented
        if extrap is None:
            result = backend.grid_sample(grid.native([batch_dims, *dims, grid.shape.non_batch.without(dims)]),
                                         coordinates._reshaped_native([batch_dims, *coordinates.shape.instance, *coordinates.shape.spatial, coordinates.shape['vector']]),
                                         'undefined')
        elif extrap.native_grid_sample_mode:
            result = backend.grid_sample(grid.native([batch_dims, *dims, grid.shape.non_batch.without(dims)]),
                                         coordinates._reshaped_native([batch_dims, *coordinates.shape.instance, *coordinates.shape.spatial, coordinates.shape['vector']]),
                                         extrap.native_grid_sample_mode)
        if result is NotImplemented:
            # pad one layer
            grid_padded = pad(grid, {dim: (1, 1) for dim in dim_names}, extrap or e_.ZERO, **pad_kwargs)
            if extrap is not None:
                from .extrapolation import _CopyExtrapolation
                if isinstance(extrap, _CopyExtrapolation):
                    inner_coordinates = extrap.transform_coordinates(coordinates, grid.shape) + 1
                else:
                    inner_coordinates = extrap.transform_coordinates(coordinates + 1, grid_padded.shape)
            else:
                inner_coordinates = coordinates + 1
            result = backend.grid_sample(grid_padded.native([batch_dims, *dims.names, grid.shape.non_batch.without(dims)]),
                                         inner_coordinates._reshaped_native([batch_dims, *coordinates.shape.instance, *coordinates.shape.spatial, coordinates.shape['vector']]),
                                         'boundary')
        if result is not NotImplemented:
            result = reshaped_tensor(result, [batch_dims, *coordinates.shape.instance, *coordinates.shape.spatial, grid.shape.non_batch.without(dims)])
            return result
    # fallback to slower grid sampling
    neighbors = _closest_grid_values(grid, coordinates, extrap or e_.ZERO, '_closest_', pad_kwargs)
    binary = meshgrid(channel, **{f'_closest_{dim}': (0, 1) for dim in dim_names}, stack_dim=channel(coordinates))
    right_weights = coordinates % 1
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    result = sum_(neighbors * weights, dim=[f"_closest_{dim}" for dim in dim_names])
    return result


def broadcast_dims(*tensors: Tensor) -> Set[str]:
    iter_dims = set()
    for tensor in tensors:
        iter_dims.update(shape(tensor).non_uniform_shape.names)
        if isinstance(tensor, TensorStack) and tensor.requires_broadcast:
            iter_dims.add(tensor._stack_dim.name)
        # --- remove iter_dims for which the sizes vary among tensors ---
        for dim in tuple(iter_dims):
            sizes = [t.shape.get_size(dim) for t in tensors if dim in t.shape]
            if not all(s == sizes[0] for s in sizes[1:]):
                iter_dims.remove(dim)
    return iter_dims


def broadcast_op(operation: Callable,
                 tensors: Union[tuple, list],
                 iter_dims: Union[set, tuple, list, Shape] = None,
                 no_return=False):
    iter_dims = broadcast_dims(*tensors) if iter_dims is None else iter_dims
    if len(iter_dims) == 0:
        return operation(*tensors)
    else:
        if isinstance(iter_dims, SHAPE_TYPES):
            iter_dims = iter_dims.names
        dim_name = next(iter(iter_dims))
        size = None
        item_names = None
        unstacked = []
        for tensor in tensors:
            if dim_name in tensor.shape:
                dim = tensor.shape[dim_name]
                unstacked_tensor = tensor._unstack(dim_name)
                unstacked.append(unstacked_tensor)
                if size is None:
                    size = len(unstacked_tensor)
                else:
                    assert size == len(unstacked_tensor)
                if item_names is None:
                    item_names = dim.slice_names
            else:
                unstacked.append(tensor)
        result_unstacked = []
        for i in range(size):
            gathered = [t[i] if isinstance(t, tuple) else t for t in unstacked]
            result_unstacked.append(broadcast_op(operation, gathered, iter_dims=set(iter_dims) - {dim_name}))
        if not no_return:
            return stack(result_unstacked, Dim(dim_name, size, dim.dim_type, item_names))


def where(condition: Union[Tensor, float, int],
          value_true: Union[Tensor, float, int, Any] = None,
          value_false: Union[Tensor, float, int, Any] = None):
    """
    Builds a tensor by choosing either values from `value_true` or `value_false` depending on `condition`.
    If `condition` is not of type boolean, non-zero values are interpreted as True.
    
    This function requires non-None values for `value_true` and `value_false`.
    To get the indices of True / non-zero values, use :func:`nonzero`.

    Args:
      condition: determines where to choose values from value_true or from value_false
      value_true: Values to pick where `condition != 0 / True`
      value_false: Values to pick where `condition == 0 / False`

    Returns:
        `Tensor` containing dimensions of all inputs.
    """
    if value_true is None:
        assert value_false is None, f"where can be used either with value_true and value_false or without both but got only value_false"
        warnings.warn("Use nonzero() instead of where() to get indices of non-zero elements.", SyntaxWarning, stacklevel=2)
        return nonzero(condition)
    from .extrapolation import Extrapolation, where as ext_where
    if isinstance(value_true, Extrapolation) or isinstance(value_false, Extrapolation):
        return ext_where(condition, value_true, value_false)
    condition = wrap(condition)
    value_true = wrap(value_true)
    value_false = wrap(value_false)
    def inner_where(c: Tensor, vt: Tensor, vf: Tensor):
        if isinstance(value_true, Layout) or isinstance(value_false, Layout):  # result must be a Layout
            shape = merge_shapes(c, vt, vf)
            result = []
            for idx in shape.meshgrid():
                result.append(vt[idx] if c[idx].any else vf[idx])
            return stack(result, shape)
        if vt._is_tracer or vf._is_tracer or c._is_tracer:
            return c * vt + (1 - c) * vf  # ToDo this does not take NaN into account
        if is_sparse(c) or is_sparse(vt) or is_sparse(vf):
            if not same_sparsity_pattern(vt, vf, allow_const=True) or not same_sparsity_pattern(c, vt, allow_const=True):
                raise NotImplementedError(f"When calling where() on sparse tensors, all arguments must have the same sparsity pattern or be dense")
            sp_dims = sparse_dims(c) & sparse_dims(vt) & sparse_dims(vf)
            d_dims = dense_dims(c) & dense_dims(vt) & dense_dims(vf)
            if d_dims and d_dims in sp_dims:  # sparse / dense conflict -> first apply sparse format
                any_sparse = c if is_sparse(c) else vt if is_sparse(vt) else vf
                sparse_ones = tensor_like(any_sparse, 1)
                c = c if is_sparse(c) else sparse_ones * c
                vt = vt if is_sparse(vt) else sparse_ones * vt
                vf = vf if is_sparse(vf) else sparse_ones * vf
            c_values = c._values if is_sparse(c) else c
            vt_values = vt._values if is_sparse(vt) else vt
            vf_values = vf._values if is_sparse(vf) else vf
            return c._with_values(where(c_values, vt_values, vf_values))
        names, shape, (c, vt, vf) = broadcastable_native_tensors(c, vt, vf)
        backend = choose_backend(c, vt, vf)
        result = backend.where(c, vt, vf)
        return Dense(result, names, shape, backend)

    return broadcast_op(inner_where, [condition, value_true, value_false])


def nonzero(value: Tensor, list_dim: Union[Shape, str, int] = instance('nonzero'), index_dim: Shape = channel('vector'), element_dims: DimFilter = channel, list_dims: DimFilter = non_batch, preserve_names=False):
    """
    Get spatial indices of non-zero / True values.
    
    Batch dimensions are preserved by this operation.
    If channel dimensions are present, this method returns the indices where any component is nonzero.

    Implementations:

    * NumPy: [`numpy.argwhere`](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html)
    * PyTorch: [`torch.nonzero`](https://pytorch.org/docs/stable/generated/torch.nonzero.html)
    * TensorFlow: [`tf.where(tf.not_equal(values, 0))`](https://www.tensorflow.org/api_docs/python/tf/where)
    * Jax: [`jax.numpy.nonzero`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nonzero.html)

    Args:
        value: spatial tensor to find non-zero / True values in.
        list_dim: Dimension listing non-zero values. If size specified, lists only the first `size` non-zero values.
            Special case: For retrieving only the first non-zero value, you may pass `1` instead of a `Shape` of size 1.
        index_dim: Index dimension.
        element_dims: Dims listing components of one value. A value is only considered `zero` if all components are 0.
        list_dims: Dims in which non-zero elements are searched. These will be stored in the item names of `index_dim`.

    Returns:
        `Tensor` of shape (batch dims..., `list_dim`=#non-zero, `index_dim`=value.shape.spatial_rank)

    """
    element_dims = value.shape.only(element_dims)
    if element_dims:
        value = sum_(abs(value), element_dims)
    list_dims = value.shape.only(list_dims) - element_dims
    if isinstance(list_dim, str):
        list_dim = auto(list_dim, instance)
    cutoff = list_dim if isinstance(list_dim, int) else list_dim.size
    if isinstance(list_dim, int) and list_dim == 1:
        list_dim = EMPTY_SHAPE
    elif isinstance(list_dim, int):
        assert list_dims.rank == 1
        list_dim = list_dims.without_sizes()
    broadcast = value.shape - list_dims - sparse_matrix_dims(value)
    def unbatched_nonzero(value: Tensor):
        if isinstance(value, CompressedSparseMatrix):
            if cutoff is not None and (sparse_dims(value) - list_dims) in value._compressed_dims:
                v0 = value._pointers[:-1]
                vs = v0 + arange(instance(_offset=cutoff))
                col = value._indices[vs]
                return ipack(rename_dims(col, '_offset', list_dim), value._compressed_dims)
            else:
                value = value.decompress()
        elif isinstance(value, CompactSparseTensor):
            if list_dims in value._compressed_dims and value._uncompressed_dims not in list_dims:
                result = value._indices
                if result.shape.only(value._compressed_dims).volume == cutoff:
                    return result
                else:
                    return result[{value._compressed_dims: slice(cutoff)}]
            else:
                raise NotImplementedError
        if isinstance(value, SparseCoordinateTensor) and cutoff is None:
            assert cutoff is None, f"Cut-off Not implemented for sparse tensors"
            nonzero_values = nonzero(value._values)
            nonzero_indices = value._indices[nonzero_values]
            index_dim_ = index_dim.with_size(channel(value._indices).item_names[0])
            return rename_dims(rename_dims(nonzero_indices, instance, list_dim), channel, index_dim_)
        elif isinstance(value, SparseCoordinateTensor):
            # value = value.compress(sparse_dims(value) - list_dims)
            raise NotImplementedError
        else:
            native = value._reshaped_native([*value.shape])
            b = choose_backend(native)
            indices = b.nonzero(native)
            if cutoff is not None:
                indices = indices[:cutoff, :]
            new_list_dim = list_dim
            if preserve_names and list_dims.rank == 1 and list_dims.item_names[0]:
                names = [list_dims.item_names[0][i] for i in indices[:, 0]]
                new_list_dim = new_list_dim.with_size(names)
            return reshaped_tensor(indices, [new_list_dim, index_dim.with_size(value.shape.name_list)])
    return broadcast_op(unbatched_nonzero, [value], iter_dims=broadcast.names)


def nonzero_slices(x: Tensor):
    """

    Args:
        x: Concrete tensor.

    Returns:

    """
    assert x.available, f"nonzero_slices requires a concrete tensor but got {x}"
    assert x.rank == 1, f"nonzero_slices requires a 1D tensor but got {x.shape}"
    if x.shape.volume == 0:
        return []
    dim = x.shape.name
    if x.dtype.kind != bool:
        x = x != 0
    slices = []
    start = None
    for i, val in enumerate(x):
        if val:
            if start is None:
                start = i
        else:
            if start is not None:
                slices.append({dim: slice(start, i)})
                start = None
    if start is not None:
        slices.append({dim: slice(start, None)})
    return slices


def reduce_(f: TensorOrTree, value, dims, require_all_dims_present=False, required_kind: type = None) -> TensorOrTree:
    if not dims:
        return value
    is_tree = not isinstance(value, Tensor) and isinstance(value, PhiTreeNode)
    if isinstance(value, (tuple, list)) and all(isinstance(v, Tensor) for v in value):
        dims = merge_shapes(batch('0'), *value).only(dims)
        is_tree = '0' not in dims
    if is_tree:
        reduce_outer = isinstance(dims, (Shape, tuple, list, str)) and '0' in parse_dim_order(dims)
        if reduce_outer:
            value = stack(value, batch('0'))
        return tree_map(lambda v: reduce_(f, v, dims, require_all_dims_present, required_kind), value, attr_type=variable_attributes)
    if isinstance(value, (tuple, list)):
        values = [wrap(v) for v in value]
        value = stack_tensors(values, batch(**{'0': len(values)}))
        dims = value.shape.only(dims)
        assert '0' in dims, "When passing a sequence of tensors to be reduced, the sequence dimension '0' must be reduced."
    elif isinstance(value, Layout):
        if value.shape.without(value._stack_dim).only(dims):  # reduce some inner
            def inner_reduce(v):
                if required_kind is not None:
                    if isinstance(v, Tensor):
                        v = cast(v, required_kind)
                    else:
                        v = required_kind(v)
                return f(wrap(v), shape(v).only(dims))

            value = tree_map(inner_reduce, value)
        if value._stack_dim.without(dims).is_empty:  # reduce all outer
            values = value._as_list()
            dims = batch('_flat_layout')
            value = wrap(values, dims)
    else:
        value = wrap(value)
    dims = value.shape.only(dims)
    if require_all_dims_present and any(d not in value.shape for d in dims):
        raise ValueError(f"Cannot sum dimensions {dims} because tensor {value.shape} is missing at least one of them")
    return f(value._simplify(), dims)


def sum_(value: TensorOrTree, dim: DimFilter = non_batch) -> TensorOrTree:
    """
    Sums `values` along the specified dimensions.

    Args:
        value: (Sparse) `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_sum, bool_to_int(value), dim, require_all_dims_present=True)


dsum = functools.partial(sum_, dim=dual)
dsum.__doc__ = """Sum dual dims of `value`, see `phiml.math.sum`."""

isum = functools.partial(sum_, dim=instance)
isum.__doc__ = """Sum instance dims of `value`, see `phiml.math.sum`."""

ssum = functools.partial(sum_, dim=spatial)
ssum.__doc__ = """Sum spatial dims of `value`, see `phiml.math.sum`."""

csum = functools.partial(sum_, dim=channel)
csum.__doc__ = """Sum channel dims of `value`, see `phiml.math.sum`."""


def _sum(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, Dense):
        result = value.backend.sum(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names)) * value.collapsed_dims.only(dims).volume
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    elif isinstance(value, TensorStack):
        reduced_inners = [_sum(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x + y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif is_sparse(value):
        return sparse_sum(value, dims)
    elif value._is_tracer:
        if dims.volume == 1:
            return value[{dims.name: 0}]
        raise NotImplementedError
    else:
        raise ValueError(type(value))


def prod(value, dim: DimFilter = non_batch) -> Tensor:
    """
    Multiplies `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_prod, value, dim, require_all_dims_present=True)


dprod = functools.partial(prod, dim=dual)
dprod.__doc__ = """Compute the product along dual dims of `value`, see `phiml.math.prod`."""

iprod = functools.partial(prod, dim=instance)
iprod.__doc__ = """Compute the product along instance dims of `value`, see `phiml.math.prod`."""

sprod = functools.partial(prod, dim=spatial)
sprod.__doc__ = """Compute the product along spatial dims of `value`, see `phiml.math.prod`."""

cprod = functools.partial(prod, dim=channel)
cprod.__doc__ = """Compute the product along channel dims of `value`, see `phiml.math.prod`."""


def _prod(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, Dense):
        result = value.backend.prod(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names)) ** value.collapsed_dims.only(dims).volume
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    elif isinstance(value, TensorStack):
        reduced_inners = [_prod(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x * y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def mean(value, dim: DimFilter = non_batch, weight: Union[Tensor, list, tuple] = None, where_no_weight=float('nan'), epsilon=1e-10) -> Tensor:
    """
    Computes the mean over `values` along the specified dimensions.

    Args:
        value: (Sparse) `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        weight: Optionally perform a weighted mean operation. Must broadcast to `value`.
        where_no_weight: Value to use when the sum of all weights are smaller than `epsilon`.
        epsilon: Only if `where_no_weight`. Threshold for using `where_no_weight`.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    if weight is not None:
        if isinstance(value, (tuple, list)):
            assert isinstance(weight, (tuple, list)), f"When computing mean over tuples or lists, the weight must also be a tuple or list"
            value = stack_tensors([wrap(v) for v in value], instance(**{'0': len(value)}))
            weight = stack_tensors([wrap(w) for w in weight], instance(**{'0': len(weight)}))
            dim = value.shape.only(dim)
            assert '0' in dim, "When passing a sequence of tensors to be reduced, the sequence dimension '0' must be reduced."
        weight_sum = sum_(weight, dim)
        if not np.isnan(where_no_weight):
            weight_sum = where(abs(weight_sum) < epsilon, 1, weight_sum)
        result = sum_(value * weight, dim) / weight_sum
        if not np.isnan(where_no_weight):
            result = where(weight_sum == 0, where_no_weight, result)
        return result
    return reduce_(_mean, value, dim)


dmean = functools.partial(mean, dim=dual)
dmean.__doc__ = """Compute the mean along dual dims of `value`, see `phiml.math.mean`."""

imean = functools.partial(mean, dim=instance)
imean.__doc__ = """Compute the mean along instance dims of `value`, see `phiml.math.mean`."""

smean = functools.partial(mean, dim=spatial)
smean.__doc__ = """Compute the mean along spatial dims of `value`, see `phiml.math.mean`."""

cmean = functools.partial(mean, dim=channel)
cmean.__doc__ = """Compute the mean along channel dims of `value`, see `phiml.math.mean`."""


def _mean(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, Dense):
        result = value.backend.mean(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names))
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    elif isinstance(value, TensorStack):
        if value._stack_dim in dims:
            total = _sum(value, dims)
            return total / dims.volume
        else:  # keep stack_dim
            return TensorStack([_mean(t, dims.without(value._stack_dim)) for t in value._tensors], value._stack_dim)
    elif is_sparse(value):
        return sparse_mean(value, dims)
    else:
        raise ValueError(type(value))


def std(value, dim: DimFilter = non_batch) -> Tensor:
    """
    Computes the standard deviation over `values` along the specified dimensions.

    *Warning*: The standard deviation of non-uniform tensors along the stack dimension is undefined.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    if not dim:
        warnings.warn("std along empty shape returns 0", RuntimeWarning, stacklevel=2)
        return zeros_like(value)
    return reduce_(_std, value, dim)


def _std(value: Tensor, dims: Shape) -> Tensor:
    if value.shape.is_uniform:
        result = value.backend.std(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names))
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    else:
        non_uniform_dims = value.shape.non_uniform_shape
        if non_uniform_dims.only(dims).is_empty:  # reduce uniform dims only
            return stack([_std(t, dims) for t in value._unstack(non_uniform_dims.name)], non_uniform_dims)
        else:
            mean_val = mean(value, dims)
            diff = value - mean_val
            variance = sum_(diff ** 2, dims) / dims.volume
            return sqrt(variance)


def any_(boolean_value, dim: DimFilter = non_batch) -> Tensor:
    """
    Tests whether any entry of `boolean_tensor` is `True` along the specified dimensions.

    Args:
        boolean_value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_any, boolean_value, dim, required_kind=bool)


def _any(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, Dense):
        result = value.backend.any(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names))
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    elif isinstance(value, TensorStack):
        reduced_inners = [_any(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x | y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif is_sparse(value):
        return sparse_sum(to_int32(value), dims) > 0
    else:
        raise ValueError(type(value))


def all_(boolean_value, dim: DimFilter = non_batch) -> Tensor:
    """
    Tests whether all entries of `boolean_tensor` are `True` along the specified dimensions.

    Args:
        boolean_value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_all, boolean_value, dim, required_kind=bool)


def _all(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, Dense):
        result = value.backend.all(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names))
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    elif isinstance(value, TensorStack):
        reduced_inners = [_all(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x & y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif is_sparse(value):
        if value.dtype.kind != bool:
            value = value != 0
        if sparse_dims(value) in dims:
            values = stored_values(value, list_dim=instance('_entries'))
            return _all(values, dims.without(sparse_dims(value)) & instance(values))
        return sparse_sum(to_int32(~value), dims) == 0
    raise ValueError(type(value))


def max_(value: TensorOrTree, dim: DimFilter = non_batch, key: Tensor = None) -> TensorOrTree:
    """
    Determines the maximum value of `values` along the specified dimensions.

    Args:
        value: (Sparse) `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        key: Optional comparison values. If specified, returns the value where `key` is maximal, see `at_max()`.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    if key is not None:
        return at_max(value, key, dim)
    return reduce_(_max, value, dim)


dmax = functools.partial(max_, dim=dual)
dmax.__doc__ = """Compute the maximum along dual dims of `value`, see `phiml.math.max`."""

imax = functools.partial(max_, dim=instance)
imax.__doc__ = """Compute the maximum along instance dims of `value`, see `phiml.math.max`."""

smax = functools.partial(max_, dim=spatial)
smax.__doc__ = """Compute the maximum along spatial dims of `value`, see `phiml.math.max`."""

cmax = functools.partial(max_, dim=channel)
cmax.__doc__ = """Compute the maximum along channel dims of `value`, see `phiml.math.max`."""


def _max(value: Tensor, dims: Shape) -> Tensor:
    if value.shape.volume == 0:
        return zeros(value.shape.without(dims), dtype=value.dtype)
    if isinstance(value, Dense):
        result = value.backend.max(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names))
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    elif isinstance(value, TensorStack):
        reduced_inners = [_max(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: maximum(x, y), reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif is_sparse(value):
        return sparse_max(value, dims)
    raise ValueError(type(value))


def min_(value, dim: DimFilter = non_batch, key: Tensor = None) -> Tensor:
    """
    Determines the minimum value of `values` along the specified dimensions.

    Args:
        value: (Sparse) `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        key: Optional comparison values. If specified, returns the value where `key` is minimal, see `at_min()`.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    if key is not None:
        return at_min(value, key, dim)
    return reduce_(_min, value, dim)


dmin = functools.partial(min_, dim=dual)
dmin.__doc__ = """Compute the minimum along dual dims of `value`, see `phiml.math.min`."""

imin = functools.partial(min_, dim=instance)
imin.__doc__ = """Compute the minimum along instance dims of `value`, see `phiml.math.min`."""

smin = functools.partial(min_, dim=spatial)
smin.__doc__ = """Compute the minimum along spatial dims of `value`, see `phiml.math.min`."""

cmin = functools.partial(min_, dim=channel)
cmin.__doc__ = """Compute the minimum along channel dims of `value`, see `phiml.math.min`."""


def _min(value: Tensor, dims: Shape) -> Tensor:
    if value.shape.volume == 0:
        return zeros(value.shape.without(dims), dtype=value.dtype)
    if isinstance(value, Dense):
        result = value.backend.min(value._native, tuple(value._names.index(n) for n in dims.names if n in value._names))
        return Dense(result, [n for n in value._names if n not in dims], value.shape - dims, value._backend)
    elif isinstance(value, TensorStack):
        reduced_inners = [_min(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: minimum(x, y), reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif is_sparse(value):
        return sparse_min(value, dims)
    raise ValueError(type(value))


def finite_min(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Finds the minimum along `dim` ignoring all non-finite values.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    value_inf = where(is_finite(value), value, float('inf'))
    result_inf = min_(value_inf, dim)
    return where(is_finite(result_inf), result_inf, default)


def finite_max(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Finds the maximum along `dim` ignoring all non-finite values.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    value_inf = where(is_finite(value), value, float('-inf'))
    result_inf = max_(value_inf, dim)
    return where(is_finite(result_inf), result_inf, default)


def finite_sum(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Sums all finite values in `value` along `dim`.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    finite = is_finite(value)
    summed = sum_(where(finite, value, 0), dim)
    return where(any_(finite, dim), summed, default)


def finite_mean(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Computes the mean value of all finite values in `value` along `dim`.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    finite = is_finite(value)
    summed = sum_(where(finite, value, 0), dim)
    count = sum_(finite, dim)
    mean_nan = summed / count
    return where(is_finite(mean_nan), mean_nan, default)


def at_max(value, key: Tensor, dim: DimFilter = non_batch):
    """
    Looks up the values of `value` at the positions where the maximum values in `key` are located along `dim`.

    See Also:
        `at_min`, `phiml.math.max`.

    Args:
        value: Tensors or trees from which to lookup and return values. These tensors are indexed at the maximum index in `key´.
            You can pass `range` (the type) to retrieve the picked indices.
        key: `Tensor` containing at least one dimension of `dim`. The maximum index of `key` is determined.
        dim: Dimensions along which to compute the maximum of `key`.

    Returns:
        The values of `other_tensors` at the positions where the maximum values in `value` are located along `dim`.
    """
    if not shape(key).only(dim):
        return value
    idx = argmax(key, dim)
    return slice_(value, idx)


def at_min(value, key: Tensor, dim: DimFilter = non_batch):
    """
    Looks up the values of `value` at the positions where the minimum values in `key` are located along `dim`.

    See Also:
        `at_max`, `phiml.math.min`.

    Args:
        value: Tensors or trees from which to lookup and return values. These tensors are indexed at the minimum index in `key´.
            You can pass `range` (the type) to retrieve the picked indices.
        key: `Tensor` containing at least one dimension of `dim`. The minimum index of `key` is determined.
        dim: Dimensions along which to compute the minimum of `key`.

    Returns:
        The values of `other_tensors` at the positions where the minimum values in `value` are located along `dim`.
    """
    if not shape(key).only(dim):
        return value
    idx = argmin(key, dim)
    return slice_(value, idx)


def argmax(x: Tensor, dim: DimFilter, index_dim=channel('index')):
    """
    Finds the maximum value along one or multiple dimensions and returns the corresponding index.

    See Also:
        `argmin`, `at_max`.

    Args:
        x: `Tensor`
        dim: Dimensions along which the maximum should be determined. These are reduced in the operation.
        index_dim: Dimension listing the index components for multidimensional argmax.

    Returns:
        Index tensor `idx`, such that `x[idx] = max(x)`.
    """
    dims = x.shape.only(dim)
    keep = x.shape.without(dims)
    assert dims, f"argmax requires dim to be present on data but {dim} does not exist on {x.shape}"
    if is_sparse(x):
        if dims in sparse_dims(x):
            max_val = max_(x, dim)
            is_max = x == max_val
            is_max_idx = nonzero(is_max, list_dim=instance('true_values'))
            scatter_val = is_max_idx[dims.only(sparse_dims(x)).name_list]
            remaining_dims = sparse_dims(x).without(dims)
            result_shape = max_val.shape & channel(scatter_val)
            if remaining_dims:
                scatter_idx = is_max_idx[remaining_dims.name_list]
                result = scatter(result_shape, scatter_idx, scatter_val, mode='update', default=-1)
            else:  # all sparse dims are reduced
                result = scatter_val.true_values[0]
            return rename_dims(result, channel(scatter_val), index_dim.with_sizes(dims.name_list))
        elif dims.isdisjoint(sparse_dims(x)):  # only argmax across values dim
            return x._with_values(argmax(x._values, dims))
        else:
            raise NotImplementedError
    broadcast = broadcast_dims(x)
    def uniform_argmin(x: Tensor):
        dims = x.shape.only(dim)
        v_native = x._reshaped_native([keep - broadcast, dims])
        idx_native = x.backend.argmax(v_native, 1, keepdims=True)
        multi_idx_native = choose_backend(idx_native).unravel_index(idx_native[:, 0], dims.sizes)
        return reshaped_tensor(multi_idx_native, [keep - broadcast, index_dim.with_size(dims.name_list)])
    return broadcast_op(uniform_argmin, [x], broadcast)


def argmin(x: Tensor, dim: DimFilter, index_dim=channel('index')):
    """
    Finds the minimum value along one or multiple dimensions and returns the corresponding index.

    See Also:
        `argmax`, `at_min`.

    Args:
        x: `Tensor`
        dim: Dimensions along which the minimum should be determined. These are reduced in the operation.
        index_dim: Dimension listing the index components for multidimensional argmin.

    Returns:
        Index tensor `idx`, such that `x[idx] = min(x)`.
    """
    dims = x.shape.only(dim)
    keep = x.shape.without(dims)
    assert dims, f"argmin requires dim to be present on data but {dim} does not exist on {x.shape}"
    if is_sparse(x):
        if dims in sparse_dims(x):
            min_val = min_(x, dim)
            is_min = x == min_val
            is_min_idx = nonzero(is_min, list_dim=instance('true_values'))
            scatter_val = is_min_idx[dims.only(sparse_dims(x)).name_list]
            remaining_dims = sparse_dims(x).without(dims)
            result_shape = min_val.shape & channel(scatter_val)
            if remaining_dims:
                scatter_idx = is_min_idx[remaining_dims.name_list]
                result = scatter(result_shape, scatter_idx, scatter_val, mode='update', default=-1)
            else:  # all sparse dims are reduced
                result = scatter_val.true_values[0]
            return rename_dims(result, channel(scatter_val), index_dim.with_sizes(dims.name_list))
        elif dims.isdisjoint(sparse_dims(x)):  # only argmin across values dim
            return x._with_values(argmin(x._values, dims))
        else:
            raise NotImplementedError
    broadcast = broadcast_dims(x)
    def uniform_argmin(x: Tensor):
        dims = x.shape.only(dim)
        v_native = x._reshaped_native([keep - broadcast, dims])
        idx_native = x.backend.argmin(v_native, 1, keepdims=True)
        multi_idx_native = choose_backend(idx_native).unravel_index(idx_native[:, 0], dims.sizes)
        return reshaped_tensor(multi_idx_native, [keep - broadcast, index_dim.with_size(dims.name_list)])
    return broadcast_op(uniform_argmin, [x], broadcast)


def quantile(value: Tensor,
             quantiles: Union[float, tuple, list, Tensor],
             dim: DimFilter = non_batch):
    """
    Compute the q-th quantile of `value` along `dim` for each q in `quantiles`.

    Implementations:

    * NumPy: [`quantile`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
    * PyTorch: [`quantile`](https://pytorch.org/docs/stable/generated/torch.quantile.html#torch.quantile)
    * TensorFlow: [`tfp.stats.percentile`](https://www.tensorflow.org/probability/api_docs/python/tfp/stats/percentile)
    * Jax: [`quantile`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.quantile.html)

    Args:
        value: `Tensor`
        quantiles: Single quantile or tensor of quantiles to compute.
            Must be of type `float`, `tuple`, `list` or `Tensor`.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to reduce the sequence of Tensors

    Returns:
        `Tensor` with dimensions of `quantiles` and non-reduced dimensions of `value`.
    """
    quantiles = wrap(quantiles, default_list_dim=instance('quantiles'))
    shared_dims = value.shape.only(quantiles.shape)
    broadcast = broadcast_dims(value, quantiles) | set(shared_dims.names)
    def uniform_quantile(value: Tensor, q: Tensor):
        dims = value.shape.only(dim)
        backend = value.backend
        if dims.volume == 0:
            return zeros((value.shape-dims) & q.shape, dtype=value.dtype) + float('nan')
        native_values = value._reshaped_native([*(value.shape-dims), dims])
        native_quantiles = q._reshaped_native([q.shape])
        native_result = backend.quantile(native_values, native_quantiles)
        if native_result is not NotImplemented:
            return reshaped_tensor(native_result, [q.shape, *value.shape.without(dims)])
        # --- fallback: custom quantile implementation ---
        v_sorted = sort(value, dims)
        q_idx = q * (v_sorted.shape.get_size(dims) - 1)
        q_idx = expand(q_idx, channel(vector=dims))
        result = grid_sample(v_sorted, q_idx, e_.ZERO_GRADIENT)
        return result
    return broadcast_op(uniform_quantile, [value, quantiles], broadcast)


def median(value, dim: DimFilter = non_batch):
    """
    Reduces `dim` of `value` by picking the median value.
    For odd dimension sizes (ambigous choice), the linear average of the two median values is computed.

    Currently implemented via `quantile()`.

    Args:
        value: `Tensor`
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor`
    """
    return reduce_(_median, value, dim)


def _median(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, Dense):
        return quantile(value, 0.5, dims)
    elif isinstance(value, TensorStack):
        reduced_inners = [_median(t, dims.without(value._stack_dim)) for t in value._tensors]
        if value._stack_dim in dims:
            raise NotImplementedError  # return median(reduced_inners)
        else:
            return TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def dot(x: Tensor,
        x_dims: DimFilter,
        y: Tensor,
        y_dims: DimFilter) -> Tensor:
    """
    Computes the dot product along the specified dimensions.
    Contracts `x_dims` with `y_dims` by first multiplying the elements and then summing them up.

    For one dimension, this is equal to matrix-matrix or matrix-vector multiplication.

    The function replaces the traditional `dot` / `tensordot` / `matmul` / `einsum` functions.

    * NumPy: [`numpy.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html), [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
    * PyTorch: [`torch.tensordot`](https://pytorch.org/docs/stable/generated/torch.tensordot.html#torch.tensordot), [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html)
    * TensorFlow: [`tf.tensordot`](https://www.tensorflow.org/api_docs/python/tf/tensordot), [`tf.einsum`](https://www.tensorflow.org/api_docs/python/tf/einsum)
    * Jax: [`jax.numpy.tensordot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tensordot.html), [`jax.numpy.einsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html)

    Args:
        x: First `Tensor`
        x_dims: Dimensions of `x` to reduce against `y`
        y: Second `Tensor`
        y_dims: Dimensions of `y` to reduce against `x`.

    Returns:
        Dot product as `Tensor`.
    """
    x_dims = x.shape.only(x_dims)
    y_dims = y.shape.only(y_dims)
    if not x_dims:
        return x * sum_(y, y_dims)
    if not y_dims:
        return sum_(x, x_dims) * y

    def tensor_dot(x, y):
        if is_sparse(x) or is_sparse(y):
            if x_dims.isdisjoint(sparse_dims(x)) and y_dims.isdisjoint(sparse_dims(y)):
                if is_sparse(x):
                    return x._op2(y, lambda vx, vy: dot(vx, x_dims, vy, y_dims), None, 'dot', '@')
                else:
                    return y._op2(x, lambda vy, vx: dot(vx, x_dims, vy, y_dims), None, 'dot', '@')
            else:
                return sparse_dot(x, x_dims, y, y_dims)
        if x._is_tracer:
            return x._matmul(x_dims, y, y_dims)
        if y._is_tracer:
            return y._matmul(y_dims, x, x_dims)
        x_native = x.native(x.shape)
        y_native = y.native(y.shape)
        backend = choose_backend(x_native, y_native)
        remaining_shape_x = x.shape.without(x_dims)
        remaining_shape_y = y.shape.without(y_dims)
        assert x_dims.volume == y_dims.volume, f"Failed to reduce {x_dims} against {y_dims} in dot product of {x.shape} and {y.shape}. Sizes do not match."
        if remaining_shape_y.isdisjoint(remaining_shape_x):  # no shared batch dimensions -> tensordot
            result_native = backend.tensordot(x_native, x.shape.indices(x_dims.names), y_native, y.shape.indices(y_dims.names))
            result_shape = remaining_shape_x + remaining_shape_y
        else:  # shared batch dimensions -> einsum
            result_shape = merge_shapes(x.shape.without(x_dims), y.shape.without(y_dims))
            REDUCE_LETTERS = list('ijklmn')
            KEEP_LETTERS = list('abcdefgh')
            x_letters = [(REDUCE_LETTERS if dim in x_dims else KEEP_LETTERS).pop(0) for dim in x.shape.names]
            letter_map = {dim: letter for dim, letter in zip(x.shape.names, x_letters)}
            REDUCE_LETTERS = list('ijklmn')
            y_letters = []
            for dim in y.shape.names:
                if dim in y_dims:
                    y_letters.append(REDUCE_LETTERS.pop(0))
                else:
                    if dim in x.shape and dim not in x_dims:
                        y_letters.append(letter_map[dim])
                    else:
                        next_letter = KEEP_LETTERS.pop(0)
                        letter_map[dim] = next_letter
                        y_letters.append(next_letter)
            keep_letters = [letter_map[dim] for dim in result_shape.names]
            subscripts = f'{"".join(x_letters)},{"".join(y_letters)}->{"".join(keep_letters)}'
            result_native = backend.einsum(subscripts, x_native, y_native)
        return Dense(result_native, result_shape.names, result_shape, backend)

    return broadcast_op(tensor_dot, [x, y])


def _backend_op1(x, unbound_method, attr_type=value_attributes) -> Union[Tensor, PhiTreeNode]:
    if isinstance(x, Tensor) and x.dtype.kind != object:
        def apply_op(native_tensor):
            backend = choose_backend(native_tensor)
            return getattr(backend, unbound_method.__name__)(backend.auto_cast(native_tensor)[0])
        apply_op.__name__ = unbound_method.__name__
        return x._op1(apply_op)
    elif x is None:
        return None
    elif isinstance(x, (PhiTreeNode, Layout, tuple, list, dict)):
        return tree_map(_backend_op1, x, unbound_method=unbound_method, attr_type=attr_type)
    else:
        backend = choose_backend(x)
        y = getattr(backend, unbound_method.__name__)(backend.auto_cast(x)[0])
        return y


def abs_(x: TensorOrTree) -> TensorOrTree:
    """
    Computes *||x||<sub>1</sub>*.
    Complex `x` result in matching precision float values.

    *Note*: The gradient of this operation is undefined for *x=0*.
    TensorFlow and PyTorch return 0 while Jax returns 1.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode`

    Returns:
        Absolute value of `x` of same type as `x`.
    """
    return _backend_op1(x, Backend.abs)


def sign(x: TensorOrTree) -> TensorOrTree:
    """
    The sign of positive numbers is 1 and -1 for negative numbers.
    The sign of 0 is undefined.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode`

    Returns:
        `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.sign)


def round_(x: TensorOrTree) -> TensorOrTree:
    """ Rounds the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` to the closest integer. """
    return _backend_op1(x, Backend.round)


def ceil(x: TensorOrTree) -> TensorOrTree:
    """ Computes *⌈x⌉* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.ceil)


def floor(x: TensorOrTree) -> TensorOrTree:
    """ Computes *⌊x⌋* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.floor)


def sqrt(x: TensorOrTree) -> TensorOrTree:
    """ Computes *sqrt(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sqrt)


def exp(x: TensorOrTree) -> TensorOrTree:
    """ Computes *exp(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.exp)


def erf(x: TensorOrTree) -> TensorOrTree:
    """ Computes the error function *erf(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.erf)


def soft_plus(x: TensorOrTree) -> TensorOrTree:
    """ Computes *softplus(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.softplus)


def factorial(x: TensorOrTree) -> TensorOrTree:
    """
    Computes *factorial(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.
    For floating-point numbers computes the continuous factorial using the gamma function.
    For integer numbers computes the exact factorial and returns the same integer type.
    However, this results in integer overflow for inputs larger than 12 (int32) or 19 (int64).
    """
    return _backend_op1(x, Backend.factorial)


def log_gamma(x: TensorOrTree) -> TensorOrTree:
    """ Computes *log(gamma(x))* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.log_gamma)


def incomplete_gamma(a: TensorOrTree, x: TensorOrTree, upper=False, regularized=True) -> TensorOrTree:
    """
    Computes the incomplete gamma function.

    Args:
        a: Positive parameter, `Tensor` or tree.
        x: Non-negative argument, `Tensor` or tree.
        upper: Whether to complete the upper integral (x to infinity) or the lower integral (0 to x).
        regularized: Whether the integral is divided by Γ(a).
    """
    call = lambda a, x: incomplete_gamma(a, x, upper=upper, regularized=regularized)
    if upper:
        reg = custom_op2(a, x, call, lambda a, x: choose_backend(a, x).gamma_inc_u(a, x), 'gamma_inc_u')
    else:
        reg = custom_op2(a, x, call, lambda a, x: choose_backend(a, x).gamma_inc_l(a, x), 'gamma_inc_l')
    return reg if regularized else reg * exp(log_gamma(a))


def to_float(x: TensorOrTree) -> TensorOrTree:
    """
    Converts the given tensor to floating point format with the currently specified precision.
    
    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.
    
    See the documentation at https://tum-pbs.github.io/PhiML/Data_Types.html

    See Also:
        `cast()`.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` to convert

    Returns:
        `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.to_float)


def to_int32(x: TensorOrTree) -> TensorOrTree:
    """ Converts the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` to 32-bit integer. """
    return _backend_op1(x, Backend.to_int32)


def to_int64(x: TensorOrTree) -> TensorOrTree:
    """ Converts the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` to 64-bit integer. """
    return _backend_op1(x, Backend.to_int64)


def to_complex(x: TensorOrTree) -> TensorOrTree:
    """
    Converts the given tensor to complex floating point format with the currently specified precision.

    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.

    See the documentation at https://tum-pbs.github.io/PhiML/Data_Types.html

    See Also:
        `cast()`.

    Args:
        x: values to convert

    Returns:
        `Tensor` of same shape as `x`
    """
    return _backend_op1(x, Backend.to_complex)


def is_finite(x: TensorOrTree) -> TensorOrTree:
    """ Returns a `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x` with values `True` where `x` has a finite value and `False` otherwise. """
    return _backend_op1(x, Backend.isfinite)


def is_nan(x: TensorOrTree) -> TensorOrTree:
    """ Returns a `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x` with values `True` where `x` is `NaN` and `False` otherwise. """
    return _backend_op1(x, Backend.isnan)


def is_inf(x: TensorOrTree) -> TensorOrTree:
    """ Returns a `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x` with values `True` where `x` is `+inf` or `-inf` and `False` otherwise. """
    return _backend_op1(x, Backend.isnan)


def nan_to_0(x: TensorOrTree) -> TensorOrTree:
    """Replaces all NaN values in `x` with `0`."""
    return where(is_nan(x), 0, x)


def real(x: TensorOrTree) -> TensorOrTree:
    """
    See Also:
        `imag()`, `conjugate()`.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Real component of `x`.
    """
    return _backend_op1(x, Backend.real)


def imag(x: TensorOrTree) -> TensorOrTree:
    """
    Returns the imaginary part of `x`.
    If `x` does not store complex numbers, returns a zero tensor with the same shape and dtype as this tensor.

    See Also:
        `real()`, `conjugate()`.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Imaginary component of `x` if `x` is complex, zeros otherwise.
    """
    return _backend_op1(x, Backend.imag)


def conjugate(x: TensorOrTree) -> TensorOrTree:
    """
    See Also:
        `imag()`, `real()`.

    Args:
        x: Real or complex `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Complex conjugate of `x` if `x` is complex, else `x`.
    """
    return _backend_op1(x, Backend.conj)


def degrees_to_radians(deg: TensorOrTree) -> TensorOrTree:
    """ Convert degrees to radians. """
    return tree_map(lambda x: x * (3.14159265358979323846 / 180), deg)


def radians_to_degrees(rad: TensorOrTree) -> TensorOrTree:
    """ Convert degrees to radians. """
    return tree_map(lambda x: x * (180 / 3.14159265358979323846), rad)


def sin(x: TensorOrTree) -> TensorOrTree:
    """ Computes *sin(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sin)


def arcsin(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *sin(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.
    For real arguments, the result lies in the range [-π/2, π/2].
    """
    return _backend_op1(x, Backend.arcsin)


def cos(x: TensorOrTree) -> TensorOrTree:
    """ Computes *cos(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.cos)


def arccos(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *cos(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.
    For real arguments, the result lies in the range [0, π].
    """
    return _backend_op1(x, Backend.arccos)


def tan(x: TensorOrTree) -> TensorOrTree:
    """ Computes *tan(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.tan)


def arctan(x: TensorOrTree, divide_by=None) -> TensorOrTree:
    """
    Computes the inverse of *tan(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.

    Args:
        x: Input. The single-argument `arctan` function cannot output π/2 or -π/2 since tan(π/2) is infinite.
        divide_by: If specified, computes `arctan(x/divide_by)` so that it can return π/2 and -π/2.
            This is equivalent to the common `arctan2` function.
    """
    if divide_by is None:
        return _backend_op1(x, Backend.arctan)
    else:
        divide_by = to_float(divide_by)
        return custom_op2(x, divide_by, arctan, lambda a, b: choose_backend(a, b).arctan2(a, b), 'arctan')


def angle(x: TensorOrTree) -> TensorOrTree:
    """
    Compute the angle of a complex number.
    This is equal to *atan(Im/Re)* for most values.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode`

    Returns:
        Angle of complex number in radians.
    """
    return arctan(imag(x), divide_by=real(x))


def sinh(x: TensorOrTree) -> TensorOrTree:
    """ Computes *sinh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sinh)


def arcsinh(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *sinh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arcsinh)


def cosh(x: TensorOrTree) -> TensorOrTree:
    """ Computes *cosh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.cosh)


def arccosh(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *cosh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arccosh)


def tanh(x: TensorOrTree) -> TensorOrTree:
    """ Computes *tanh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.tanh)


def arctanh(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *tanh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arctanh)


def log(x: TensorOrTree) -> TensorOrTree:
    """ Computes the natural logarithm of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.log)


def log2(x: TensorOrTree) -> TensorOrTree:
    """ Computes *log(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` with base 2. """
    return _backend_op1(x, Backend.log2)


def log10(x: TensorOrTree) -> TensorOrTree:
    """ Computes *log(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` with base 10. """
    return _backend_op1(x, Backend.log10)


def sigmoid(x: TensorOrTree) -> TensorOrTree:
    """ Computes the sigmoid function of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sigmoid)


def softmax(x, reduce: DimFilter):
    """Compute the softmax of `x` over any dimension. The softmax is e^x / ∑ e^x ."""
    e = exp(x)
    return e / sum_(e, reduce)


def cast_same(*values: Tensor) -> Tuple[Tensor]:
    """
    Casts all tensors to the same `DType`.
    If all data types are of the same kind, returns the largest occurring data type.
    Otherwise casts `bool` &rarr; `int` &rarr; `float` &rarr; `complex`.

    Args:
        *values: tensors to cast

    Returns:
        Tuple of Tensors with same data type.
    """
    assert all(isinstance(v, Tensor) for v in values), f"Only Tensor arguments allowed but got {values}"
    dtypes = [v.dtype for v in values]
    if any(dt != dtypes[0] for dt in dtypes):
        common_type = combine_types(*dtypes, fp_precision=get_precision())
        return tuple([cast(v, common_type) for v in values])
    else:
        return values


def safe_div(x: Union[Number, Tensor], y: Union[Number, Tensor]):
    """ Computes *x/y* with the `Tensor`s `x` and `y` but returns 0 where *y=0*. """
    return custom_op2(x, y,
                      l_operator=safe_div,
                      l_native_function=lambda x_, y_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      r_operator=lambda y_, x_: safe_div(x_, y_),
                      r_native_function=lambda y_, x_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      op_name='divide_no_nan')


def maximum(x: Union[Tensor, float], y: Union[Tensor, float], allow_none=False):
    """ Computes the element-wise maximum of `x` and `y`. """
    if allow_none:
        if x is None:
            return y
        elif y is None:
            return x
    return custom_op2(x, y, maximum, lambda x_, y_: choose_backend(x_, y_).maximum(x_, y_), op_name='maximum')


def minimum(x: Union[Tensor, float], y: Union[Tensor, float], allow_none=False):
    """ Computes the element-wise minimum of `x` and `y`. """
    if allow_none:
        if x is None:
            return y
        elif y is None:
            return x
    return custom_op2(x, y, minimum, lambda x_, y_: choose_backend(x_, y_).minimum(x_, y_), op_name='minimum')


def clip(x: Tensor, lower_limit: Union[float, Tensor] = 0, upper_limit: Union[float, Tensor, Shape] = 1):
    """ Limits the values of the `Tensor` `x` to lie between `lower_limit` and `upper_limit` (inclusive). """
    if isinstance(upper_limit, SHAPE_TYPES):
        assert x.shape.channel_rank == 1, f"When passing a Shape for upper_limit, x must have exactly one channel dim but got {x.shape}"
        upper_limit = wrap(upper_limit.sizes, channel(x))
    if isinstance(lower_limit, Number) and isinstance(upper_limit, Number):
        def clip_(x):
            return x._op1(lambda native: choose_backend(native).clip(native, lower_limit, upper_limit))
        return broadcast_op(clip_, [x])
    else:
        return maximum(lower_limit, minimum(x, upper_limit))


def convolve(value: Tensor,
             kernel: Tensor,
             extrapolation: 'e_.Extrapolation' = None,
             dims: DimFilter = spatial) -> Tensor:
    """
    Computes the convolution of `value` and `kernel` along the specified dims.

    Dual dims of `kernel` are reduced against the corresponding primal dims of `value`.
    All other primal dims of `value` are treated as batch.

    Args:
        value: `Tensor` whose shape includes all spatial dimensions of `kernel`.
        kernel: `Tensor` used as convolutional filter.
        dims: Which dimensions to convolve over. Defaults to all spatial dims.
        extrapolation: If not None, pads `value` so that the result has the same shape as `value`.

    Returns:
        `Tensor` with all non-reduced dims of `value` and additional non-dual dims from `kernel`.
    """
    assert all(dim in value.shape for dim in kernel.shape.spatial.names), f"Value must have all spatial dimensions of kernel but got value {value} kernel {kernel}"
    dims = kernel.shape.only(dims)
    assert dims.dual_rank == 0, f"convolve dims must not be of type dual but got {dims}"
    in_dims = value.shape.only(dual(kernel).as_batch().names)
    out_dims = non_dual(kernel) - dims - batch(value)
    batch_dims = (value.shape - dims - in_dims) & (non_dual(kernel) - dims - out_dims)
    if extrapolation is not None and extrapolation != e_.ZERO:
        value = pad(value, {dim: (kernel.shape.get_size(dim) // 2, (kernel.shape.get_size(dim) - 1) // 2) for dim in dims.names}, extrapolation)
    backend = backend_for(value, kernel)
    native_kernel = kernel.native((batch_dims if batch(kernel) else EMPTY_SHAPE, out_dims, dual(kernel), *dims))
    native_value = value.native((batch_dims, in_dims, *dims.names))
    native_result = backend.conv(native_value, native_kernel, zero_padding=extrapolation == e_.ZERO)
    result = reshaped_tensor(native_result, (batch_dims, out_dims, *dims))
    return result


def boolean_mask(x, dim: DimFilter, mask: Tensor, preserve_names=False):
    """
    Discards values `x.dim[i]` where `mask.dim[i]=False`.
    All dimensions of `mask` that are not `dim` are treated as batch dimensions.

    Alternative syntax: `x.dim[mask]`.

    Implementations:

    * NumPy: Slicing
    * PyTorch: [`masked_select`](https://pytorch.org/docs/stable/generated/torch.masked_select.html)
    * TensorFlow: [`tf.boolean_mask`](https://www.tensorflow.org/api_docs/python/tf/boolean_mask)
    * Jax: Slicing

    Args:
        x: `Tensor` or `phiml.math.magic.Sliceable`.
        dim: Dimension of `x` to along which to discard slices.
        mask: Boolean `Tensor` marking which values to keep. Must have the dimension `dim` matching `x´.
        preserve_names: This only supports uniform 1D slicing. Batched slicing will remove item names if incompatible.

    Returns:
        Selected values of `x` as `Tensor` with dimensions from `x` and `mask`.
    """
    if x is None:
        return None
    dim, original_dim = shape(mask).only(dim), dim
    assert dim, f"mask dimension '{original_dim}' must be present on the mask {mask.shape}"
    assert dim.rank == 1, f"boolean mask only supports 1D selection"
    if not isinstance(x, Tensor) and isinstance(x, PhiTreeNode):
        return tree_map(boolean_mask, x, all_attributes, dim=dim, mask=mask, preserve_names=preserve_names, include_non_attrs=False, treat_layout_as_leaf=True)
    if isinstance(x, Layout):
        if x._stack_dim.without(dim):
            from ._functional import map_
            return map_(boolean_mask, x, dims=x._stack_dim - dim, dim=dim - x._stack_dim, mask=mask, preserve_names=preserve_names)
        if dim in x._stack_dim:
            indices = np.nonzero(mask.numpy())[0]
            gathered = [x._obj[i] for i in indices]
            size = len(gathered) if not preserve_names or x._stack_dim.item_names[0] is None else [x._stack_dim.item_names[0][i] for i in indices]
            return Layout(gathered, dim.with_size(size))
        raise NotImplementedError
    if is_sparse(x):
        indices = nonzero(mask, list_dim=instance('_boolean_mask'))
        result = x[indices]
        return result.__replace_dims__(('_boolean_mask',), mask.shape.non_channel)
    if not isinstance(x, Tensor) or is_sparse(x):
        keep_slices = nonzero_slices(mask)
        x_slices = [x[s] for s in keep_slices]
        return concat(x_slices, dim.name)
    if isinstance(x, TensorStack) and dim.name in broadcast_dims(x):
        indices = np.nonzero(mask.numpy())[0]
        items = x._unstack(dim.name)
        return TensorStack([items[i] for i in indices], dim)
    
    def uniform_boolean_mask(x: Tensor, mask_1d: Tensor):
        if dim in x.shape:
            x_native = x.native(x.shape.names)  # order does not matter
            mask_native = mask_1d.native()  # only has 1 dim
            backend = choose_backend(x_native, mask_native)
            result_native = backend.boolean_mask(x_native, mask_native, axis=x.shape.index(dim))
            new_shape = x.shape.with_sizes(backend.staticshape(result_native))  # ToDo add selected item names!!!
            if preserve_names and dim.item_names[0]:
                sel_names = [n for n, sel in zip(dim.item_names[0], mask_native) if sel]
                new_shape = new_shape.with_dim_size(dim, sel_names)
            return Dense(result_native, new_shape.names, new_shape, backend)
        else:
            total = int(sum_(to_int64(mask_1d), mask_1d.shape))
            new_shape = mask_1d.shape.with_sizes([total])
            return expand(x, new_shape)

    return broadcast_op(uniform_boolean_mask, [x, mask], iter_dims=set(mask.shape.without(dim).names) | broadcast_dims(x, mask))


def gather(values, indices: Tensor, dims: Union[DimFilter, None] = None, pref_index_dim='index'):
    """
    Gathers the entries of `values` at positions described by `indices`.
    All non-channel dimensions of `indices` that are part of `values` but not indexed are treated as batch dimensions.

    See Also:
        `scatter()`.

    Args:
        values: `Tensor` or `phiml.math.matic.PhiTreeNode` containing values to gather.
        indices: `int` `Tensor`. Multidimensional position references in `values`.
            Must contain a single channel dimension for the index vector matching the number of dimensions to index.
            This channel dimension should list the dimension names to index as item names unless explicitly specified as `dims`.
        dims: (Optional) Dimensions indexed by `indices`.
            Alternatively, the dimensions can be specified as the item names of the channel dimension of `indices`.
            If `None` and no index item names are specified, will default to all spatial dimensions or all instance dimensions, depending on which ones are present (but not both).
        pref_index_dim: In case `indices` has multiple channel dims, use this dim as the index, treating the others as batch.
            Has no effect if `indices` only has one channel dim.

    Returns:
        `Tensor` with combined batch dimensions, channel dimensions of `values` and spatial/instance dimensions of `indices`.
    """
    if values is None:
        return None
    if not isinstance(values, Tensor):
        return tree_map(lambda v: gather(v, indices, dims), values)
    index_dim = channel(indices)
    if index_dim.rank >= 2:
        assert pref_index_dim in index_dim, f"When indices has multiple channel dims, pref_index_dim must select one of them but got {pref_index_dim} which is not in {index_dim}"
        index_dim = index_dim.only(pref_index_dim)
    if dims is None:
        if index_dim and index_dim.item_names[0]:
            dims = index_dim.item_names[0]
        else:  # Fallback to spatial / instance
            assert values.shape.instance.is_empty or values.shape.spatial.is_empty, f"Specify gather dimensions for values with both instance and spatial dimensions. Got {values.shape}"
            dims = values.shape.instance if values.shape.spatial.is_empty else values.shape.spatial
            assert dims, f"Specify gather dimensions for values with neither instance nor spatial dimensions. Got {values.shape}"
    dims = parse_dim_order(dims)
    assert dims, f"No indexing dimensions for tensor {values.shape} given indices {indices.shape}"
    if dims not in values.shape:
        return expand(values, indices.shape - index_dim)
    if len(dims) > 1:
        assert index_dim.rank == 1, f"indices must have a single channel dimension listing the indexed dims {dims} but got {indices.shape}."
    assert index_dim.volume == len(dims), f"channel dim of indices must have size equal to the number of indexed dims {dims} but got {index_dim} which has {index_dim.volume} entries"
    if indices.dtype.kind == bool:
        indices = to_int32(indices)
    if isinstance(values, Layout) and dims in values._stack_dim:
        index_list = unstack(rename_dims(indices, index_dim, 'index_'), indices.shape - index_dim)
        v_list = [values[{n: int(v) for n, v in zip(index_dim.item_names[0], i)}] for i in index_list]
        return stack(v_list, indices.shape - index_dim)
    if values._is_tracer or is_sparse(values):
        if not index_dim:
            index_dim = channel(gather=dims)
            indices = expand(indices, index_dim)
        if not index_dim.item_names[0]:
            indices = indices._with_shape_replaced(indices.shape.with_dim_size(index_dim, dims))
        if values._is_tracer:
            return values._gather(indices)
        if is_sparse(values):
            if isinstance(values, TensorStack):
                if dims in values._stack_dim:
                    gathered = [values[{dims[0]: i}] for i in indices]
                    return stack(gathered, indices.shape-index_dim)
                raise NotImplementedError
            return sparse_gather(values, indices, index_dim)
    elif is_sparse(indices):  # only indices sparse -> gather on sparse pattern
        gathered = gather(values, indices._values, dims=dims, pref_index_dim=index_dim)
        return indices._with_values(gathered)
    broadcast = broadcast_dims(values, indices)
    treat_as_batch = indices.shape.only(values.shape) - dims - index_dim
    batch_ = ((values.shape.batch & indices.shape.batch).without(dims) & treat_as_batch) - broadcast
    channel_ = values.shape - dims - batch_ - broadcast
    if broadcast.intersection(set(dims)):  # Cannot broadcast because that would iterate over dims!
        if values.shape.is_uniform:
            broadcast = broadcast - set(dims)
        else:  # We have to slice the items, then stack the results
            # if batch_ or treat_as_batch:
            #     raise NotImplementedError  # ToDo iterate over batches
            result = []
            for single_index in unstack(indices, indices.shape - index_dim):
                index_slice = {d: i for d, i in zip(index_dim.item_names[0], single_index)}
                result.append(values[index_slice])
            return stack(result, indices.shape - index_dim)
    def uniform_gather(values: Tensor, indices: Tensor):
        index_list_dims = indices.shape - index_dim - batch_
        channel_ = values.shape - dims - batch_ - broadcast
        squeeze_index_list = False
        if not index_list_dims:
            index_list_dims = instance(_single_index=1)
            squeeze_index_list = True
        backend = backend_for(values, indices)
        native_values = values.native([batch_, *dims, channel_], True)
        native_indices = indices._reshaped_native([batch_, *index_list_dims, index_dim])
        native_result = backend.batched_gather_nd(native_values, native_indices)
        result = reshaped_tensor(native_result, [batch_, *index_list_dims, channel_], convert=False)
        if squeeze_index_list:
            result = result[{'_single_index': 0}]
        return result
    return broadcast_op(uniform_gather, [values, indices], iter_dims=broadcast)


def scatter(base_grid: Union[Tensor, Shape],
            indices: Union[Tensor, dict],
            values: Union[Tensor, float],
            mode: Union[str, Callable] = 'update',
            outside_handling: str = 'check',
            indices_gradient=False,
            default=None,
            treat_as_batch=None):
    """
    Scatters `values` into `base_grid` at `indices`.
    instance dimensions of `indices` and/or `values` are reduced during scattering.
    Depending on `mode`, this method has one of the following effects:

    * `mode='update'`: Replaces the values of `base_grid` at `indices` by `values`. The result is undefined if `indices` contains duplicates.
    * `mode='add'`: Adds `values` to `base_grid` at `indices`. The values corresponding to duplicate indices are accumulated.
    * `mode='mean'`: Replaces the values of `base_grid` at `indices` by the mean of all `values` with the same index.

    Implementations:

    * NumPy: Slice assignment / `numpy.add.at`
    * PyTorch: [`torch.scatter`](https://pytorch.org/docs/stable/generated/torch.scatter.html), [`torch.scatter_add`](https://pytorch.org/docs/stable/generated/torch.scatter_add.html)
    * TensorFlow: [`tf.tensor_scatter_nd_add`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_add), [`tf.tensor_scatter_nd_update`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update)
    * Jax: [`jax.lax.scatter_add`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter_add.html), [`jax.lax.scatter`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter.html)

    See Also:
        `gather()`.

    Args:
        base_grid: `Tensor` into which `values` are scattered.
        indices: `Tensor` of n-dimensional indices at which to place `values`.
            Must have a single channel dimension with size matching the number of spatial dimensions of `base_grid`.
            This dimension is optional if the spatial rank is 1.
            Must also contain all `scatter_dims`.
        values: `Tensor` of values to scatter at `indices`.
        mode: Scatter mode as `str` or function.
            Supported modes are 'add', 'mean', 'update', 'max', 'min', 'prod', 'any', 'all'.
            The corresponding functions are the built-in `sum`, `max´, `min`, as well as the reduce functions in `phiml.math`.
        outside_handling: Defines how indices lying outside the bounds of `base_grid` are handled.

            * `'check'`: Raise an error if any index is out of bounds.
            * `'discard'`: Outside indices are ignored.
            * `'clamp'`: Outside indices are projected onto the closest point inside the grid.
            * `'undefined'`: All points are expected to lie inside the grid. Otherwise an error may be thrown or an undefined tensor may be returned.
        indices_gradient: Whether to allow the gradient of this operation to be backpropagated through `indices`.
        default: Default value to use for bins into which no value is scattered.
            By default, `NaN` is used for the modes `update` and `mean`, `0` for `sum`, `inf` for min and `-inf` for max.
            This will upgrade the data type to `float` if necessary.
        treat_as_batch: Dimensions which should be treated like dims by this operation.
            This can be used for scattering vectors along instance dims into a grid.
            Normally, instance dims on `values` and `indices` would not be matched to `base_grid` but when treated as batch, they will be.

    Returns:
        Copy of `base_grid` with updated values at `indices`.
    """
    if callable(mode):
        mode = {sum: 'add', max: 'max', min: 'min', sum_: 'add', max_: 'max', min_: 'min', mean: 'mean', prod: 'prod', any_: 'any', all_: 'all'}[mode]
    if mode == 'prod':
        log_base_grid = log(base_grid) if isinstance(base_grid, Tensor) else base_grid
        log_default = None if default is None else log(default)
        log_result = scatter(log_base_grid, indices, log(values), 'add', outside_handling, indices_gradient, log_default)
        return exp(log_result)
    elif mode == 'any':
        b_base_grid = cast(base_grid, bool) if isinstance(base_grid, Tensor) else base_grid
        b_values = cast(values, bool)
        i_result = scatter(b_base_grid, indices, b_values, 'add', outside_handling, indices_gradient, False)
        return cast(i_result, bool)
    elif mode == 'all':
        not_base_grid = ~cast(base_grid, bool) if isinstance(base_grid, Tensor) else base_grid
        not_values = ~cast(values, bool)
        i_result = scatter(not_base_grid, indices, not_values, 'add', outside_handling, indices_gradient, False)
        return ~cast(i_result, bool)
    assert mode in ('update', 'add', 'mean', 'max', 'min'), f"Invalid scatter mode: '{mode}'"
    assert outside_handling in ('discard', 'clamp', 'undefined', 'check')
    assert isinstance(indices_gradient, bool)
    if isinstance(indices, dict):  # update a slice
        if len(indices) == 1 and isinstance(next(iter(indices.values())), (str, int, slice)):  # update a range
            dim, sel = next(iter(indices.items()))
            full_dim = base_grid.shape[dim]
            if isinstance(sel, str):
                sel = full_dim.item_names[0].index(sel)
            if isinstance(sel, int):
                sel = slice(sel, sel+1)
            assert isinstance(sel, slice), f"Selection must be a str, int or slice but got {type(sel)}"
            values = expand(values, full_dim.after_gather({dim: sel}))
            parts = [
                base_grid[{dim: slice(sel.start)}],
                values,
                base_grid[{dim: slice(sel.stop, None)}]
            ]
            return concat(parts, dim)
        else:
            raise NotImplementedError("scattering into non-continuous values not yet supported by dimension")
    grid_shape = base_grid if isinstance(base_grid, SHAPE_TYPES) else base_grid.shape
    assert channel(indices).rank < 2
    if channel(indices) and channel(indices).item_names[0]:
        indexed_dims = channel(indices).item_names[0]
        assert indexed_dims in grid_shape, f"Scatter indices {indices.shape} point to missing dimensions in grid {grid_shape}"
        if indexed_dims != grid_shape.only(indexed_dims).names:
            indices = indices[{channel: grid_shape.only(indexed_dims).names}]
        indexed_dims = grid_shape.only(indexed_dims)
    else:
        indexed_dims = grid_shape.spatial or grid_shape.instance
        assert channel(indices).rank == 1 or (grid_shape.spatial_rank + grid_shape.instance_rank == 1 and indices.shape.channel_rank == 0), f"indices must have a channel dimension listing the indexed dims {indexed_dims} but got {indices.shape}. You can create it via vec({', '.join([d+'=...' for d in indexed_dims.names])}) or channel(index='{','.join(indexed_dims.names)}'). If you have raveled indices, use unpack_dim(indices, channel, base_grid.shape['{','.join(indexed_dims.names)}'])."
        assert channel(indices).volume == indexed_dims.rank
    values = wrap(values)
    batches = values.shape.non_channel.non_instance & indices.shape.non_channel.non_instance
    batches &= values.shape.only(treat_as_batch) & indices.shape.only(treat_as_batch)
    # --- Set up grid ---
    if isinstance(base_grid, SHAPE_TYPES):
        with backend_for(indices, values):
            base_grid = zeros(base_grid & batches & values.shape.channel, dtype=values.dtype)
        if default is not None:
            base_grid += default
        elif mode in ['update', 'mean']:
            base_grid += float('nan')
        elif mode == 'max':
            base_grid -= float('inf')
        elif mode == 'min':
            base_grid += float('inf')
        else:
            assert mode == 'add'  # initialize with zeros
    # --- Handle outside indices ---
    if not channel(indices):
        indices = expand(indices, channel(_index=indexed_dims.name_list))
    limit = wrap(indexed_dims, channel(indices)) - 1
    if outside_handling == 'check':
        from ._functional import when_available
        def check(indices):
            assert_close(True, (indices >= 0) & (indices < (limit+1)))
        when_available(check, indices)
    elif outside_handling == 'clamp':
        indices = clip(indices, 0, limit)
    elif outside_handling == 'discard':
        indices_linear = pack_dims(indices, instance, instance(_scatter_instance=1))
        indices_inside = min_((round_(indices_linear) >= 0) & (round_(indices_linear) < wrap(indexed_dims, channel(indices_linear))), channel)
        indices_linear = boolean_mask(indices_linear, '_scatter_instance', indices_inside)
        if instance(values).rank > 0:
            values_linear = pack_dims(values, instance, instance(_scatter_instance=1))
            values_linear = boolean_mask(values_linear, '_scatter_instance', indices_inside)
            values = unpack_dim(values_linear, '_scatter_instance', instance(values))
        indices = unpack_dim(indices_linear, '_scatter_instance', instance(indices))
        if indices.shape.is_non_uniform:
            raise NotImplementedError()
    broadcast = broadcast_dims(base_grid, indices, values)
    def scatter_forward(base_grid: Tensor, indices: Tensor, values: Tensor, indexed_dims=indexed_dims):
        indexed_dims = base_grid.shape[indexed_dims] - broadcast
        batches = values.shape.non_channel.non_instance & indices.shape.non_channel.non_instance
        batches &= values.shape.only(treat_as_batch) & indices.shape.only(treat_as_batch)
        batches -= broadcast
        channels = (grid_shape - indexed_dims - batches - broadcast) & values.shape.channel
        lists = indices.shape.instance & values.shape.instance
        if values._is_tracer:
            if indices._is_tracer or base_grid._is_tracer:
                raise NotImplementedError("scattering linear tracer into linear tracer not supported")
            if not channel(indices):
                indices = expand(indices, channel(scatter_idx=indexed_dims))
            return values._scatter(base_grid, indices)
        indices = to_int32(round_(indices))
        backend = backend_for(indices, values, base_grid)
        native_grid = base_grid._reshaped_native([batches, *indexed_dims, channels])
        native_values = values._reshaped_native([batches, lists, channels])
        native_indices = indices._reshaped_native([batches, lists, indices.shape.channel])
        if mode != 'mean':
            native_result = backend.scatter(native_grid, native_indices, native_values, mode=mode)
        else:  # mean
            zero_grid = backend.zeros_like(native_grid)
            summed = backend.scatter(zero_grid, native_indices, native_values, mode='add')
            count = backend.scatter(zero_grid, native_indices, backend.ones_like(native_values), mode='add')
            native_result = summed / backend.maximum(count, 1)
            native_result = backend.where(count == 0, native_grid, native_result)
        return reshaped_tensor(native_result, [batches, *indexed_dims, channels], check_sizes=True, convert=False)

    def scatter_backward(args: dict, _output, d_output):
        from ._nd import spatial_gradient
        values_grad = gather(d_output, args['indices'])
        spatial_gradient_indices = gather(spatial_gradient(d_output, dims=indexed_dims), args['indices'])
        indices_grad = mean(spatial_gradient_indices * args['values'], 'vector_')
        return None, indices_grad, values_grad

    from ._functional import custom_gradient
    scatter_function = custom_gradient(scatter_forward, scatter_backward) if indices_gradient else scatter_forward
    result = broadcast_op(scatter_function, [base_grid, indices, values], broadcast)
    return result


def ravel_index(index: Tensor, resolution: Shape, dim=channel, mode='undefined') -> Tensor:
    """
    Computes a scalar index from a vector index.

    Args:
        index: `Tensor` with one channel dim.
        resolution: `Shape`
        mode: `'undefined'`, `'periodic'`, `'clamp'` or an `int` to use for all invalid indices.

    Returns:
        `Tensor`
    """
    index_dim = index.shape.only(dim)
    assert index_dim.rank == 1, f"index must have exaclty one index dim but got {index_dim}"
    nat_idx = index._reshaped_native([*(index.shape-index_dim), index_dim])
    if index_dim.item_names[0]:
        sizes = [resolution.get_size(dim) for dim in index_dim.item_names[0]]
    else:
        assert resolution.rank == index_dim.size
        sizes = resolution.sizes
    nat_result = index.backend.ravel_multi_index(nat_idx, sizes, mode)
    return reshaped_tensor(nat_result, [index.shape - index_dim])



def histogram(values: Tensor, bins: Shape or Tensor = spatial(bins=30), weights=1, same_bins: DimFilter = None, eps=1e-5):
    """
    Compute a histogram of a distribution of values.

    *Important Note:* In its current implementation, values outside the range of bins may or may not be added to the outermost bins.

    Args:
        values: `Tensor` listing the values to be binned along spatial or instance dimensions.
            `values´ may not contain channel or dual dimensions.
        bins: Either `Shape` specifying the number of equally-spaced bins to use or bin edge positions as `Tensor` with a spatial or instance dimension.
        weights: `Tensor` assigning a weight to every value in `values` that will be added to the bin, default 1.
        same_bins: Only used if `bins` is given as a `Shape`.
            Use the same bin sizes and positions across these batch dimensions.
            By default, bins will be chosen independently for each example.

    Returns:
        hist: `Tensor` containing all batch dimensions and the `bins` dimension with dtype matching `weights`.
        bin_edges: `Tensor`
        bin_center: `Tensor`
    """
    assert isinstance(values, Tensor), f"values must be a Tensor but got {type(values)}"
    assert channel(values).is_empty, f"Only 1D histograms supported but values have a channel dimension: {values.shape}"
    assert dual(values).is_empty, f"values cannot contain dual dimensions but got shape {values.shape}"
    weights = wrap(weights)
    if isinstance(bins, SHAPE_TYPES):
        def equal_bins(v):
            lo, up = finite_min(v, shape), finite_max(v, shape)
            margin = eps * (up - lo)
            return linspace(lo, up+margin, bins.with_size(bins.size + 1))
        bins = broadcast_op(equal_bins, [values], iter_dims=(batch(values) & batch(weights)).without(same_bins))
    assert isinstance(bins, Tensor), f"bins must be a Tensor but got {type(bins)}"
    assert non_batch(bins).rank == 1, f"bins must contain exactly one spatial or instance dimension listing the bin edges but got shape {bins.shape}"
    assert channel(bins).rank == dual(bins).rank == 0, f"bins cannot have any channel or dual dimensions but got shape {bins.shape}"
    tensors = [values, bins] if weights is None else [values, weights, bins]
    backend = backend_for(*tensors)

    def histogram_uniform(values: Tensor, bin_edges: Tensor, weights):
        batch_dims = batch(values) & batch(bin_edges) & batch(weights)
        value_dims = non_batch(values) & non_batch(weights)
        values_native = values._reshaped_native([batch_dims, value_dims])
        weights_native = weights._reshaped_native([batch_dims, value_dims])
        bin_edges_native = bin_edges._reshaped_native([batch_dims, non_batch(bin_edges)])
        hist_native = backend.histogram1d(values_native, weights_native, bin_edges_native)
        hist = reshaped_tensor(hist_native, [batch_dims, non_batch(bin_edges).with_size(non_batch(bin_edges).size - 1)])
        return hist
        # return stack_tensors([bin_edges, hist], channel(vector=[bin_edges.shape.name, 'hist']))

    bin_center = (bins[{non_batch(bins).name: slice(1, None)}] + bins[{non_batch(bins).name: slice(0, -1)}]) / 2
    bin_center = expand(bin_center, channel(vector=non_batch(bins).names))
    bin_edges = stack_tensors([bins], channel(values)) if channel(values) else bins
    return broadcast_op(histogram_uniform, [values, bins, weights]), bin_edges, bin_center


def fft(x: Tensor, dims: DimFilter = spatial) -> Tensor:
    """
    Performs a fast Fourier transform (FFT) on all spatial dimensions of x.
    
    The inverse operation is `ifft()`.

    Implementations:

    * NumPy: [`np.fft.fft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html),
      [`numpy.fft.fft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html),
      [`numpy.fft.fftn`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html)
    * PyTorch: [`torch.fft.fft`](https://pytorch.org/docs/stable/fft.html)
    * TensorFlow: [`tf.signal.fft`](https://www.tensorflow.org/api_docs/python/tf/signal/fft),
      [`tf.signal.fft2d`](https://www.tensorflow.org/api_docs/python/tf/signal/fft2d),
      [`tf.signal.fft3d`](https://www.tensorflow.org/api_docs/python/tf/signal/fft3d)
    * Jax: [`jax.numpy.fft.fft`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft.html),
      [`jax.numpy.fft.fft2`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft2.html)
      [`jax.numpy.fft.fft`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fftn.html)

    Args:
        x: Uniform complex or float `Tensor` with at least one spatial dimension.
        dims: Dimensions along which to perform the FFT.
            If `None`, performs the FFT along all spatial dimensions of `x`.

    Returns:
        *Ƒ(x)* as complex `Tensor`
    """
    dims = x.shape.only(dims)
    x_native = x.native(x.shape)
    result_native = x.backend.fft(x_native, x.shape.indices(dims.names))
    return Dense(result_native, x.shape.names, x.shape, x.backend)


def ifft(k: Tensor, dims: DimFilter = spatial):
    """
    Inverse of `fft()`.

    Args:
        k: Complex or float `Tensor` with at least one spatial dimension.
        dims: Dimensions along which to perform the inverse FFT.
            If `None`, performs the inverse FFT along all spatial dimensions of `k`.

    Returns:
        *Ƒ<sup>-1</sup>(k)* as complex `Tensor`
    """
    dims = k.shape.only(dims)
    k_native = k.native(k.shape)
    result_native = k.backend.ifft(k_native, k.shape.indices(dims.names))
    return Dense(result_native, k.shape.names, k.shape, k.backend)


def dtype(x) -> DType:
    """
    Returns the data type of `x`.

    Args:
        x: `Tensor` or native tensor.

    Returns:
        `DType`
    """
    if isinstance(x, Tensor):
        return x.dtype
    else:
        return choose_backend(x).dtype(x)


def always_close(t1: Union[Number, Tensor, bool], t2: Union[Number, Tensor, bool], rel_tolerance=1e-5, abs_tolerance=0, equal_nan=False) -> bool:
    """
    Checks whether two tensors are guaranteed to be `close` in all values.
    Unlike `close()`, this function can be used with JIT compilation and with tensors of incompatible shapes.
    Incompatible tensors are never close.

    If one of the given tensors is being traced, the tensors are only equal if they reference the same native tensor.
    Otherwise, an element-wise equality check is performed.

    See Also:
        `close()`.

    Args:
        t1: First tensor or number to compare.
        t2: Second tensor or number to compare.
        rel_tolerance: Relative tolerance, only used if neither tensor is traced.
        abs_tolerance: Absolute tolerance, only used if neither tensor is traced.
        equal_nan: If `True`, tensors are considered close if they are NaN in the same places.

    Returns:
        `bool`
    """
    if t1 is t2:
        return True
    if t1 is None or t2 is None:
        return t1 is None and t2 is None
    t1 = wrap(t1)
    t2 = wrap(t2)
    if t1.available != t2.available:
        return False
    if t1.available and t2.available:
        try:
            return close(t1, t2, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, equal_nan=equal_nan)
        except IncompatibleShapes:
            return False
    elif isinstance(t1, Dense) and isinstance(t2, Dense):
        return t1._native is t2._native
    else:
        return t1 is t2


def close(*tensors, rel_tolerance: Union[float, Tensor] = 1e-5, abs_tolerance: Union[float, Tensor] = 0, equal_nan=False, reduce=shape) -> bool:
    """
    Checks whether all tensors have equal values within the specified tolerance.
    
    Does not check that the shapes exactly match but if shapes are incompatible, returns `False`.
    Unlike with `always_close()`, all shapes must be compatible and tensors with different shapes are reshaped before comparing.

    See Also:
        `always_close()`.

    Args:
        *tensors: At least two  `Tensor` or tensor-like objects or `None`.
            The shapes of all tensors must be compatible but not all tensors must have all dimensions.
            If any argument is `None`, returns `True` only if all are `None`.
        rel_tolerance: Relative tolerance
        abs_tolerance: Absolute tolerance
        equal_nan: If `True`, tensors are considered close if they are NaN in the same places.

    Returns:
        `bool`, whether all given tensors are equal to the first tensor within the specified tolerance.
    """
    if tensors[0] is None:
        return all(o is None for o in tensors)
    if any(o is None for o in tensors):
        return False
    if all(t is tensors[0] for t in tensors):
        return True
    tensors = [wrap(t) for t in tensors]
    if any([not tensors[0].shape.is_compatible(t.shape) for t in tensors[1:]]):
        return False
    c = True
    for other in tensors[1:]:
        c &= _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, equal_nan=equal_nan, reduce=reduce)
    return c


def equal(*objects, equal_nan=False) -> bool:
    """
    Checks whether all objects are equal.

    See Also:
        `close()`, `always_close()`.

    Args:
        *objects: Objects to compare. Can be tensors or other objects or `None`
        equal_nan: If all objects are tensor-like, whether to count `NaN` values as equal.

    Returns:
        `bool`, whether all given objects are equal to the first one.
    """
    if objects[0] is None:
        return all(o is None for o in objects)
    if any(o is None for o in objects):
        return False
    if all(o is objects[0] for o in objects):
        return True
    try:
        tensors = [wrap(o) for o in objects]
        if any(t.dtype.kind == object for t in tensors):
            raise ValueError
    except ValueError:  # not all are tensor-like
        return all(o == objects[0] for o in objects)
    return close(*tensors, rel_tolerance=0, abs_tolerance=0, equal_nan=equal_nan)


def _close(tensor1: Tensor, tensor2: Tensor, rel_tolerance: Union[float, Tensor] = 1e-5, abs_tolerance: Union[float, Tensor] = 0, equal_nan=False, reduce=shape):
    reduce = tensor1.shape.only(reduce).names + tensor2.shape.only(reduce).names + shape(rel_tolerance).only(reduce).names + shape(abs_tolerance).only(reduce).names
    non_reduced = tensor1.shape.without(reduce) & tensor2.shape.without(reduce) & shape(rel_tolerance).without(reduce) & shape(abs_tolerance).without(reduce)
    if non_reduced:
        return broadcast_op(_close, [tensor1, tensor2, wrap(rel_tolerance), wrap(abs_tolerance), wrap(equal_nan)], non_reduced)
    if tensor2 is tensor1:
        return True
    iter_dims = tensor1.shape.non_uniform_shape & tensor2.shape.non_uniform_shape & shape(rel_tolerance) & shape(abs_tolerance)
    if iter_dims:
        for i in iter_dims.meshgrid():
            if not _close(tensor1[i], tensor2[i], slice_(rel_tolerance, i), slice_(abs_tolerance, i)):
                return False
        return True
    if is_sparse(tensor1) or is_sparse(tensor2):
        if not is_sparse(tensor1) or not is_sparse(tensor2):
            tensor1 = dense(tensor1)
            tensor2 = dense(tensor2)
        else:  # both sparse
            if type(tensor1) != type(tensor2):
                raise NotImplementedError("Checking sparse equality only supported for same sparse format")
            if not _close(tensor1._indices, tensor2._indices, rel_tolerance=0, abs_tolerance=0):
                return False
            if not _close(tensor1._values, tensor2._values, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, equal_nan=equal_nan):
                return False
            if isinstance(tensor1, CompressedSparseMatrix) and isinstance(tensor2, CompressedSparseMatrix):
                if not _close(tensor1._pointers, tensor2._pointers, rel_tolerance=0, abs_tolerance=0):
                    return False
            return True
    _, new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = choose_backend(native1).numpy(native1)
    np2 = choose_backend(native2).numpy(native2)
    return np.allclose(np1, np2, float(rel_tolerance), float(abs_tolerance), equal_nan=bool(equal_nan))


def assert_close(*values,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0,
                 msg: str = "",
                 verbose: bool = True,
                 equal_nan=True):
    """
    Checks that all given tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.
    
    Does not check that the shapes match as long as they can be broadcast to a common shape.

    Args:
        values: Tensors or native tensors or numbers or sequences of numbers.
        rel_tolerance: Relative tolerance.
        abs_tolerance: Absolute tolerance.
        msg: Optional error message.
        verbose: Whether to print conflicting values.
        equal_nan: If `False`, `NaN` values will always trigger an assertion error.
    """
    if not values:
        return
    ml_tensors = [t for t in values if isinstance(t, Tensor)]
    if ml_tensors:
        values = [compatible_tensor(t, ml_tensors[0].shape)._simplify() for t in values]  # use Tensor to infer dimensions
        for other in values[1:]:
            _assert_close(values[0], other, rel_tolerance, abs_tolerance, msg, verbose)
    elif all(isinstance(v, PhiTreeNode) for v in values):
        tree0, tensors0 = disassemble_tree(values[0], cache=False, attr_type=value_attributes)
        for value in values[1:]:
            tree, tensors_ = disassemble_tree(value, cache=False, attr_type=value_attributes)
            assert tree0 == tree, f"Tree structures do not match: {tree0} and {tree}"
            for t0, t in zip(tensors0, tensors_):
                _assert_close(t0, t, rel_tolerance, abs_tolerance, msg, verbose)
    else:
        np_values = [choose_backend(t).numpy(t) for t in values]
        for other in np_values[1:]:
            np.testing.assert_allclose(np_values[0], other, rel_tolerance, abs_tolerance, err_msg=msg, verbose=verbose, equal_nan=equal_nan)


def _assert_close(tensor1: Tensor, tensor2: Tensor, rel_tolerance: float, abs_tolerance: float, msg: str, verbose: bool):
    if tensor2 is tensor1:
        return
    # if isinstance(tensor2, (int, float, bool)):
    #     np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)
    if isinstance(tensor1, Layout):
        tensor1._assert_close(tensor2, rel_tolerance, abs_tolerance, msg, verbose)
    elif isinstance(tensor2, Layout):
        tensor2._assert_close(tensor1, rel_tolerance, abs_tolerance, msg, verbose)
    elif is_sparse(tensor1):
        if is_sparse(tensor2) and type(tensor2) == type(tensor1):
            _assert_close(stored_values(tensor1), stored_values(tensor2), rel_tolerance, abs_tolerance, msg, verbose)
            _assert_close(stored_indices(tensor1), stored_indices(tensor2), 0, 0, msg, verbose)
            if isinstance(tensor2, CompressedSparseMatrix) and isinstance(tensor1, CompressedSparseMatrix):
                _assert_close(tensor1._pointers, tensor2._pointers, 0, 0, msg, verbose)
        else:
            _assert_close(dense(tensor1), tensor2, rel_tolerance, abs_tolerance, msg, verbose)
    elif is_sparse(tensor2):
        return _assert_close(tensor2, tensor1, rel_tolerance, abs_tolerance, msg, verbose)
    else:
        def inner_assert_close(tensor1, tensor2):
            _, new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
            np1 = tensor1.backend.numpy(native1)
            np2 = tensor2.backend.numpy(native2)
            if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
                np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance, err_msg=msg, verbose=verbose)

        broadcast_op(inner_assert_close, [tensor1, tensor2], no_return=True)


def _native_wrapper(tensor_function: Callable, create_native_function: Callable, persistent_refs=False):
    INPUT_TENSORS = []
    OUTPUT_TENSORS = []

    def native_function(*natives):
        natives = list(natives)
        values = [t._op1(lambda _: natives.pop(0)) for t in INPUT_TENSORS]
        assert len(natives) == 0, "Not all arguments were converted"
        result = tensor_function(*values)
        results = [result] if not isinstance(result, (tuple, list)) else result
        OUTPUT_TENSORS.clear()
        OUTPUT_TENSORS.extend(results)
        return sum([v._natives() for v in results], ())

    backend = default_backend()
    traced = create_native_function(native_function, backend)
    if traced is NotImplemented:
        warnings.warn(f"Backend '{backend}' not supported. Returning original function.", RuntimeWarning)
        return tensor_function, None, INPUT_TENSORS, OUTPUT_TENSORS

    def wrapper(*values: Tensor):
        INPUT_TENSORS.clear()
        INPUT_TENSORS.extend(values)
        values = [cached(v) for v in values]
        natives = sum([v._natives() for v in values], ())
        results_native = list(traced(*natives))
        results = [t._with_natives_replaced(results_native) for t in OUTPUT_TENSORS]
        if not persistent_refs:
            INPUT_TENSORS.clear()
            # OUTPUT_TENSORS.clear()  outputs need to be saved because native_function may be called only the first time. Will get garbage collected once the function is not referenced anymore.
        assert len(results_native) == 0
        return results[0] if len(results) == 1 else results

    return wrapper, traced, INPUT_TENSORS, OUTPUT_TENSORS


def stop_gradient(x):
    """
    Disables gradients for the given tensor.
    This may switch off the gradients for `x` itself or create a copy of `x` with disabled gradients.

    Implementations:

    * PyTorch: [`x.detach()`](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)
    * TensorFlow: [`tf.stop_gradient`](https://www.tensorflow.org/api_docs/python/tf/stop_gradient)
    * Jax: [`jax.lax.stop_gradient`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.stop_gradient.html)

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` for which gradients should be disabled.

    Returns:
        Copy of `x`.
    """
    if isinstance(x, SHAPE_TYPES):
        return x
    return _backend_op1(x, Backend.stop_gradient, attr_type=variable_attributes)


def pairwise_differences(positions: Tensor,
                         max_distance: Union[float, Tensor] = None,
                         format: Union[str, Tensor] = 'dense',
                         domain: Optional[Tuple[Tensor, Tensor]] = None,
                         periodic: Union[bool, Tensor] = False,
                         method: str = 'auto',
                         default: float = float('nan'),
                         avg_neighbors=8.) -> Tensor:
    """
    Computes the distance matrix containing the pairwise position differences between each pair of points.
    The matrix will consist of the channel and batch dimension of `positions` and the primal dimensions plus their dual counterparts, spanning the matrix.
    Points that are further apart than `max_distance` (if specified) are assigned an invalid value given by `default`.
    The diagonal of the matrix (self-distance) consists purely of zero-vectors and is always stored explicitly.
    The neighbors of the positions are listed along the dual dimension(s) of the matrix, and vectors point *towards* the neighbors.

    This function can operate in *dense* mode or *sparse* mode, depending on `format`.
    If `format=='dense'` or a dense `Tensor`, all possible pair-wise distances are considered and a full-rank tensor is returned.
    The value of `method` is ignored in that case.

    Otherwise, if `format` is a sparse format identifier or sparse `Tensor`, only a subset of distances is considered, depending on `method`.
    In this case, the result is a sparse matrix with the same dimensionos as the dense tensor would have had.

    **JIT behavior:** This function can be JIT compiled with all backends.
    However, as the exact number of neighbors is unknown beforehand, all sparse methods rely on a variable-size buffer.
    PyTorch and TensorFlow allow variable shapes and behave the same way with JIT compilation as without.
    JAX, however, requires all tensor shapes to be known beforehand.
    This function will guess the required buffer size based on `avg_neighbors` and track the actually required sizes.
    When using `phiml.math.jit_compile`, this will automatically trigger a re-tracing when a buffer overflow is detected.
    User calling `jax.jit` manually must retrieve these sizes from the buffer API and implement buffer overflow handling.

    Args:
        positions: `Tensor`.
            Channel dimensions are interpreted as position components.
            Instance and spatial dimensions list nodes.
        max_distance: Scalar or `Tensor` specifying a max_radius for each point separately.
            Can contain additional batch dimensions but spatial/instance dimensions must match `positions` if present.
            If not specified, uses an infinite cutoff radius, i.e. all points will be considered neighbors.
        format: Matrix format as `str` or concrete sparsity pattern as `Tensor`.
            Allowed strings are `'dense'', `'sparse'`, `'csr'`, `'coo'`, `'csc'`.
            When a `Tensor` is passed, it needs to have all instance and spatial dims as `positions` as well as corresponding dual dimensions.
            The distances will be evaluated at all stored entries of the `format` tensor.
        domain: Lower and upper corner of the bounding box. All positions must lie within this box.
            This must be specified to use with periodic boundaries.
        periodic: Which domain boundaries should be treated as periodic, i.e. particles on opposite sides are neighbors.
            Can be specified as a `bool` for all sides or as a vector-valued boolean `Tensor` to specify periodicity by direction.
        default: Value for distances greater than `max_distance`. Only for dense distance matrices.
        method: Neighbor search algorithm; only used if `format` is a sparse format or `Tensor`.
            The default, `'auto'` lets the runtime decide on the best method. Supported methods:

            * `'sparse'`: GPU-supported hash grid implementation with fully sparse connectivity.
            * `'scipy-kd'`: SciPy's [kd-tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html#scipy.spatial.KDTree.query_ball_point) implementation.

        avg_neighbors: Expected average number of neighbors. This is only relevant for hash grid searches, where it influences the default buffer sizes.

    Returns:
        Distance matrix as sparse or dense `Tensor`, depending on `format`.
        For each spatial/instance dimension in `positions`, the matrix also contains a dual dimension of the same name and size.
        The matrix also contains all batch dimensions of `positions` and the channel dimension of `positions`.

    Examples:
        >>> pos = vec(x=0, y=tensor([0, 1, 2.5], instance('particles')))
        >>> dx = pairwise_differences(pos, format='dense', max_distance=2)
        >>> dx.particles[0]
        (x=0.000, y=0.000); (x=0.000, y=1.000); (x=0.000, y=0.000) (~particlesᵈ=3, vectorᶜ=x,y)
    """
    assert isinstance(positions, Tensor), f"positions must be a Tensor but got {type(positions)}"
    assert channel(positions).rank == 1, f"positions must have exactly one channel dimension but got {positions.shape}"
    primal_dims = positions.shape.non_batch.non_channel.non_dual
    dual_dims = primal_dims.as_dual()
    if isinstance(periodic, bool):
        any_periodic = periodic
        periodic = expand(periodic, channel(positions))
    else:
        assert isinstance(periodic, Tensor), f"periodic must be a bool or Tensor but got {periodic}"
        assert periodic.shape.names == channel(positions).names
        assert periodic.shape.item_names == channel(positions).item_names
        any_periodic = periodic.any
    # --- Dense ---
    if (isinstance(format, str) and format == 'dense') or (isinstance(format, Tensor) and get_format(format) == 'dense'):
        if isinstance(format, Tensor):
            dual_dims = dual(format)
        dx = unpack_dim(pack_dims(positions, non_batch(positions).non_channel.non_dual, instance('_tmp')), '_tmp', dual_dims) - positions
        if max_distance is not None:
            if any_periodic:
                domain_size = domain[1] - domain[0]
                dx_periodic = (dx + domain_size / 2) % domain_size - domain_size / 2
                dx = where(periodic, dx_periodic, dx)
            neighbors = sum_(dx ** 2, channel) <= max_distance ** 2
            dx = where(neighbors, dx, default)
        return dx
    # --- sparse with known connectivity ---
    if isinstance(format, Tensor):  # sparse connectivity specified, no neighborhood search required
        assert max_distance is None, "max_distance not allowed when connectivity is specified (passing a Tensor for format)"
        assert is_sparse(format)
        if any_periodic:
            from .extrapolation import PERIODIC
            def periodic_dist(p1, p2):
                p_dist = PERIODIC.shortest_distance(p1-domain[0], p2-domain[0], domain[1] - domain[0])
                return where(periodic, p_dist, p2 - p1)
            return map_pairs(periodic_dist, positions, format)
        return map_pairs(lambda p1, p2: p2 - p1, positions, format)
    # --- Sparse neighbor search ---
    assert max_distance is not None, "max_distance must be specified when computing distance in sparse format"
    max_distance = wrap(max_distance)
    index_dtype = DType(int, 32)
    backend = backend_for(positions, max_distance)
    batch_shape = batch(positions) & batch(max_distance)
    if not dual_dims.well_defined:
        assert dual_dims.rank == 1, f"others_dims sizes must be specified when passing more then one dimension but got {dual_dims}"
        dual_dims = dual_dims.with_size(primal_dims.volume)
    if domain is not None:
        assert isinstance(domain, tuple) and len(domain) == 2, f"Domain needs to be of the form (lower_corner, upper_corner) but got {domain}"
        domain = (wrap(domain[0]), wrap(domain[1]))
        if channel(positions).size > 1:
            assert domain[0].shape.names == channel(positions).names, f"Domain must have exactly the channel dimensions of positions but got {domain[0]}"
            assert domain[1].shape.names == channel(positions).names, f"Domain must have exactly the channel dimensions of positions but got {domain[1]}"
        domain = (domain[0]._reshaped_native([channel(domain[0])]), domain[1]._reshaped_native([channel(domain[1])]))
    if method == 'auto':
        method = 'sparse'
    assert method in ['sparse', 'scipy-kd'], f"Invalid neighbor search method: '{method}'"
    if any_periodic:
        assert domain is not None, f"domain must be specified when periodic=True"
        if method in ['scipy-kd']:
            warnings.warn(f"Neighbor search method '{method}' is not compatible with periodic boundaries.", RuntimeWarning, stacklevel=2)
            method = 'sparse'
    def uniform_neighbor_search(positions: Tensor, max_distance: Tensor):
        native_positions = positions._reshaped_native([primal_dims, channel(positions)])
        native_max_dist = max_distance.native()
        if method == 'sparse':
            from phiml.backend._partition import find_neighbors_sparse
            nat_rows, nat_cols, nat_deltas = find_neighbors_sparse(native_positions, native_max_dist, domain, periodic=periodic, default=default, index_dtype=index_dtype, avg_neighbors=avg_neighbors)
            nat_indices = backend.stack([nat_rows, nat_cols], -1)
            indices = reshaped_tensor(nat_indices, [instance('pairs'), channel(vector=primal_dims.names + dual_dims.names)], convert=False)
            deltas = reshaped_tensor(nat_deltas, [instance('pairs'), channel(positions)], convert=False)
            return SparseCoordinateTensor(indices, deltas, primal_dims & dual_dims, can_contain_double_entries=False, indices_sorted=True, indices_constant=False)
        elif method == 'scipy-kd':
            from phiml.backend._partition import find_neighbors_scipy_kd
            nat_idx, nat_ptr, nat_deltas = find_neighbors_scipy_kd(native_positions, native_max_dist, avg_neighbors, index_dtype)
            indices = reshaped_tensor(nat_idx, [instance('pairs')], convert=False)
            pointers = reshaped_tensor(nat_ptr, [instance('pointers')], convert=False)
            deltas = reshaped_tensor(nat_deltas, [instance('pairs'), channel(positions)], convert=False)
            if format == 'csc':  # the matrix is symmetric, so we can transpose to match desired result
                uncompressed, compressed = primal_dims, dual_dims
            else:
                uncompressed, compressed = dual_dims, primal_dims
                deltas = -deltas
            return CompressedSparseMatrix(indices, pointers, deltas, uncompressed, compressed, indices_constant=False)
        # elif method == 'semi-sparse':
        #     from phiml.backend._partition import find_neighbors_semi_sparse
        #     native_positions = positions.native([primal_dims, channel(positions)])
        #     native_max_dist = max_distance.native()
        #     nat_rows, nat_cols, nat_vals, req_pair_count, req_max_occupancy = find_neighbors_semi_sparse(native_positions, native_max_dist, None, periodic=False, default=default)
        # elif method == 'matscipy':
        #     positions.backend.numpy_call()
        #     from phiml.backend._partition import find_neighbors_matscipy
        #     nat_rows, nat_cols, nat_vals = find_neighbors_matscipy(native_positions, native_max_dist, None, periodic=False)
        # elif method == 'sklearn':
        #     assert positions.available, f"Cannot jit-compile matscipy neighborhood search"
        #     from phiml.backend._partition import find_neighbors_sklearn
        #     nat_rows, nat_cols, nat_vals = find_neighbors_sklearn(native_positions, native_max_dist)
        else:
            raise ValueError(method)

    matrix = broadcast_op(uniform_neighbor_search, [positions, max_distance], iter_dims=batch_shape)
    # --- Assemble sparse matrix ---
    return to_format(matrix, format)


def map_pairs(map_function: Callable, values: Tensor, connections: Tensor):
    """
    Evaluates `map_function` on all pairs of elements present in the sparsity pattern of `connections`.

    Args:
        map_function: Function with signature `(Tensor, Tensor) -> Tensor`.
        values: Values to evaluate `map_function` on.
            Needs to have a spatial or instance dimension but must not have a dual dimension.
        connections: Sparse tensor.

    Returns:
        `Tensor` with the sparse dimensions of `connections` and all non-instance dimensions returned by `map_function`.
    """
    assert dual(values).is_empty, f"values must not have a dual dimension but got {values.shape}"
    indices = stored_indices(connections, invalid='clamp')
    origin_dim, neighbors_dim = channel(indices).item_names[0]
    if origin_dim not in values.shape:
        origin_dim, neighbors_dim = neighbors_dim, origin_dim
    assert origin_dim in values.shape, f"No dimension of connections {connections.shape} is present in values {values.shape}"
    origin = values[{origin_dim: indices[origin_dim]}]
    target = values[{origin_dim: indices[neighbors_dim]}]
    result = map_function(origin, target)
    return tensor_like(connections, result, value_order='as existing')


def with_diagonal(matrix: Tensor, values: Union[float, Tensor], check_square=True):
    """
    Create a copy of `matrix`, replacing the diagonal elements.
    If `matrix` is sparse, diagonal zeros (and possibly other explicitly stored zeros) will be dropped from the sparse matrix.

    This function currently only supports sparse COO,CSR,CSC SciPy matrices.

    Args:
        matrix: `Tensor` with at least one dual dim.
        values: Diagonal values
        check_square: If `True` allow this function only for square matrices.

    Returns:
        `Tensor`
    """
    col_dims = matrix.shape.dual
    row_dims = matrix.shape.only(col_dims.as_channel())
    if not row_dims:
        row_dims = primal(matrix)
    if not row_dims:
        row_dims = batch(matrix)
    if check_square:
        assert row_dims.volume == col_dims.volume, f"matrix is not square (check_square=True). rows={row_dims}, cols={col_dims}"
    if is_sparse(matrix):
        assert matrix.backend.name == 'numpy', f"with_diagonal currently only supports SciPy matrices"
        values = wrap(values)
        result = []
        for idx in (batch(values) & batch(matrix)).meshgrid():
            scipy_matrix = matrix[idx].native()
            values = values[idx].native()
            scipy_matrix.setdiag(values)
            if close(0, values):
                scipy_matrix.eliminate_zeros()
            result.append(wrap(scipy_matrix, row_dims.after_gather(idx), col_dims.after_gather(idx)))
        return stack(result, batch(values) & batch(matrix))
    else:
        raise NotImplementedError("with_diagonal currently only supports sparse matrices")


def eigenvalues(matrix: Tensor, eigen_dim=channel('eigenvalues')):
    """
    Computes the eigenvalues of a square matrix.
    The matrix columns are listed along dual dimensions and the rows are listed along the corresponding non-dual dimensions.
    Row dims are matched by name if possible, else all primal dims are used.

    Args:
        matrix: Square matrix. Must have at least one dual dim and corresponding non-dual dim.
        eigen_dim: Dimension along which eigenvalues should be listed.

    Returns:
        `Tensor` listing the eigenvalues along `eigen_dim`.
    """
    cols = dual(matrix)
    assert cols, f"Matrix must have at least one dual dim listing the columns"
    rows = matrix.shape.only(cols.as_batch().name_list)
    if not rows:
        rows = primal(matrix)
    assert rows.volume == cols.volume, f"Matrix rows {rows} don't match cols {cols}"
    batch_dims = matrix.shape.without(cols).without(rows)
    native_matrix = matrix._reshaped_native([*batch_dims, rows, cols])
    native_result = matrix.backend.eigvals(native_matrix)
    return reshaped_tensor(native_result, [*batch_dims, eigen_dim], convert=False)


def svd(x: Tensor, feature_dim: DimFilter = channel, list_dim: DimFilter = None, latent_dim=channel('singular'), full_matrices=False):
    """
    Singular value decomposition.

    The original matrix is approximated by `(latent_to_value * singular.T) @ latents` or `latent_to_value @ (singular * latents)`.

    **Warning:** Even for well-defined SVDs, different backend use different sign conventions, causing results to differ.

    Args:
        x: Matrix containing `feature_dim` and `list_dim`.
        feature_dim: Dimensions that list the features (columns).
        list_dim: Dimensions that list the data points (rows).
        latent_dim: Latent dimension. If a size is specified, truncates the SVD to this size.
        full_matrices: If `True`, return full-sized (square) matrices for latent_by_example and latent_to_value. These may not match the singular values.

    Returns:
        latents: Latent vectors of each item listed. `Tensor` with `list_dim` and `latent_dim`.
        singular: List of singular values. `Tensor` with `latent_dim`.
        features: Stacked normalized features / trends. This matrix can be used to compute the original value from a latent vector. `Tensor` with `latent_dim` and `feature_dim`.
    """
    feature_dim = x.shape.only(feature_dim)
    if list_dim is not None:
        list_dim = x.shape.only(list_dim)
    else:
        if non_batch(x) - feature_dim:
            list_dim = non_batch(x) - feature_dim
        else:
            list_dim = x.shape - feature_dim
    assert feature_dim, f"No valid feature dim specified: {feature_dim} for data {x}"
    assert list_dim, f"No valid list dim specified: {list_dim} for data {x}"
    batch_dims = x.shape - feature_dim - list_dim
    latent_dim = auto(latent_dim, channel) if isinstance(latent_dim, str) else latent_dim
    native = x._reshaped_native([batch_dims, list_dim, feature_dim])
    u, s, v = x.backend.svd(native, full_matrices=full_matrices)
    truncate = latent_dim.size
    if truncate is not None:
        if s.shape[1] < truncate:
            warnings.warn(f"Trying to truncate SVD but there are too few values: {s.shape[1]} < {truncate}")
        u = u[:, :, :truncate]
        s = s[:, :truncate]
        v = v[:, :truncate, :]
    latent_by_example = reshaped_tensor(u, [batch_dims, list_dim, latent_dim])
    singular_values = reshaped_tensor(s, [batch_dims, latent_dim])
    latent_to_value = reshaped_tensor(v, [batch_dims, latent_dim.as_dual(), feature_dim])
    return latent_by_example, singular_values, latent_to_value


def count_occurrences(values: Tensor, query: Tensor, feature_dims: DimFilter = channel) -> Tensor:
    """
    For each query item, counts how often this value occurs in `values`.

    See Also:
        `contains()`.

    Args:
        values: Data `Tensor` containing all `feature_dims`.
            All non-batch and dims not specified as `feature_dims` are flattened.
        query: Items to count the occurrences of. Must contain all `feature_dims`.
        feature_dims: One item is considered to be the set of all values along `feature_dims`.
            The number of items in a tensor is given by all dims except `feature_dims`.

    Returns:
        Integer `Tensor` matching `query` without `feature_dims`.
    """
    feature_dims = values.shape.only(feature_dims)
    assert feature_dims in query
    batches = batch(values) & batch(query)
    values_nat = values.native([batches, ..., feature_dims])
    query_nat = query.native([batches, ..., feature_dims])
    def np_count(query_np: np.ndarray, values_np: np.ndarray):
        query_and_values = np.concatenate([query_np, values_np], 1)
        result_np = []
        for i in range(batches.volume):
            unique, inverse, counts = np.unique(query_and_values[i], axis=0, return_counts=True, return_inverse=True)
            combined_occurrences = counts[inverse][:query_np.shape[1]]
            unique, inverse, counts = np.unique(query_np[i], axis=0, return_counts=True, return_inverse=True)
            query_occurrences = counts[inverse]
            result_np.append(combined_occurrences - query_occurrences)
        return np.stack(result_np).astype(np.int32)
    result_nat = choose_backend(query_nat, values_nat).numpy_call(np_count, (batches.volume, (non_batch(query) - feature_dims).volume), DType(int, 32), query_nat, values_nat)
    return reshaped_tensor(result_nat, [batches, non_batch(query) - feature_dims], convert=False)


def contains(values: Tensor, query: Tensor, feature_dims: DimFilter = channel) -> Tensor:
    """
    For each query item, checks whether it is contained in `values`.

    See Also:
        `count_occurrences()`.

    Args:
        values: Data `Tensor` containing all `feature_dims`.
            All non-batch and dims not specified as `feature_dims` are flattened.
        query: Items to count the occurrences of. Must contain all `feature_dims`.
        feature_dims: One item is considered to be the set of all values along `feature_dims`.
            The number of items in a tensor is given by all dims except `feature_dims`.

    Returns:
        Integer `Tensor` matching `query` without `feature_dims`.
    """
    return count_occurrences(values, query, feature_dims=feature_dims) > 0


def count_intersections(values: Tensor, arg_dims: DimFilter, list_dims: DimFilter = instance, feature_dims: DimFilter = channel) -> Tensor:
    """
    Counts the number of elements that are part of each pair of lists.

    Args:
        values:
        arg_dims: Dims enumerating the input lists.
        list_dims: Dims listing the elements.
        feature_dims: Vector dims of one element. Elements are equal if all values along `feature_dims` are equal.

    Returns:
        `Tensor`.
    """
    assert arg_dims is not batch
    feature_dims = values.shape.only(feature_dims)
    arg_dims = values.shape.only(arg_dims)
    if feature_dims:
        if feature_dims.volume == 1:
            values = unstack(values, feature_dims)[0]
        else:
            raise NotImplementedError
    batch_dims = values.shape - arg_dims - list_dims - feature_dims
    result = []
    for b in batch_dims.meshgrid():
        lists = unstack(values[b], arg_dims)
        np_lists = [l.numpy([list_dims]) for l in lists]
        n = len(np_lists)
        shared_counts = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                intersection = np.intersect1d(np_lists[i], np_lists[j])
                shared_counts[i, j] = shared_counts[j, i] = len(intersection)
        result.append(wrap(shared_counts, arg_dims & arg_dims.as_dual()))
    return stack(result, batch_dims)
