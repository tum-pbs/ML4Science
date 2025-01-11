import copy
import dataclasses
import warnings
from functools import partial
from numbers import Number
from typing import TypeVar, Tuple, Dict, Union, Optional, Sequence, Any, Callable

from . import channel, EMPTY_SHAPE
from ._shape import Shape, DimFilter, batch, instance, shape, non_batch, merge_shapes, concat_shapes, spatial, parse_dim_order, dual, auto, parse_shape_spec, DIM_FUNCTIONS, \
    INV_CHAR, concat_shapes_, Dim, DEBUG_CHECKS, SHAPE_TYPES, primal
from .magic import Sliceable, Shaped, Shapable, PhiTreeNode
from ..backend import choose_backend, NoBackendFound
from ..backend._dtype import DType


PhiTreeNodeType = TypeVar('PhiTreeNodeType')  # Defined in phiml.math.magic: tuple, list, dict, custom


class MagicNotImplemented(Exception): pass


def slice_(value: PhiTreeNodeType, slices: Union[Dict[str, Union[int, slice, str, tuple, list, Any]], Any]) -> PhiTreeNodeType:
    """
    Slices a `Tensor` or `phiml.math.magic.PhiTreeNode` along named dimensions.

    See Also:
        `unstack`.

    Args:
        value: `Tensor` or `phiml.math.magic.PhiTreeNode` or `Number` or `None`.
        slices: `dict` mapping dimension names to slices. A slice can be one of the following:

            * An index (`int`)
            * A range (`slice`)
            * An item name (`str`)
            * Multiple item names (comma-separated `str`)
            * Multiple indices or item names (`tuple` or `list`)

    Returns:
        `Tensor` or `phiml.math.magic.PhiTreeNode` of the same type as `value`.

    Examples:
        >>> math.slice([vec(x=0, y=1), vec(x=2, y=3)], {'vector': 'y'})
        [1, 3]
    """
    if isinstance(value, (bool, Number, str)) or value is None:
        return value
    if isinstance(value, tuple):
        return tuple([slice_(v, slices) for v in value])
    if isinstance(value, list):
        return [slice_(v, slices) for v in value]
    if isinstance(value, dict):
        return {k: slice_(v, slices) for k, v in value.items()}
    if isinstance(value, SHAPE_TYPES):
        return value.after_gather(slices)
    if value is range:
        from ._tensors import Tensor
        if isinstance(slices, Tensor):
            return slices
        raise NotImplementedError("range only supported for index slicing")
    if hasattr(value, '__getitem__'):
        return value[slices]
    if isinstance(value, PhiTreeNode):
        attrs = {key: getattr(value, key) for key in all_attributes(value)}
        new_attrs = {k: slice_(v, slices) for k, v in attrs.items()}
        return copy_with(value, **new_attrs)
    raise ValueError(f"value must be a PhiTreeNode but got {type(value)}")


def unstack(value, dim: DimFilter) -> tuple:
    """
    Un-stacks a `Sliceable` along one or multiple dimensions.

    If multiple dims are given, the order of elements will be according to the dimension order in `dim`, i.e. elements along the last dimension will be neighbors in the returned `tuple`.
    If no dimension is given or none of the given dims exists on `value`, returns a list containing only `value`.

    See Also:
        `phiml.math.slice`.

    Args:
        value: `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
        dim: Dimensions as `Shape` or comma-separated `str` or dimension type, i.e. `channel`, `spatial`, `instance`, `batch`.

    Returns:
        `tuple` of objects matching the type of `value`.

    Examples:
        >>> unstack(expand(0, spatial(x=5)), 'x')
        (0.0, 0.0, 0.0, 0.0, 0.0)
    """
    if DEBUG_CHECKS:
        assert isinstance(value, Sliceable) and isinstance(value, Shaped), f"Cannot unstack {type(value).__name__}. Must be Sliceable and Shaped, see https://tum-pbs.github.io/PhiML/phiml/math/magic.html"
    dims = shape(value).only(dim, reorder=True)
    if dims.rank == 0:
        return value,
    if dims.rank == 1:
        if hasattr(value, '__unstack__'):
            result = value.__unstack__(dims.names)
            if result is not NotImplemented:
                if DEBUG_CHECKS:
                    assert isinstance(result, tuple), f"__unstack__ must return a tuple but got {type(result)}"
                    assert all([isinstance(item, Sliceable) for item in result]), f"__unstack__ must return a tuple of Sliceable objects but not all items were sliceable in {result}"
                return result
        return tuple([slice_(value, {dims.name: i}) for i in range(dims.size)])
    else:  # multiple dimensions
        if hasattr(value, '__pack_dims__'):
            packed_dim = batch('_unstack')
            value_packed = value.__pack_dims__(dims, packed_dim, pos=None)
            if value_packed is not NotImplemented:
                return unstack(value_packed, packed_dim)
        unstack_dim = _any_uniform_dim(dims)
        first_unstacked = unstack(value, unstack_dim)
        inner_unstacked = [unstack(v, dims.without(unstack_dim)) for v in first_unstacked]
        return sum(inner_unstacked, ())


def _any_uniform_dim(dims: Shape):
    for dim in dims:
        if dim.is_uniform:
            return dim
    raise ValueError(f"Uniform dimension required but found only non-uniform dims {dims}")


def stack(values: Union[Sequence[PhiTreeNodeType], Dict[str, PhiTreeNodeType]], dim: Union[Shape, str], expand_values=False, simplify=False, layout_non_matching=False, **kwargs) -> PhiTreeNodeType:
    """
    Stacks `values` along the new dimension `dim`.
    All values must have the same spatial, instance and channel dimensions. If the dimension sizes vary, the resulting tensor will be non-uniform.
    Batch dims will be added as needed.

    Stacking tensors is performed lazily, i.e. the memory is allocated only when needed.
    This makes repeated stacking and slicing along the same dimension very efficient, i.e. jit-compiled functions will not perform these operations.

    Args:
        values: Collection of `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
            If a `dict`, keys must be of type `str` and are used as item names along `dim`.
        dim: `Shape` with a least one dimension. None of these dims can be present with any of the `values`.
            If `dim` is a single-dimension shape, its size is determined from `len(values)` and can be left undefined (`None`).
            If `dim` is a multi-dimension shape, its volume must be equal to `len(values)`.
        expand_values: If `True`, will first add missing dims to all values, not just batch dimensions.
            This allows tensors with different dims to be stacked.
            The resulting tensor will have all dims that are present in `values`.
            If `False`, this may return a non-numeric object instead.
        simplify: If `True` and all values are equal, returns one value without adding the dimension.
        layout_non_matching: If non-matching values should be stacked using a Layout object, i.e. should be put into a named list instead.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        `Tensor` containing `values` stacked along `dim`.

    Examples:
        >>> stack({'x': 0, 'y': 1}, channel('vector'))
        (x=0, y=1)

        >>> stack([math.zeros(batch(b=2)), math.ones(batch(b=2))], channel(c='x,y'))
        (x=0.000, y=1.000); (x=0.000, y=1.000) (bᵇ=2, cᶜ=x,y)

        >>> stack([vec(x=1, y=0), vec(x=2, y=3.)], batch('b'))
        (x=1.000, y=0.000); (x=2.000, y=3.000) (bᵇ=2, vectorᶜ=x,y)
    """
    assert len(values) > 0, f"stack() got empty sequence {values}"
    if simplify and len(values) == 1:
        return next(iter(values.values())) if isinstance(values, dict) else values[0]
    if not dim:
        assert len(values) == 1, f"Only one element can be passed as `values` if no dim is passed but got {values}"
        return next(iter(values.values())) if isinstance(values, dict) else values[0]
    if not isinstance(dim, SHAPE_TYPES):
        dim = auto(dim)
    values_ = tuple(values.values()) if isinstance(values, dict) else values
    if simplify:
        if all(v is None for v in values_):
            return None
        if all(type(v) == type(values_[0]) for v in values_[1:]):
            from ._tensors import equality_by_shape_and_value
            with equality_by_shape_and_value(equal_nan=True):
                if all(v == values_[0] for v in values_[1:]):
                    return values_[0]
    shapes = [shape(v) for v in values_]
    if not expand_values:
        v0_dims = set(shapes[0].non_batch.names)
        for s in shapes[1:]:
            if set(s.non_batch.names) != v0_dims:  # shapes don't match
                if layout_non_matching:
                    from ._tensors import layout
                    return layout(values, dim)
                else:
                    raise ValueError(f"Non-batch dims must match but got: {v0_dims} and {s.non_batch.names}. Manually expand tensors or set expand_values=True")
    # --- Add missing dims ---
    if expand_values:
        all_dims = merge_shapes(*shapes, allow_varying_sizes=True)
        if isinstance(values, dict):
            values = {k: expand(v, all_dims - s) for (k, v), s in zip(values.items(), shapes)}
        else:
            values = [expand(v, all_dims - s) for v, s in zip(values, shapes)]
    else:
        all_batch_dims = merge_shapes(*[s.batch for s in shapes], allow_varying_sizes=True)
        if isinstance(values, dict):
            values = {k: expand(v, all_batch_dims - s) for (k, v), s in zip(values.items(), shapes)}
        else:
            values = [expand(v, all_batch_dims - s) for v, s in zip(values, shapes)]
    if dim.rank == 1:
        assert dim.size == len(values) or dim.size is None, f"stack dim size must match len(values) or be undefined but got {dim} for {len(values)} values"
        if dim.size is None:
            dim = dim.with_size(len(values))
        if isinstance(values, dict):
            dim_item_names = tuple([k.name if isinstance(k, SHAPE_TYPES) else k for k in values.keys()])
            assert all(isinstance(k, str) for k in dim_item_names), f"dict keys must be of type str but got {dim_item_names}"
            values = tuple(values.values())
            dim = dim.with_size(dim_item_names)
        # --- First try __stack__ ---
        for v in values:
            if hasattr(v, '__stack__'):
                result = v.__stack__(values, dim, **kwargs)
                if result is not NotImplemented:
                    if DEBUG_CHECKS:
                        assert isinstance(result, SHAPE_TYPES) if isinstance(v, SHAPE_TYPES) else isinstance(result, Shapable), "__stack__ must return a Shapable object"
                    return result
        # --- Next: try stacking attributes for tree nodes ---
        if any(dataclasses.is_dataclass(v) for v in values):
            from ..dataclasses._merge import dc_stack
            return dc_stack(values, dim, expand_values=expand_values, simplify=simplify, layout_non_matching=layout_non_matching, **kwargs)
        if all(isinstance(v, dict) for v in values):
            keys = set(values[0])
            if all(set(v) == keys for v in values[1:]):
                new_dict = {}
                for k in keys:
                    k_values = [v[k] for v in values]
                    new_dict[k] = stack(k_values, dim, expand_values=expand_values, simplify=simplify, **kwargs)
                return new_dict
            raise NotImplementedError
        if all(isinstance(v, PhiTreeNode) for v in values):
            attributes = all_attributes(values[0])
            if attributes and all(all_attributes(v) == attributes for v in values):
                new_attrs = {}
                for a in attributes:
                    a_values = [getattr(v, a) for v in values]
                    if all(v is a_values[0] for v in a_values[1:]):
                        new_attrs[a] = expand(a_values[0], dim, **kwargs) if a_values[0] is not None else a_values[0]
                    else:
                        new_attrs[a] = stack(a_values, dim, expand_values=expand_values, simplify=simplify, **kwargs)
                return copy_with(values[0], **new_attrs)
            else:
                warnings.warn(f"Failed to concat values using value attributes because attributes differ among values {values}")
        # --- Fallback: use expand and concat ---
        for v in values:
            if not hasattr(v, '__stack__') and hasattr(v, '__concat__') and hasattr(v, '__expand__'):
                expanded_values = tuple([expand(v, dim.with_size(1 if dim.item_names[0] is None else dim.item_names[0][i]), **kwargs) for i, v in enumerate(values)])
                if len(expanded_values) > 8:
                    warnings.warn(f"stack() default implementation is slow on large dims ({dim.name}={len(expanded_values)}). Please implement __stack__()", RuntimeWarning, stacklevel=2)
                result = v.__concat__(expanded_values, dim.name, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), "__concat__ must return a Shapable object"
                    return result
        # --- else maybe all values are native scalars ---
        from ._tensors import wrap
        try:
            values = tuple([wrap(v) for v in values])
        except ValueError:
            raise MagicNotImplemented(f"At least one item in values must be Shapable but got types {[type(v) for v in values]}")
        return values[0].__stack__(values, dim, **kwargs)
    else:  # multi-dim stack
        assert dim.volume == len(values), f"When passing multiple stack dims, their volume must equal len(values) but got {dim} for {len(values)} values"
        if isinstance(values, dict):
            warnings.warn(f"When stacking a dict along multiple dimensions, the key names are discarded. Got keys {tuple(values.keys())}", RuntimeWarning, stacklevel=2)
            values = tuple(values.values())
        # --- if any value implements Shapable, use stack and unpack_dim ---
        for v in values:
            if hasattr(v, '__stack__') and hasattr(v, '__unpack_dim__'):
                stack_dim = batch('_stack')
                stacked = v.__stack__(values, stack_dim, **kwargs)
                if stacked is not NotImplemented:
                    assert isinstance(stacked, Shapable), "__stack__ must return a Shapable object"
                    assert hasattr(stacked, '__unpack_dim__'), "If a value supports __unpack_dim__, the result of __stack__ must also support it."
                    reshaped = stacked.__unpack_dim__(stack_dim.name, dim, **kwargs)
                    if reshaped is NotImplemented:
                        warnings.warn("__unpack_dim__ is overridden but returned NotImplemented during multi-dimensional stack. This results in unnecessary stack operations.", RuntimeWarning, stacklevel=2)
                    else:
                        return reshaped
        # --- Fallback: multi-level stack ---
        for dim_ in reversed(dim):
            values = [stack(values[i:i + dim_.size], dim_, **kwargs) for i in range(0, len(values), dim_.size)]
        return values[0]


def concat(values: Sequence[PhiTreeNodeType], dim: Union[str, Shape], expand_values=False, **kwargs) -> PhiTreeNodeType:
    """
    Concatenates a sequence of `phiml.math.magic.Shapable` objects, e.g. `Tensor`, along one dimension.
    All values must have the same spatial, instance and channel dims and their sizes must be equal, except for `dim`.
    Batch dims will be added as needed.

    Args:
        values: Tuple or list of `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
        dim: Concatenation dimension, must be present in all `values`.
            The size along `dim` is determined from `values` and can be set to undefined (`None`).
            Alternatively, a `str` of the form `'t->name:t'` can be specified, where `t` is on of `b d i s c` denoting the dimension type.
            This first packs all dims of the input into a new dim with given name and type, then concatenates the values along this dim.
        expand_values: If `True`, will first add missing dims to all values, not just batch dimensions.
            This allows tensors with different dims to be concatenated.
            The resulting tensor will have all dims that are present in `values`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Concatenated `Tensor`

    Examples:
        >>> concat([math.zeros(batch(b=10)), math.ones(batch(b=10))], 'b')
        (bᵇ=20) 0.500 ± 0.500 (0e+00...1e+00)

        >>> concat([vec(x=1, y=0), vec(z=2.)], 'vector')
        (x=1.000, y=0.000, z=2.000) float64
    """
    assert len(values) > 0, f"concat() got empty sequence {values}"
    if isinstance(dim, SHAPE_TYPES):
        dim = dim.name
    assert isinstance(dim, str), f"dim must be a str or Shape but got '{dim}' of type {type(dim)}"
    if '->' in dim:
        dim_type, dim = [s.strip() for s in dim.split('->', 1)]
        dim_type = DIM_FUNCTIONS[INV_CHAR[dim_type]]
        dim = auto(dim, dim_type)
        values = [pack_dims(v, dim_type, dim) for v in values]
        dim = dim.name
    else:
        dim = auto(dim, channel).name
    # --- Filter 0-length values ---
    values = [v for v in values if shape(v).get_size(dim) > 0]
    if len(values) == 1:
        return values[0]
    #  --- Add missing dimensions ---
    if expand_values:
        all_dims = merge_shapes(*values, allow_varying_sizes=True)
        all_dims = all_dims.with_dim_size(dim, 1, keep_item_names=False)
        values = [expand(v, all_dims.without(shape(v))) for v in values]
    else:
        for v in values:
            assert dim in shape(v), f"concat dim '{dim}' must be present in the shapes of all values bot got value {type(v).__name__} with shape {shape(v)}"
        for v in values[1:]:
            assert set(non_batch(v).names) == set(non_batch(values[0]).names), f"Concatenated values must have the same non-batch dims but got {non_batch(values[0])} and {non_batch(v)}"
        all_batch_dims = merge_shapes(*[shape(v).batch.without(dim) for v in values])
        values = [expand(v, all_batch_dims) for v in values]
    # --- First try __concat__ ---
    for v in values:
        if isinstance(v, Shapable):
            if hasattr(v, '__concat__'):
                result = v.__concat__(values, dim, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), f"__concat__ must return a Shapable object but got {type(result).__name__} from {type(v).__name__} {v}"
                    return result
    # --- Next: try concat attributes for tree nodes ---
    if all(isinstance(v, PhiTreeNode) for v in values):
        attributes = all_attributes(values[0])
        if attributes and all(all_attributes(v) == attributes for v in values):
            new_attrs = {}
            for a in attributes:
                common_shape = merge_shapes(*[shape(getattr(v, a)).without(dim) for v in values])
                a_values = [expand(getattr(v, a), common_shape & shape(v).only(dim)) for v in values]  # expand by dim if missing, and dims of others
                new_attrs[a] = concat(a_values, dim, **kwargs)
            return copy_with(values[0], **new_attrs)
        else:
            warnings.warn(f"Failed to concat values using value attributes because attributes differ among values {values}")
    # --- Fallback: slice and stack ---
    try:
        unstacked = sum([unstack(v, dim) for v in values], ())
    except MagicNotImplemented:
        raise MagicNotImplemented(f"concat: No value implemented __concat__ and not all values were Sliceable along {dim}. values = {[type(v) for v in values]}")
    if len(unstacked) > 8:
        warnings.warn(f"concat() default implementation is slow on large dims ({dim}={len(unstacked)}). Please implement __concat__()", RuntimeWarning, stacklevel=2)
    dim = shape(values[0])[dim].with_size(None)
    try:
        return stack(unstacked, dim, **kwargs)
    except MagicNotImplemented:
        raise MagicNotImplemented(f"concat: No value implemented __concat__ and slices could not be stacked. values = {[type(v) for v in values]}")


def ncat(values: Sequence[PhiTreeNodeType], dim: Shape, expand_values=False) -> PhiTreeNodeType:
    """
    Concatenate named components along `dim`.

    Args:
        values: Each value can contain multiple components of `dim` if `dim` is present in its shape.
            Else, it is interpreted as a single component whose name will be determined from the leftover item names of `dim`.
        dim: Single dimension that has item names matching components of `values`.
        expand_values: If `True`, will add all missing dims to values, not just batch dimensions.
            This allows tensors with different dims to be concatenated.
            The resulting tensor will have all dims that are present in `values`.
            If `False`, this may return a non-numeric object instead.

    Returns:
        Same type as any value from `values`.
    """
    order = dim.item_names[0]
    assert dim.rank == 1 and order, f"dim needs to be a single dimension with item names but got {dim}"
    named = {}
    unnamed = []
    for value in values:
        s = shape(value)
        if dim in s:
            for n in s[dim].item_names[0]:
                named[n] = value[{dim.name: n}]
        else:
            unnamed.append(value)
    missing = [n for n in order if n not in named]
    assert len(missing) == len(unnamed), f"Components do not match dim {dim}. Given: {len(unnamed)} for remaining names {missing}"
    named.update({n: v for v, n in zip(unnamed, missing)})
    components = [named[n] for n in order]
    return stack(components, dim, expand_values=expand_values)


def tcat(values: Sequence[PhiTreeNodeType], dim_type: Callable, expand_values=False, default_name='tcat') -> PhiTreeNodeType:
    """
    Concatenate values by dim type.
    This function first packs all dims of `dim_type` into one dim, then concatenates all `values`.
    Values that do not have a dim of `dim_type` are considered a size-1 slice.

    The name of the first matching dim of `dim_type` is used as the concatenated output dim name.
    If no value has a matching dim, `default_name` is used instead.

    Args:
        values: Values to be concatenated.
        dim_type: Dimension type along which to concatenate.
        expand_values: Whether to add missing other non-batch dims to values as needed.
        default_name: Concatenation dim name if none of the values have a matching dim.

    Returns:
        Same type as any value.
    """
    dims = [dim_type(v) for v in values]
    present_names = tuple(set(sum([s.names for s in dims], ())))
    if len(present_names) == 1:
        dim_name = present_names[0]
    elif len(present_names) > 1:
        if default_name in present_names:
            dim_name = default_name
        else:
            dim_name = present_names[0]
    else:
        dim_name = default_name
    single = dim_type(**{dim_name: 1})
    flat_values = [pack_dims(v, dim_type, dim_type(dim_name)) if s else expand(v, single) for v, s in zip(values, dims)]
    return concat(flat_values, dim_name, expand_values=expand_values)


ccat = partial(tcat, dim_type=channel, default_name='ccat')
ccat.__doc__ = "Concatenate values along their channel dim, see `tcat`."
icat = partial(tcat, dim_type=instance, default_name='icat')
icat.__doc__ = "Concatenate values along their instance dim, see `tcat`."
dcat = partial(tcat, dim_type=dual, default_name='dcat')
dcat.__doc__ = "Concatenate values along their dual dim, see `tcat`."
scat = partial(tcat, dim_type=spatial, default_name='scat')
scat.__doc__ = "Concatenate values along their spatial dim, see `tcat`."


def expand(value, *dims: Union[Shape, str], **kwargs):
    """
    Adds dims to a `Tensor` or tensor-like object by implicitly repeating the tensor values along the new dimensions.
    If `value` already contains any of the new dimensions, a size and type check is performed for these instead.

    If any of `dims` varies along a dimension that is present neither in `value` nor on `dims`, it will also be added to `value`.

    This function replaces the usual `tile` / `repeat` functions of
    [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.tile.html),
    [PyTorch](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.repeat),
    [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/tile) and
    [Jax](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html).

    Additionally, it replaces the traditional `unsqueeze` / `expand_dims` functions.

    Args:
        value: `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
            For tree nodes, expands all value attributes by `dims` or the first variable attribute if no value attributes are set.
        *dims: Dimensions to be added as `Shape`
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.
    """
    if not dims:
        return value
    dims = concat_shapes_(*[d if isinstance(d, SHAPE_TYPES) else parse_shape_spec(d) for d in dims])
    combined = merge_shapes(value, dims)  # check that existing sizes match
    if not dims.without(shape(value)):  # no new dims to add
        if set(dims) == set(shape(value).only(dims)):  # sizes and item names might differ, though
            return value
    dims &= combined.non_uniform_shape  # add missing non-uniform dims
    # --- First try __expand__
    if hasattr(value, '__expand__'):
        result = value.__expand__(dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        new_attributes = {a: expand(getattr(value, a), dims, **kwargs) for a in all_attributes(value)}
        return copy_with(value, **new_attributes)
    # --- Fallback: stack ---
    if hasattr(value, '__stack__'):
        if dims.volume > 8:
            warnings.warn(f"expand() default implementation is slow on large shapes {dims}. Please implement __expand__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
        for dim in reversed(dims):
            value = stack((value,) * dim.size, dim, **kwargs)
            assert value is not NotImplemented, "Value must implement either __expand__ or __stack__"
        return value
    try:  # value may be a native scalar
        from ._tensors import expand_tensor, wrap
        value = wrap(value)
    except ValueError:
        raise AssertionError(f"Cannot expand non-shapable object {type(value)}")
    return expand_tensor(value, dims)


def squeeze(x: PhiTreeNodeType, dims: DimFilter) -> PhiTreeNodeType:
    """
    Remove specific singleton (volume=1) dims from `x`.

    Args:
        x: Tensor or composite type / tree.
        dims: Singleton dims to remove.

    Returns:
        Same type as `x`.
    """
    dims = shape(x).only(dims)
    if not dims:
        return x
    assert dims.volume == 1, f"Cannot squeeze non-singleton dims {dims} from {x}"
    return x[{d: 0 for d in dims.names}]


def rename_dims(value: PhiTreeNodeType,
                dims: DimFilter,
                names: DimFilter,
                **kwargs) -> PhiTreeNodeType:
    """
    Change the name and optionally the type of some dims of `value`.

    Dimensions that are not present on value will be ignored. The corresponding new dims given by `names` will not be added.

    Args:
        value: `Shape` or `Tensor` or `Shapable`.
        dims: Existing dims of `value` as comma-separated `str`, `tuple`, `list`, `Shape` or filter function.
        names: Either

            * Sequence of names matching `dims` as `tuple`, `list` or `str`. This replaces only the dimension names but leaves the types untouched.
            * `Shape` matching `dims` to replace names and types.
            * Dimension type function to replace only types.

        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.
    """
    if isinstance(value, SHAPE_TYPES):
        old_dims, new_dims = _shape_replace(value, dims, names)
        return value.replace(old_dims, new_dims)
    elif isinstance(value, (Number, bool)):
        return value
    if DEBUG_CHECKS:
        assert isinstance(value, Shapable) and isinstance(value, Shaped), f"value must be a Shape or Shapable but got {type(value).__name__}"
    old_dims, new_dims = _shape_replace(shape(value), dims, names)
    if not new_dims:
        return value
    if new_dims.names == old_dims.names and new_dims == old_dims:
        return value
    # --- First try __replace_dims__ ---
    if hasattr(value, '__replace_dims__'):
        result = value.__replace_dims__(old_dims.names, new_dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        return tree_map(rename_dims, value, all_attributes, treat_layout_as_leaf=True, dims=old_dims, names=new_dims, **kwargs)
    # --- Fallback: unstack and stack ---
    if shape(value).only(old_dims).volume > 8:
        warnings.warn(f"rename_dims() default implementation is slow on large dims ({old_dims}). Please implement __replace_dims__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
    for old_name, new_dim in zip(old_dims.names, new_dims):
        value = stack(unstack(value, old_name), new_dim, **kwargs)
    return value


def _shape_replace(shape: Shape, dims: DimFilter, new: DimFilter) -> Tuple[Shape, Shape]:  # _replace_names_and_types
    if callable(dims):
        existing = dims(shape)
    elif isinstance(dims, SHAPE_TYPES):
        existing = dims.only(shape)
    else:
        dims = parse_dim_order(dims)
        existing = shape.only(dims, reorder=True)
    if not existing:
        return EMPTY_SHAPE, EMPTY_SHAPE
    # --- Replace based on type(new) ---
    if isinstance(new, str) and new.startswith('(') and new.endswith(')'):
        item_names = [s.strip() for s in new[1:-1].split(',')]
        new = concat_shapes_(*[d.with_size(item_names) for d in existing])
    elif isinstance(new, str):
        new = parse_dim_order(new)
        assert len(new) == len(existing), f"Number of names {new} does not match dims to replace {existing}"
        new = concat_shapes_(*[Dim(n, dim.size, dim.dim_type, dim.slice_names) for dim, n in zip(existing, new)])
    elif callable(new):
        new = new(**existing.untyped_dict)
        assert len(existing) == len(new), f"Number of names {new} does not match dims to replace {dims}"
    elif isinstance(new, (tuple, list)):
        assert len(new) == len(existing), f"Number of names {new} does not match dims to replace {dims}"
        new_dims = []
        for dim, n in zip(existing, new):
            if isinstance(n, str):
                new_dims.append(Dim(n, dim.size, dim.dim_type, dim.slice_names))
            elif isinstance(n, Shape):
                new_dims.append(n.with_size(dim.slice_names or dim.size))
            else:
                raise ValueError(f"Invalid item in names: {n}")
        new = concat_shapes_(*new_dims)
    elif isinstance(new, Shape):
        if not callable(dims):
            if isinstance(dims, Shape):
                existing_idx = dims.indices(existing.names)
            elif isinstance(dims, (tuple, list)):
                existing_idx = [dims.index(n) for n in existing.names]
            else:
                raise NotImplementedError
            new = new[existing_idx]
        if not new.well_defined:
            new = new.with_sizes(existing.sizes)
    else:
        raise ValueError(new)
    return existing, new



def b2i(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *batch* dims of `value` to *instance* dimensions. See `rename_dims`. """
    return rename_dims(value, batch, instance)


def c2b(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *channel* dims of `value` to *batch* dimensions. See `rename_dims`. """
    return rename_dims(value, channel, batch)


def s2b(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *spatial* dims of `value` to *batch* dimensions. See `rename_dims`. """
    return rename_dims(value, spatial, batch)


def si2d(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *spatial* and *instance* dims of `value` to *dual* dimensions. See `rename_dims`. """
    return rename_dims(value, lambda s: s.non_channel.non_dual.non_batch, dual)


def c2d(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *channel* dims of `value` to *dual* dimensions. See `rename_dims`. """
    return rename_dims(value, channel, dual)


def p2d(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *primal* dims (instance, spatial, channel) of `value` to *dual* dimensions. See `rename_dims`. """
    return rename_dims(value, primal, dual)


def i2b(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *instance* dims of `value` to *batch* dimensions. See `rename_dims`. """
    return rename_dims(value, instance, batch)


def d2i(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *dual* dims of `value` to *instance* dimensions. See `rename_dims`. """
    return rename_dims(value, dual, instance)


def d2s(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *dual* dims of `value` to *spatial* dimensions. See `rename_dims`. """
    return rename_dims(value, dual, spatial)


def pack_dims(value, dims: DimFilter, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """
    Compresses multiple dims into a single dimension by concatenating the elements.
    Elements along the new dims are laid out according to the order of `dims`.
    If the order of `dims` differs from the current dimension order, the tensor is transposed accordingly.
    This function replaces the traditional `reshape` for these cases.

    The type of the new dimension will be equal to the types of `dims`.
    If `dims` have varying types, the new dimension will be a batch dimension.

    If none of `dims` exist on `value`, `packed_dim` will be added only if it is given with a definite size and `value` is not a primitive type.

    See Also:
        `unpack_dim()`

    Args:
        value: `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`.
        dims: Dimensions to be compressed in the specified order.
        packed_dim: Single-dimension `Shape`.
        pos: Index of new dimension. `None` for automatic, `-1` for last, `0` for first.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> pack_dims(math.zeros(spatial(x=4, y=3)), spatial, instance('points'))
        (pointsⁱ=12) const 0.0
    """
    if isinstance(value, (Number, bool)):
        return value
    if DEBUG_CHECKS:
        assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    packed_dim = auto(packed_dim, dims if callable(dims) else None) if isinstance(packed_dim, str) else packed_dim
    dims = shape(value).only(dims, reorder=True)
    if packed_dim in shape(value):
        assert packed_dim in dims, f"Cannot pack dims into new dimension {packed_dim} because it already exists on value {value} and is not packed."
    if len(dims) == 0 or all(dim not in shape(value) for dim in dims):
        return value if packed_dim.size is None else expand(value, packed_dim, **kwargs)  # Inserting size=1 can cause shape errors
    elif len(dims) == 1 and packed_dim.rank == 1:
        return rename_dims(value, dims, packed_dim, **kwargs)
    elif len(dims) == 1 and packed_dim.rank > 1:
        return unpack_dim(value, dims, packed_dim, **kwargs)
    # --- First try __pack_dims__ ---
    if hasattr(value, '__pack_dims__'):
        result = value.__pack_dims__(dims, packed_dim, pos, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        return tree_map(pack_dims, value, attr_type=all_attributes, dims=dims, packed_dim=packed_dim, pos=pos, **kwargs)
    # --- Fallback: unstack and stack ---
    if shape(value).only(dims).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dims ({shape(value).only(dims)}). Please implement __pack_dims__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
    return stack(unstack(value, dims), packed_dim, **kwargs)


def dpack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=dual)"""
    return pack_dims(value, dual, packed_dim, pos=pos, **kwargs)


def spack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=spatial)"""
    return pack_dims(value, spatial, packed_dim, pos=pos, **kwargs)


def ipack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=instance)"""
    return pack_dims(value, instance, packed_dim, pos=pos, **kwargs)


def cpack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=channel)"""
    return pack_dims(value, channel, packed_dim, pos=pos, **kwargs)


def unpack_dim(value, dim: DimFilter, *unpacked_dims: Union[Shape, Sequence[Shape]], **kwargs):
    """
    Decompresses a dimension by unstacking the elements along it.
    This function replaces the traditional `reshape` for these cases.
    The compressed dimension `dim` is assumed to contain elements laid out according to the order of `unpacked_dims`.

    If `dim` does not exist on `value`, this function will return `value` as-is. This includes primitive types.

    See Also:
        `pack_dims()`

    Args:
        value: `phiml.math.magic.Shapable`, such as `Tensor`, for which one dimension should be split.
        dim: Single dimension to be decompressed.
        *unpacked_dims: Either vararg `Shape`, ordered dims to replace `dim`, fulfilling `unpacked_dims.volume == shape(self)[dim].rank`.
            This results in a single tensor output.
            Alternatively, pass a `tuple` or `list` of shapes to unpack a dim into multiple tensors whose combined volumes match `dim.size`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> unpack_dim(math.zeros(instance(points=12)), 'points', spatial(x=4, y=3))
        (xˢ=4, yˢ=3) const 0.0
    """
    if len(unpacked_dims) == 1 and isinstance(unpacked_dims[0], (tuple, list)):
        from ._ops import unflatten_unpack
        return unflatten_unpack(value, dim, unpacked_dims[0])
    if isinstance(value, (Number, bool)):
        return value
    if DEBUG_CHECKS:
        assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    dim = shape(value).only(dim)
    if dim.is_empty:
        return value  # Nothing to do, maybe expand?
    assert dim.rank == 1, f"unpack_dim requires as single dimension to be unpacked but got {dim}"
    dim = dim.name
    unpacked_dims = concat_shapes_(*unpacked_dims)
    if unpacked_dims.rank == 0:
        return value[{dim: 0}]  # remove dim
    elif unpacked_dims.rank == 1:
        return rename_dims(value, dim, unpacked_dims, **kwargs)
    # --- First try __unpack_dim__
    if hasattr(value, '__unpack_dim__'):
        result = value.__unpack_dim__(dim, unpacked_dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode) and all_attributes(value):
        new_attributes = {a: unpack_dim(getattr(value, a), dim, unpacked_dims, **kwargs) for a in all_attributes(value)}
        return copy_with(value, **new_attributes)
    # --- Fallback: unstack and stack ---
    if shape(value).only(dim).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dims ({shape(value).only(dim)}). Please implement __unpack_dim__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
    unstacked = unstack(value, dim)
    for dim in reversed(unpacked_dims):
        unstacked = [stack(unstacked[i:i+dim.size], dim, **kwargs) for i in range(0, len(unstacked), dim.size)]
    return unstacked[0]


def flatten(value, flat_dim: Shape = instance('flat'), flatten_batch=False, **kwargs):
    """
    Returns a `Tensor` with the same values as `value` but only a single dimension `flat_dim`.
    The order of the values in memory is not changed.

    Args:
        value: `phiml.math.magic.Shapable`, such as `Tensor`.
            If a non-`phiml.math.magic.Shaped` object or one with an empty `Shape` is passed, it is returned without alteration.
        flat_dim: Dimension name and type as `Shape` object. The size is ignored.
        flatten_batch: Whether to flatten batch dims as well.
            If `False`, batch dims are kept, only onn-batch dims are flattened.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> flatten(math.zeros(spatial(x=4, y=3)))
        (flatⁱ=12) const 0.0
    """
    assert isinstance(flat_dim, SHAPE_TYPES) and flat_dim.rank == 1, flat_dim
    if not isinstance(value, Shaped):
        return value
    if shape(value).is_empty:
        return value
    assert isinstance(value, Shapable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    # --- First try __flatten__ ---
    if hasattr(value, '__flatten__'):
        result = value.__flatten__(flat_dim, flatten_batch, **kwargs)
        if result is not NotImplemented:
            return result
    # There is no tree node implementation for flatten because pack_dims is just as fast
    # --- Fallback: pack_dims ---
    return pack_dims(value, shape(value) if flatten_batch else non_batch(value), flat_dim, **kwargs)


def variable_attributes(obj) -> Tuple[str, ...]:
    if hasattr(obj, '__variable_attrs__'):
        result = obj.__variable_attrs__()
        assert isinstance(result, tuple), f"__variable_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif hasattr(obj, '__all_attrs__'):
        result = obj.__all_attrs__()
        assert isinstance(result, tuple), f"__all_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif hasattr(obj, '__value_attrs__'):
        result = obj.__value_attrs__()
        assert isinstance(result, tuple), f"__value_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif dataclasses.is_dataclass(obj):
        if hasattr(obj, 'variable_attrs'):
            result = obj.variable_attrs
            assert isinstance(result, tuple), f"dataclass.variable_attrs must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
            return result
        return all_attributes(obj)
    else:
        raise ValueError(f"Not a PhiTreeNode: {type(obj).__name__}")


def value_attributes(obj) -> Tuple[str, ...]:
    if hasattr(obj, '__value_attrs__'):
        result = obj.__value_attrs__()
        assert isinstance(result, tuple), f"__value_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif hasattr(obj, '__all_attrs__'):
        result = obj.__all_attrs__()
        assert isinstance(result, tuple), f"__all_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    if dataclasses.is_dataclass(obj):
        if hasattr(obj, 'value_attrs'):
            result = obj.value_attrs
            assert isinstance(result, tuple), f"dataclass.value_attrs must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
            return result
        return all_attributes(obj)
    raise ValueError(f"{type(obj).__name__} must implement '__value_attrs__()' or be a dataclass to be used with value functions.")


def all_attributes(obj, assert_any=False) -> Tuple[str, ...]:
    if hasattr(obj, '__all_attrs__'):
        result = obj.__all_attrs__()
        assert isinstance(result, tuple), f"__value_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    if not isinstance(obj, PhiTreeNode):
        raise ValueError(f"Not a PhiTreeNode: {type(obj).__name__}")
    result = set()
    if hasattr(obj, '__variable_attrs__'):
        result.update(obj.__variable_attrs__())
    if hasattr(obj, '__value_attrs__'):
        result.update(obj.__value_attrs__())
    if dataclasses.is_dataclass(obj) and not hasattr(obj, '__variable_attrs__') and not hasattr(obj, '__value_attrs__'):
        from phiml.dataclasses import data_fields
        result.update([f.name for f in data_fields(obj)])
    if assert_any:
        assert result, f"{type(obj).__name__} is not a valid tree node because it has no tensor-like attributes."
    return tuple(sorted(result))


def replace(obj: PhiTreeNodeType, **updates) -> PhiTreeNodeType:
    """
    Creates a copy of the given `phiml.math.magic.PhiTreeNode` with updated values as specified in `updates`.

    If `obj` overrides `__with_attrs__`, the copy will be created via that specific implementation.
    Otherwise, the `copy` module and `setattr` will be used.

    Args:
        obj: `phiml.math.magic.PhiTreeNode`
        **updates: Values to be replaced.

    Returns:
        Copy of `obj` with updated values.
    """
    if isinstance(obj, (Number, bool)):
        return obj
    elif hasattr(obj, '__with_attrs__'):
        result = obj.__with_attrs__(**updates)
        if result is not NotImplemented:
            return result
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **updates)
    else:
        cpy = copy.copy(obj)
        for attr, value in updates.items():
            setattr(cpy, attr, value)
        return cpy


copy_with = replace


# Other Ops

MagicType = TypeVar('MagicType')
OtherMagicType = TypeVar('OtherMagicType')


def cast(x: MagicType, dtype: Union[DType, type]) -> OtherMagicType:
    """
    Casts `x` to a different data type.

    Implementations:

    * NumPy: [`x.astype()`](numpy.ndarray.astype)
    * PyTorch: [`x.to()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to)
    * TensorFlow: [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/cast)
    * Jax: [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)

    See Also:
        `to_float`, `to_int32`, `to_int64`, `to_complex`.

    Args:
        x: `Tensor`
        dtype: New data type as `phiml.math.DType`, e.g. `DType(int, 16)`.

    Returns:
        `Tensor` with data type `dtype`
    """
    if not isinstance(dtype, DType):
        dtype = DType.as_dtype(dtype)
    if hasattr(x, '__cast__'):
        return x.__cast__(dtype)
    elif isinstance(x, (Number, bool)):
        return dtype.kind(x)
    elif isinstance(x, PhiTreeNode):
        attrs = {key: getattr(x, key) for key in value_attributes(x)}
        new_attrs = {k: cast(v, dtype) for k, v in attrs.items()}
        return copy_with(x, **new_attrs)
    try:
        backend = choose_backend(x)
        return backend.cast(x, dtype)
    except NoBackendFound:
        if dtype.kind == bool:
            return bool(x)
        raise ValueError(f"Cannot cast object of type '{type(x).__name__}'")


def bool_to_int(x: MagicType, bits=32):
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, Number):
        return x
    if hasattr(x, 'dtype') and isinstance(x.dtype, DType):
        return cast(x, DType(int, bits)) if x.dtype.kind == bool else x
    elif isinstance(x, PhiTreeNode):
        return tree_map(bool_to_int, x, bits=32)
    try:
        backend = choose_backend(x)
        return backend.cast(x, DType(int, bits)) if backend.dtype(x).kind == bool else x
    except NoBackendFound:
        raise ValueError(f"Cannot cast object of type '{type(x).__name__}'")


def tree_map(f, tree, attr_type=value_attributes, include_non_attrs=True, treat_layout_as_leaf=False, **f_kwargs):
    """
    Recursively iterates over Layouts, lists, tuples, dicts and the value attributes of PhiTreeNodes.
    Calls `f` on `Tensor` instances and everything else not in the above list.

    Args:
        f: Function mapping `Tensor`s or leaves
        tree: Nested tree or leaf
        attr_type: Which attributes to use for PhiTreeNodes. Typically, either `value_attributes` or `variable_attributes`.
        include_non_attrs: Whether to invoke `f` on non-attribute types, such as `str`, `Shape` or primitives.
        treat_layout_as_leaf: Whether to call `f` directly on `Layout` instances instead of their children.
        **f_kwargs: Keyword arguments for `f`.

    Returns:
        Tree matching structure of `tree` with transformed leaf values.
    """
    from ._tensors import Tensor, Layout
    if isinstance(tree, Layout):
        if treat_layout_as_leaf:
            return f(tree, **f_kwargs)
        else:
            return tree._op1(lambda x: tree_map(f, x, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs))
    if isinstance(tree, Tensor) or tree is None:
        return f(tree, **f_kwargs)
    if isinstance(tree, list):
        return [tree_map(f, e, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for e in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(f, e, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for e in tree])
    elif isinstance(tree, dict):
        return {k: tree_map(f, e, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for k, e in tree.items()}
    elif isinstance(tree, PhiTreeNode):
        attrs = {key: getattr(tree, key) for key in attr_type(tree)}
        new_attrs = {k: tree_map(f, v, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for k, v in attrs.items()}
        return copy_with(tree, **new_attrs)
    else:
        from phiml.dataclasses._dataclasses import NON_ATTR_TYPES
        if include_non_attrs or not isinstance(tree, NON_ATTR_TYPES):
            return f(tree, **f_kwargs)  # try anyway
        return tree


def find_differences(tree1, tree2, compare_tensors_by_id=False, attr_type=value_attributes, tensor_equality=None) -> Sequence[Tuple[str, str, Any, Any]]:
    """
    Compares `tree1` and `tree2` and returns all differences in the form `(difference_description: str, variable_identifier: str, value1, value2)`.

    Args:
        tree1: Nested tree or leaf
        tree2: Nested tree or leaf
        compare_tensors_by_id: Whether `phiml.math.Tensor` objects should be compared by identity or values.
        attr_type: What attributes to compare, either `value_attributes` or `variable_attributes`.
        tensor_equality: Function that compares two tensors for equality. `None` defaults to `equal`.

    Returns:
        List of differences, each represented as a `tuple`.
    """
    result = []
    _recursive_diff(tree1, tree2, '', result, compare_tensors_by_id, attr_type, tensor_equality)
    return result


def _recursive_diff(a, b, path: str, result: list, compare_tensors_by_id=False, attr_type=value_attributes, tensor_equality=None):
    if a is b:
        return
    if (a is None) != (b is None):
        result.append(("Only one tree has a value, other is None", path, a, b))
        return
    from ._tensors import Tensor, Layout
    if isinstance(a, Layout):
        raise NotImplementedError
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            result.append(("Only one value is a Tensor", path, a, b))
            return
        if compare_tensors_by_id:
            if a is not b:
                result.append(("Tensor ids do not match", path, a, b))
        else:
            if tensor_equality is None:
                from ._ops import equal
                tensor_equality = partial(equal, equal_nan=True)
            if a.shape != b.shape:
                result.append(("Tensor shapes do not match", path, a, b))
            elif not tensor_equality(a, b):
                result.append(("Tensor values do not match", path, a, b))
    elif type(a) != type(b):
        result.append(("Types do not match", path, a, b))
        return
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            result.append(("Lengths do not match", path, a, b))
        else:
            for i, (ae, be) in enumerate(zip(a, b)):
                _recursive_diff(ae, be, f"{path}[{i}]", result, compare_tensors_by_id, attr_type, tensor_equality)
    elif isinstance(a, dict):
        if set(a) != set(b):
            result.append(("Keys do not match", path, a, b))
        else:
            for k, av in a.items():
                bv = b[k]
                _recursive_diff(av, bv, f"{path}[{k}]", result, compare_tensors_by_id, attr_type, tensor_equality)
    elif isinstance(a, PhiTreeNode):
        a_attrs = attr_type(a)
        if set(a_attrs) != set(attr_type(b)):
            result.append(("Keys do not match", path, a, b))
        else:
            for k in a_attrs:
                av = getattr(a, k)
                bv = getattr(b, k)
                _recursive_diff(av, bv, f"{path}.{k}", result, compare_tensors_by_id, attr_type, tensor_equality)
    else:
        try:
            backend = choose_backend(a, b)
            if backend.shape(a) != backend.shape(b):
                result.append(("Native tensor shapes do not match", path, a, b))
            equal_tensor = backend.equal(a, b) | (backend.isnan(a) & backend.isnan(b))
            equal = backend.numpy(backend.all(equal_tensor))
            if not equal:
                result.append(("Native tensor values do not match", path, a, b))
        except NoBackendFound:
            if a != b:
                result.append(("Values do not match", path, a, b))
