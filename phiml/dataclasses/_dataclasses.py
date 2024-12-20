import collections
import dataclasses
import inspect
from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, Callable, Tuple, List, Set, Iterable, Optional, get_origin, get_args, Dict, Sequence, Union, Any

from phiml.dataclasses._dep import get_unchanged_cache
from phiml.math import DimFilter, shape, Shape
from phiml.math._magic_ops import slice_, variable_attributes
from phiml.math._shape import SHAPE_TYPES
from phiml.math._tensors import disassemble_tree, Tensor, assemble_tree, equality_by_shape_and_value, equality_by_ref
from phiml.math.magic import slicing_dict, BoundDim

PhiMLDataclass = TypeVar("PhiMLDataclass")


def sliceable(cls=None, /, *, dim_attrs=True, keepdims=None, dim_repr=True):
    """
    Decorator for frozen dataclasses, adding slicing functionality by defining `__getitem__`.
    This enables slicing similar to tensors, gathering and boolean masking.

    Args:
        dim_attrs: Whether to generate `__getattr__` that allows slicing via the syntax `instance.dim[...]` where `dim` is the name of any dim present on `instance`.
        keepdims: Which dimensions should be kept with size 1 taking a single slice along them. This will preserve item names.
        dim_repr: Whether to replace the default `repr` of a dataclass by a simplified one based on the object's shape.
    """
    def wrap(cls):
        assert dataclasses.is_dataclass(cls), f"@sliceable must be used on a @dataclass, i.e. declared above it."
        assert cls.__dataclass_params__.frozen, f"@sliceable dataclasses must be frozen. Declare as @dataclass(frozen=True)"
        assert data_fields(cls), f"PhiML dataclasses must have at least one field storing a Shaped object, such as a Tensor, tree of Tensors or compatible dataclass."
        if not hasattr(cls, '__getitem__'):
            def __dataclass_getitem__(obj, item):
                return getitem(obj, item, keepdims=keepdims)
            cls.__getitem__ = __dataclass_getitem__
        if dim_attrs and not hasattr(cls, '__getattr__'):
            def __dataclass_getattr__(obj, name: str):
                if name in ('shape', '__shape__', '__all_attrs__', '__variable_attrs__', '__value_attrs__', '__setstate__'):  # these can cause infinite recursion
                    raise AttributeError(f"'{type(obj)}' instance has no attribute '{name}'")
                if name in shape(obj):
                    return BoundDim(obj, name)
                else:
                    raise AttributeError(f"'{type(obj)}' instance has no attribute '{name}'")
            cls.__getattr__ = __dataclass_getattr__
        if dim_repr:
            def __dataclass_repr__(obj):
                try:
                    content = shape(obj)
                    if not content:
                        content = f"{', '.join([f'{f.name}={getattr(obj, f.name)}' for f in dataclasses.fields(cls)])}"
                except BaseException as err:
                    content = f"Unknown shape: {type(err).__name__}"
                return f"{type(obj).__name__}[{content}]"
            cls.__repr__ = __dataclass_repr__
        return cls
    return wrap(cls) if cls is not None else wrap  # See if we're being called as @dataclass or @dataclass().


def data_eq(cls=None, /, *, rel_tolerance=0., abs_tolerance=0., equal_nan=True, compare_tensors_by_ref=False):
    def wrap(cls):
        assert cls.__dataclass_params__.eq, f"@data_eq can only be used with dataclasses with eq=True."
        cls.__default_dataclass_eq__ = cls.__eq__
        def __tensor_eq__(obj, other):
            if compare_tensors_by_ref:
                with equality_by_ref():
                    return cls.__default_dataclass_eq__(obj, other)
            with equality_by_shape_and_value(rel_tolerance, abs_tolerance, equal_nan):
                return cls.__default_dataclass_eq__(obj, other)
        cls.__eq__ = __tensor_eq__
        # __ne__ calls `not __eq__()` by default
        return cls
    return wrap(cls) if cls is not None else wrap  # See if we're being called as @dataclass or @dataclass().


NON_ATTR_TYPES = str, int, float, complex, bool, Shape, slice, Callable


def data_fields(obj) -> Sequence[dataclasses.Field]:
    """
    List all dataclass Fields of `obj` that are considered data, i.e. can hold (directly or indirectly) one or multiple `Tensor` instances.
    This includes fields referencing other dataclasses.

    Args:
        obj: Dataclass type or instance.

    Returns:
        Sequence of `dataclasses.Field`.
    """
    return [f for f in dataclasses.fields(obj) if is_data_field(f)]


def non_data_fields(obj) -> Sequence[dataclasses.Field]:
    """
    List all dataclass Fields of `obj` that cannot hold tensors (directly or indirectly).

    Args:
        obj: Dataclass type or instance.

    Returns:
        Sequence of `dataclasses.Field`.
    """
    return [f for f in dataclasses.fields(obj) if not is_data_field(f)]


def config_fields(obj) -> Sequence[dataclasses.Field]:
    """
    List all dataclass Fields of `obj` that are not considered data_fields or special.
    These cannot hold any Tensors or shaped objects.

    Args:
        obj: Dataclass type or instance.

    Returns:
        Sequence of `dataclasses.Field`.
    """
    return [f for f in dataclasses.fields(obj) if not is_data_field(f) and f.name not in ('variable_attrs', 'value_attrs')]


def special_fields(obj) -> Sequence[dataclasses.Field]:
    """
    List all special dataclass Fields of `obj`, i.e. fields that don't store data related to the object but rather meta-information relevant to PhiML.

    These include `variable_attrs` and `value_attrs`.

    Args:
        obj: Dataclass type or instance.

    Returns:
        Sequence of `dataclasses.Field`.
    """
    return [f for f in dataclasses.fields(obj) if f.name in ('variable_attrs', 'value_attrs')]


def is_data_field(field: dataclasses.Field):
    if field.name in ('variable_attrs', 'value_attrs'):  # this check is not strictly necessary since the types of special fields cannot hold tensors
        return False
    primitives = _get_primitive_types(field.type)
    return any(p not in NON_ATTR_TYPES for p in primitives)


def _get_primitive_types(field_type) -> list:
    """Returns None for unknown types."""
    if field_type is Ellipsis:
        return []
    origin_type = get_origin(field_type)
    if origin_type in {list, List, tuple, Tuple, set, Set, Iterable, Optional, collections.abc.Sequence}:
        args = get_args(field_type)  # The arguments passed to the generic (e.g., List[int] -> (int,))
        return sum([_get_primitive_types(a) for a in args], []) if args else [None]
    elif origin_type in {Dict, dict}:
        k_type, v_type = get_args(field_type)
        return _get_primitive_types(v_type)
    else:
        return [field_type]


def replace(obj: PhiMLDataclass, /, call_metaclass=False, **changes) -> PhiMLDataclass:
    """
    Create a copy of `obj` with some fields replaced.
    Unlike `dataclasses.replace()`, this function also transfers `@cached_property` members if their dependencies are not affected.

    Args:
        obj: Dataclass instance.
        call_metaclass: Whether to copy `obj` by invoking `type(obj).__call__`.
            If `obj` defines a metaclass, this will allow users to define custom constructors for dataclasses.
        **changes: New field values to replace old ones.

    Returns:
        Copy of `obj` with replaced values.
    """
    cls = obj.__class__
    kwargs = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    kwargs.update(**changes)
    if call_metaclass:
        new_obj = cls(**kwargs)
    else:  # This allows us override the dataclass constructor with a metaclass for user convenience, but not call it internally.
        new_obj = cls.__new__(cls)
        new_obj.__init__(**kwargs)
    cache = get_unchanged_cache(obj, set(changes.keys()))
    new_obj.__dict__.update(cache)
    return new_obj


@dataclass
class DataclassTreeNode:
    cls: type
    attr_type: Callable
    extracted: Dict[str, Any]  # trees without variable tensors
    not_extracted: Dict[str, Any]  # original values of non-extracted properties
    cache: Dict[str, Any]


def disassemble(obj: PhiMLDataclass, attr_type=variable_attributes):
    extract_names = attr_type(obj)
    keys = {}
    values = []
    for attr in extract_names:
        key, value = disassemble_tree(getattr(obj, attr), False, attr_type)
        keys[attr] = key
        values.extend(value)
    non_attributes = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.name not in keys}
    cache = get_unchanged_cache(obj, set(extract_names))
    return DataclassTreeNode(type(obj), attr_type, keys, non_attributes, cache), values


def assemble(container: DataclassTreeNode, values: List[Tensor]):
    extracted = {a: assemble_tree(v, values, container.attr_type) for a, v in container.extracted.items()}
    instance = container.cls.__new__(container.cls)
    instance.__init__(**extracted, **container.not_extracted)
    instance.__dict__.update(container.cache)
    return instance


def getitem(obj: PhiMLDataclass, item, keepdims: DimFilter = None) -> PhiMLDataclass:
    """
    Slice / gather a dataclass by broadcasting the operation to its data_fields.

    You may call this from `__getitem__` to allow the syntax `my_class[component_str]`, `my_class[slicing_dict]`, `my_class[boolean_tensor]` and `my_class[index_tensor]`.

    ```python
    def __getitem__(self, item):
        return getitem(self, item)
    ```

    Args:
        obj: Dataclass instance to slice / gather.
        item: One of the supported tensor slicing / gathering values.
        keepdims: Dimensions that will not be removed during slicing.
            When selecting a single slice, these dims will remain with size 1.

    Returns:
        Slice of `obj` of same type.
    """
    assert dataclasses.is_dataclass(obj), f"obj must be a dataclass but got {type(obj)}"
    item = slicing_dict(obj, item)
    if keepdims:
        keep = shape(obj).only(keepdims)
        for dim, sel in item.items():
            if dim in keep:
                if isinstance(sel, int):
                    item[dim] = slice(sel, sel+1)
                elif isinstance(sel, str) and ',' not in sel:
                    item[dim] = [sel]
    if not item:
        return obj
    attrs = data_fields(obj)
    kwargs = {f.name: slice_(getattr(obj, f.name), item) if f in attrs else getattr(obj, f.name) for f in dataclasses.fields(obj)}
    cls = type(obj)
    new_obj = cls.__new__(cls, **kwargs)
    new_obj.__init__(**kwargs)
    cache = {k: slice_(v, item) for k, v in obj.__dict__.items() if isinstance(getattr(type(obj), k, None), cached_property) and not isinstance(v, SHAPE_TYPES)}
    new_obj.__dict__.update(cache)
    return new_obj


def equal(obj1, obj2, rel_tolerance=0., abs_tolerance=0., equal_nan=True):
    """
    Checks if two

    Args:
        obj1:
        obj2:
        rel_tolerance:
        abs_tolerance:
        equal_nan:

    Returns:

    """
    cls = type(obj1)
    eq_fn = cls.__default_dataclass_eq__ if hasattr(cls, '__default_dataclass_eq__') else cls.__eq__
    with equality_by_shape_and_value(rel_tolerance, abs_tolerance, equal_nan):
        return eq_fn(obj1, obj2)
