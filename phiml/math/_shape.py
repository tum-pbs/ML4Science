import re
import sys
import warnings
from dataclasses import dataclass, replace
from functools import cached_property
from numbers import Number
from typing import Tuple, Callable, List, Union, Any, Sequence, Optional, Dict, Protocol, runtime_checkable

from .. import math


BATCH_DIM = 'batch'
SPATIAL_DIM = 'spatial'
CHANNEL_DIM = 'channel'
INSTANCE_DIM = 'instance'
DUAL_DIM = 'dual'
DIM_TYPES = (BATCH_DIM, DUAL_DIM, INSTANCE_DIM, SPATIAL_DIM, CHANNEL_DIM)
TYPE_INDEX = {t: i for i, t in enumerate(DIM_TYPES)}
PRIMAL_TYPES = {INSTANCE_DIM, SPATIAL_DIM, CHANNEL_DIM}

SUPERSCRIPT = {SPATIAL_DIM: "ˢ", CHANNEL_DIM: "ᶜ", INSTANCE_DIM: "ⁱ", BATCH_DIM: "ᵇ", DUAL_DIM: "ᵈ", None: "⁻"}  # ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ
CHAR = {SPATIAL_DIM: "s", CHANNEL_DIM: "c", INSTANCE_DIM: "i", BATCH_DIM: "b", DUAL_DIM: "d", None: "-"}  # ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ
INV_CHAR = {v: k for k, v in CHAR.items()}

DEBUG_CHECKS = []


def enable_debug_checks():
    """
    Once called, additional type checks are enabled.
    This may result in a noticeable drop in performance.
    """
    DEBUG_CHECKS.append(True)


class ShapeMeta(type(Protocol)):

    def __instancecheck__(self, obj):
        return isinstance(obj, (Dim, PureShape, MixedShape))

    def __subclasscheck__(self, subclass):
        return subclass in {Dim, PureShape, MixedShape}


@runtime_checkable
class Shape(Protocol, metaclass=ShapeMeta):

    @property
    def names(self) -> Sequence[str]:
        """
        Ordered dimension names as `tuple[str]`.

        See Also:
            `Shape.name`.
        """
        ...

    @property
    def sizes(self) -> Sequence:
        """
        Ordered dimension sizes as `tuple`.
        The size of a dimension can be an `int` or a `Tensor` for [non-uniform shapes](https://tum-pbs.github.io/PhiML/Non_Uniform.html).

        See Also:
            `Shape.get_size()`, `Shape.size`, `Shape.shape`.
        """
        ...

    @property
    def types(self) -> Sequence[Callable]:
        ...

    @property
    def dim_types(self) -> Sequence[str]:
        ...

    @property
    def item_names(self) -> Sequence[Optional[Sequence[str]]]:
        ...

    @property
    def name_list(self) -> List[str]:
        ...

    @property
    def untyped_dict(self) -> dict:
        """
        Returns:
            `dict` containing dimension names as keys.
                The values are either the item names as `tuple` if available, otherwise the size.
        """
        ...

    def __len__(self):
        ...

    def __contains__(self, item):
        ...

    def isdisjoint(self, other: Union['Shape', tuple, list, str]):
        """ Shapes are disjoint if all dimension names of one shape do not occur in the other shape. """
        ...

    def __iter__(self):
        ...

    def index(self, dim: Union[str, 'Shape', None]) -> int:
        """
        Finds the index of the dimension within this `Shape`.

        See Also:
            `Shape.indices()`.

        Args:
            dim: Dimension name or single-dimension `Shape`.

        Returns:
            Index as `int`.
        """
        ...

    def indices(self, names: Sequence[str]) -> Tuple[int, ...]:
        """
        Finds the indices of the given dimensions within this `Shape`.

        See Also:
            `Shape.index()`.

        Args:
            names: Sequence of dim names as `tuple` or `list`. No name can occur in `names` more than once.

        Returns:
            Indices as `tuple[int]`.
        """
        ...

    def get_size(self, dim: Union[str, 'Shape', int], default=None):
        """

        Args:
            dim: Dimension, either as name `str` or single-dimension `Shape` or index `int`.
            default: (Optional) If the dim does not exist, return this value instead of raising an error.

        Returns:
            Size associated with `dim` as `int` or `Tensor`.
        """
        ...

    def get_dim_type(self, dim: str) -> str:
        """
        Args:
            dim: Dimension, either as name `str` or single-dimension `Shape`.

        Returns:
            Dimension type, one of `batch`, `spatial`, `instance`, `channel`.
        """
        ...

    def get_item_names(self, dim: Union[str, 'Shape', int], fallback_spatial=False) -> Union[tuple, None]:
        """
        Args:
            fallback_spatial: If `True` and no item names are defined for `dim` and `dim` is a channel dimension, the spatial dimension names are interpreted as item names along `dim` in the order they are listed in this `Shape`.
            dim: Dimension, either as `int` index, `str` name or single-dimension `Shape`.

        Returns:
            Item names as `tuple` or `None` if not defined.
        """
        ...

    def flipped(self, dims: Union[List[str], Tuple[str]]):
        ...

    def __getitem__(self, selection):
        ...

    @property
    def reversed(self):
        return ...

    @property
    def batch(self) -> 'Shape':
        """
        Filters this shape, returning only the batch dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def non_batch(self) -> 'Shape':
        """
        Filters this shape, returning only the non-batch dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def spatial(self) -> 'Shape':
        """
        Filters this shape, returning only the spatial dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def non_spatial(self) -> 'Shape':
        """
        Filters this shape, returning only the non-spatial dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def instance(self) -> 'Shape':
        """
        Filters this shape, returning only the instance dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def non_instance(self) -> 'Shape':
        """
        Filters this shape, returning only the non-instance dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def channel(self) -> 'Shape':
        """
        Filters this shape, returning only the channel dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def non_channel(self) -> 'Shape':
        """
        Filters this shape, returning only the non-channel dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def dual(self) -> 'Shape':
        """
        Filters this shape, returning only the dual dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def non_dual(self) -> 'Shape':
        """
        Filters this shape, returning only the non-dual dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def primal(self) -> 'Shape':
        """
        Filters this shape, returning only the dual dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def non_primal(self) -> 'Shape':
        """
        Filters this shape, returning only batch and dual dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def non_singleton(self) -> 'Shape':
        """
        Filters this shape, returning only non-singleton dimensions as a new `Shape` object.
        Dimensions are singleton if their size is exactly `1`.

        Returns:
            New `Shape` object
        """
        ...

    @property
    def singleton(self) -> 'Shape':
        """
        Filters this shape, returning only singleton dimensions as a new `Shape` object.
        Dimensions are singleton if their size is exactly `1`.

        Returns:
            New `Shape` object
        """
        ...

    def as_channel(self):
        """Returns a copy of this `Shape` with all dimensions of type *channel*."""
        ...

    def as_batch(self):
        """Returns a copy of this `Shape` with all dimensions of type *batch*."""
        ...

    def as_spatial(self):
        """Returns a copy of this `Shape` with all dimensions of type *spatial*."""
        ...

    def as_instance(self):
        """Returns a copy of this `Shape` with all dimensions of type *instance*."""
        ...

    def as_dual(self):
        """Returns a copy of this `Shape` with all dimensions of type *dual*."""
        ...

    def as_type(self, new_type: Callable):
        """Returns a copy of this `Shape` with all dimensions of the given type, either `batch`, `dual`, `spatial`, `instance`, or `channel` ."""
        ...

    def transpose(self, dim_type: str):
        ...

    @property
    def name(self) -> str:
        """
        Only for Shapes containing exactly one single dimension.
        Returns the name of the dimension.

        See Also:
            `Shape.names`.
        """
        ...

    @property
    def size(self):
        """
        Only for Shapes containing exactly one single dimension.
        Returns the size of the dimension.

        See Also:
            `Shape.sizes`, `Shape.get_size()`.
        """
        ...

    @property
    def type(self) -> Callable:
        """
        Only for Shapes containing exactly one single dimension.
        Returns the type of the dimension.

        See Also:
            `Shape.get_type()`.
        """
        ...

    @property
    def dim_type(self) -> str:
        ...

    def mask(self, names: Union[tuple, list, set, 'Shape']):
        """
        Returns a binary sequence corresponding to the names of this Shape.
        A value of 1 means that a dimension of this Shape is contained in `names`.

        Args:
          names: instance of dimension
          names: tuple or list or set:

        Returns:
          binary sequence

        """
        ...

    def without(self, dims: 'DimFilter') -> 'Shape':
        """
        Builds a new shape from this one that is missing all given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.

        The complementary operation is `Shape.only()`.

        Args:
          dims: Single dimension (str) or instance of dimensions (tuple, list, Shape)
          dims: Dimensions to exclude as `str` or `tuple` or `list` or `Shape`. Dimensions that are not included in this shape are ignored.

        Returns:
          Shape without specified dimensions
        """
        ...

    def __and__(self, other) -> 'Shape':
        ...

    def __add__(self, other) -> 'Shape':
        ...

    def __sub__(self, other) -> 'Shape':
        ...

    def only(self, dims: 'DimFilter', reorder=False) -> 'Shape':
        """
        Builds a new shape from this one that only contains the given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.

        The complementary operation is :func:`Shape.without`.

        Args:
          dims: comma-separated dimension names (str) or instance of dimensions (tuple, list, Shape) or filter function.
          reorder: If `False`, keeps the dimension order as defined in this shape.
            If `True`, reorders the dimensions of this shape to match the order of `dims`.

        Returns:
          Shape containing only specified dimensions

        """
        ...

    def is_compatible(self, *others: 'Shape') -> bool:
        """
        Checks if this shape and the others can be broadcast.

        Args:
            others: Other shapes.

        Returns:
            `True` only if all shapes are compatible.
        """
        ...

    @property
    def rank(self) -> int:
        """
        Returns the number of dimensions.
        Equal to `len(shape)`.

        See `Shape.is_empty`, `Shape.batch_rank`, `Shape.spatial_rank`, `Shape.channel_rank`.
        """
        ...

    @property
    def batch_rank(self) -> int:
        """ Number of batch dimensions """
        ...

    @property
    def instance_rank(self) -> int:
        ...

    @property
    def spatial_rank(self) -> int:
        """ Number of spatial dimensions """
        ...

    @property
    def dual_rank(self) -> int:
        """ Number of spatial dimensions """
        ...

    @property
    def channel_rank(self) -> int:
        """ Number of channel dimensions """
        ...

    @property
    def well_defined(self):
        """
        Returns `True` if no dimension size is `None`.

        Shapes with undefined sizes may be used in `phiml.math.tensor()`, `phiml.math.wrap()`, `phiml.math.stack()` or `phiml.math.concat()`.

        To create an undefined size, call a constructor function (`batch()`, `spatial()`, `channel()`, `instance()`)
        with positional `str` arguments, e.g. `spatial('x')`.
        """
        ...

    @property
    def defined(self):
        ...

    @property
    def undefined(self):
        ...

    @property
    def shape(self) -> 'Shape':
        """
        Higher-order `Shape`.
        The returned shape will always contain the channel dimension `dims` with a size equal to the `Shape.rank` of this shape.

        For uniform shapes, `Shape.shape` will only contain the dimension `dims` but the shapes of [non-uniform shapes](https://tum-pbs.github.io/PhiML/Non_Uniform.html)
        may contain additional dimensions.

        See Also:
            `Shape.is_uniform`.

        Returns:
            `Shape`.
        """
        ...

    @property
    def is_uniform(self) -> bool:
        """
        A shape is uniform if it all sizes have a single integer value.

        See Also:
            `Shape.is_non_uniform`, `Shape.shape`.
        """
        ...

    @property
    def is_non_uniform(self) -> bool:
        """
        A shape is non-uniform if the size of any dimension varies along another dimension.

        See Also:
            `Shape.is_uniform`, `Shape.shape`.
        """
        ...

    @property
    def non_uniform(self) -> 'Shape':
        """
        Returns only the non-uniform dimensions of this shape, i.e. the dimensions whose size varies along another dimension.

        See Also
            `Shape.non_uniform_shape`
        """
        ...

    @property
    def non_uniform_shape(self):
        """
        Returns the stack dimensions of non-uniform shapes.
        This is equal to `Shape.shape` excluding the `dims` dimension.

        For example, when stacking `(x=3)` and `(x=2)` along `vector`, the resulting shape is non_uniform.
        Its `non_uniform_shape` is `vector` and its `non_uniform` dimension is `x`.

        See Also
            `Shape.non_uniform`.
        """
        ...

    def with_size(self, size: Union[int, Sequence[str]]):
        """
        Only for single-dimension shapes.
        Returns a `Shape` representing this dimension but with a different size.

        See Also:
            `Shape.with_sizes()`.

        Args:
            size: Replacement size for this dimension.

        Returns:
            `Shape`
        """
        ...

    def with_sizes(self, sizes: Union[Sequence[int], Sequence[Tuple[str, ...]], 'Shape', int], keep_item_names=True):
        """
        Returns a new `Shape` matching the dimension names and types of `self` but with different sizes.

        See Also:
            `Shape.with_size()`.

        Args:
            sizes: One of

                * `tuple` / `list` of same length as `self` containing replacement sizes or replacement item names.
                * `Shape` of any rank. Replaces sizes for dimensions shared by `sizes` and `self`.
                * `int`: new size for all dimensions

            keep_item_names: If `False`, forgets all item names.
                If `True`, keeps item names where the size does not change.

        Returns:
            `Shape` with same names and types as `self`.
        """
        ...

    def without_sizes(self):
        """
        Returns:
            `Shape` with all sizes undefined (`None`)
        """
        ...

    def with_dim_size(self, dim: Union[str, 'Shape'], size: Union[int, 'math.Tensor', str, tuple, list], keep_item_names=True):
        """
        Returns a new `Shape` that has a different size for `dim`.

        Args:
            dim: Dimension for which to replace the size, `Shape` or `str`.
            size: New size, `int` or `Tensor`

        Returns:
            `Shape` with same names and types as `self`.
        """
        ...

    def replace(self, dims: Union['Shape', str, tuple, list], new: 'Shape') -> 'Shape':
        """
        Returns a copy of `self` with `dims` replaced by `new`.
        Dimensions that are not present in `self` are ignored.

        The dimension order is preserved.

        Args:
            dims: Dimensions to replace.
            new: New dimensions, must have same length as `dims` if `len(dims) > 1`.
                If a `Shape` is given, replaces the dimension types and item names as well.

        Returns:
            `Shape` with same rank and dimension order as `self`.
        """
        ...

    def replace_selection(self, names: Sequence[str], new: 'Shape') -> 'Shape':
        """
        Replace some of the dims of this shape.

        Args:
            names: Sequence of dim names.
            new: Replacement dims, must have same length as `old`.

        Returns:
            Copy of `self` with replaced dims.
        """
        ...

    @property
    def volume(self) -> Union[int, None]:
        """
        Returns the total number of values contained in a tensor of this shape.
        This is the product of all dimension sizes.

        Returns:
            volume as `int` or `Tensor` or `None` if the shape is not `Shape.well_defined`
        """
        ...

    @property
    def is_empty(self) -> bool:
        """ True if this shape has no dimensions. Equivalent to `Shape.rank` `== 0`. """
        ...

    def prepare_gather(self, dim: str, selection: Union[slice, int, 'Shape', str, tuple, list]) -> Union[slice, List[int]]:
        """
        Parse a slice object for a specific dimension.

        Args:
            dim: Name of dimension to slice.
            selection: Slice object.

        Returns:

        """
        ...

    def prepare_renaming_gather(self, dim: str, selection: Union[slice, int, 'Shape', str, tuple, list]):
        ...

    def after_gather(self, selection: dict) -> 'Shape':
        ...

    def meshgrid(self, names=False):
        """
        Builds a sequence containing all multi-indices within a tensor of this shape.
        All indices are returned as `dict` mapping dimension names to `int` indices.

        The corresponding values can be retrieved from Tensors and other Sliceables using `tensor[index]`.

        This function currently only supports uniform tensors.

        Args:
            names: If `True`, replace indices by their item names if available.

        Returns:
            `dict` iterator.
        """
        ...


DimFilter = Union[str, Sequence, set, Shape, Callable, None]
try:
    DimFilter.__doc__ = """Dimension filters can be used with `Shape.only()` and `Shype.without()`, making them the standard tool for specifying sets of dimensions.

    The following types can be used as dimension filters:

    * `Shape` instances
    * `tuple` or `list` objects containing dimension names as `str`
    * Single `str` listing comma-separated dimension names
    * Any function `filter(Shape) -> Shape`, such as `math.batch()`, `math.non_batch()`, `math.spatial()`, etc.
    """  # docstring must be set explicitly
except AttributeError:  # on older Python versions, this is not possible
    pass


if sys.version_info >= (3, 10):
    _dataclass_kwargs = {'slots': True}
else:
    _dataclass_kwargs = {}


@dataclass(frozen=True, **_dataclass_kwargs)
class Dim:
    name: str
    size: Union[int, Any]
    dim_type: str
    slice_names: Optional[Tuple[str, ...]]

    def __post_init__(self):
        if DEBUG_CHECKS:
            assert isinstance(self.name, str)
            assert self.dim_type in DIM_TYPES
            from ._tensors import Tensor
            if isinstance(self.size, Tensor):
                assert self.size.rank > 0
            else:
                assert self.size is None or isinstance(self.size, int)
            if self.dim_type == DUAL_DIM:
                assert self.name.startswith('~'), f"Dual dimensions must start with '~' but got '{self.name}' in {self}"
            assert isinstance(self.slice_names, tuple) or self.slice_names is None
            if self.slice_names is not None:
                assert all(isinstance(n, str) for n in self.slice_names)
                try:
                    int(self.size)
                except Exception:
                    raise AssertionError(f"When item names are present, the size must be an integer type")
                assert len(self.slice_names) == self.size, f"Number of item names ({len(self.slice_names)}) does not match size {self.size}"
                assert len(set(self.slice_names)) == len(self.slice_names), f"Duplicate item names in shape {self} at dim '{self.name}': {self.slice_names}"

    @property
    def dims(self):
        return {self.name: self}

    def __len__(self):
        return 1
    @property
    def rank(self):
        return 1

    def __bool__(self):
        return True
    @property
    def is_empty(self) -> bool:
        return False

    @property
    def volume(self) -> Union[int, None]:
        if self.size is None or isinstance(self.size, int):
            return self.size
        return int((self.size / self.non_uniform_shape.volume).sum)

    @property
    def names(self):
        return self.name,
    @property
    def name_list(self):
        return [self.name]
    @property
    def sizes(self):
        return self.size,
    @property
    def types(self):
        return self.type,
    @property
    def dim_types(self):
        return self.dim_type,
    @property
    def item_names(self):
        return self.slice_names,

    @property
    def untyped_dict(self):
        return {self.name: self.slice_names if self.slice_names is not None else self.size}

    @property
    def type(self) -> str:
        return DIM_FUNCTIONS[self.dim_type]

    @property
    def is_uniform(self) -> bool:
        return isinstance(self.size, int)
    @property
    def is_non_uniform(self) -> bool:
        return self.size is not None and not isinstance(self.size, int)
    @property
    def non_uniform(self):
        return EMPTY_SHAPE if isinstance(self.size, int) else self
    @property
    def non_uniform_shape(self):
        return EMPTY_SHAPE if isinstance(self.size, int) else shape(self.size)

    @property
    def singleton(self):
        return self if _size_equal(self.size, 1) else EMPTY_SHAPE
    @property
    def non_singleton(self):
        return EMPTY_SHAPE if _size_equal(self.size, 1) else self

    @property
    def well_defined(self):
        return self.size is not None
    @property
    def defined(self):
        return self if self.size is not None else EMPTY_SHAPE
    @property
    def undefined(self):
        return self if self.size is None else EMPTY_SHAPE

    @property
    def batch_rank(self) -> int:
        return 1 if self.dim_type == BATCH_DIM else 0
    @property
    def instance_rank(self) -> int:
        return 1 if self.dim_type == INSTANCE_DIM else 0
    @property
    def spatial_rank(self) -> int:
        return 1 if self.dim_type == SPATIAL_DIM else 0
    @property
    def dual_rank(self) -> int:
        return 1 if self.dim_type == DUAL_DIM else 0
    @property
    def channel_rank(self) -> int:
        return 1 if self.dim_type == CHANNEL_DIM else 0

    @property
    def batch(self):
        return self if self.dim_type == BATCH_DIM else EMPTY_SHAPE
    @property
    def dual(self):
        return self if self.dim_type == DUAL_DIM else EMPTY_SHAPE
    @property
    def instance(self):
        return self if self.dim_type == INSTANCE_DIM else EMPTY_SHAPE
    @property
    def spatial(self):
        return self if self.dim_type == SPATIAL_DIM else EMPTY_SHAPE
    @property
    def channel(self):
        return self if self.dim_type == CHANNEL_DIM else EMPTY_SHAPE
    @property
    def primal(self):
        return self if self.dim_type in PRIMAL_TYPES else EMPTY_SHAPE

    @property
    def non_batch(self):
        return self if self.dim_type != BATCH_DIM else EMPTY_SHAPE
    @property
    def non_dual(self):
        return self if self.dim_type != DUAL_DIM else EMPTY_SHAPE
    @property
    def non_instance(self):
        return self if self.dim_type != INSTANCE_DIM else EMPTY_SHAPE
    @property
    def non_spatial(self):
        return self if self.dim_type != SPATIAL_DIM else EMPTY_SHAPE
    @property
    def non_channel(self):
        return self if self.dim_type != CHANNEL_DIM else EMPTY_SHAPE
    @property
    def non_primal(self):
        return self if self.dim_type not in PRIMAL_TYPES else EMPTY_SHAPE

    def __repr__(self):
        if self.slice_names:
            items_str = ",".join(self.slice_names)
            size_str = items_str if len(items_str) <= 12 else f"{self.size}:{self.slice_names[0][:5]}..."
        else:
            size_str = self.size
        return f"({self.name}{SUPERSCRIPT.get(self.dim_type, '?')}={size_str})"

    def __eq__(self, other):
        if isinstance(other, Dim):
            return self.name == other.name and self.dim_type == other.dim_type and self.slice_names == other.slice_names and _size_equal(self.size, other.size)
        if isinstance(other, MixedShape):
            return len(other) == 1 and self == getattr(other, self.dim_type)
        return False

    def __contains__(self, item):
        if isinstance(item, Dim):
            return item.name == self.name
        if isinstance(item, str):
            return item == self.name
        if isinstance(item, (tuple, list)):
            return len(item) == 1 and item[0] == self.name
        return not item or (len(item) == 1 and item.name == self.name)

    def index(self, dim: Union[str, 'Shape', None]) -> Optional[int]:
        if dim is None:
            return None
        if isinstance(dim, SHAPE_TYPES):
            dim = dim.name
        if isinstance(dim, str):
            if dim != self.name:
                raise ValueError(f"Shape {self} has no dimension '{dim}'")
            return 0
        raise ValueError(f"index() requires a single dimension as input but got {dim}")

    def indices(self, names: Sequence[str]) -> Tuple[int, ...]:
        return (0,) if names else ()

    def __getitem__(self, selection):
        if isinstance(selection, SHAPE_TYPES):
            selection = selection.names
        if isinstance(selection, (tuple, list)):
            if not selection:
                return EMPTY_SHAPE
            assert len(selection) == 1, f"Only one dim contained in {self} but tried to access {selection}"
            selection = selection[0]
        if isinstance(selection, int):
            assert selection in {0, -1}, f"Tried to access index {selection} in {self}"
            return self
        elif isinstance(selection, slice):
            if selection.start in {0, None} and selection.stop in {1, None}:
                return self
            return EMPTY_SHAPE
        elif isinstance(selection, str):
            assert selection == self.name, f"Name {selection} not contained in {self}"
            return self
        raise AssertionError("Can only access shape elements as shape[int], shape[str], shape[slice], shape[Sequence] or shape[Shape]")

    def __iter__(self):
        return iter([self])

    def get_size(self, dim: Union[str, 'Shape'], default=None):
        dim = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        if dim == self.name:
            return self.size
        if default is not None:
            return default
        raise KeyError(f"get_size() failed because '{dim}' is not part of {self} and no default value was provided")

    def get_dim_type(self, dim: str) -> str:
        return self.dim_type

    def get_item_names(self, dim: Union[str, 'Shape'], fallback_spatial=False) -> Union[tuple, None]:
        dim = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        if dim != self.name:
            raise KeyError(f"get_item_names() failed because '{dim}' is not part of {self}")
        return self.slice_names  # fallback_spatial requires shape.spatial and dim.type==channel

    def __and__(self, other):
        if other is dual:
            return self & self.primal.as_dual()
        if isinstance(other, (Dim, PureShape)) and other.dim_type == self.dim_type:
            return pure_merge(self, other, allow_varying_sizes=False)
        elif isinstance(other, (Dim, PureShape)):
            if not other:
                return self
            by_type = [EMPTY_SHAPE] * len(DIM_TYPES)
            by_type[TYPE_INDEX[self.dim_type]] = self
            by_type[TYPE_INDEX[other.dim_type]] = other
            return MixedShape(*by_type, dims={self.name: self, **other.dims})
        return NotImplemented

    def is_compatible(self, other):
        if self.name not in other:
            return True
        dim = other[self.name]
        if not _size_equal(self.size, dim.size):
            return False
        return self.dim_type == dim.dim_type

    def isdisjoint(self, other) -> bool:
        if isinstance(other, SHAPE_TYPES):
            return self.name not in other
        return self.name not in parse_dim_order(other)

    def only(self, dims: 'DimFilter', reorder=False):
        if dims is None:  # keep none
            return EMPTY_SHAPE
        if callable(dims):
            return dims(self)
        if isinstance(dims, str):
            dims = parse_dim_order(dims)
        if isinstance(dims, SHAPE_TYPES):
            return self if self.name in dims else EMPTY_SHAPE
        assert isinstance(dims, (tuple, list, set))
        if all(isinstance(d, int) for d in dims):
            return self if 0 in dims else EMPTY_SHAPE
        for d in dims:
            if callable(d):
                if d(self):
                    return self
            elif isinstance(d, str):
                if d == self.name:
                    return self
            elif isinstance(d, SHAPE_TYPES):
                if self.name in d:
                    return self
            else:
                raise ValueError(f"Format not understood for Shape.only(): {dims}")
        return EMPTY_SHAPE

    def __add__(self, other):
        if isinstance(other, int):
            return Dim(self.name, self.size + other, self.dim_type, None)
        return concat_shapes_(self, other)

    def __sub__(self, other):
        if isinstance(other, int):
            return Dim(self.name, self.size - other, self.dim_type, None)
        return self.without(other)

    def without(self, dims: 'DimFilter'):
        if dims is None:  # subtract none
            return self
        elif callable(dims):
            dims = dims(self)
        if isinstance(dims, (str, SHAPE_TYPES)):
            dims = parse_dim_order(dims)
            return EMPTY_SHAPE if self.name in dims else self
        if isinstance(dims, (tuple, list, set)) and all([isinstance(d, str) for d in dims]):
            return EMPTY_SHAPE if self.name in dims else self
        elif isinstance(dims, (tuple, list, set)):
            for d in dims:
                if not self.without(d):
                    return EMPTY_SHAPE
            return self
        else:
            raise ValueError(dims)

    def meshgrid(self, names=False):
        assert self.is_uniform, f"Shape.meshgrid() is currently not supported for non-uniform tensors, {self}"
        if isinstance(self.size, int) and self.size == 0:
            return ()
        if names and self.slice_names is not None:
            for sln in self.slice_names:
                yield {self.name: sln}
        else:  # indices
            for i in range(self.size):
                yield {self.name: i}

    def with_size(self, size, keep_item_names=True):
        if isinstance(size, SHAPE_TYPES):
            names = size.get_item_names(self.name)
            size = names if names is not None else size.get_size(self.name)
        if isinstance(size, str):
            names = parse_dim_order(size)
            return Dim(self.name, len(names), self.dim_type, names)
        if isinstance(size, (tuple, list)):
            assert all(isinstance(s, str) for s in size)
            return Dim(self.name, len(size), self.dim_type, tuple(size))
        if size is None:
            return Dim(self.name, None, self.dim_type, None)
        if keep_item_names and _size_equal(self.size, size):
            return self
        from ._tensors import Tensor
        if not isinstance(size, Tensor):
            size = int(size)
        return Dim(self.name, size, self.dim_type, None)

    def with_dim_size(self, dim: Union[str, 'Shape'], size: Union[int, 'math.Tensor', str, tuple, list], keep_item_names=True):
        name = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        assert name == self.name, f"Cannot set dim size of {dim} on {self}"
        return self.with_size(size, keep_item_names=keep_item_names)

    def with_sizes(self, sizes: Union[Sequence[int], Sequence[Tuple[str, ...]], 'Shape', int], keep_item_names=True):
        if isinstance(sizes, SHAPE_TYPES):
            if self.name not in sizes:
                return self
            slices = sizes.get_item_names(self.name)
            return self.with_size(slices if slices is not None else sizes.get_size(self.name))
        if isinstance(sizes, int):
            return self.with_size(sizes)
        assert len(sizes) == 1, f"Too many sizes for shape {self}: {sizes}"
        return self.with_size(sizes[0])

    def without_sizes(self):
        return Dim(self.name, None, self.dim_type, None)

    def flipped(self, dims: Union[List[str], Tuple[str]]):
        if self.slice_names is None or self.name not in dims:
            return self
        return Dim(self.name, self.size, self.dim_type, self.slice_names[::-1])

    def replace(self, dims: Union['Shape', str, tuple, list], new: 'Shape'):
        old_names = parse_dim_order(dims)
        if len(old_names) == 1 and self.name == old_names[0]:
            return self._replace(new)
        elif self.name in old_names:
            idx = old_names.index(self.name)
            return self._replace(new[idx])
        else:
            return self

    def _replace(self, new: 'Shape'):
        if self.slice_names is None or new.slice_names is not None:
            return new
        if len(new) != 1 or not _size_equal(self.size, new.size):
            return new
        else:  # keep item names from self
            return Dim(new.name, new.size, new.dim_type, self.slice_names)

    def replace_selection(self, names: Sequence[str], new: Shape):
        if self.name not in names:
            return self
        new = new[names.index(self.name)]
        if self.slice_names is None or new.item_names[0] is not None or not _size_equal(self.size, new.size):
            return new
        return Dim(new.name, new.size, new.dim_type, self.slice_names)

    def as_batch(self):
        name = _apply_prefix(self.name, BATCH_DIM) if self.dim_type == DUAL_DIM else self.name
        return Dim(name, self.size, BATCH_DIM, self.slice_names)
    def as_dual(self):
        return Dim(_apply_prefix(self.name, DUAL_DIM), self.size, DUAL_DIM, self.slice_names)
    def as_instance(self):
        name = _apply_prefix(self.name, INSTANCE_DIM) if self.dim_type == DUAL_DIM else self.name
        return Dim(name, self.size, INSTANCE_DIM, self.slice_names)
    def as_spatial(self):
        name = _apply_prefix(self.name, SPATIAL_DIM) if self.dim_type == DUAL_DIM else self.name
        return Dim(name, self.size, SPATIAL_DIM, self.slice_names)
    def as_channel(self):
        name = _apply_prefix(self.name, CHANNEL_DIM) if self.dim_type == DUAL_DIM else self.name
        return Dim(name, self.size, CHANNEL_DIM, self.slice_names)
    def as_type(self, new_type: Callable):
        dim_type = TYPE_BY_FUNCTION[new_type]
        return Dim(_apply_prefix(self.name, dim_type), self.size, dim_type, self.slice_names)

    def transpose(self, dim_type: str):
        assert dim_type in {SPATIAL_DIM, INSTANCE_DIM, CHANNEL_DIM}
        if self.dim_type == DUAL_DIM:
            return Dim(self.name[1:], self.size, dim_type, self.slice_names)
        if self.dim_type == dim_type:
            return Dim('~'+self.name, self.size, DUAL_DIM, self.slice_names)
        return self


@dataclass(frozen=True)  # slots not compatible with @cached_property
class PureShape:
    dim_type: str
    dims: Dict[str, Dim]

    def __post_init__(self):
        if DEBUG_CHECKS:
            assert len(self.dims) != 1
            assert self.dim_type in DIM_TYPES
            for n, dim in self.dims.items():
                assert n == dim.name
                assert dim.dim_type == self.dim_type

    def __len__(self):  # this is also used for bool(self)
        return len(self.dims)
    @property
    def rank(self):
        return len(self.dims)

    @property
    def is_empty(self) -> bool:
        return not self.dims

    @property
    def volume(self) -> Union[int, None]:
        result = 1
        for size in self.sizes:
            if size is None:
                return None
            result *= size
        from ._tensors import Tensor
        if not isinstance(result, Tensor):
            return result
        result /= self.non_uniform_shape.volume  # We summed up the items -> undo multiplication
        return int(result.sum)

    @cached_property
    def names(self):
        return tuple(self.dims)
    @property
    def name_list(self):
        return list(self.dims)
    @property
    def sizes(self):
        return tuple([d.size for d in self.dims.values()])
    @property
    def types(self):
        return [d.type for d in self.dims.values()]
    @property
    def dim_types(self):
        return [d.dim_type for d in self.dims.values()]
    @property
    def item_names(self):
        return [d.slice_names for d in self.dims.values()]

    @property
    def untyped_dict(self):
        return {k: v for dim in self.dims.values() for k, v in dim.untyped_dict.items()}

    @property
    def name(self):
        raise AssertionError("Shape.name is only defined for shapes of rank 1.")
    @property
    def size(self):
        raise AssertionError("Shape.size is only defined for shapes of rank 1.")
    @property
    def type(self):
        raise AssertionError("Shape.type is only defined for shapes of rank 1.")

    @property
    def is_uniform(self) -> bool:
        return all(dim.is_uniform for dim in self.dims.values())
    @property
    def is_non_uniform(self) -> bool:
        return any(dim.is_non_uniform for dim in self.dims.values())
    @property
    def non_uniform(self) -> 'Shape':
        return PureShape(self.dim_type, {name: dim for name, dim in self.dims.items() if dim.is_non_uniform})
    @property
    def non_uniform_shape(self):
        result = EMPTY_SHAPE
        for size in self.sizes:
            if not isinstance(size, int):
                result &= size.shape
        return result

    @property
    def singleton(self):
        dims = {n: dim for n, dim in self.dims.items() if _size_equal(dim.size, 1)}
        return next(iter(dims.values())) if len(dims) == 1 else PureShape(self.dim_type, dims)
    @property
    def non_singleton(self):
        dims = {n: dim for n, dim in self.dims.items() if not _size_equal(dim.size, 1)}
        return next(iter(dims.values())) if len(dims) == 1 else PureShape(self.dim_type, dims)

    @property
    def well_defined(self):
        for size in self.sizes:
            if size is None:
                return False
        return True
    @property
    def defined(self):
        dims = {n: dim for n, dim in self.dims.items() if dim.size is not None}
        return next(iter(dims.values())) if len(dims) == 1 else PureShape(self.dim_type, dims)
    @property
    def undefined(self):
        dims = {n: dim for n, dim in self.dims.items() if dim.size is None}
        return next(iter(dims.values())) if len(dims) == 1 else PureShape(self.dim_type, dims)

    @property
    def batch_rank(self) -> int:
        return len(self.dims) if self.dim_type == BATCH_DIM else 0
    @property
    def instance_rank(self) -> int:
        return len(self.dims) if self.dim_type == INSTANCE_DIM else 0
    @property
    def spatial_rank(self) -> int:
        return len(self.dims) if self.dim_type == SPATIAL_DIM else 0
    @property
    def dual_rank(self) -> int:
        return len(self.dims) if self.dim_type == DUAL_DIM else 0
    @property
    def channel_rank(self) -> int:
        return len(self.dims) if self.dim_type == CHANNEL_DIM else 0

    @property
    def batch(self):
        return self if self.dim_type == BATCH_DIM else EMPTY_SHAPE
    @property
    def dual(self):
        return self if self.dim_type == DUAL_DIM else EMPTY_SHAPE
    @property
    def instance(self):
        return self if self.dim_type == INSTANCE_DIM else EMPTY_SHAPE
    @property
    def spatial(self):
        return self if self.dim_type == SPATIAL_DIM else EMPTY_SHAPE
    @property
    def channel(self):
        return self if self.dim_type == CHANNEL_DIM else EMPTY_SHAPE
    @property
    def primal(self):
        return self if self.dim_type in PRIMAL_TYPES else EMPTY_SHAPE

    @property
    def non_batch(self):
        return self if self.dim_type != BATCH_DIM else EMPTY_SHAPE
    @property
    def non_dual(self):
        return self if self.dim_type != DUAL_DIM else EMPTY_SHAPE
    @property
    def non_instance(self):
        return self if self.dim_type != INSTANCE_DIM else EMPTY_SHAPE
    @property
    def non_spatial(self):
        return self if self.dim_type != SPATIAL_DIM else EMPTY_SHAPE
    @property
    def non_channel(self):
        return self if self.dim_type != CHANNEL_DIM else EMPTY_SHAPE
    @property
    def non_primal(self):
        return self if self.dim_type not in PRIMAL_TYPES else EMPTY_SHAPE

    def __repr__(self):
        strings = [repr(dim)[1:-1] for dim in self.dims.values()]
        return '(' + ', '.join(strings) + ')'

    def __eq__(self, other):
        if isinstance(other, PureShape):
            return self.dim_type == other.dim_type and self.dims == other.dims
        if isinstance(other, MixedShape):
            return other.__eq__(self)
        return False

    def __hash__(self):
        return hash(tuple(self.dims.values()))

    def __contains__(self, item):
        if isinstance(item, Dim):
            return item.name in self.dims
        if isinstance(item, (str, tuple, list)):
            dims = parse_dim_order(item)
            return all(dim in self.dims for dim in dims)
        return all([d in self.dims for d in item.names])

    def index(self, dim: Union[str, 'Shape', None]) -> Optional[int]:
        if dim is None:
            return None
        elif isinstance(dim, str):
            if dim not in self.dims:
                raise ValueError(f"Shape {self} has no dimension '{dim}'")
            return self.names.index(dim)
        elif isinstance(dim, SHAPE_TYPES):
            assert len(dim) == 1, f"index() requires a single dimension as input but got {dim}. Use indices() for multiple dimensions."
            return self.names.index(dim.name)
        raise ValueError(f"index() requires a single dimension as input but got {dim}")

    def indices(self, names: Sequence[str]) -> Tuple[int, ...]:
        order = self.names
        return tuple(order.index(n) for n in names)

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return list(self.dims.values())[selection]
        elif isinstance(selection, slice):
            return concat_shapes_(*list(self.dims.values())[selection])
        elif isinstance(selection, str):
            if ',' in selection:
                selection = [self.index(s.strip()) for s in selection.split(',')]
            else:
                selection = self.index(selection)
            return self[selection]
        elif isinstance(selection, SHAPE_TYPES):
            selection = selection.names
        if isinstance(selection, (tuple, list)):
            selection = [self.names[sel] if isinstance(sel, int) else sel for sel in selection]
            if DEBUG_CHECKS:
                assert all(isinstance(s, str) for s in selection)
            return concat_shapes_(*[self.dims[n] for n in selection])
        raise AssertionError("Can only access shape elements as shape[int], shape[str], shape[slice], shape[Sequence] or shape[Shape]")

    def __iter__(self):
        return iter(self.dims.values())

    def get_size(self, dim: Union[str, 'Shape'], default=None):
        name = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        if default is None:
            return self.dims[name].size
        else:
            return self.dims[name].size if name in self.dims else default

    def get_dim_type(self, dim: str) -> str:
        return self.dim_type

    def get_item_names(self, dim: Union[str, 'Shape'], fallback_spatial=False) -> Union[tuple, None]:
        name = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        return self.dims[name].slice_names  # fallback_spatial requires shape.spatial and dim.type==channel

    def __and__(self, other):
        if other is dual:
            return concat_shapes_(self, self.primal.as_dual())
        if isinstance(other, (Dim, PureShape)) and other.dim_type == self.dim_type:
            return pure_merge(self, other, allow_varying_sizes=False)
        elif isinstance(other, (Dim, PureShape)):
            if not self:
                return other
            if not other:
                return self
            by_type = [EMPTY_SHAPE] * len(DIM_TYPES)
            by_type[TYPE_INDEX[self.dim_type]] = self
            by_type[TYPE_INDEX[other.dim_type]] = other
            return MixedShape(*by_type, dims={**self.dims, **other.dims})
        return NotImplemented

    def is_compatible(self, other: Shape):
        return all(dim.is_compatible(other) for dim in self.dims.values())

    def isdisjoint(self, other) -> bool:
        if isinstance(other, Dim):
            return other.name not in self.dims
        if isinstance(other, PureShape):
            return other.dim_type != self.dim_type or self.dims.keys().isdisjoint(other.dims)
        elif isinstance(other, MixedShape):
            return other.isdisjoint(self)
        return self.dims.keys().isdisjoint(set(parse_dim_order(other)))

    def only(self, dims: 'DimFilter', reorder=False):
        if not self.dims or dims is None:
            return EMPTY_SHAPE
        if isinstance(dims, Dim):
            return dims if dims.name in self.dims else EMPTY_SHAPE
        if callable(dims):
            return dims(self)
        if isinstance(dims, str):
            dims = parse_dim_order(dims)
        elif isinstance(dims, SHAPE_TYPES):
            dims = [dims]
        names = []
        for d in dims:
            if isinstance(d, str):
                names.append(d)
            elif isinstance(d, SHAPE_TYPES):
                names.extend(d.names)
            elif callable(d):
                names.extend(d(self).names)
            else:
                raise ValueError(f"Format not understood for Shape.only(): {dims}")
        names = [d for d in names if d in self.dims]
        if not names:
            return EMPTY_SHAPE
        if len(names) == 1:
            return self.dims[names[0]]
        if reorder:
            return PureShape(self.dim_type, {n: self.dims[n] for n in names})
        else:
            return PureShape(self.dim_type, {n: dim for n, dim in self.dims.items() if n in names})

    def __add__(self, other):
        if isinstance(other, int):
            assert self.dim_type != BATCH_DIM, f"Shape arithmetic not allowed for batch dims {self}"
            return PureShape(self.dim_type, {n: dim + other for n, dim in self.dims.items()})
        return concat_shapes_(self, other)

    def __sub__(self, other):
        if isinstance(other, int):
            assert self.dim_type != BATCH_DIM, f"Shape arithmetic not allowed for batch dims {self}"
            return PureShape(self.dim_type, {n: dim - other for n, dim in self.dims.items()})
        return self.without(other)

    def without(self, dims: 'DimFilter'):
        if dims is None or not self.dims:  # subtract none
            return self
        elif callable(dims):
            dims = dims(self)
        if isinstance(dims, (str, SHAPE_TYPES)):
            names = parse_dim_order(dims)
            dims = {n: dim for n, dim in self.dims.items() if n not in names}
            return next(iter(dims.values())) if len(dims) == 1 else PureShape(self.dim_type, dims)
        if isinstance(dims, (tuple, list, set)) and all([isinstance(d, str) for d in dims]):
            dims = {n: dim for n, dim in self.dims.items() if n not in dims}
            return next(iter(dims.values())) if len(dims) == 1 else PureShape(self.dim_type, dims)
        elif isinstance(dims, (tuple, list, set)):
            result = self
            for wo in dims:
                result = result.without(wo)
            return result
        else:
            raise ValueError(dims)

    def meshgrid(self, names=False):
        assert self.is_uniform, f"Shape.meshgrid() is currently not supported for non-uniform tensors, {self}"
        if self.volume == 0:
            return ()
        indices = [0] * len(self.dims)
        while True:
            if names:
                yield {dim: (names[index] if names is not None else index) for dim, index, names in zip(self.names, indices, self.item_names)}
            else:
                yield {dim: index for dim, index in zip(self.names, indices)}
            for i in range(self.rank-1, -1, -1):
                indices[i] = (indices[i] + 1) % self.sizes[i]
                if indices[i] != 0:
                    break
            else:
                return

    def with_size(self, size, keep_item_names=True):
        raise AssertionError(f"Shape.with_size() is only defined for shapes of rank 1 but got {self}")

    def with_dim_size(self, dim: Union[str, 'Shape'], size: Union[int, 'math.Tensor', str, tuple, list], keep_item_names=True):
        name = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        dims = dict(self.dims)
        dims[dim] = dims[dim].with_size(size)
        return PureShape(self.dim_type, {n: dim.with_size(size) if n == name else dim for n, dim in self.dims.items()})

    def with_sizes(self, sizes: Union[Sequence[int], Sequence[Tuple[str, ...]], 'Shape', int], keep_item_names=True):
        if not self.dims:
            assert not sizes or isinstance(sizes, SHAPE_TYPES)
            return self
        if isinstance(sizes, (int, str)):
            dims = {n: dim.with_size(sizes, keep_item_names) for n, dim in self.dims.items()}
        elif isinstance(sizes, SHAPE_TYPES):
            dims = {n: dim.with_size(sizes.get_size(n), keep_item_names) if n in sizes else dim for n, dim in self.dims.items()}
        else:
            dims = {dim.name: dim.with_size(size, keep_item_names) for dim, size in zip(self.dims.values(), sizes)}
        return PureShape(self.dim_type, dims)

    def without_sizes(self):
        return PureShape(self.dim_type, {n: dim.without_sizes() for n, dim in self.dims.items()})

    def flipped(self, dims: Union[List[str], Tuple[str]]):
        dims = {n: dim.flipped(dims) for n, dim in self.dims.items()}
        return PureShape(self.dim_type, dims)

    def replace(self, dims: Union['Shape', str, tuple, list], new: 'Shape'):
        dims = parse_dim_order(dims)
        if len(dims) == len(new):
            dim_list = list(self.dims.values())
            for old, new_dim in zip(dims, new):
                new_dim = self.dims[old]._replace(new_dim)
                dim_list[self.index(old)] = new_dim
        elif len(new) > 1 and len(dims) == 1:
            dim_list = list(self.dims.values())
            i = self.index(dims[0])
            dim_list[i:i+1] = new
        else:
            assert len(new) == 1
            if len(dims) == len(self.dims):
                return new
            i0 = self.index(dims[0])
            dim_list = [d for n, d in self.dims.items() if n not in dims]
            dim_list.insert(i0, new)
        return concat_shapes_(*dim_list)

    def replace_selection(self, names: Sequence[str], new: Shape):
        dim_list = list(self.dims.values())
        for old_name, new_dim in zip(names, new):
            if old_name in self.dims:
                new_dim = self.dims[old_name]._replace(new_dim)
                dim_list[self.index(old_name)] = new_dim
        return concat_shapes_(*dim_list)

    def as_batch(self):
        dims = [dim.as_batch() for dim in self.dims.values()]
        return PureShape(BATCH_DIM, {dim.name: dim for dim in dims})
    def as_dual(self):
        dims = [dim.as_dual() for dim in self.dims.values()]
        return PureShape(DUAL_DIM, {dim.name: dim for dim in dims})
    def as_instance(self):
        dims = [dim.as_instance() for dim in self.dims.values()]
        return PureShape(INSTANCE_DIM, {dim.name: dim for dim in dims})
    def as_spatial(self):
        dims = [dim.as_spatial() for dim in self.dims.values()]
        return PureShape(SPATIAL_DIM, {dim.name: dim for dim in dims})
    def as_channel(self):
        dims = [dim.as_channel() for dim in self.dims.values()]
        return PureShape(CHANNEL_DIM, {dim.name: dim for dim in dims})
    def as_type(self, new_type: Callable):
        return {batch: self.as_batch, dual: self.as_dual, instance: self.as_instance, spatial: self.as_spatial, channel: self.as_channel}[new_type]()

    def transpose(self, dim_type: str):
        if not self:
            return self
        if self.dim_type not in {DUAL_DIM, dim_type}:
            return self
        dims = [dim.transpose(dim_type) for dim in self.dims.values()]
        return PureShape(dims[0].dim_type, {dim.name: dim for dim in dims})


@dataclass(frozen=True, **_dataclass_kwargs)
class MixedShape:
    batch: Union[PureShape, Dim]
    dual: Union[PureShape, Dim]
    instance: Union[PureShape, Dim]
    spatial: Union[PureShape, Dim]
    channel: Union[PureShape, Dim]
    dims: Dict[str, Dim]  # dim order

    def __post_init__(self):
        assert self

    def __len__(self):
        return len(self.dims)
    @property
    def rank(self):
        return len(self.dims)

    @property
    def is_empty(self) -> bool:
        return not self.dims

    @property
    def volume(self) -> Union[int, None]:
        result = 1
        for size in self.sizes:
            if size is None:
                return None
            result *= size
        from ._tensors import Tensor
        if not isinstance(result, Tensor):
            return result
        result /= self.non_uniform_shape.volume  # We summed up the items -> undo multiplication
        return int(result.sum)

    @property
    def names(self):
        return tuple(self.dims)
    @property
    def name_list(self):
        return list(self.dims)
    @property
    def sizes(self) -> tuple:
        return sum([dim.sizes for dim in self.dims.values()], ())
    @property
    def types(self):
        return [d.type for d in self.dims.values()]
    @property
    def dim_types(self):
        return [d.dim_type for d in self.dims.values()]
    @property
    def item_names(self):
        return sum([dim.item_names for dim in self.dims.values()], ())

    @property
    def untyped_dict(self):
        return {k: v for dim in self.dims.values() for k, v in dim.untyped_dict.items()}

    @property
    def name(self):
        assert len(self.dims) == 1, f"Shape.name is only defined for shapes of rank 1 but has dims {self}"
        return next(iter(self.dims))
    @property
    def size(self):
        assert len(self.dims) == 1, f"Shape.size is only defined for shapes of rank 1 but has dims {self}"
        return next(iter(self.dims.values())).size
    @property
    def type(self) -> str:
        assert len(self.dims) == 1, f"Shape.type is only defined for shapes of rank 1 but has dims {self}"
        return next(iter(self.dims.values())).type
    @property
    def dim_type(self):
        assert len(self.dims) == 1, f"Shape.dim_type is only defined for shapes of rank 1 but has dims {self}"
        return next(iter(self.dims.values())).dim_type

    @property
    def is_uniform(self) -> bool:
        return all(dim.is_uniform for dim in self.dims.values())
    @property
    def is_non_uniform(self) -> bool:
        return any(dim.is_non_uniform for dim in self.dims.values())
    @property
    def non_uniform(self) -> 'Shape':
        result = EMPTY_SHAPE
        for dim in self.dims.values():
            if not isinstance(dim.size, int):
                result &= dim
        return result
    @property
    def non_uniform_shape(self):
        result = EMPTY_SHAPE
        for size in self.sizes:
            if not isinstance(size, int):
                result &= size.shape
        return result

    @property
    def singleton(self):
        dims = {n: dim for n, dim in self.dims.items() if _size_equal(dim.size, 1)}
        return next(iter(dims.values())) if len(dims) == 1 else merge_shapes(dims)
    @property
    def non_singleton(self):
        dims = {n: dim for n, dim in self.dims.items() if not _size_equal(dim.size, 1)}
        return next(iter(dims.values())) if len(dims) == 1 else merge_shapes(dims)

    @property
    def well_defined(self):
        for size in self.sizes:
            if size is None:
                return False
        return True
    @property
    def defined(self):
        return concat_shapes_(*[dim for dim in self.dims.values() if dim.size is not None])
    @property
    def undefined(self):
        return concat_shapes_(*[dim for dim in self.dims.values() if dim.size is None])

    @property
    def batch_rank(self) -> int:
        return len(self.batch)
    @property
    def instance_rank(self) -> int:
        return len(self.instance)
    @property
    def spatial_rank(self) -> int:
        return len(self.spatial)
    @property
    def dual_rank(self) -> int:
        return len(self.dual)
    @property
    def channel_rank(self) -> int:
        return len(self.channel)

    @property
    def primal(self):
        dims = {**self.instance.dims, **self.spatial.dims, **self.channel.dims}
        return MixedShape(EMPTY_SHAPE, EMPTY_SHAPE, self.instance, self.spatial, self.channel, dims) if dims else EMPTY_SHAPE
    @property
    def non_primal(self):
        dims = {**self.batch.dims, **self.dual.dims}
        return MixedShape(self.batch, self.dual, EMPTY_SHAPE, EMPTY_SHAPE, EMPTY_SHAPE, dims) if dims else EMPTY_SHAPE

    @property
    def non_batch(self):
        dims = {n: dim for n, dim in self.dims.items() if dim.dim_type != BATCH_DIM}
        return MixedShape(EMPTY_SHAPE, self.dual, self.instance, self.spatial, self.channel, dims) if dims else EMPTY_SHAPE
    @property
    def non_dual(self):
        dims = {n: dim for n, dim in self.dims.items() if dim.dim_type != DUAL_DIM}
        return MixedShape(self.batch, EMPTY_SHAPE, self.instance, self.spatial, self.channel, dims) if dims else EMPTY_SHAPE
    @property
    def non_instance(self):
        dims = {n: dim for n, dim in self.dims.items() if dim.dim_type != INSTANCE_DIM}
        return MixedShape(self.batch, self.dual, EMPTY_SHAPE, self.spatial, self.channel, dims) if dims else EMPTY_SHAPE
    @property
    def non_spatial(self):
        dims = {n: dim for n, dim in self.dims.items() if dim.dim_type != SPATIAL_DIM}
        return MixedShape(self.batch, self.dual, self.instance, EMPTY_SHAPE, self.channel, dims) if dims else EMPTY_SHAPE
    @property
    def non_channel(self):
        dims = {n: dim for n, dim in self.dims.items() if dim.dim_type != CHANNEL_DIM}
        return MixedShape(self.batch, self.dual, self.instance, self.spatial, EMPTY_SHAPE, dims) if dims else EMPTY_SHAPE

    def __repr__(self):
        return '(' + ', '.join([repr(dim)[1:-1] for dim in self.dims.values()]) + ')'

    def __eq__(self, other):
        if isinstance(other, (Dim, PureShape)):
            pure = getattr(self, other.dim_type)
            if len(pure) != len(self.dims):
                return False
            return pure == other
        if isinstance(other, MixedShape):
            return self.dims == other.dims
        return False

    def __hash__(self):
        return hash(tuple(self.dims.values()))

    def __contains__(self, item):
        if isinstance(item, Dim):
            return item.name in self.dims
        if isinstance(item, (str, tuple, list)):
            dims = parse_dim_order(item)
            return all(dim in self.dims for dim in dims)
        return all([d in self.dims for d in item.names])

    def index(self, dim: Union[str, 'Shape', None]) -> Optional[int]:
        if dim is None:
            return None
        elif isinstance(dim, str):
            if dim not in self.dims:
                raise ValueError(f"Shape {self} has no dimension '{dim}'")
            return self.names.index(dim)
        elif isinstance(dim, SHAPE_TYPES):
            assert len(dim) == 1, f"index() requires a single dimension as input but got {dim}. Use indices() for multiple dimensions."
            return self.names.index(dim.name)
        raise ValueError(f"index() requires a single dimension as input but got {dim}")

    def indices(self, names: Sequence[str]) -> Tuple[int, ...]:
        order = self.names
        return tuple(order.index(n) for n in names)

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return list(self.dims.values())[selection]
        elif isinstance(selection, slice):
            return concat_shapes_(*list(self.dims.values())[selection])
        elif isinstance(selection, str):
            if ',' in selection:
                selection = [self.index(s.strip()) for s in selection.split(',')]
            else:
                selection = self.index(selection)
            return self[selection]
        elif isinstance(selection, SHAPE_TYPES):
            selection = selection.names
        if isinstance(selection, (tuple, list)):
            selection = [self.names[sel] if isinstance(sel, int) else sel for sel in selection]
            if DEBUG_CHECKS:
                assert all(isinstance(s, str) for s in selection)
            return concat_shapes_(*[self.dims[n] for n in selection])
        raise AssertionError("Can only access shape elements as shape[int], shape[str], shape[slice], shape[Sequence] or shape[Shape]")

    def __iter__(self):
        return iter(self.dims.values())

    def get_size(self, dim: Union[str, 'Shape'], default=None):
        name = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        if default is None:
            return self.dims[name].size
        else:
            return self.dims[name].size if name in self.dims else default

    def get_dim_type(self, dim: str) -> str:
        return self.dims[dim].dim_type

    def get_item_names(self, dim: Union[str, 'Shape'], fallback_spatial=False) -> Union[tuple, None]:
        name = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        dim = self.dims[name]
        if dim.slice_names is not None:
            return dim.slice_names
        elif fallback_spatial and dim.dim_type == CHANNEL_DIM and _size_equal(len(self.spatial), dim.size):
            return self.spatial.names
        return None

    def __and__(self, other):
        if other is dual:
            return self & self.primal.as_dual()
        if isinstance(other, (Dim, PureShape)):
            if not other:
                return self
            group = getattr(self, other.dim_type)
            merged = pure_merge(group, other, allow_varying_sizes=False)
            dims = {**self.dims, **merged.dims}
            return replace(self, dims=dims, **{other.dim_type: merged})
        return merge_shapes(self, other)

    __rand__ = __and__

    def is_compatible(self, other: Shape):
        return all(dim.is_compatible(other) for dim in self.dims.values())

    def isdisjoint(self, other) -> bool:
        if isinstance(other, Dim):
            return other.name not in self.dims
        if isinstance(other, PureShape):
            return other.isdisjoint(getattr(self, other.dim_type))
        if isinstance(other, MixedShape):
            return self.dims.keys().isdisjoint(other.dims)
        return self.dims.keys().isdisjoint(set(parse_dim_order(other)))

    @property
    def __empty__(self):
        return EMPTY_SHAPE

    def only(self, dims: 'DimFilter', reorder=False):
        if isinstance(dims, (Dim, PureShape)):
            return getattr(self, dims.dim_type).only(dims, reorder=reorder)
        if callable(dims):
            return dims(self)
        if isinstance(dims, str) and ',' not in dims:
            return self.dims.get(dims, EMPTY_SHAPE)
        if reorder:
            result = [self.only(sel) for sel in parse_dim_order(dims)]
            return concat_shapes_(*result)
        return concat_shapes_(*[dim for dim in self.dims.values() if dim.only(dims)])

    def __add__(self, other):
        if isinstance(other, int):
            assert not self.batch, f"Shape arithmetic not allowed for batch dims {self}"
            return concat_shapes_(*[dim + other for dim in self.dims.values()])
        return concat_shapes_(self, other)

    def __sub__(self, other):
        if isinstance(other, int):
            assert not self.batch, f"Shape arithmetic not allowed for batch dims {self}"
            return concat_shapes_(*[dim - other for dim in self.dims.values()])
        return self.without(other)

    def without(self, dims: 'DimFilter'):
        b = self.batch.without(dims)
        d = self.dual.without(dims)
        i = self.instance.without(dims)
        s = self.spatial.without(dims)
        c = self.channel.without(dims)
        type_count = bool(b) + bool(d) + bool(i) + bool(s) + bool(c)
        if type_count == 0:
            return EMPTY_SHAPE
        if type_count == 1:
            return b if b else (d if d else (i if i else (s if s else c)))  # if only one has entries, return it
        dims = {n: dim for n, dim in self.dims.items() if dim.without(dims)}
        return MixedShape(b, d, i, s, c, dims)

    def meshgrid(self, names=False):
        assert self.is_uniform, f"Shape.meshgrid() is currently not supported for non-uniform tensors, {self}"
        if self.volume == 0:
            return ()
        indices = [0] * len(self.dims)
        while True:
            if names:
                yield {dim: (names[index] if names is not None else index) for dim, index, names in zip(self.names, indices, self.item_names)}
            else:
                yield {dim: index for dim, index in zip(self.names, indices)}
            for i in range(self.rank-1, -1, -1):
                indices[i] = (indices[i] + 1) % self.sizes[i]
                if indices[i] != 0:
                    break
            else:
                return

    def with_size(self, size, keep_item_names=True):
        assert len(self.dims) == 1
        return next(iter(self.dims.values())).with_size(size, keep_item_names=keep_item_names)

    def with_dim_size(self, dim: Union[str, 'Shape'], size: Union[int, 'math.Tensor', str, tuple, list], keep_item_names=True):
        dim = dim.name if isinstance(dim, SHAPE_TYPES) else dim
        dims = dict(self.dims)
        dims[dim] = dims[dim].with_size(size, keep_item_names=keep_item_names)
        return concat_shapes_(*[dims[n] for n in self.dims])

    def with_sizes(self, sizes: Union[Sequence[int], Sequence[Tuple[str, ...]], 'Shape', int], keep_item_names=True):
        if isinstance(sizes, (int, SHAPE_TYPES)):
            b = self.batch.with_sizes(sizes)
            d = self.dual.with_sizes(sizes)
            i = self.instance.with_sizes(sizes)
            s = self.spatial.with_sizes(sizes)
            c = self.channel.with_sizes(sizes)
        else:
            assert len(sizes) == len(self.dims)
            sizes = {n: s for n, s in zip(self.dims, sizes)}
            b = self.batch.with_sizes([sizes[n] for n in self.batch.names])
            d = self.dual.with_sizes([sizes[n] for n in self.dual.names])
            i = self.instance.with_sizes([sizes[n] for n in self.instance.names])
            s = self.spatial.with_sizes([sizes[n] for n in self.spatial.names])
            c = self.channel.with_sizes([sizes[n] for n in self.channel.names])
        dims = {**b.dims, **d.dims, **i.dims, **s.dims, **c.dims}
        return MixedShape(b, d, i, s, c, {n: dims[n] for n in self.dims})

    def without_sizes(self):
        return concat_shapes_(*[dim.without_sizes() for dim in self.dims.values()])

    def flipped(self, dims: Union[List[str], Tuple[str]]):
        return concat_shapes_(*[dim.flipped(dims) for n, dim in self.dims.items()])

    def replace(self, dims: Union['Shape', str, tuple, list], new: 'Shape'):
        dims = parse_dim_order(dims)
        dim_list = list(self.dims.values())
        if len(dims) == len(new):
            for old, new_dim in zip(dims, new):
                new_dim = self.dims[old]._replace(new_dim)
                dim_list[self.index(old)] = new_dim
        elif len(new) > 1 and len(dims) == 1:
            i = self.index(dims[0])
            dim_list[i:i+1] = new
        else:
            if len(dims) == len(self.dims):
                return new
            i0 = self.index(dims[0])
            dim_list = [d for n, d in self.dims.items() if n not in dims]
            dim_list.insert(i0, new)
        return concat_shapes_(*dim_list)

    def replace_selection(self, names: Sequence[str], new: Shape):
        if DEBUG_CHECKS:
            assert all(isinstance(n, str) for n in names)
        dim_list = list(self.dims.values())
        for old_name, new_dim in zip(names, new):
            if old_name in self.dims:
                new_dim = self.dims[old_name].replace_selection(old_name, new_dim)
                dim_list[self.index(old_name)] = new_dim
        return concat_shapes_(*dim_list)

    def as_batch(self):
        dims = [dim.as_batch() for dim in self.dims.values()]
        return PureShape(BATCH_DIM, {dim.name: dim for dim in dims}) if len(dims) > 1 else dims[0]
    def as_dual(self):
        dims = [dim.as_dual() for dim in self.dims.values()]
        return PureShape(DUAL_DIM, {dim.name: dim for dim in dims}) if len(dims) > 1 else dims[0]
    def as_instance(self):
        dims = [dim.as_instance() for dim in self.dims.values()]
        return PureShape(INSTANCE_DIM, {dim.name: dim for dim in dims}) if len(dims) > 1 else dims[0]
    def as_spatial(self):
        dims = [dim.as_spatial() for dim in self.dims.values()]
        return PureShape(SPATIAL_DIM, {dim.name: dim for dim in dims}) if len(dims) > 1 else dims[0]
    def as_channel(self):
        dims = [dim.as_channel() for dim in self.dims.values()]
        return PureShape(CHANNEL_DIM, {dim.name: dim for dim in dims}) if len(dims) > 1 else dims[0]
    def as_type(self, new_type: Callable):
        return {batch: self.as_batch, dual: self.as_dual, instance: self.as_instance, spatial: self.as_spatial, channel: self.as_channel}[new_type]()

    def transpose(self, dim_type: str):
        pure = getattr(self, dim_type)
        if not pure:
            return self
        return concat_shapes_(*[dim.transpose(dim_type) for dim in self.dims.values()])



EMPTY_SHAPE = PureShape('__empty__', {})
""" Empty shape, `()` """

SHAPE_TYPES = (Dim, PureShape, MixedShape)


class IncompatibleShapes(Exception):
    """
    Raised when the shape of a tensor does not match the other arguments.
    """
    def __init__(self, message, *shapes: Shape):
        Exception.__init__(self, message)
        self.shapes = shapes


def parse_dim_names(obj: Union[str, Sequence[str], Shape], count: int) -> tuple:
    if isinstance(obj, str):
        parts = obj.split(',')
        result = []
        for part in parts:
            part = part.strip()
            if part == '...':
                result.extend([None] * (count - len(parts) + 1))
            elif part == ':':
                result.append(None)
            else:
                result.append(part)
        assert len(result) == count, f"Number of specified names in '{obj}' does not match number of dimensions ({count})"
        return tuple(result)
    elif isinstance(obj, SHAPE_TYPES):
        assert len(obj) == count, f"Number of specified names in {obj} does not match number of dimensions ({count})"
        return obj.names
    elif isinstance(obj, Sequence):
        assert len(obj) == count, f"Number of specified names in {obj} does not match number of dimensions ({count})"
        return tuple(obj)
    raise ValueError(obj)


def parse_dim_order(order: Union[str, tuple, list, Shape]) -> Sequence[str]:
    if isinstance(order, SHAPE_TYPES):
        return order.names
    if isinstance(order, str) and ',' not in order:
        return order,
    if isinstance(order, (tuple,list)):
        return order
    elif isinstance(order, str):
        parts = order.split(',')
        parts = [p.strip() for p in parts if p]
        return tuple(parts)
    raise ValueError(order)


def _construct_shape(dim_type: str, *args, **kwargs):
    dims = {}
    for arg in args:
        parts = [s.strip() for s in arg.split(',')]
        for name in parts:
            name = _apply_prefix(name, dim_type)
            assert name not in dims, f"Duplicate dimension name {name}"
            dims[name] = Dim(name, None, dim_type, None)
    for name, size in kwargs.items():
        name = _apply_prefix(name, dim_type)
        assert name not in dims, f"Duplicate dimension name '{name}'"
        if isinstance(size, str):
            items = tuple([i.strip() for i in size.split(',')])
            if not items[-1]:
                items = items[:-1]
            size = len(items)
        elif isinstance(size, (tuple, list)):
            assert all(isinstance(s, str) for s in size), f"Item names must all be of type 'str' but got '{size}'"
            items = tuple(size)
            size = len(items)
        elif isinstance(size, SHAPE_TYPES):
            items = size.names
            size = size.rank
        elif size is None or isinstance(size, int):
            # keep size
            items = None
        else:
            items = None
            from ._tensors import Tensor
            if isinstance(size, Tensor):
                size = int(size) if size.shape.volume == 1 else size
            else:
                try:
                    size = int(size)
                except ValueError:
                    raise ValueError(f"Cannot construct dimension from {type(size).__name__}. Only int, tuple, list, str or Shape allowed. Got {size}")
        dims[name] = Dim(name, size, dim_type, items)
    return next(iter(dims.values())) if len(dims) == 1 else PureShape(dim_type, dims)


def _apply_prefix(name: str, dim_type: str):
    match = re.search("\\w", name)
    assert match, f"Dimension name must contain at least one letter or underscore but got '{name}'"
    proper_name_index = match.start()
    prefix = '~' if dim_type == DUAL_DIM else ''
    return prefix + name[proper_name_index:]


def shape(obj, allow_unshaped=False) -> Shape:
    """
    If `obj` is a `Tensor` or `phiml.math.magic.Shaped`, returns its shape.
    If `obj` is a `Shape`, returns `obj`.

    This function can be passed as a `dim` argument to an operation to specify that it should act upon all dimensions.

    Args:
        obj: `Tensor` or `Shape` or `Shaped`
        allow_unshaped: If `True`, returns an empty shape for unsupported objects, else raises a `ValueError`.

    Returns:
        `Shape`
    """
    if isinstance(obj, SHAPE_TYPES):
        return obj
    if hasattr(obj, 'shape') and isinstance(obj.shape, SHAPE_TYPES):
        return obj.shape
    if hasattr(obj, '__shape__'):
        return obj.__shape__()
    if isinstance(obj, (Number, bool)):
        return EMPTY_SHAPE
    if obj is None:
        return EMPTY_SHAPE
    if isinstance(obj, (tuple, list)) and all(isinstance(item, (int, float, complex, bool)) for item in obj):
        return channel(vector=len(obj))
    from .magic import PhiTreeNode, Shaped, BoundDim
    if isinstance(obj, BoundDim):
        return shape(obj.obj)[obj.name]
    if isinstance(obj, (tuple, list)) and all(isinstance(item, (PhiTreeNode, Shaped)) for item in obj):
        return merge_shapes(*obj, allow_varying_sizes=True)
    if isinstance(obj, dict) and all(isinstance(item, (PhiTreeNode, Shaped)) for item in obj):
        return merge_shapes(*obj.values(), allow_varying_sizes=True)
    if isinstance(obj, PhiTreeNode):
        from ._magic_ops import all_attributes
        return merge_shapes(*[getattr(obj, a) for a in all_attributes(obj, assert_any=True)], allow_varying_sizes=True)
    from ..backend import choose_backend, NoBackendFound
    try:
        backend = choose_backend(obj)
        shape_tuple = backend.staticshape(obj)
        if len(shape_tuple) == 0:
            return EMPTY_SHAPE
        elif len(shape_tuple) == 1:
            return channel('vector')
        else:
            raise ValueError(f"Cannot auto-complete shape of {backend} tensor with shape {shape_tuple}. Only 0D and 1D tensors have a Φ-ML shape by default.")
    except NoBackendFound:
        if allow_unshaped:
            return EMPTY_SHAPE
        raise ValueError(f'shape() requires Shaped or Shape argument but got {type(obj)}')


def spatial(*args, **dims: Union[int, str, tuple, list, Shape, 'Tensor']) -> Shape:
    """
    Returns the spatial dimensions of an existing `Shape` or creates a new `Shape` with only spatial dimensions.

    Usage for filtering spatial dimensions:
    >>> spatial_dims = spatial(shape)
    >>> spatial_dims = spatial(tensor)

    Usage for creating a `Shape` with only spatial dimensions:
    >>> spatial_shape = spatial('undef', x=2, y=3)
    (x=2, y=3, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `batch`, `instance`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type spatial.
    """
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(SPATIAL_DIM, *args, **dims)
    elif len(args) == 1 and isinstance(args[0], SHAPE_TYPES):
        return args[0].spatial
    assert len(args) == 1, f"spatial() must be called either as a selector spatial(Shape) or spatial(Tensor) or as a constructor spatial(*names, **dims). Got *args={args}, **dims={dims}"
    return shape(args[0]).spatial


def channel(*args, **dims: Union[int, str, tuple, list, Shape, 'Tensor']) -> Shape:
    """
    Returns the channel dimensions of an existing `Shape` or creates a new `Shape` with only channel dimensions.

    Usage for filtering channel dimensions:
    >>> channel_dims = channel(shape)
    >>> channel_dims = channel(tensor)

    Usage for creating a `Shape` with only channel dimensions:
    >>> channel_shape = channel('undef', vector=2)
    (vector=2, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `spatial`, `batch`, `instance`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type channel.
    """
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(CHANNEL_DIM, *args, **dims)
    elif len(args) == 1 and isinstance(args[0], SHAPE_TYPES):
        return args[0].channel
    assert len(args) == 1, f"channel() must be called either as a selector channel(Shape) or channel(Tensor) or as a constructor channel(*names, **dims). Got *args={args}, **dims={dims}"
    return shape(args[0]).channel


def batch(*args, **dims: Union[int, str, tuple, list, Shape, 'Tensor']) -> Shape:
    """
    Returns the batch dimensions of an existing `Shape` or creates a new `Shape` with only batch dimensions.

    Usage for filtering batch dimensions:
    >>> batch_dims = batch(shape)
    >>> batch_dims = batch(tensor)

    Usage for creating a `Shape` with only batch dimensions:
    >>> batch_shape = batch('undef', batch=2)
    (batch=2, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `spatial`, `instance`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type batch.
    """
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(BATCH_DIM, *args, **dims)
    elif len(args) == 1 and isinstance(args[0], SHAPE_TYPES):
        return args[0].batch
    assert len(args) == 1, f"batch() must be called either as a selector batch(Shape) or batch(Tensor) or as a constructor batch(*names, **dims). Got *args={args}, **dims={dims}"
    return shape(args[0]).batch


def instance(*args, **dims: Union[int, str, tuple, list, Shape, 'Tensor']) -> Shape:
    """
    Returns the instance dimensions of an existing `Shape` or creates a new `Shape` with only instance dimensions.

    Usage for filtering instance dimensions:
    >>> instance_dims = instance(shape)
    >>> instance_dims = instance(tensor)

    Usage for creating a `Shape` with only instance dimensions:
    >>> instance_shape = instance('undef', points=2)
    (points=2, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `batch`, `spatial`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type instance.
    """
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(INSTANCE_DIM, *args, **dims)
    elif len(args) == 1 and isinstance(args[0], SHAPE_TYPES):
        return args[0].instance
    assert len(args) == 1, f"instance() must be called either as a selector instance(Shape) or instance(Tensor) or as a constructor instance(*names, **dims). Got *args={args}, **dims={dims}"
    return shape(args[0]).instance


def dual(*args, **dims: Union[int, str, tuple, list, Shape, 'Tensor']) -> Shape:
    """
    Returns the dual dimensions of an existing `Shape` or creates a new `Shape` with only dual dimensions.

    Dual dimensions are assigned the prefix `~` to distinguish them from regular dimensions.
    This way, a regular and dual dimension of the same name can exist in one `Shape`.

    Dual dimensions represent the input space and are typically only present on matrices or higher-order matrices.
    Dual dimensions behave like batch dimensions in regular operations, if supported.
    During matrix multiplication, they are matched against their regular counterparts by name (ignoring the `~` prefix).

    Usage for filtering dual dimensions:

    >>> dual_dims = dual(shape)
    >>> dual_dims = dual(tensor)

    Usage for creating a `Shape` with only dual dimensions:

    >>> dual('undef', points=2)
    (~undefᵈ=None, ~pointsᵈ=2)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `batch`, `spatial`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type dual.
    """
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(DUAL_DIM, *args, **dims)
    elif len(args) == 1 and isinstance(args[0], SHAPE_TYPES):
        return args[0].dual
    assert len(args) == 1, f"dual() must be called either as a selector dual(Shape) or dual(Tensor) or as a constructor dual(*names, **dims). Got *args={args}, **dims={dims}"
    return shape(args[0]).dual


def auto(spec: Union[str, Shape], default_type: Callable = None) -> Shape:
    """
    Create a single-dimension `Shape` from a specification string.
    
    Args:
        spec: String specifying the dimension name and type.
            The type can be specified by a trailing superscript letter or `:` followed by the regular letter.
            Examples: `vector:c` or `vectorᶜ` indicate channel dimensions.
            Leading `~` indicate dual dimensions.
        default_type: Fallback type if no type is specified.
            If not provided, an error will be thrown if `spec` does not specify the type.

    Returns:
        `Shape`
    """
    if isinstance(spec, SHAPE_TYPES):
        return spec  # allow multi-dim Shapes as well, as the main application is stacking
    assert isinstance(spec, str), f"spec must be a Shape or str but got {type(spec)}"
    assert ',' not in spec, f"auto dim only supported for single dimensions"
    return parse_shape_spec(spec, default_type=default_type)


class InvalidShapeSpec(ValueError):
    pass


SPEC_PATTERNS = {
    'name_type_items': re.compile(r'(~?)(\w+):(\w+)=\(([^)]*)\)'),
    'name_type': re.compile(r'(~?)(\w+):(\w+)'),
    'name_items': re.compile(r'(~?)(\w+)=\(([^)]*)\)'),
    'items': re.compile(r'(~?)\(([^)]*)\)'),
    'dual_name': re.compile(r'(~\w+)'),
    'single_letter': re.compile(r'(\w)(?=,|$)'),
    'name_only': re.compile(r'(\w+)(?=,|$)')
}


def parse_shape_spec(input_string, default_type: Callable = None) -> Shape:
    dims = []
    pos = 0
    while pos < len(input_string):
        if match := SPEC_PATTERNS['name_type_items'].match(input_string, pos):
            tilde, n, t, values = match.groups()
            if tilde and t not in ('d', 'dual'):
                raise InvalidShapeSpec(input_string, f"Dimension names starting with ~ must be of type dual. Failed at index {pos}: {input_string[pos:]}")
            elif not tilde and t in ('d', 'dual'):
                raise InvalidShapeSpec(input_string, f"Dual dims must start with ~. Failed at index {pos}: {input_string[pos:]}")
            items = [n.strip() for n in values.split(',') if n.strip()]
            dims.append(Dim('~' + n if tilde else n, len(items), INV_CHAR[t] if len(t) == 1 else t, tuple(items)))
            pos = match.end() + 1
        elif match := SPEC_PATTERNS['name_type'].match(input_string, pos):
            tilde, n, t = match.groups()
            if tilde and t not in ('d', 'dual'):
                raise InvalidShapeSpec(input_string, f"Dimension names starting with ~ must be of type dual. Failed at index {pos}: {input_string[pos:]}")
            elif not tilde and t in ('d', 'dual'):
                raise InvalidShapeSpec(input_string, f"Dual dims must start with ~. Failed at index {pos}: {input_string[pos:]}")
            # Check if the next character is an equal sign followed by parentheses
            next_char_pos = pos + len(match.group())
            if next_char_pos < len(input_string) and input_string[next_char_pos] == '=':
                raise ValueError(f"Invalid format at position {pos}: values must be inside parentheses")
            dims.append(Dim('~' + n if tilde else n, None, INV_CHAR[t] if len(t) == 1 else t, None))
            pos = match.end() + 1
        elif match := SPEC_PATTERNS['name_items'].match(input_string, pos):
            tilde, n, values = match.groups()
            items = [n.strip() for n in values.split(',') if n.strip()]
            dims.append(Dim('~' + n if tilde else n, len(items), DUAL_DIM if tilde else CHANNEL_DIM, tuple(items)))
            pos = match.end() + 1
        elif match := SPEC_PATTERNS['items'].match(input_string, pos):
            tilde, values = match.groups()
            items = [n.strip() for n in values.split(',') if n.strip()]
            dims.append(Dim('~vector' if tilde else 'vector', len(items), DUAL_DIM if tilde else CHANNEL_DIM, tuple(items)))
            pos = match.end() + 1
        elif match := SPEC_PATTERNS['dual_name'].match(input_string, pos):
            n, = match.groups()
            dims.append(Dim(n, None, DUAL_DIM, None))
            pos = match.end() + 1
        elif match := SPEC_PATTERNS['single_letter'].match(input_string, pos):
            n, = match.groups()
            dims.append(Dim(n, None, SPATIAL_DIM, None))
            pos = match.end() + 1
        elif default_type is not None and (match := SPEC_PATTERNS['name_only'].match(input_string, pos)):
            n, = match.groups()
            default_type_str = TYPE_BY_FUNCTION[default_type]
            dims.append(Dim(n, None, default_type_str, None))
            pos = match.end() + 1
        else:
            raise InvalidShapeSpec(input_string, f"Failed to parse from index {pos}: '{input_string[pos:]}'. Dims must be specified as name:type or name:type=(item_names...). Names and types may only be omitted if component names are given.")
    return concat_shapes_(*dims)


DIM_FUNCTIONS = {BATCH_DIM: batch, SPATIAL_DIM: spatial, INSTANCE_DIM: instance, CHANNEL_DIM: channel, DUAL_DIM: dual}
TYPE_BY_FUNCTION = {v: k for k, v in DIM_FUNCTIONS.items()}


def merge_shapes(*objs: Union[Shape, Any], allow_varying_sizes=False) -> Shape:
    """
    Combines `shapes` into a single `Shape`, grouping dimensions by type.
    If dimensions with equal names are present in multiple shapes, their types and sizes must match.

    The shorthand `shape1 & shape2` merges shapes with `check_exact=[spatial]`.

    See Also:
        `concat_shapes()`.

    Args:
        *objs: `Shape` or `Shaped` objects to combine.
        allow_varying_sizes: If `True`, merges incompatible dims by setting their size to `None` and erasing their item names.
            If `False`, raises an error for incompatible dims.

    Returns:
        Merged `Shape`

    Raises:
        IncompatibleShapes if the shapes are not compatible
    """
    if not objs:
        return EMPTY_SHAPE
    shapes = [shape(obj) for obj in objs]
    is_pure = not any(isinstance(s, MixedShape) for s in shapes)
    if is_pure:
        is_pure = len(set([s.dim_type for s in shapes])) == 1
    if is_pure:
        return pure_merge(*shapes, allow_varying_sizes=allow_varying_sizes)
    else:
        b = pure_merge(*[s.batch for s in shapes], allow_varying_sizes=allow_varying_sizes)
        d = pure_merge(*[s.dual for s in shapes], allow_varying_sizes=allow_varying_sizes)
        i = pure_merge(*[s.instance for s in shapes], allow_varying_sizes=allow_varying_sizes)
        s = pure_merge(*[s.spatial for s in shapes], allow_varying_sizes=allow_varying_sizes)
        c = pure_merge(*[s.channel for s in shapes], allow_varying_sizes=allow_varying_sizes)
        dims = {**b.dims, **d.dims, **i.dims, **s.dims, **c.dims}
        return MixedShape(b, d, i, s, c, dims) if dims else EMPTY_SHAPE


def pure_merge(*shapes: Shape, allow_varying_sizes: bool) -> Shape:
    all_dims = []
    non_empty = []
    for s in shapes:
        if isinstance(s, Dim):
            all_dims.append(s)
            non_empty.append(s)
        elif isinstance(s, PureShape) and s:
            all_dims.extend(s.dims.values())
            non_empty.append(s)
        else:
            assert isinstance(s, PureShape), f"Only Dim and PureShape allowed in pure_merge() but got {type(s)}"
    if not all_dims:
        return EMPTY_SHAPE
    if len(non_empty) == 1:
        return non_empty[0]
    dims = {}
    for dim in all_dims:
        if dim.name not in dims:
            dims[dim.name] = dim
        else:  # check size match
            prev_dim = dims[dim.name]
            sizes_match = _size_equal(dim.size, prev_dim.size)
            if allow_varying_sizes:
                if not sizes_match:
                    dims[dim.name] = Dim(dim.name, None, dim.dim_type, None)
            else:
                if not sizes_match:
                    raise IncompatibleShapes(f"Cannot merge shapes {shapes} because dimension '{dim.name}' exists with different sizes.", *shapes)
                names1 = prev_dim.slice_names
                names2 = dim.slice_names
                if names1 is not None and names2 is not None and len(names1) > 1:
                    if names1 != names2:
                        if set(names1) == set(names2):
                            raise IncompatibleShapes(
                                f"Inconsistent component order on {dim.name}: '{','.join(names1)}' vs '{','.join(names2)}' in dimension '{dim.name}'. Failed to merge shapes {shapes}",
                                *shapes)
                        else:
                            raise IncompatibleShapes(f"Cannot merge shapes {shapes} because dimension '{dim.name}' exists with different item names.", *shapes)
                elif names1 is None and names2 is not None:
                    dims[dim.name] = dim  # override prev_dim with dim because it has item names
    return next(iter(dims.values())) if len(dims) == 1 else PureShape(all_dims[0].dim_type, dims)


def non_batch(obj) -> Shape:
    """
    Returns the non-batch dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    return shape(obj).non_batch


def non_spatial(obj) -> Shape:
    """
    Returns the non-spatial dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    return shape(obj).non_spatial


def non_instance(obj) -> Shape:
    """
    Returns the non-instance dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    return shape(obj).non_instance


def non_channel(obj) -> Shape:
    """
    Returns the non-channel dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    return shape(obj).non_channel


def non_dual(obj) -> Shape:
    """
    Returns the non-dual dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    return shape(obj).non_dual


def non_primal(obj) -> Shape:
    """
    Returns the batch and dual dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    return shape(obj).non_primal


def primal(obj) -> Shape:
    """
    Returns the instance, spatial and channel dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    return shape(obj).primal


def _size_equal(s1, s2):
    if s1 is None:
        return s2 is None
    if s2 is None:
        return False
    if isinstance(s1, int):
        return isinstance(s2, int) and s2 == s1
    else:
        return math.close(s1, s2)


def concat_shapes(*shapes: Union[Shape, Any]) -> Shape:
    """
    Creates a `Shape` listing the dimensions of all `shapes` in the given order.

    See Also:
        `merge_shapes()`.

    Args:
        *shapes: Shapes to concatenate. No two shapes must contain a dimension with the same name.

    Returns:
        Combined `Shape`.
    """
    shapes = [obj if isinstance(obj, SHAPE_TYPES) else shape(obj) for obj in shapes]
    return concat_shapes_(*shapes)


def concat_shapes_(*shapes: Shape) -> Shape:
    shapes = [s for s in shapes if s]
    if not shapes:
        return EMPTY_SHAPE
    if len(shapes) == 1:
        return shapes[0]
    all_dims = []
    by_type = {BATCH_DIM: EMPTY_SHAPE, DUAL_DIM: EMPTY_SHAPE, INSTANCE_DIM: EMPTY_SHAPE, SPATIAL_DIM: EMPTY_SHAPE, CHANNEL_DIM: EMPTY_SHAPE}
    for s in shapes:
        if s:
            all_dims.extend(s.dims.values())
            if isinstance(s, (Dim, PureShape)):
                by_type[s.dim_type] &= s
            else:
                by_type[BATCH_DIM] &= s.batch
                by_type[DUAL_DIM] &= s.dual
                by_type[INSTANCE_DIM] &= s.instance
                by_type[SPATIAL_DIM] &= s.spatial
                by_type[CHANNEL_DIM] &= s.channel
    dims = {dim.name: dim for dim in all_dims}
    if len(dims) != len(all_dims):
        raise IncompatibleShapes(f"Cannot concatenate shapes {list(shapes)}. Duplicate dimension names are not allowed.")
    return MixedShape(dims=dims, **by_type)


def shape_stack(stack_dim: Shape, *shapes: Shape, stack_dim_first=True):
    """ Returns the shape of a tensor created by stacking tensors with `shapes`. """
    if stack_dim.rank > 1:
        assert stack_dim.volume == len(shapes), f"stack_dim {stack_dim} does not match number of shapes: {len(shapes)}"
    elif len(stack_dim) == 1:
        stack_dim = Dim(stack_dim.name, len(shapes), stack_dim.dim_type, stack_dim.item_names[0])
    else:
        assert len(shapes) == 1
        return shapes[0]
    if not shapes:
        return stack_dim
    if len(shapes) == 1:
        return (stack_dim + shapes[0]) if stack_dim_first else (shapes[0] + stack_dim)
    dims = {}
    conflicts = set()  # names with conflicting item names
    # --- Find conflicting slice names (only where sizes match up) ---
    for s in shapes:
        for dim in s:
            prev = dims.get(dim.name, None)
            if prev:
                if dim.slice_names != prev.slice_names:
                    if prev.slice_names is None:
                        if dim.name not in conflicts:
                            dims[dim.name] = dim  # update with item names
                    else:
                        warnings.warn(f"Stacking shapes with incompatible item names will result in item names being lost. For {dim.name} Got {prev.slice_names} and {dim.slice_names}", RuntimeWarning)
                        conflicts.add(dim.name)
                        dims[dim.name] = Dim(dim.name, dim.size, dim.dim_type, None)
                # merge item names, not conflicts
                pass
            else:
                dims[dim.name] = dim
    # --- Stack sizes ---
    sdims = []
    for name, dim in dims.items():
        dim_sizes = [(s.get_size(name) if name in s else None) for s in shapes]
        valid_dim_sizes = [s.get_size(name) for s in shapes if name in s]
        if all([math.close(s, valid_dim_sizes[0]) for s in valid_dim_sizes[1:]]):
            sdims.append(dim)  # all equal, keep as is
        else:
            from ._magic_ops import stack
            dim_sizes_or_1 = [1 if s is None else s for s in dim_sizes]
            size = stack(dim_sizes_or_1, stack_dim, expand_values=True)
            sdims.append(Dim(name, size, dim.dim_type, None))
    return concat_shapes_(stack_dim, *sdims) if stack_dim_first else concat_shapes_(*sdims, stack_dim)


def unstack(self: Shape, dim: str) -> Sequence[Shape]:
    """
    Slices this `Shape` along a dimension.
    The dimension listing the sizes of the shape is referred to as `'dims'`.

    Non-uniform tensor shapes may be unstacked along other dimensions as well, see
    https://tum-pbs.github.io/PhiML/Non_Uniform.html

    Args:
        dim: dimension to unstack

    Returns:
        slices of this shape
    """
    assert dim != 'dims'
    if dim not in self and self.is_uniform:
        return self,
    elif self.is_uniform:
        return (self-dim,) * self.get_size(dim)
    # --- non-uniform case ---
    from ._tensors import Tensor
    if dim in self:
        inner = self.without(dim)
        dim_size = self.get_size(dim)
    else:
        inner = self
        dim_size = self.shape.get_size(dim)
    sizes = []
    for size in inner.sizes:
        if isinstance(size, Tensor) and dim in size.shape:
            sizes.append(size._unstack(dim))
            dim_size = size.shape.get_size(dim)
        else:
            sizes.append(size)
    assert isinstance(dim_size, int)
    result = []
    for i in range(dim_size):
        sizes_i = [int(size[i]) if isinstance(size, tuple) else size for size in sizes]
        result.append(inner.with_sizes(sizes_i))
    return result


def prepare_gather(self: Shape, dim: str, selection: Union[slice, int, 'Shape', str, tuple, list]) -> Union[slice, List[int]]:
    """
    Parse a slice object for a specific dimension.

    Args:
        dim: Name of dimension to slice.
        selection: Slice object.

    Returns:

    """
    if isinstance(selection, SHAPE_TYPES):
        selection = selection.name if selection.rank == 1 else selection.names
    if isinstance(selection, str) and ',' in selection:
        selection = parse_dim_order(selection)
    if isinstance(selection, str):  # single item name
        item_names = self.get_item_names(dim, fallback_spatial=True)
        assert item_names is not None, f"No item names defined for dim '{dim}' in tensor {self.shape} and dimension size does not match spatial rank."
        assert selection in item_names, f"Accessing tensor.{dim}['{selection}'] failed. Item names are {item_names}."
        selection = item_names.index(selection)
    if isinstance(selection, (tuple, list)):
        selection = list(selection)
        if any([isinstance(s, str) for s in selection]):
            item_names = self.get_item_names(dim, fallback_spatial=True)
            for i, s in enumerate(selection):
                if isinstance(s, str):
                    assert item_names is not None, f"Accessing tensor.{dim}['{s}'] failed because no item names are present on tensor {self.shape}"
                    assert s in item_names, f"Accessing tensor.{dim}['{s}'] failed. Item names are {item_names}."
                    selection[i] = item_names.index(s)
        if not selection:  # empty
            selection = slice(0, 0)
    return selection


def prepare_renaming_gather(self: Shape, dim: str, selection: Union[slice, int, 'Shape', str, tuple, list]):
    if isinstance(selection, str) and '->' in selection:
        selection, new_names = selection.split('->')
        if new_names == '?':
            return prepare_gather(self, dim, selection), self[dim]._with_item_names((None,))
        else:
            return prepare_gather(self, dim, selection), self[dim].with_size(new_names)
    else:
        return prepare_gather(self, dim, selection), None


def after_gather(self, selection: dict) -> 'Shape':
    from . import Tensor
    if self.is_non_uniform:
        sizes = [(s[selection] if isinstance(s, Tensor) else s) for s in self.sizes]
        sizes = [(int(s) if isinstance(s, Tensor) and s.rank == 0 else s) for s in sizes]
        result = self.with_sizes(sizes)
    else:
        result = self
    for sel_dim, sel in selection.items():
        if sel_dim not in self.names:
            continue
        sel = prepare_gather(self, sel_dim, sel)
        if isinstance(sel, int):
            result = result.without(sel_dim)
        elif isinstance(sel, slice):
            step = int(sel.step) if sel.step is not None else 1
            start = int(sel.start) if sel.start is not None else (0 if step > 0 else self.get_size(sel_dim)-1)
            stop = int(sel.stop) if sel.stop is not None else (self.get_size(sel_dim) if step > 0 else -1)
            if stop < 0 and step > 0:
                stop += self.get_size(sel_dim)
                assert stop >= 0
            if start < 0 and step > 0:
                start += self.get_size(sel_dim)
                assert start >= 0
            stop = min(stop, self.get_size(sel_dim))
            new_size = math.to_int64(math.ceil(math.wrap((stop - start) / step)))
            if new_size.rank == 0:
                new_size = int(new_size)  # NumPy array not allowed because not hashable
            keep_item_names = step == -1 and sel.stop is None and sel.start is None
            #         result = result.flipped([sel_dim])
            old_item_names = self.get_item_names(sel_dim)
            if old_item_names is not None:
                if not keep_item_names:
                    result = result.with_dim_size(sel_dim, tuple(self.get_item_names(sel_dim)[sel]))
            else:
                result = result.with_dim_size(sel_dim, new_size, keep_item_names=True)
        elif isinstance(sel, (tuple, list)):
            if self.get_item_names(sel_dim) is not None:
                result = result.with_dim_size(sel_dim, tuple([self.get_item_names(sel_dim)[i] for i in sel]))
            else:
                result = result.with_dim_size(sel_dim, len(sel))
        elif isinstance(sel, Tensor):
            if sel.dtype.kind == bool:
                raise NotImplementedError("Shape.after_gather(Tensor[bool]) not yet implemented")
                # from ._ops import nonzero
                # sel = nonzero(sel)
            if sel.dtype.kind == int:
                assert len(selection) == 1, f"When slicing a Shape with Tensor[int], only one sel item is allowed but got {sel}"
                sel_shape = shape(sel)
                assert sel_shape.channel_rank == 1 and sel_shape.channel.item_names[0], f"Shape.after_gather(Tensor[int]) requires indices to have a single channel dim with item names but got {sel}"
                indexed = sel_shape.channel.item_names[0]
                assert indexed in self, f"All indexed dims {indexed} must be part of sliced Shape {self}"
                from ._ops import slice_
                sizes = [slice_(s, sel) for s in self.sizes]
                return self.with_sizes(sizes).without(indexed) & sel_shape.non_channel
        else:
            raise NotImplementedError(f"{type(sel)} not supported. Only (int, slice) allowed.")
    return result


def resolve_index(self, index: Dict[str, Union[slice, int, 'Shape', str, tuple, list]]) -> Dict[str, Union[slice, int, tuple, list]]:
    """
    Replaces item names by the corresponding indices.

    Args:
        index: n-dimensional index or slice.

    Returns:
        Same index but without any reference to item names.
    """
    return {dim: self.prepare_gather(dim, s) for dim, s in index.items()}


def after_pad(self, widths: dict) -> 'Shape':
    sizes = list(self.sizes)
    for dim, (lo, up) in widths.items():
        if dim in self.names:
            sizes[self.index(dim)] += int(lo + up)
    return self.with_sizes(sizes)


def mask(self: Shape, names: Union[tuple, list, set, 'Shape']):
    names = [names] if isinstance(names, str) else names.names
    return [int(name in names) for name in self.names]


def transposed(self):
    raise NotImplementedError
    if self.channel_rank > 0:
        replacement = {DUAL_DIM: CHANNEL_DIM, CHANNEL_DIM: DUAL_DIM}
    elif self.instance_rank > 0:
        replacement = {DUAL_DIM: INSTANCE_DIM, INSTANCE_DIM: DUAL_DIM}
    elif self.spatial_rank > 0:
        replacement = {DUAL_DIM: SPATIAL_DIM, SPATIAL_DIM: DUAL_DIM}
    elif self.dual_rank > 0:
        warnings.warn(f"Transposing {self} is ill-defined because there are not primal dims. Replacing dual dims by channel dims.", SyntaxWarning)
        replacement = {DUAL_DIM: CHANNEL_DIM}
    else:
        raise ValueError(f"Cannot transpose shape {self} as it has no channel or instance or spatial dims.")
    return self._with_types(tuple([replacement.get(t, t) for t in self.types]))


def to_dict(self: Shape, include_sizes=True):
    result = dict(names=self.names, types=self.dim_types, item_names=self.item_names)
    if include_sizes:
        assert self.is_uniform, f"do_dict(Shape) only supports uniform shapes but got {self}"
        result['sizes'] = self.sizes
    return result


def from_dict(dict_: dict):
    names = dict_['names']
    sizes = list(dict_['sizes']) if 'sizes' in dict_ else [None] * len(names)
    types = dict_['types']
    item_names = tuple([None if n is None else tuple(n) for n in dict_['item_names']])
    for i, n in enumerate(item_names):
        if n and sizes[i] is None:
            sizes[i] = len(n)
    return concat_shapes_(*[Dim(n, s, t, i) for n, s, t, i in zip(names, sizes, types, item_names)])


Shape._from_dict = from_dict


def first_index(shape: Shape):
    return next(iter(shape.meshgrid()))


for cls in [Dim, PureShape, MixedShape]:
    cls.unstack = unstack
    cls.after_gather = after_gather
    cls.after_pad = after_pad
    cls.first_index = first_index
    cls.prepare_gather = prepare_gather
    cls._to_dict = to_dict
    cls.transposed = transposed
    cls.mask = mask
