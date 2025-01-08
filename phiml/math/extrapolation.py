"""
Extrapolations are used for padding tensors and sampling coordinates lying outside the tensor bounds.
Standard extrapolations are listed as global variables in this module.

Extrapolations are an important part of sampled fields such as grids.
See the documentation at https://tum-pbs.github.io/PhiML/Fields.html#extrapolations .
"""
import warnings
from abc import ABC
from numbers import Number
from typing import Union, Dict, Callable, Tuple, Optional, Sequence

from .magic import slicing_dict
from ..backend._backend import get_spatial_derivative_order
from ..backend import choose_backend
from ._shape import Shape, channel, spatial, EMPTY_SHAPE, merge_shapes, dual, non_dual, instance, parse_dim_names, parse_dim_order, after_pad
from ._magic_ops import concat, stack, expand, rename_dims
from ._tensors import Tensor, Dense, TensorStack, wrap, to_dict as tensor_to_dict, from_dict as tensor_from_dict
from . import _ops as math  # TODO this executes _ops.py, can we avoid this?


class Extrapolation:
    """
    Extrapolations are used to determine values of grids or other structures outside the sampled bounds.
    They play a vital role in padding and sampling.
    """

    def __init__(self, pad_rank):
        """
        Args:
            pad_rank: low-ranking extrapolations are handled first during mixed-extrapolation padding.
                The typical order is periodic=1, boundary=2, symmetric=3, reflect=4, constant=5.
        """
        self.pad_rank = pad_rank

    @property
    def shape(self):
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """
        Serialize this extrapolation to a dictionary that is serializable (JSON-writable).
        
        Use `from_dict()` to restore the Extrapolation object.
        """
        raise NotImplementedError(self.__class__)

    def spatial_gradient(self) -> 'Extrapolation':
        """
        Returns the extrapolation for the spatial gradient of a tensor/field with this extrapolation.

        Returns:
            `Extrapolation` or `NotImplemented`
        """
        raise NotImplementedError(self.__class__)

    def valid_outer_faces(self, dim) -> Tuple[bool, bool]:
        """
        Use `determines_boundary_values()` instead.

         `(lower: bool, upper: bool)` indicating whether the values sampled at the outer-most faces of a staggered grid with this extrapolation are valid, i.e. need to be stored and are not redundant. """
        return not self.determines_boundary_values(dim+'-'), not self.determines_boundary_values(dim+'+')

    def determines_boundary_values(self, boundary_key: str) -> bool:
        """
        Tests whether this extrapolation fully determines the values at the boundary faces of the outermost cells or elements.
        If so, the values need not be stored along with the inside values.

        Override this function instead of `valid_outer_faces()`.

        Args:
            boundary_key: Boundary name as `str`.

        Returns:
            Whether the value is fully determined by the boundary and need not be stored elsewhere.
        """
        raise NotImplementedError(self.__class__)

    @property
    def is_flexible(self) -> bool:
        """
        Whether the outside values are affected by the inside values.
        Only `True` if there are actual outside values, i.e. PERIODIC is not flexible.

        This property is important for pressure solves to determine whether the total divergence is fixed or can be adjusted during the solve.
        """
        raise NotImplementedError(self.__class__)

    def pad(self, value: Tensor, widths: dict, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        """
        Pads a tensor using values from `self.pad_values()`.

        If `value` is a linear tracer, assume pad_values() to produce constant values, independent of `value`.
        To change this behavior, override this method.

        Args:
            value: `Tensor` to be padded
            widths: `dict` mapping `dim: str -> (lower: int, upper: int)`
            already_padded: Used when padding a tensor with multiple extrapolations.
                Contains all widths that have already been padded prior to this call.
                This causes the shape of `value` to be different from the original tensor passed to `math.pad()`.
            kwargs: Additional keyword arguments for padding, passed on to `pad_values()`.

        Returns:
            Padded `Tensor`
        """
        from ._trace import ShiftLinTracer
        if isinstance(value, ShiftLinTracer):
            lower = {dim: -lo for dim, (lo, _) in widths.items()}
            return value.shift(lower, new_shape=after_pad(value.shape, widths), val_fun=lambda v: ZERO.pad(v, widths, already_padded=already_padded, **kwargs), bias_fun=lambda b: self.pad(b, widths, already_padded=already_padded, **kwargs), nonzero_edge=False)
        already_padded = {} if already_padded is None else dict(already_padded)
        for dim, width in widths.items():
            assert (w > 0 for w in width), "Negative widths not allowed in Extrapolation.pad(). Use math.pad() instead."
            values = []
            if width[False] > 0:
                values.append(self.pad_values(value, width[False], dim, False, already_padded=already_padded, **kwargs))
            values.append(value)
            if width[True] > 0:
                values.append(self.pad_values(value, width[True], dim, True, already_padded=already_padded, **kwargs))
            value = concat(values, dim)
            if dim in already_padded:
                already_padded[dim] = tuple(i+j for i, j in zip(already_padded[dim], width))
            else:
                already_padded[dim] = width
        return value

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        """
        Determines the values with which the given tensor would be padded at the specified using this extrapolation.

        Args:
            value: `Tensor` to be padded.
            width: `int > 0`: Number of cells to pad along `dimension`.
            dim: Dimension name as `str`.
            upper_edge: `True` for upper edge, `False` for lower edge.
            already_padded: Used when padding a tensor with multiple extrapolations.
                Contains all widths that have already been padded prior to this call.
                This causes the shape of `value` to be different from the original tensor passed to `math.pad()`.

        Returns:
            `Tensor` that can be concatenated to `value` along `dimension`
        """
        raise NotImplementedError(self.__class__)

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, boundary: str, **kwargs) -> Tensor:
        """
        Determines pad values for boundary nodes of a sparsely connected graph.

        Args:
            value: `Tensor` to pad. Dense tensor containing an entry for each non-boundary node of the graph, laid out along a dual dim.
            connectivity: Sliced graph connectivity as sparse matrix. Only the relevant entries along the primal node dim are given.
            boundary: Boundary name to pad.
            **kwargs: Additional provided arguments, such as `mesh`.

        Returns:
            Values with which to pad `value`, laid out along the dual dim of `value`.
        """
        raise NotImplementedError(self.__class__)

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        """
        If `self.is_copy_pad`, transforms outside coordinates to the index from which the value is copied.
        
        Otherwise, the grid tensor is assumed to hold the correct boundary values for this extrapolation at the edge.
        Coordinates are then snapped to the valid index range.
        This is the default implementation.

        Args:
            coordinates: integer coordinates in index space
            shape: tensor shape

        Returns:
            Transformed coordinates
        """
        res = shape[coordinates.shape.get_item_names('vector')] if 'vector' in coordinates.shape and coordinates.shape.get_item_names('vector') else shape.spatial
        return math.clip(coordinates, 0, math.wrap(res - 1, channel('vector')))

    def is_copy_pad(self, dim: str, upper_edge: bool):
        """:return: True if all pad values are copies of existing values in the tensor to be padded"""
        return False

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return None

    def shortest_distance(self, start: Tensor, end: Tensor, domain_size: Tensor):
        """
        Computes the shortest distance between two points.
        Both points are assumed to lie within the domain

        Args:
            start: Start position.
            end: End position.
            domain_size: Domain side lengths as vector.

        Returns:
            Shortest distance from `start` to `end`.
        """
        return end - start

    def __getitem__(self, item):
        return self

    def _getitem_with_domain(self, item: dict, dim: str, upper_edge: bool, all_dims: Sequence[str]) -> 'Extrapolation':
        return self[item]

    def __eq__(self, other):
        raise NotImplementedError(self.__class__)

    def __hash__(self):
        raise NotImplementedError(self.__class__)  # must be overridden by all subclasses that implement __eq__

    def __ne__(self, other):
        return not self == other

    def __abs__(self):
        raise NotImplementedError(self.__class__)

    def __neg__(self):
        raise NotImplementedError(self.__class__)

    def __add__(self, other):
        raise NotImplementedError(self.__class__)

    def __radd__(self, other):
        raise NotImplementedError(self.__class__)

    def __sub__(self, other):
        raise NotImplementedError(self.__class__)

    def __rsub__(self, other):
        raise NotImplementedError(self.__class__)

    def __mul__(self, other):
        raise NotImplementedError(self.__class__)

    def __rmul__(self, other):
        raise NotImplementedError(self.__class__)

    def __truediv__(self, other):
        raise NotImplementedError(self.__class__)

    def __rtruediv__(self, other):
        raise NotImplementedError(self.__class__)


class ConstantExtrapolation(Extrapolation):
    """
    Extrapolate with a constant value.
    """

    def __init__(self, value: Union[Tensor, float]):
        Extrapolation.__init__(self, 5)
        self.value = wrap(value)
        """ Extrapolation value """
        assert self.value.dtype.kind in (bool, int, float, complex), f"Numeric value required for constant extrapolation but got '{value}'"

    @property
    def shape(self):
        return self.value.shape

    def __repr__(self):
        return repr(self.value)

    def to_dict(self) -> dict:
        return {'type': 'constant', 'value': self.value.numpy()}

    def __value_attrs__(self):
        return 'value',

    def __getitem__(self, item):
        return ConstantExtrapolation(self.value[item])

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'ConstantExtrapolation':
        if all(isinstance(v, ConstantExtrapolation) for v in values):
            return ConstantExtrapolation(stack([v.value for v in values], dim, **kwargs))
        else:
            return NotImplemented

    def spatial_gradient(self):
        return ZERO

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        return True

    @property
    def is_flexible(self) -> bool:
        return False

    def pad(self, value: Tensor, widths: dict, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        """Pads a tensor using constant values."""
        value = value._simplify()
        if isinstance(value, Dense):
            pad_value = self._get_pad_value(already_padded)
            backend = choose_backend(value._native, *pad_value._natives())
            for dim in pad_value.shape.non_batch.names:
                assert dim in value.shape, f"Cannot pad tensor {value.shape} with extrapolation {pad_value.shape} because non-batch dimension '{dim}' is missing."
            if pad_value.rank == 0:
                equal_values = math.always_close(self.value, value, 0, 0, equal_nan=True)
                if not equal_values:
                    required_dims = value._shape.only(tuple(widths.keys()))
                    value = value._cached(required_dims)
                should_pad_native = any(dim in value._names for dim in widths) and pad_value.shape.volume == 1
                if should_pad_native:
                    ordered_pad_widths = order_by_shape(value._names, widths, default=(0, 0))
                    result_native = backend.pad(value._native, ordered_pad_widths, 'constant', pad_value.native())
                else:
                    result_native = value._native
                if result_native is not NotImplemented:
                    return Dense(result_native, value._names, after_pad(value._shape, widths), value._backend)
            return Extrapolation.pad(self, value, widths, already_padded=already_padded, **kwargs)
        elif isinstance(value, TensorStack):
            if not value.requires_broadcast:
                return self.pad(value._contiguous(), widths)
            inner_widths = {dim: w for dim, w in widths.items() if dim != value._stack_dim.name}
            tensors = [self[{value._stack_dim.name: i}].pad(t, inner_widths) for i, t in enumerate(value.dimension(value._stack_dim.name))]
            return TensorStack(tensors, value._stack_dim)
        else:
            return Extrapolation.pad(self, value, widths, already_padded=already_padded, **kwargs)

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        shape = value.shape.after_gather({dim: slice(0, width)})
        pad_value = self._get_pad_value(already_padded)
        return math.expand(pad_value, shape)

    def _get_pad_value(self, already_padded: Optional[dict]):
        if get_spatial_derivative_order() == 0:
            if already_padded:
                return ZERO.pad(self.value, already_padded)
            else:
                return self.value
        else:
            return math.wrap(0)

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        return math.expand(self.value, dual(connectivity) & non_dual(value))

    def __eq__(self, other):
        return isinstance(other, ConstantExtrapolation) and math.always_close(self.value, other.value, equal_nan=True)

    def __hash__(self):
        return hash(self.__class__)

    def is_zero(self):
        return self == ZERO

    def is_one(self):
        return self == ONE

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return 'zeros' if self.is_zero() else None

    def __add__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value + other.value)
        elif self.is_zero():
            return other
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value - other.value)
        elif self.is_zero():
            return -other
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value - self.value)
        elif self.is_zero():
            return other
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value * other.value)
        elif self.is_one():
            return other
        elif self.is_zero():
            return self
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value / other.value)
        elif self.is_zero():
            return self
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value / self.value)
        elif self.is_one():
            return other
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value < other.value)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value > other.value)
        else:
            return NotImplemented

    def __abs__(self):
        return ConstantExtrapolation(abs(self.value))

    def __neg__(self):
        return ConstantExtrapolation(-self.value)


class _CopyExtrapolation(Extrapolation, ABC):

    @property
    def shape(self):
        return EMPTY_SHAPE

    def is_copy_pad(self, dim: str, upper_edge: bool):
        return True

    def to_dict(self) -> dict:
        return {'type': repr(self)}

    @property
    def backend_pad_mode(self) -> Optional[str]:
        return repr(self)

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return str(self)

    def __value_attrs__(self):
        return ()

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        return False

    @property
    def _is_dim_separable(self):
        """
        If `True`, the extrapolation values only depend on values of the same row/column.
        If `False`, collapsed dimensions have to be expanded during padding.
        """
        return True

    def pad(self, value: Tensor, widths: dict, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        value = value._simplify()
        from ._trace import ShiftLinTracer
        if isinstance(value, Dense):
            if not self._is_dim_separable:
                required_dims = value._shape.only(tuple(widths.keys()))
                value = value._cached(required_dims)
            should_pad_native = any(dim in value._names for dim in widths)
            if should_pad_native:
                ordered_pad_widths = order_by_shape(value._names, widths, default=(0, 0))
                result_native = value._backend.pad(value._native, ordered_pad_widths, self.backend_pad_mode)
            else:
                result_native = value._native
            if result_native is not NotImplemented:
                return Dense(result_native, value._names, after_pad(value._shape, widths), value._backend)
            return Extrapolation.pad(self, value, widths)
        elif isinstance(value, TensorStack):
            if not value.requires_broadcast:
                return self.pad(value._contiguous(), widths)
            inner_widths = {dim: w for dim, w in widths.items() if dim != value._stack_dim.name}
            tensors = [self.pad(t, inner_widths) for t in value.dimension(value._stack_dim.name)]
            return TensorStack(tensors, value._stack_dim)
        elif isinstance(value, ShiftLinTracer):
            return self._pad_linear_tracer(value, widths)
        else:
            raise NotImplementedError(f'{type(value)} not supported')

    def _pad_linear_tracer(self, value, widths: dict):
        raise NotImplementedError(self.__class__)

    def __eq__(self, other):
        return type(other) == type(self)

    def __hash__(self):
        return hash(self.__class__)

    def _op(self, other, op):
        if type(other) == type(self):
            return self
        if isinstance(other, ConstantExtrapolation):  # some operations can be handled by ConstantExtrapolation, e.g. * 0
            op = getattr(other, op.__name__)
            return op(self)
        else:
            return NotImplemented

    def __abs__(self):
        return self  # assume also applied to values

    def __neg__(self):
        return self  # assume also applied to values

    def __add__(self, other):
        return self._op(other, ConstantExtrapolation.__add__)

    def __radd__(self, other):
        return self._op(other, ConstantExtrapolation.__add__)

    def __mul__(self, other):
        return self._op(other, ConstantExtrapolation.__mul__)

    def __rmul__(self, other):
        return self._op(other, ConstantExtrapolation.__mul__)

    def __sub__(self, other):
        return self._op(other, ConstantExtrapolation.__rsub__)

    def __rsub__(self, other):
        return self._op(other, ConstantExtrapolation.__sub__)

    def __truediv__(self, other):
        return self._op(other, ConstantExtrapolation.__rtruediv__)

    def __rtruediv__(self, other):
        return self._op(other, ConstantExtrapolation.__truediv__)

    def __lt__(self, other):
        return self._op(other, ConstantExtrapolation.__gt__)

    def __gt__(self, other):
        return self._op(other, ConstantExtrapolation.__lt__)


class _ZeroGradient(_CopyExtrapolation):
    """Uses the closest defined value for points lying outside the defined region."""

    _CACHED_LOWER_MASKS = {}
    _CACHED_UPPER_MASKS = {}

    def __repr__(self):
        return 'zero-gradient'

    @property
    def backend_pad_mode(self) -> Optional[str]:
        return 'boundary'

    @property
    def native_grid_sample_mode(self) -> Union[str, None]:
        return 'boundary'

    def spatial_gradient(self):
        return ZERO

    @property
    def is_flexible(self) -> bool:
        return True

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        if upper_edge:
            edge = value[{dim: slice(-1, None)}]
        else:
            edge = value[{dim: slice(1)}]
        return concat([edge] * width, value.shape[dim])

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        """
        *Warning*:
        This implementation discards corners, i.e. values that lie outside the original tensor in more than one dimension.
        These are typically sliced off in differential operators. Corners are instead assigned the value 0.
        To take corners into account, call pad() for each axis individually. This is inefficient with ShiftLinTracer.

        Args:
          value: ShiftLinTracer:
          widths: dict: 

        Returns:

        """
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        result = value.shift(lower, new_shape=after_pad(value.shape, widths), val_fun=lambda v: ZERO.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))  # inner values  ~half the computation time
        for bound_dim, (bound_lo, bound_hi) in widths.items():
            for i in range(bound_lo):  # i=0 means outer
                # this sets corners to 0
                lower = {dim: -i if dim == bound_dim else -lo for dim, (lo, _) in widths.items()}
                mask = self._lower_mask(value.shape.only(tuple(lower)), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))
                result += boundary
            for i in range(bound_hi):
                lower = {dim: i - lo - hi if dim == bound_dim else -lo for dim, (lo, hi) in widths.items()}
                mask = self._upper_mask(value.shape.only(tuple(lower)), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))  # ~ half the computation time
                result += boundary  # this does basically nothing if value is the identity
        return result

    def _lower_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        # key = (shape, tuple(widths.keys()), tuple(widths.values()), bound_dim, bound_lo, bound_hi, i)
        # if key in _BoundaryExtrapolation._CACHED_LOWER_MASKS:
        #     result = math.tensor(_BoundaryExtrapolation._CACHED_LOWER_MASKS[key])
        #     _BoundaryExtrapolation._CACHED_LOWER_MASKS[key] = result
        #     return result
        # else:
            mask = ZERO.pad(math.zeros(shape), {bound_dim: (bound_lo - i - 1, 0)})
            mask = ONE.pad(mask, {bound_dim: (1, 0)})
            mask = ZERO.pad(mask, {dim: (i, bound_hi) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
            # _BoundaryExtrapolation._CACHED_LOWER_MASKS[key] = mask
            return mask

    def _upper_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        # key = (shape, tuple(widths.keys()), tuple(widths.values()), bound_dim, bound_lo, bound_hi, i)
        # if key in _BoundaryExtrapolation._CACHED_UPPER_MASKS:
        #     result = math.tensor(_BoundaryExtrapolation._CACHED_UPPER_MASKS[key])
        #     _BoundaryExtrapolation._CACHED_UPPER_MASKS[key] = result
        #     return result
        # else:
            mask = ZERO.pad(math.zeros(shape), {bound_dim: (0, bound_hi - i - 1)})
            mask = ONE.pad(mask, {bound_dim: (0, 1)})
            mask = ZERO.pad(mask, {dim: (bound_lo, i) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
            # _BoundaryExtrapolation._CACHED_UPPER_MASKS[key] = mask
            return mask

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        from ._sparse import stored_indices, is_sparse
        from ._ops import arange, nonzero, scatter
        dual_dim = dual(connectivity).name
        # --- Gather the edge values ---
        if is_sparse(connectivity):
            indices = stored_indices(connectivity, invalid='discard')
        else:
            indices = nonzero(connectivity)
        primal_dim = [n for n in channel(indices).item_names[0] if not n.startswith('~')][0]
        assert primal_dim not in value.shape, f"sparse_pad_values only implemented for vectors, not matrices"
        gathered = value[{dual_dim: indices[primal_dim]}]
        # --- Scatter, but knowing there is only one entry per row & col, we can simply permute ---
        inv_perm = scatter(dual(connectivity), indices[[dual_dim]], arange(instance(indices)), default=0, outside_handling='undefined')
        return gathered[{instance(gathered).name: inv_perm}]


class _PeriodicExtrapolation(_CopyExtrapolation):
    """ Periodic extrapolation in n dimensions. """

    def __repr__(self):
        return 'periodic'

    def spatial_gradient(self):
        return self

    def determines_boundary_values(self, boundary_key: str) -> bool:
        if boundary_key.endswith('+') or boundary_key.endswith('-'):
            is_upper = boundary_key.endswith('+')
            return is_upper
        else:
            raise AssertionError(f"Periodic extrapolation only supported for grids but got boundary key '{boundary_key}'")

    @property
    def is_flexible(self) -> bool:
        return False

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        return coordinates % wrap(shape.spatial, channel(coordinates))[coordinates.vector.item_name_list]

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        if upper_edge:
            return value[{dim: slice(width)}]
        else:
            return value[{dim: slice(-width, None)}]

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        value_sizes = [value.shape.get_size(n) for n in widths]
        source_sizes = [value._source.shape.get_size(n) for n in widths]
        if value_sizes != source_sizes:
            raise NotImplementedError("Periodicity does not match input: %s but input has %s. This can happen when padding an already padded or sliced tensor." % (value.shape.only(tuple(widths)), value._source.shape.only(tuple(widths))))
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        return value.shift(lower, new_shape=after_pad(value.shape, widths), val_fun=lambda v: self.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))

    def shortest_distance(self, start: Tensor, end: Tensor, domain_size: Tensor):
        dx = end - start
        return (dx + domain_size / 2) % domain_size - domain_size / 2

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise AssertionError("Periodic extrapolation cannot be used with sparse connectivity. Instead, add periodicity to the connectivity matrix.")


class _SymmetricExtrapolation(_CopyExtrapolation):
    """Mirror with the boundary value occurring twice."""

    def __repr__(self):
        return 'symmetric'

    def spatial_gradient(self):
        return -self

    @property
    def is_flexible(self) -> bool:
        return True

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        coordinates = coordinates % (2 * shape)
        return ((2 * shape - 1) - abs((2 * shape - 1) - 2 * coordinates)) // 2

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        if upper_edge:
            return value[{dim: slice(-1, -width-1, -1)}]
        else:
            return value[{dim: slice(width-1, None, -1)}]

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        """
        *Warning*:
        This implementation discards corners, i.e. values that lie outside the original tensor in more than one dimension.
        These are typically sliced off in differential operators. Corners are instead assigned the value 0.
        To take corners into account, call pad() for each axis individually. This is inefficient with ShiftLinTracer.

        Args:
          value: ShiftLinTracer:
          widths: dict:

        Returns:

        """
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        result = value.shift(lower, new_shape=after_pad(value.shape, widths), val_fun=lambda v: ZERO.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))  # inner values  ~half the computation time
        for bound_dim, (bound_lo, bound_hi) in widths.items():
            for i in range(bound_lo):  # i=0 means outer
                # this sets corners to 0
                lower = {dim: bound_lo-1-2*i if dim == bound_dim else -lo for dim, (lo, _) in widths.items()}
                mask = self._lower_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))
                result += boundary
            for i in range(bound_hi):
                lower = {dim: -(bound_hi-1-2*i) - bound_lo - bound_hi if dim == bound_dim else -lo for dim, (lo, hi) in widths.items()}
                mask = self._upper_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))  # ~ half the computation time
                result += boundary  # this does basically nothing if value is the identity
        return result

    def _lower_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        mask = ZERO.pad(math.zeros(shape), {bound_dim: (bound_lo - i - 1, 0)})
        mask = ONE.pad(mask, {bound_dim: (1, 0)})
        mask = ZERO.pad(mask, {dim: (i, bound_hi) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
        return mask

    def _upper_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        mask = ZERO.pad(math.zeros(shape), {bound_dim: (0, bound_hi - i - 1)})
        mask = ONE.pad(mask, {bound_dim: (0, 1)})
        mask = ZERO.pad(mask, {dim: (bound_lo, i) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
        return mask

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError


class _AntiSymmetricExtrapolation(_SymmetricExtrapolation):
    """Like _SymmetricExtrapolation but symmetric counterparts are negated for padding"""

    def __repr__(self):
        return 'antisymmetric'

    def pad_values(self, *args, **kwargs) -> Tensor:
        return -super().pad_values(*args, **kwargs)

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        """
        *Warning*:
        This implementation discards corners, i.e. values that lie outside the original tensor in more than one dimension.
        These are typically sliced off in differential operators. Corners are instead assigned the value 0.
        To take corners into account, call pad() for each axis individually. This is inefficient with ShiftLinTracer.

        Args:
          value: ShiftLinTracer:
          widths: dict:

        Returns:

        """
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        result = value.shift(lower, new_shape=after_pad(value.shape, widths), val_fun=lambda v: ZERO.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))  # inner values  ~half the computation time
        for bound_dim, (bound_lo, bound_hi) in widths.items():
            for i in range(bound_lo):  # i=0 means outer
                # this sets corners to 0
                lower = {dim: bound_lo-1-2*i if dim == bound_dim else -lo for dim, (lo, _) in widths.items()}
                mask = self._lower_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))
                result -= boundary
            for i in range(bound_hi):
                lower = {dim: -(bound_hi-1-2*i) - bound_lo - bound_hi if dim == bound_dim else -lo for dim, (lo, hi) in widths.items()}
                mask = self._upper_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))  # ~ half the computation time
                result -= boundary  # this does basically nothing if value is the identity
        return result

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError


class _ReflectExtrapolation(_CopyExtrapolation):
    """Mirror of inner elements. The boundary value is not duplicated."""

    def __repr__(self):
        return 'reflect'

    def spatial_gradient(self):
        return -self

    @property
    def is_flexible(self) -> bool:
        return True

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        if upper_edge:
            return value[{dim: slice(-2, -2-width, -1)}]
        else:
            return value[{dim: slice(width, 0, -1)}]

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        coordinates = coordinates % (2 * shape - 2)
        return (shape - 1) - math.abs_((shape - 1) - coordinates)

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        """
        *Warning*:
        This implementation discards corners, i.e. values that lie outside the original tensor in more than one dimension.
        These are typically sliced off in differential operators. Corners are instead assigned the value 0.
        To take corners into account, call pad() for each axis individually. This is inefficient with ShiftLinTracer.

        Args:
          value: ShiftLinTracer:
          widths: dict:

        Returns:

        """
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        result = value.shift(lower, new_shape=after_pad(value.shape, widths), val_fun=lambda v: ZERO.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))  # inner values  ~half the computation time
        for bound_dim, (bound_lo, bound_hi) in widths.items():
            for i in range(bound_lo):  # i=0 means outer
                # this sets corners to 0
                lower = {dim: bound_lo-2*i if dim == bound_dim else -lo for dim, (lo, _) in widths.items()}
                mask = self._lower_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))
                result += boundary
            for i in range(bound_hi):
                lower = {dim: -(bound_hi-2*i) - bound_lo - bound_hi if dim == bound_dim else -lo for dim, (lo, hi) in widths.items()}
                mask = self._upper_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))  # ~ half the computation time
                result += boundary  # this does basically nothing if value is the identity
        return result

    def _lower_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        mask = ZERO.pad(math.zeros(shape), {bound_dim: (bound_lo - i - 1, 0)})
        mask = ONE.pad(mask, {bound_dim: (1, 0)})
        mask = ZERO.pad(mask, {dim: (i, bound_hi) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
        return mask

    def _upper_mask(self, shape, widths, bound_dim, bound_lo, bound_hi, i):
        mask = ZERO.pad(math.zeros(shape), {bound_dim: (0, bound_hi - i - 1)})
        mask = ONE.pad(mask, {bound_dim: (0, 1)})
        mask = ZERO.pad(mask, {dim: (bound_lo, i) if dim == bound_dim else (lo, hi) for dim, (lo, hi) in widths.items()})
        return mask

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError


class _AntiReflectExtrapolation(_ReflectExtrapolation):
    """Like _ReflectExtrapolation but symmetric counterparts are negated for padding"""

    def __repr__(self):
        return 'antireflect'

    def pad_values(self, *args, **kwargs) -> Tensor:
        return -super().pad_values(*args, **kwargs)

    def _pad_linear_tracer(self, value: 'ShiftLinTracer', widths: dict) -> 'ShiftLinTracer':
        """
        *Warning*:
        This implementation discards corners, i.e. values that lie outside the original tensor in more than one dimension.
        These are typically sliced off in differential operators. Corners are instead assigned the value 0.
        To take corners into account, call pad() for each axis individually. This is inefficient with ShiftLinTracer.

        Args:
          value: ShiftLinTracer:
          widths: dict:

        Returns:

        """
        lower = {dim: -lo for dim, (lo, _) in widths.items()}
        result = value.shift(lower, new_shape=after_pad(value.shape, widths), val_fun=lambda v: ZERO.pad(v, widths), bias_fun=lambda b: ZERO.pad(b, widths))  # inner values  ~half the computation time
        for bound_dim, (bound_lo, bound_hi) in widths.items():
            for i in range(bound_lo):  # i=0 means outer
                # this sets corners to 0
                lower = {dim: bound_lo-2*i if dim == bound_dim else -lo for dim, (lo, _) in widths.items()}
                mask = self._lower_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))
                result -= boundary
            for i in range(bound_hi):
                lower = {dim: -(bound_hi-2*i) - bound_lo - bound_hi if dim == bound_dim else -lo for dim, (lo, hi) in widths.items()}
                mask = self._upper_mask(value.shape.only(result.dependent_dims), widths, bound_dim, bound_lo, bound_hi, i)
                boundary = value.shift(lower, new_shape=result.shape, val_fun=lambda v: self.pad(v, widths) * mask, bias_fun=lambda b: ZERO.pad(b, widths))  # ~ half the computation time
                result -= boundary  # this does basically nothing if value is the identity
        return result

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError


class _SymmetricGradientExtrapolation(Extrapolation):

    def __init__(self):
        super().__init__(pad_rank=3)
        warnings.warn('symmetric-gradient extrapolation is experimental. Use with caution.', DeprecationWarning)

    @property
    def shape(self):
        return EMPTY_SHAPE

    def to_dict(self) -> dict:
        return {'type': 'symmetric-gradient'}

    def spatial_gradient(self) -> 'Extrapolation':
        raise NotImplementedError

    def valid_outer_faces(self, dim) -> Tuple[bool, bool]:
        raise NotImplementedError  # probably return True, True but this hasn't been used on grids yet

    @property
    def is_flexible(self) -> bool:
        return True

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        anti_s = ANTIREFLECT.pad_values(value, width, dim, upper_edge, already_padded=already_padded, **kwargs)
        edge = value[{dim: -1}] if upper_edge else value[{dim: 0}]
        return anti_s + 2 * edge

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, _SymmetricGradientExtrapolation)

    def __hash__(self):
        return 209385671

    def __abs__(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError


class _NoExtrapolation(Extrapolation):  # singleton

    @property
    def shape(self):
        return EMPTY_SHAPE

    def to_dict(self) -> dict:
        return {'type': 'none'}

    def pad(self, value: Tensor, widths: dict, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        return value

    def spatial_gradient(self) -> 'Extrapolation':
        return self

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        return False

    def __value_attrs__(self):
        return ()

    @property
    def is_flexible(self) -> bool:
        raise AssertionError(f"is_flexible not defined by {self.__class__}")

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        return math.zeros(value.shape._replace_single_size(dim, 0))

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, _NoExtrapolation)

    def __hash__(self):
        return hash(self.__class__)

    def __repr__(self):
        return "none"

    def __abs__(self):
        return self

    def __neg__(self):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self


class Undefined(Extrapolation):
    """
    The extrapolation is unknown and must be replaced before usage.
    Any access to outside values will raise an AssertionError.
    """

    def __init__(self, derived_from: Extrapolation):
        super().__init__(-1)
        self.derived_from = derived_from

    @property
    def shape(self):
        return EMPTY_SHAPE

    def to_dict(self) -> dict:
        return {'type': 'undefined', 'derived_from': self.derived_from.to_dict()}

    def pad(self, value: Tensor, widths: dict, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        for (lo, up) in widths.items():
            assert lo == 0 and up == 0, "Undefined extrapolation"
        return value

    def spatial_gradient(self) -> 'Extrapolation':
        return self

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        return self.derived_from.determines_boundary_values(boundary_key)

    @property
    def is_flexible(self) -> bool:
        raise AssertionError("Undefined extrapolation")

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        raise AssertionError("Undefined extrapolation")

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise AssertionError("Undefined extrapolation")

    def __eq__(self, other):
        return isinstance(other, Undefined) and other.derived_from == self.derived_from

    def __hash__(self):
        return hash(self.__class__) + hash(self.derived_from)

    def __repr__(self):
        return "undefined"

    def __abs__(self):
        return self

    def __neg__(self):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self


ZERO = ConstantExtrapolation(0)
""" Extrapolates with the constant value 0 (Dirichlet boundary condition). """
ONE = ConstantExtrapolation(1)
""" Extrapolates with the constant value 1 (Dirichlet boundary condition). """
PERIODIC = _PeriodicExtrapolation(1)
""" Extends a grid by tiling it (Periodic boundary condition). """
ZERO_GRADIENT = _ZeroGradient(2)
""" Extends a grid with its edge values (Neumann boundary condition). The value of a point lying outside the grid is determined by the closest grid value(s). """
BOUNDARY = ZERO_GRADIENT
# undocumented, use ZERO_GRADIENT instead
SYMMETRIC = _SymmetricExtrapolation(3)
""" Extends a grid by tiling it. Every other copy of the grid is flipped. Edge values occur twice per seam. """
ANTISYMMETRIC = _AntiSymmetricExtrapolation(3)
""" Like SYMMETRIC but extends a grid with the negative value of the corresponding counterpart instead. """
REFLECT = _ReflectExtrapolation(4)
""" Like SYMMETRIC but the edge values are not copied and only occur once per seam. """
ANTIREFLECT = _AntiReflectExtrapolation(4)
""" Like REFLECT but extends a grid with the negative value of the corresponding counterpart instead. """
SYMMETRIC_GRADIENT = _SymmetricGradientExtrapolation()
""" Extrapolates in a continuous manner. The normal component of the spatial gradient is symmetric at the boundaries. The outer-most valid difference is duplicated. """

NONE = _NoExtrapolation(-1)
""" Raises AssertionError when used to determine outside values. Padding operations will have no effect with this extrapolation. """


_PRIMITIVES = {  # used by as_boundary() and from_dict()
    'periodic': PERIODIC,
    'zero': ZERO,
    'one': ONE,
    'zero-gradient': ZERO_GRADIENT,
    '∇=0': ZERO_GRADIENT,
    'boundary': ZERO_GRADIENT,  # deprecated
    'symmetric': SYMMETRIC,
    'symmetric-gradient': SYMMETRIC_GRADIENT,
    'antisymmetric': ANTISYMMETRIC,
    'reflect': REFLECT,
    'antireflect': ANTISYMMETRIC,
}


def as_extrapolation(obj) -> Extrapolation:
    """
    Creates an `Extrapolation` from a descriptor object.

    Args:
        obj: Extrapolation specification, one of the following:

            * `Extrapolation`
            * Primitive name as `str`: periodic, zero, one, zero-gradient, symmetric, symmetric-gradient, antisymmetric, reflect, antireflect
            * `dict` containing exactly the keys `'normal'` and `'tangential'`
            * `dict` mapping spatial dimension names to extrapolations

    Returns:
        `Extrapolation`
    """
    if isinstance(obj, Extrapolation):
        return obj
    if obj is None:
        return NONE
    if isinstance(obj, str):
        assert obj in _PRIMITIVES, f"Unrecognized extrapolation type: '{obj}'"
        return _PRIMITIVES[obj]
    if isinstance(obj, dict):
        if 'normal' in obj or 'tangential' in obj:
            assert 'normal' in obj and 'tangential' in obj, f"Normal/tangential dict requires both entries 'normal' and 'tangential' but got {obj}"
            assert len(obj) == 2, f"Normal/tangential dict must only contain entries 'normal' and 'tangential' but got {obj}"
            normal = as_extrapolation(obj['normal'])
            tangential = as_extrapolation(obj['tangential'])
            return combine_by_direction(normal=normal, tangential=tangential)
        else:
            return combine_sides(obj)
    return ConstantExtrapolation(obj)


def combine_sides(boundary_dict: Dict[str, Extrapolation] = None, **extrapolations: Union[Extrapolation, tuple, Number]) -> Extrapolation:
    """
    Specify extrapolations for each side / face of a box.

    Args:
        boundary_dict: Extrapolations by boundary names.
        **extrapolations: map from dim: str -> `Extrapolation` or `tuple` (lower, upper)

    Returns:
        `Extrapolation`
    """
    boundary_dict = dict(boundary_dict) if boundary_dict is not None else {}
    boundary_dict.update(extrapolations)
    result = {}
    for dim, ext in boundary_dict.items():
        if isinstance(ext, tuple):
            assert len(ext) == 2, "Tuple must contain exactly two elements, (lower, upper)"
            result[dim+'-'] = as_extrapolation(ext[0])
            result[dim+'+'] = as_extrapolation(ext[1])
        else:
            result[dim] = as_extrapolation(ext)
    if len(set(result.values())) == 1:  # All equal -> return any
        return next(iter(result.values()))
    else:
        return _MixedExtrapolation(result)


class _MixedExtrapolation(Extrapolation):
    """
    A mixed extrapolation uses different extrapolations for different sides.
    Each side is identified by its name ´str`.
    """

    def __init__(self, ext_by_boundary: Dict[str, Extrapolation]):
        """
        Args:
            ext_by_boundary: key: str -> Extrapolation
        """
        super().__init__(pad_rank=None)
        assert all(isinstance(e, Extrapolation) for e in ext_by_boundary.values())
        assert all(not isinstance(e, _MixedExtrapolation) for e in ext_by_boundary.values()), f"Nested mixed extrapolations not supported"
        assert all(isinstance(k, str) for k in ext_by_boundary.keys())
        assert len(set(ext_by_boundary.values())) >= 2, f"Extrapolation can be simplified: {ext_by_boundary}"
        self.ext = ext_by_boundary

    @property
    def shape(self):
        base_shape = merge_shapes(*self.ext.values())
        if 'vector' not in base_shape and any(isinstance(e, _NormalTangentialExtrapolation) for e in self.ext.values()):  # we need to add the vector dimension via _NormalTangentialExtrapolation._get_shape()
            return merge_shapes(*[e._get_shape(self._dims) if isinstance(e, _NormalTangentialExtrapolation) else e.shape for e in self.ext.values()])
        return base_shape

    def to_dict(self) -> dict:
        return {
            'type': 'mixed_v2',
            'dims': {k: ext.to_dict() for k, ext in self.ext.items()}
        }

    def __value_attrs__(self):
        return 'ext',

    def __eq__(self, other):
        if not isinstance(other, _MixedExtrapolation):
            return False
        return self.ext == other.ext

    def __hash__(self):
        return hash(frozenset(self.ext.items()))

    def __repr__(self):
        return repr(self.ext)

    def spatial_gradient(self) -> Extrapolation:
        return combine_sides({k: ext.spatial_gradient() for k, ext in self.ext.items()})

    def determines_boundary_values(self, boundary_key: str) -> bool:
        return self._at_boundary(boundary_key).determines_boundary_values(boundary_key)

    def is_copy_pad(self, dim: str, upper_edge: bool):
        return self._at_boundary(dim+('+' if upper_edge else '-')).is_copy_pad(dim, upper_edge)

    @property
    def is_flexible(self) -> bool:
        return any([ext.is_flexible for ext in self.ext.values()])

    def pad(self, value: Tensor, widths: dict, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        """Pads a tensor using multiple extrapolations."""
        extrapolations = set(self.ext.values())
        extrapolations = tuple(sorted(extrapolations, key=lambda e: e.pad_rank))
        already_padded = {} if already_padded is None else dict(already_padded)
        for ext in extrapolations:
            ext_widths = {dim: (l if self._at_boundary(dim+'-') == ext else 0, u if self._at_boundary(dim+'+') == ext else 0) for dim, (l, u) in widths.items()}
            value = ext.pad(value, ext_widths, already_padded=already_padded, **kwargs)
            already_padded.update(ext_widths)
        return value

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        extrap: Extrapolation = self._at_boundary(dim + ('+' if upper_edge else '-'))
        return extrap.pad_values(value, width, dim, upper_edge, already_padded=already_padded, **kwargs)

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        extrap: Extrapolation = self._at_boundary(dim)
        return extrap.sparse_pad_values(value, connectivity, dim, **kwargs)

    def transform_coordinates(self, coordinates: Tensor, shape: Shape, **kwargs) -> Tensor:
        result = []
        for dim in shape.spatial:
            dim_coords = coordinates[[dim.name]]
            le = self._at_boundary(dim.name+'-')
            ue = self._at_boundary(dim.name+'+')
            if le == ue:
                result.append(le.transform_coordinates(dim_coords, dim, **kwargs))
            else:  # separate boundary for lower and upper face
                lower = le.transform_coordinates(dim_coords, dim, **kwargs)
                upper = ue.transform_coordinates(dim_coords, dim, **kwargs)
                result.append(math.where(dim_coords <= 0, lower, upper))
        if 'vector' in result[0].shape:
            return concat(result, channel('vector'))
        else:
            return stack(result, channel('vector'))

    def _at_boundary(self, boundary_key: str):
        if boundary_key not in self.ext and boundary_key[-1] in '+-' and boundary_key[:-1] in self.ext:
            boundary_key = boundary_key[:-1]
        return self.ext[boundary_key]

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        return combine_sides({b: ext._getitem_with_domain(item, b[:-1], b.endswith('+'), self._dims) for b, ext in self.ext.items()})

    def _getitem_with_domain(self, item: dict, dim: str, upper_edge: bool, all_dims: Sequence[str]) -> 'Extrapolation':
        for b, ext in self.ext.items():
            if b in [dim, dim+('+' if upper_edge else '-')]:
                return ext._getitem_with_domain(item, b[:-1], b.endswith('+'), all_dims)
        raise KeyError((dim, upper_edge))

    @property
    def _dims(self):
        return tuple(set(b[:-1] if b[-1] in '+-' else b for b in self.ext))

    def __add__(self, other):
        return self._op2(other, lambda e1, e2: e1 + e2)

    def __radd__(self, other):
        return self._op2(other, lambda e1, e2: e2 + e1)

    def __sub__(self, other):
        return self._op2(other, lambda e1, e2: e1 - e2)

    def __rsub__(self, other):
        return self._op2(other, lambda e1, e2: e2 - e1)

    def __mul__(self, other):
        return self._op2(other, lambda e1, e2: e1 * e2)

    def __rmul__(self, other):
        return self._op2(other, lambda e1, e2: e2 * e1)

    def __truediv__(self, other):
        return self._op2(other, lambda e1, e2: e1 / e2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda e1, e2: e2 / e1)

    def _op2(self, other, operator):
        if isinstance(other, _MixedExtrapolation):
            assert self.ext.keys() == other.ext.keys()
            return combine_sides({k: operator(ext, other.ext[k]) for k, ext in self.ext.items()})
        else:
            return combine_sides({k: operator(ext, other) for k, ext in self.ext.items()})

    def __abs__(self):
        return combine_sides({k: abs(ext) for k, ext in self.ext.items()})

    def __neg__(self):
        return combine_sides({k: -ext for k, ext in self.ext.items()})


class _NormalTangentialExtrapolation(Extrapolation):

    def __init__(self, normal: Extrapolation, tangential: Extrapolation):
        super().__init__(pad_rank=min(normal.pad_rank, tangential.pad_rank))
        self.normal = normal
        self.tangential = tangential

    @property
    def shape(self):
        return merge_shapes(self.normal, self.tangential)

    def _get_shape(self, domain_dims: Sequence[str]):
        return self.shape & channel(vector=domain_dims)

    def __getitem__(self, item):
        if 'vector' in item:
            raise AssertionError(f"Cannot slice normal/tangential extrapolation along vector without providing domain dimensions. Use domain_slice() instead.")
        return self

    def to_dict(self) -> dict:
        return {
            'type': 'normal-tangential',
            'normal': self.normal.to_dict(),
            'tangential': self.tangential.to_dict(),
        }

    def __value_attrs__(self):
        return 'normal', 'tangential'

    def __repr__(self):
        return f"normal={self.normal}, tangential={self.tangential}"

    def spatial_gradient(self) -> 'Extrapolation':
        return combine_by_direction(self.normal.spatial_gradient(), self.tangential.spatial_gradient())

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        return self.normal.determines_boundary_values(boundary_key)

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError

    def is_copy_pad(self, dim: str, upper_edge: bool):
        return False  # normal and tangential might copy from different places, so no.

    @property
    def is_flexible(self) -> bool:
        return self.normal.is_flexible

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        if 'vector' not in value.shape:
            warnings.warn(f'{self} adding a vector dimension to tensor {value.shape}')
            from . import expand
            value = expand(value, channel(vector=spatial(value).names))
        assert value.vector.item_names is not None, "item_names must be present when padding with normal-tangential"
        result = []
        for component_name, component in zip(value.vector.item_names, value.vector):
            ext = self.normal if component_name == dim else self.tangential
            result.append(ext.pad_values(component, width, dim, upper_edge, already_padded=already_padded, **kwargs))
        from ._magic_ops import stack
        result = stack(result, value.shape.only('vector'))
        return result

    def _getitem_with_domain(self, item: dict, dim: str, upper_edge: bool, all_dims: Sequence[str]):
        if 'vector' not in item:
            return self
        component = item['vector']
        if isinstance(component, int):
            component = all_dims[component]
        if component == dim:
            return self.normal
        else:
            return self.tangential

    def __eq__(self, other):
        return isinstance(other, _NormalTangentialExtrapolation) and self.normal == other.normal and self.tangential == other.tangential

    def __hash__(self):
        return hash(self.normal) + hash(self.tangential)

    def __add__(self, other):
        return self._op2(other, lambda e1, e2: e1 + e2)

    def __radd__(self, other):
        return self._op2(other, lambda e1, e2: e2 + e1)

    def __sub__(self, other):
        return self._op2(other, lambda e1, e2: e1 - e2)

    def __rsub__(self, other):
        return self._op2(other, lambda e1, e2: e2 - e1)

    def __mul__(self, other):
        return self._op2(other, lambda e1, e2: e1 * e2)

    def __rmul__(self, other):
        return self._op2(other, lambda e1, e2: e2 * e1)

    def __truediv__(self, other):
        return self._op2(other, lambda e1, e2: e1 / e2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda e1, e2: e2 / e1)

    def _op2(self, other, operator):
        if isinstance(other, _NormalTangentialExtrapolation):
            return combine_by_direction(normal=operator(self.normal, other.normal), tangential=operator(self.tangential, other.tangential))
        else:
            return combine_by_direction(normal=operator(self.normal, other), tangential=operator(self.tangential, other))

    def __abs__(self):
        return combine_by_direction(normal=abs(self.normal), tangential=abs(self.tangential))

    def __neg__(self):
        return combine_by_direction(normal=-self.normal, tangential=-self.tangential)


def combine_by_direction(normal: Union[Extrapolation, float, Tensor], tangential: Union[Extrapolation, float, Tensor]) -> Extrapolation:
    """
    Use a different extrapolation for the normal component of vector-valued tensors.

    Args:
        normal: Extrapolation for the component that is orthogonal to the boundary.
        tangential: Extrapolation for the component that is tangential to the boundary.

    Returns:
        `Extrapolation`
    """
    normal = as_extrapolation(normal)
    tangential = as_extrapolation(tangential)
    return normal if normal == tangential else _NormalTangentialExtrapolation(normal, tangential)


def get_normal(ext: Extrapolation) -> Extrapolation:
    """Returns only the extrapolation for the surface normal component."""
    if isinstance(ext, _NormalTangentialExtrapolation):
        return ext.normal
    elif isinstance(ext, _MixedExtrapolation):
        return combine_sides({k: get_normal(ext) for k, ext in ext.ext.items()})
    else:
        return ext


def get_tangential(ext: Extrapolation) -> Extrapolation:
    """Returns only the extrapolation for components tangential to the boundary surface."""
    if isinstance(ext, _NormalTangentialExtrapolation):
        return ext.tangential
    elif isinstance(ext, _MixedExtrapolation):
        return combine_sides({k: get_tangential(ext) for k, ext in ext.ext.items()})
    else:
        return ext


class _ConditionalExtrapolation(Extrapolation):

    def __init__(self, mask: Tensor, true_ext: Extrapolation, false_ext: Extrapolation):
        super().__init__(false_ext.pad_rank)
        self.mask = mask
        self.false_ext = false_ext
        self.true_ext = true_ext

    def to_dict(self) -> dict:
        return {
            'type': 'conditional',
            'mask': tensor_to_dict(self.mask),
            'true': self.true_ext.to_dict(),
            'false': self.false_ext.to_dict(),
        }

    @property
    def shape(self):
        return self.mask.shape & self.true_ext.shape & self.false_ext.shape

    def spatial_gradient(self) -> 'Extrapolation':
        return where(self.mask, self.true_ext.spatial_gradient(), self.false_ext.spatial_gradient())

    def determines_boundary_values(self, boundary_key: Union[Tuple[str, bool], str]) -> bool:
        x = self.true_ext.determines_boundary_values(boundary_key)
        y = self.false_ext.determines_boundary_values(boundary_key)
        assert x == y, f"determines_boundary_values not defined for incompatible extrapolations true/false: {self.true_ext} / {self.false_ext}"
        return x

    def sparse_pad_values(self, value: Tensor, connectivity: Tensor, dim: str, **kwargs) -> Tensor:
        raise NotImplementedError

    @property
    def is_flexible(self) -> bool:
        return self.false_ext.is_flexible or self.true_ext.is_flexible

    def pad_values(self, value: Tensor, width: int, dim: str, upper_edge: bool, already_padded: dict = None, **kwargs) -> Tensor:
        mask = self.mask
        if already_padded:
            mask = ZERO_GRADIENT.pad(mask, already_padded)
        p_false = self.false_ext.pad_values(value, width, dim, upper_edge, **kwargs)
        p_true = self.true_ext.pad_values(value, width, dim, upper_edge, **kwargs)
        return math.where(mask, p_true, p_false)

    def pad(self, value: Tensor, widths: dict, already_padded: Optional[dict] = None, **kwargs) -> Tensor:
        from phiml.math._trace import ShiftLinTracer
        if isinstance(value, ShiftLinTracer):
            mask = self.mask
            if already_padded:
                mask = ZERO_GRADIENT.pad(mask, already_padded)
            p_false = self.false_ext.pad(value, widths, **kwargs)
            p_true = self.true_ext.pad(value, widths, **kwargs)
            return p_true * mask + p_false * (1 - mask)
        return Extrapolation.pad(self, value, widths, already_padded=already_padded, **kwargs)

    def __repr__(self):
        return f"{self.true_ext} or {self.false_ext}"

    def __eq__(self, other):
        return isinstance(other, _ConditionalExtrapolation) and math.always_close(self.mask, other.mask) and self.true_ext == other.true_ext and self.false_ext == other.false_ext

    def __hash__(self):
        return hash(self.true_ext) + hash(self.false_ext)

    def __abs__(self):
        return where(self.mask, abs(self.true_ext), abs(self.false_ext))

    def __neg__(self):
        return where(self.mask, -self.true_ext, -self.false_ext)

    def _op2(self, other, op: Callable):
        if isinstance(other, _ConditionalExtrapolation) and math.close(other.mask, self.mask):
            return where(self.mask, op(self.true_ext, other.true_ext), op(self.false_ext, other.false_ext))
        return where(self.mask, op(self.true_ext, other), op(self.false_ext, other))

    def __add__(self, other):
        return self._op2(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self._op2(other, lambda y, x: x + y)

    def __sub__(self, other):
        return self._op2(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._op2(other, lambda y, x: x - y)

    def __mul__(self, other):
        return self._op2(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self._op2(other, lambda y, x: x * y)

    def __truediv__(self, other):
        return self._op2(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self._op2(other, lambda y, x: x / y)


def where(mask: Tensor, true_ext, false_ext) -> Extrapolation:
    """
    Uses `true_ext` where `mask` is true and `false_ext` where mask is false.
    If the extrapolations are not consistent in which boundary faces are uniquely determined, the result cannot be used for boundary faces.

    You may also call `math.where()` instead of this function.

    Args:
        mask: `Tensor` with dimensions matching the tensor that is being padded.
        true_ext: Extrapolation to use where `mask==True`.
        false_ext: Extrapolation to use where `mask==False`.

    Returns:
        `Extrapolation`
    """
    true_ext = as_extrapolation(true_ext)
    false_ext = as_extrapolation(false_ext)
    from phiml.math import always_close
    if always_close(mask, True):
        return true_ext
    elif always_close(mask, False):
        return false_ext
    if true_ext == false_ext:
        return true_ext
    return _ConditionalExtrapolation(mask, true_ext, false_ext)


def from_dict(dictionary: dict) -> Extrapolation:
    """
    Loads an `Extrapolation` object from a dictionary that was created using `Extrapolation.to_dict()`.

    Args:
        dictionary: serializable dictionary holding all extrapolation properties

    Returns:
        Loaded extrapolation
    """
    etype = dictionary['type']
    if etype in _PRIMITIVES:
        return _PRIMITIVES[etype]
    elif etype == 'constant':
        return ConstantExtrapolation(dictionary['value'])
    elif etype == 'mixed':
        dims: Dict[str, tuple] = dictionary['dims']
        extrapolations = {k: (from_dict(l), from_dict(u)) for k, (l, u) in dims.items()}
        return combine_sides(**extrapolations)
    elif etype == 'mixed_v2':
        dims: Dict[str, dict] = dictionary['dims']
        extrapolations = {k: from_dict(ext) for k, ext in dims.items()}
        return _MixedExtrapolation(extrapolations)
    elif etype == 'normal-tangential':
        normal = from_dict(dictionary['normal'])
        tangential = from_dict(dictionary['tangential'])
        return _NormalTangentialExtrapolation(normal, tangential)
    elif etype == 'conditional':
        mask = tensor_from_dict(dictionary['mask'])
        true_ext = from_dict(dictionary['true'])
        false_ext = from_dict(dictionary['false'])
        return _ConditionalExtrapolation(mask, true_ext, false_ext)
    elif etype == 'none':
        return NONE
    elif etype == 'undefined':
        derived_from = from_dict(dictionary['derived_from'])
        return Undefined(derived_from)
    else:
        raise ValueError(dictionary)


def order_by_shape(names: Sequence[str], sequence, default=None) -> Union[tuple, list]:
    """
    If sequence is a dict with dimension names as keys, orders its values according to this shape.

    Otherwise, the sequence is returned unchanged.

    Args:
      sequence: Sequence or dict to be ordered
      default: default value used for dimensions not contained in sequence

    Returns:
      ordered sequence of values
    """
    if isinstance(sequence, dict):
        result = [sequence.get(dim, default) for dim in names]
        return result
    elif isinstance(sequence, (tuple, list)):
        assert len(sequence) == len(names)
        return sequence
    else:  # just a constant
        return sequence


def map(f: Callable[[Extrapolation], Extrapolation], extrapolation):
    """
    Applies a function to all leaf extrapolations in `extrapolation`.
    Non-leaves are those created by `combine_sides()` and `combine_by_direction()`.

    The tree will be collapsed if possible.

    Args:
        f: Function mapping a leaf `Extrapolation` to another `Extrapolation`.
        extrapolation: Input tree for `f`.

    Returns:
        `Extrapolation`
    """
    if isinstance(extrapolation, _MixedExtrapolation):
        return combine_sides({k: map(f, ext) for k, ext in extrapolation.ext.items()})
    elif isinstance(extrapolation, _NormalTangentialExtrapolation):
        return combine_by_direction(map(f, extrapolation.normal), map(f, extrapolation.tangential))
    elif isinstance(extrapolation, _ConditionalExtrapolation):
        return where(extrapolation.mask, map(f, extrapolation.true_ext), map(f, extrapolation.false_ext))
    else:
        return f(extrapolation)


def remove_constant_offset(extrapolation):
    """
    Removes all constant offsets from an extrapolation.
    This also includes `NaN` values in constants (unlike `ext - ext`).

    Args:
        extrapolation: `Extrapolation` object.

    Returns:
        `Extrapolation` that has no constant offsets
    """
    def const_to_zero(extrapolation):
        if isinstance(extrapolation, ConstantExtrapolation):
            return ZERO
        else:
            return extrapolation
    return map(const_to_zero, extrapolation)


def domain_slice(ext: Extrapolation, item: dict, domain_dims: Union[Shape, tuple, list, str]) -> Extrapolation:
    """
    Slices a domain, similar to `ext[item]` but with additional information about the domain.
    In some cases, `ext[item]` will fail, e.g. slicing a normal/tangential extrapolation along `vector`.

    Args:
        ext: `Extrapolation`
        item: slicing dict
        domain_dims: All spatial dimensions.

    Returns:
        `Extrapolation`
    """
    if isinstance(ext, (ConstantExtrapolation, _CopyExtrapolation)):
        return ext[item]
    sides = {}
    domain_dims = [d[:-1] if d[-1] in '+-' else d for d in parse_dim_order(domain_dims)]
    dim_set = set()
    domain_dims = [d for d in domain_dims if not (d in dim_set or dim_set.add(d))]
    for dim in domain_dims:
        lo = ext._getitem_with_domain(item, dim, False, domain_dims)
        up = ext._getitem_with_domain(item, dim, True, domain_dims)
        if lo == up:
            sides[dim] = lo
        else:
            sides[dim+'-'] = lo
            sides[dim+'+'] = up
    return combine_sides(sides)
