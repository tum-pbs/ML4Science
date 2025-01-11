import operator
from collections import namedtuple
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, Sequence, List, Collection

import numpy
import numpy as np
from phiml.backend import get_precision
from scipy.sparse import csr_matrix

from ..backend import choose_backend, NUMPY, Backend
from ._ops import backend_for, concat_tensor, scatter, zeros_like
from ._shape import Shape, parse_dim_order, merge_shapes, spatial, instance, batch, concat_shapes, EMPTY_SHAPE, dual, channel, non_batch, primal, non_channel, DEBUG_CHECKS, \
    after_gather, concat_shapes_
from ._magic_ops import stack, expand, rename_dims, unpack_dim, unstack, value_attributes
from ._tensors import Tensor, wrap, disassemble_tree, disassemble_tensors, assemble_tree, TensorStack, may_vary_along, \
    discard_constant_dims, variable_shape, Dense, equality_by_shape_and_value
from ._sparse import SparseCoordinateTensor, is_sparse, sparse_dims, same_sparsity_pattern, sparse_tensor, stored_indices, stored_values, add_sparse_batch_dim
from . import _ops as math
from ..backend._dtype import combine_types

TracerSource = namedtuple('TracerSource', ['shape', 'dtype', 'name', 'index'])


class Shift:
    def __init__(self, by_dim: Dict[str, int]):
        self.by_dim = {dim: shift for dim, shift in by_dim.items() if shift != 0}
        self._hash = hash(tuple(sorted(self.by_dim.items())))

    def __repr__(self):
        if not self.by_dim:
            return "self"
        return ", ".join(f"{dim}{amount:+}" for dim, amount in self.by_dim.items())

    def __eq__(self, other):
        return isinstance(other, Shift) and self.by_dim == other.by_dim

    def __hash__(self):
        return self._hash

    def __getitem__(self, item):
        if isinstance(item, Shape):
            item = item.name
        assert isinstance(item, str)
        return self.by_dim.get(item, 0)

    def __add__(self, other):
        d1 = self.by_dim
        d2 = other.by_dim if isinstance(other, Shift) else other
        all_dims = set(d1.keys()).union(d2.keys())
        result = {dim: d1.get(dim, 0) + d2.get(dim, 0) for dim in all_dims}
        return Shift(result)

    def __bool__(self):
        return bool(self.by_dim)


class ShiftLinTracer(Tensor):
    """
    Tracer object for linear and affine functions.
    The sparsity pattern is assumed equal for all grid cells and is reflected in `val` (e.g. for a 5-point stencil, `val` has 5 items).
    The Tensors stored in `val` include position-dependent dimensions, allowing for different stencils at different positions.
    Dimensions not contained in any `val` Tensor are treated as independent (batch dimensions).
    """

    def __init__(self,
                 source: TracerSource,
                 values_by_shift: Dict[Shift, Tensor],
                 shape: Shape,
                 bias: Tensor,
                 renamed: Dict[str, str],
                 nz_edge: Dict[Shift, bool]):
        """
        Args:
            source: placeholder tensor
            values_by_shift: `dict` mapping relative shifts (`frozendict`) to value Tensors.
                Shape keys only contain non-zero shift dims. Missing dims are interpreted as independent.
            shape: shape of this tensor
            bias: Constant Tensor to be added to the multiplication output, A*x + b.
                A bias naturally arises at boundary cells with non-trivial boundary conditions if no ghost cells are added to the matrix.
                When non-zero, this tracer technically represents an affine function, not a linear one.
                However, the bias can be subtracted from the solution vector when solving a linear system, allowing this function to be solved with regular linear system solvers.
        """
        super().__init__()
        if DEBUG_CHECKS:
            assert isinstance(source, TracerSource)
            assert isinstance(renamed, dict)
            assert isinstance(bias, Tensor)
            assert all(isinstance(shift, Shift) for shift in values_by_shift)
            assert all(isinstance(shift, Shift) for shift in nz_edge)
            assert not bias._is_tracer, f"bias cannot be a linear tracer"
            assert set(values_by_shift) == set(nz_edge), f"nonzero-edge labels {nz_edge} don't match values {set(values_by_shift)}"
            for shift_, v in values_by_shift.items():
                assert v.shape.only(shape, reorder=True) == shape.only(v.shape), f"Tracer with shape {shape} must have matching values but got {v.shape}"  # values must match shape
        self._source = source
        self.val = values_by_shift
        self._nz_edge = nz_edge
        self._bias = bias
        self._shape = shape
        self._renamed = renamed  # new_name -> old_name

    def __repr__(self):
        return f"{self.__class__.__name__} {self._shape}"

    @property
    def _out_name_to_original(self) -> Dict[str, str]:
        return self._renamed

    @property
    def dtype(self):
        return combine_types(self._source.dtype, *[v.dtype for v in self.val.values()])

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape: Shape):
        renamed = {new_dim: self._renamed[old_dim] for old_dim, new_dim in zip(self._shape.names, new_shape.names)}
        bias = rename_dims(self._bias, self._shape, new_shape)
        return ShiftLinTracer(self._source, self.val, new_shape, bias, renamed, self._nz_edge)

    @property
    def _is_tracer(self) -> bool:
        return True

    @property
    def backend(self) -> Backend:
        return backend_for(self._bias, *self.val.values())

    def _getitem(self, selection: dict):
        starts = {dim: (item.start or 0) if isinstance(item, slice) else item for dim, item in selection.items()}
        new_shape = after_gather(self._shape, selection)
        return self.shift(starts, new_shape, lambda v: v[selection], lambda b: b[selection], nonzero_edge=False)

    def shift(self, shifts: Dict[str, int],
              new_shape: Shape,
              val_fun: Callable,
              bias_fun: Callable = None,
              nonzero_edge=True):
        """
        Shifts all values of this tensor by `shifts`.
        Values shifted outside will be mapped with periodic boundary conditions when the matrix is built.

        Args:
            shifts: Offsets by dimension
            new_shape: Shape of the shifted tensor, must match the shape returned by `val_fun`.
            val_fun: Function to apply to the matrix values, may change the tensor shapes
            bias_fun: Function to apply to the bias vector, may change the tensor shape
            nonzero_edge: Whether any of the edge values is non-zero. Edge values are those that map out-of-bounds positions onto inside positions.

        Returns:
            Shifted tensor, possibly with altered values.
        """
        val = {}
        nz_edge = {}
        for shift, values in self.val.items():
            assert isinstance(shift, Shift)
            nze = self._nz_edge[shift]
            for dim, delta in reversed(tuple(shifts.items())):
                if dim not in values.shape:
                    values = math.expand(values, self._shape.only(dim))  # dim order may be scrambled
                if delta:
                    shift += {dim: delta}
            val[shift] = val_fun(values)
            nz_edge[shift] = nze or nonzero_edge
        bias = expand(self._bias, self.shape)
        bias = bias_fun(bias)
        bias = discard_constant_dims(bias)
        return ShiftLinTracer(self._source, val, new_shape, bias, self._renamed, nz_edge)

    def _unstack(self, dimension):
        raise NotImplementedError()

    def __neg__(self):
        return ShiftLinTracer(self._source, {shift: -values for shift, values in self.val.items()}, self._shape, -self._bias, self._renamed, self._nz_edge)

    def _op1(self, native_function):
        # __neg__ is the only proper linear op1 and is implemented above.
        if native_function.__name__ == 'isfinite':
            test_output = self.apply(math.ones(self._source.shape, dtype=self._source.dtype))
            return math.is_finite(test_output)
        else:
            raise NotImplementedError('Only linear operations are supported')

    def _op2(self, other, op: Callable, switch_args: bool) -> 'Tensor':
        if is_sparse(other):
            return NotImplemented
        if isinstance(other, SparseLinTracer):
            return to_sparse_tracer(self, other)._op2(other, op, switch_args)
        assert op in {operator.add, operator.sub, operator.mul, operator.truediv}, f"Unsupported operation encountered while tracing linear function: {op}"
        zeros_for_missing_self = op != operator.add and not (op == operator.sub and switch_args)  # perform `operator` where `self == 0`
        zeros_for_missing_other = op != operator.add and not (op == operator.sub and not switch_args)  # perform `operator` where `other == 0`
        if isinstance(other, Tensor) and other._is_tracer:
            if not isinstance(other, ShiftLinTracer):
                raise NotImplementedError
            assert self._source is other._source, "Multiple linear tracers are not yet supported."
            assert set(self._shape) == set(other._shape), f"Tracers have different shapes: {self._shape} and {other._shape}"
            values = {}
            nz_edge = {}
            for dim_shift in self.val.keys():
                if dim_shift in other.val:
                    values[dim_shift] = op(self.val[dim_shift], other.val[dim_shift])
                    nz_edge[dim_shift] = self._nz_edge[dim_shift] or other._nz_edge[dim_shift]
                else:
                    if zeros_for_missing_other:
                        values[dim_shift] = op(self.val[dim_shift], math.zeros_like(self.val[dim_shift]))
                    else:
                        values[dim_shift] = self.val[dim_shift]
                    nz_edge[dim_shift] = self._nz_edge[dim_shift]
            for dim_shift, other_values in other.val.items():
                if dim_shift not in self.val:
                    if zeros_for_missing_self:
                        values[dim_shift] = op(math.zeros_like(other_values), other_values)
                    else:
                        values[dim_shift] = other_values
                    nz_edge[dim_shift] = other._nz_edge[dim_shift]
            bias = op(self._bias, other._bias)
            return ShiftLinTracer(self._source, values, self._shape, bias, self._renamed, nz_edge)
        else:
            other = self._tensor(other)
            if op in {operator.mul, operator.truediv}:
                values = {}
                for dim_shift, val in self.val.items():
                    values[dim_shift] = op(val, other)
                bias = op(self._bias, other)
                return ShiftLinTracer(self._source, values, self._shape & other.shape, bias, self._renamed, self._nz_edge)
            elif op in {operator.add, operator.sub}:
                bias = op(self._bias, other)
                return ShiftLinTracer(self._source, self.val, self._shape & other.shape, bias, self._renamed, self._nz_edge)
            else:
                raise ValueError(f"Unsupported operation encountered while tracing linear function: {op}")

    def _natives(self) -> tuple:
        """
        This function should only be used to determine the compatible backends, this tensor should be regarded as not available.
        """
        return sum([v._natives() for v in self.val.values()], ()) + self._bias._natives()

    def _spec_dict(self) -> dict:
        raise LinearTraceInProgress(self)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        if all(isinstance(v, ShiftLinTracer) for v in values):
            # if shifts along other dims match and along `dim` are in the correct order -> stack values
            shifts = values[0].val
            stacked = [[v] for shift, v in shifts.items()]
            nz_edge = [[values[0]._nz_edge[shift]] for shift in shifts]
            plus_one = Shift({dim.name: 1})
            for i, v in enumerate(values[1:]):
                shifts = [shift + plus_one for shift in shifts]
                if len(shifts) != len(v.val) or set(shifts) != set(v.val):
                    return Tensor.__stack__(values, dim, **_kwargs)
                for i, shift in enumerate(shifts):
                    stacked[i].append(v.val[shift])
                    nz_edge[i].append(v._nz_edge[shift])
            stacked = {shift: stack(vals, dim) for shift, vals in zip(values[0].val, stacked)}
            bias = stack([v._bias for v in values], dim, expand_values=True, **_kwargs)
            nz_edge = {shift: any(nz) for shift, nz in zip(values[0].val, nz_edge)}
            return ShiftLinTracer(values[0]._source, stacked, values[0].shape & dim, bias, values[0]._renamed, nz_edge)
        return Tensor.__stack__(values, dim, **_kwargs)

    def _matmul(self, self_dims: Shape, matrix: Tensor, matrix_dims: Shape) -> Tensor:
        if is_sparse(matrix):
            return to_gather_tracer(self).matmul(self_dims, matrix, matrix_dims)
        raise NotImplementedError

    def _upgrade_gather(self):
        if len(self.val) > 1 or next(iter(self.val)):
            return to_sparse_tracer(self, None)
        else:
            return to_gather_tracer(self)

    def _gather(self, indices: Tensor) -> Tensor:
        return self._upgrade_gather()._gather(indices)

    def _scatter(self, base: Tensor, indices: Tensor) -> Tensor:
        return self._upgrade_gather()._scatter(base, indices)

    def min_rank_deficiency(self) -> Tensor:
        trimming_dict = {}
        for dim in pattern_dim_names(self):
            shifts = [shift[dim] for shift in self.val]
            lo = -min(shifts)
            hi = -max(shifts) or None
            trimming_dict[dim] = slice(lo, hi)
        trimmed_vals = [v[trimming_dict] for v in self.val.values()]
        if all(v.available for v in trimmed_vals):
            stencil_sum = sum(trimmed_vals)
            stencil_abs = sum([abs(v) for v in trimmed_vals])
            eps = {16: 1e-2, 32: 1e-5, 64: 1e-10}[get_precision()]
            balanced_stencil = math.close(0, stencil_sum, rel_tolerance=0, abs_tolerance=eps * math.mean(stencil_abs), reduce=pattern_dim_names(self))
        else:
            balanced_stencil = True  # cannot be determined here because values can vary. Assume could be rank-deficient to print warning
        deficiency = 0
        for shift, nonzero in self._nz_edge.items():
            if shift and nonzero:
                deficiency += 1
        return math.where(balanced_stencil, deficiency, 0)


class GatherLinTracer(Tensor):
    """
    Represents the operation `source[selection] * diag + bias`.
    """

    def __init__(self, source: TracerSource, diag, bias: Tensor, shape: Shape, selection: Optional[Tensor], renamed: Dict[str, str]):
        super().__init__()
        assert isinstance(diag, Tensor)
        assert isinstance(bias, Tensor)
        assert not bias._is_tracer, f"bias cannot be a linear tracer"
        assert bias.shape in shape
        assert selection is None or selection.dtype.kind == int
        assert diag.shape in shape
        assert selection is None or selection.shape.volume > 0
        self._source = source
        self._diag = diag  # full matrix or diagonal elements only
        self._bias = bias  # matches self.shape
        self._selection = selection  # Can index one or multiple dimensions of the source. Must retain source dimensions.
        self._shape = shape
        self._renamed = renamed  # dims renamed before matrix mul. new_name -> old_name
        if DEBUG_CHECKS:
            if selection is not None:
                assert selection.min >= 0, f"Negative selection indices: {selection}"
                for dim in channel(selection).item_names[0]:
                    assert selection[dim].max < self._source.shape.get_size(dim), f"Too large selection indices for source tensor {self._source.shape}: {selection}"

    def __repr__(self):
        return f"{self.__class__.__name__} {self._shape}"

    def _matmul(self, self_dims: Shape, matrix: Tensor, matrix_dims: Shape) -> Tensor:
        shape = matrix.shape.without(matrix_dims) & self._shape.without(self_dims)
        if self_dims not in self._diag.shape:  # self is constant along self_dims
            matrix = math.sum_(matrix, matrix_dims)
        diag = matrix * self._diag
        diag = rename_dims(diag, matrix_dims, rename_dims(self_dims, [*self._renamed.keys()], [*self._renamed.values()]).as_dual())
        renamed = {n: o for n, o in self._renamed.items() if n not in self_dims}
        return GatherLinTracer(self._source, diag, self._bias, shape, self._selection, renamed)

    def _gather(self, indices: Tensor):
        """
        Args:
            indices: has 1 channel and 1 non-channel/non-instance
        """
        dims = channel(indices).item_names[0]
        shape = self.shape.without(dims) & indices.shape.non_channel
        renamed = {n: o for n, o in self._renamed.items() if n not in dims}
        bias = expand(self._bias, self.shape.only(dims))[indices]
        diag = expand(self._diag, self.shape.only(dims))[indices]
        if self._selection is not None:
            indices = self._selection[indices]
        old_sel_dims = [self._renamed.get(d, d) for d in channel(indices).item_names[0]]
        indices_shape = indices.shape.with_dim_size(channel(indices), old_sel_dims)
        indices = indices._with_shape_replaced(indices_shape)
        return GatherLinTracer(self._source, diag, bias, shape, indices, renamed)

    def _scatter(self, base: Tensor, indices: Tensor) -> Tensor:
        return to_sparse_tracer(self, None)._scatter(base, indices)

    def __neg__(self):
        return GatherLinTracer(self._source, -self._diag, -self._bias, self._shape, self._selection, self._renamed)

    def _op1(self, native_function):
        # __neg__ is the only proper linear op1 and is implemented above.
        if native_function.__name__ == 'isfinite':
            finite = math.is_finite(self._source) & math.all_(math.is_finite(self._diag), self._source.shape)
            raise NotImplementedError
        else:
            raise NotImplementedError('Only linear operations are supported')

    def _op2(self, other, op: Callable, switch_args: bool) -> 'Tensor':
        assert op in {operator.add, operator.sub, operator.mul, operator.truediv}, f"Unsupported operation '{op}' encountered while tracing linear function"
        if isinstance(other, ShiftLinTracer):
            other = to_gather_tracer(other)
        if isinstance(other, GatherLinTracer):
            assert op in {operator.add, operator.sub}, f"Non-linear operation '{op}' cannot be converted to matrix"
            if not math.always_close(self._selection, other._selection):
                return to_sparse_tracer(self, other)._op2(other, op, switch_args)
            diag = op(self._diag, other._diag)
            bias = op(self._bias, other._bias)
            return GatherLinTracer(self._source, diag, bias, self._shape, self._selection, self._renamed)
        if isinstance(other, SparseLinTracer) or is_sparse(other):
            return NotImplemented
        else:
            other = self._tensor(other)
            if op in {operator.mul, operator.truediv}:
                matrix = op(self._diag, other)
                bias = op(self._bias, other)
                return GatherLinTracer(self._source, matrix, bias, self._shape & other.shape, self._selection, self._renamed)
            elif op in {operator.add, operator.sub}:
                bias = op(self._bias, other)
                return GatherLinTracer(self._source, self._matrix, bias, self._shape & other.shape, self._selection, self._renamed)
            else:
                raise ValueError(f"Unsupported operation {op} encountered while tracing linear function")

    @property
    def _is_tracer(self) -> bool:
        return True

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return combine_types(self._source.dtype, self._diag.dtype)

    @property
    def backend(self) -> Backend:
        return backend_for(self._diag, self._bias)

    def _with_shape_replaced(self, new_shape: Shape):
        renamed = dict(self._renamed)
        renamed.update({n: self._renamed.get(o, o) for n, o in zip(new_shape.names, self._shape.names)})
        return GatherLinTracer(self._source, self._diag, self._bias, new_shape, self._selection, renamed)

    @property
    def _out_name_to_original(self) -> Dict[str, str]:
        return self._renamed

    def _natives(self) -> tuple:
        """
        This function should only be used to determine the compatible backends, this tensor should be regarded as not available.
        """
        return self._diag._natives()

    def _get_selection(self, selection_dims, list_dim: Shape = instance('selection'), index_dim: Shape = channel('gather')):
        original_dims = [self._renamed.get(d, d) for d in selection_dims]
        if self._selection is not None:
            assert selection_dims == set(channel(self._selection).item_names[0])
            return rename_dims(self._selection, non_batch(self._selection).non_channel, list_dim)
        else:
            sel_src_shape = self._source.shape.only(original_dims)
            return expand(math.range_tensor(list_dim.with_size(sel_src_shape.volume)), index_dim.with_size(sel_src_shape.names))


class SparseLinTracer(Tensor):

    def __init__(self, source: TracerSource, matrix: SparseCoordinateTensor, bias: Tensor, shape: Shape):
        super().__init__()
        assert isinstance(matrix, Tensor)
        assert bias.shape in shape
        assert matrix.shape.only(shape) == shape.only(matrix.shape, reorder=True)
        if any(d.endswith('_src') for d in matrix.shape.names):
            assert any(not d.endswith('_src') for d in matrix.shape.names)  # If there is an input dim, there needs to be an output dim
        for dim in source.shape:
            if '~' + dim.name + '_src' in matrix.shape:
                assert (dim in matrix.shape) == (dim in shape), f"Inconsistent traced output dim {dim} for tracer {shape} with matrix {matrix.shape}"
        self._source = source
        self._matrix = matrix  # full matrix or diagonal elements only
        self._bias = bias  # should always match self.shape
        self._shape = shape

    def __repr__(self):
        return f"{self.__class__.__name__} {self._shape}"

    def _get_matrix(self, sparsify: Shape):
        in_dims = [d for d in self._matrix.shape if d.name.endswith('_src')]
        renamed = [d.name[:-4] for d in self._matrix.shape if d.name.endswith('_src')]
        return rename_dims(self._matrix, in_dims, renamed)

    def _matmul(self, self_dims: Shape, matrix: Tensor, matrix_dims: Shape) -> Tensor:
        reduced_dims = dependent_out_dims(self).only(self_dims)  # these dimensions
        shape = self._shape.without(self_dims) & matrix.shape.without(matrix_dims)
        if reduced_dims:
            raise NotImplementedError
        from ._ops import dot
        missing_self_dims = self_dims.without(self._matrix.shape)
        if missing_self_dims:
            new_source_dims = concat_shapes_(*[dual(**{n + '_src': v for n, v in d.untyped_dict.items()}) for d in missing_self_dims])
            batched = add_sparse_batch_dim(self._matrix, new_source_dims, missing_self_dims)  # to preserve the source dim
        else:
            batched = self._matrix
        new_sparse = dot(batched, self_dims, matrix, matrix_dims)
        # reduced_sparse = math.sum_(new_sparse, new_dims)
        bias = dot(self._bias, self_dims, matrix, matrix_dims)
        return SparseLinTracer(self._source, new_sparse, bias, shape)
        # for dim in new_dims:
        #     matrix_slices = unstack(matrix, matrix_dims)
        #     new_tracers = [self._matrix * s for s in matrix_slices]  # duplicate tracer entries for each slice
        #     new_tracer = stack(new_tracers, self_dims)
        #     new_tracer = math.sum_(new_tracer, self_dims)

    def _gather(self, indices: Tensor):
        """
        Args:
            indices: has 1 channel and 1 non-channel/non-instance
        """
        matrix = self._matrix[indices]
        bias = self._bias[indices]
        shape = self._shape.without(channel(indices).item_names[0]) & non_channel(indices)
        return SparseLinTracer(self._source, matrix, bias, shape)

    def _scatter(self, base: Tensor, indices: Tensor) -> Tensor:
        full_shape = base.shape
        add_bias = discard_constant_dims(base)
        min_shape = base.shape.only(channel(indices).item_names[0])
        row_dims = sparse_dims(self._matrix).only(self._shape)
        col_dims = sparse_dims(self._matrix).only([n for n in self._matrix.sparse_dims.names if n.endswith('_src')])
        rows = rename_dims(indices[self._matrix._indices[row_dims.name_list]], channel, 'sparse_idx')
        cols = self._matrix._indices[col_dims.name_list]
        transformed_indices = concat_tensor([rows, cols], 'sparse_idx')
        dense_shape = sparse_dims(self._matrix).without(row_dims) & min_shape
        matrix = SparseCoordinateTensor(transformed_indices, self._matrix._values, dense_shape, can_contain_double_entries=True, indices_sorted=False, indices_constant=True)
        bias = scatter(min_shape, indices, self._bias, mode='add', outside_handling='undefined') + add_bias
        return SparseLinTracer(self._source, matrix, bias, full_shape)

    def __neg__(self):
        return SparseLinTracer(self._source, -self._matrix, -self._bias, self._shape)

    def _op1(self, native_function):
        # __neg__ is the only proper linear op1 and is implemented above.
        if native_function.__name__ == 'isfinite':
            finite = math.is_finite(self._source) & math.all_(math.is_finite(self._matrix), self._source.shape)
            raise NotImplementedError
        else:
            raise NotImplementedError('Only linear operations are supported')

    def _op2(self, other, op: Callable, switch_args: bool) -> 'Tensor':
        other = self._tensor(other)
        assert op in {operator.add, operator.sub, operator.mul, operator.truediv}, f"Unsupported operation {op} encountered while tracing linear function"
        if other._is_tracer and not isinstance(other, SparseLinTracer):
            other = to_sparse_tracer(other, self)
        if isinstance(other, SparseLinTracer):
            assert op in {operator.add, operator.sub}, f"Non-linear operation '{op}' cannot be converted to matrix"
            bias = op(self._bias, other._bias)
            matrix_dims = sparse_dims(self._matrix) & sparse_dims(other._matrix)
            self_matrix = expand_matrix(self._matrix, matrix_dims)
            other_matrix = expand_matrix(other._matrix, matrix_dims)
            matrix = op(self_matrix, other_matrix)  # ToDo if other has no dependence on vector, it would also be in the output
            shape = self._shape & other._shape
            return SparseLinTracer(self._source, matrix, bias, shape)
        else:
            # other = self._tensor(other)
            if op in {operator.mul, operator.truediv}:
                matrix = op(self._matrix, other)
                bias = op(self._bias, other)
                return SparseLinTracer(self._source, matrix, bias, self._shape & other.shape)
            elif op in {operator.add, operator.sub}:
                bias = op(self._bias, other)
                return SparseLinTracer(self._source, self._matrix, bias, self._shape & other.shape)
            else:
                raise ValueError(f"Unsupported operation {op} encountered while tracing linear function")

    @property
    def _is_tracer(self) -> bool:
        return True

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return combine_types(self._source.dtype, self._matrix.dtype)

    @property
    def backend(self) -> Backend:
        return backend_for(self._matrix, self._bias)

    def _with_shape_replaced(self, new_shape: Shape):
        matrix = rename_dims(self._matrix, self._shape, new_shape)
        bias = rename_dims(self._bias, self._shape, new_shape)
        return SparseLinTracer(self._source, matrix, bias, new_shape)

    @property
    def _out_name_to_original(self) -> Dict[str, str]:
        return {}

    def _natives(self) -> tuple:
        """
        This function should only be used to determine the compatible backends, this tensor should be regarded as not available.
        """
        return self._matrix._natives()


def concat_tracers(tracers: Sequence[Tensor], dim: str):
    full_size = sum([t_.shape.get_size(dim) for t_ in tracers])
    shape = merge_shapes([t.shape.with_dim_size(dim, full_size) for t in tracers])
    if any(isinstance(t, SparseLinTracer) for t in tracers):
        any_tracer = [t for t in tracers if isinstance(t, SparseLinTracer)][0]
        tracers = [to_sparse_tracer(t, any_tracer) if t._is_tracer else t for t in tracers]
        biases = []
        indices = []
        values = []
        offset = 0
        for t in tracers:
            if t._is_tracer:
                offset_vec = [offset if d == dim else 0 for d in channel(t._matrix._indices).item_names[0]]
                indices.append(stored_indices(t._matrix, invalid='keep') + offset_vec)
                values.append(stored_values(t._matrix, invalid='keep'))
                biases.append(expand(t._bias, t.shape[dim]))
            else:  # constant
                # mapped_shape = rename_dims(t.shape, tuple(any_tracer._renamed), [any_tracer._source.shape[o] for o in any_tracer._renamed.values()])
                biases.append(expand(discard_constant_dims(t), t.shape[dim]))
            offset += t.shape[dim].size
        full_bias = math.concat(biases, dim, expand_values=True)
        indices = concat_tensor(indices, 'entries')
        values = concat_tensor(values, 'entries')
        can_contain_double_entries = any([t._matrix._can_contain_double_entries for t in tracers if t._is_tracer])
        dense_shape = any_tracer._matrix._dense_shape.with_dim_size(dim, full_size)
        matrix = sparse_tensor(indices, values, dense_shape, can_contain_double_entries, indices_sorted=False, indices_constant=True)
        return SparseLinTracer(any_tracer._source, matrix, full_bias, shape)
    elif any(isinstance(t, GatherLinTracer) for t in tracers):
        tracers = [to_gather_tracer(t) if t._is_tracer else t for t in tracers]
        any_tracer = [t for t in tracers if t._is_tracer][0]
        src_dim = any_tracer._renamed.get(dim, dim)
        selection_dims = set(sum([channel(t._selection).item_names[0] for t in tracers if t._is_tracer and t._selection is not None], ()))
        if not selection_dims:
            selection_dims = {any_tracer._renamed.get(dim, dim)}
        selections = []
        diags = []
        biases = []
        for t in tracers:
            if t._is_tracer:
                selections.append(t._get_selection(selection_dims))
                diags.append(expand(t._diag, t.shape[dim]))
                biases.append(expand(t._bias, t.shape[dim]))
            else:  # constant
                mapped_shape = rename_dims(t.shape, tuple(any_tracer._renamed), [any_tracer._source.shape[o] for o in any_tracer._renamed.values()])
                selections.append(math.zeros(instance(selection=mapped_shape.only(src_dim).volume), channel(gather=list(selection_dims)), dtype=(int, 32)))
                diags.append(math.zeros(t.shape[dim]))
                biases.append(expand(discard_constant_dims(t), t.shape[dim]))
        full_diag = concat_tensor(diags, dim)
        full_bias = math.concat(biases, dim, expand_values=True)
        full_selection = concat_tensor(selections, 'selection')
        full_selection = unpack_dim(full_selection, 'selection', shape[dim])
        renamed = any_tracer._renamed
        assert non_channel(full_selection).size == full_size
        return GatherLinTracer(tracers[0]._source, full_diag, full_bias, shape, full_selection, renamed)
    else:  # only ShiftLinTracer + constants
        if True:  # ToDo this block minimizes zeros but removes the ShiftLinTracer structure. May be slower in the long run...
            tracers = [to_gather_tracer(t) if t._is_tracer else t for t in tracers]
            return concat_tracers(tracers, dim)
        # any_tracer = [t for t in tracers if t._is_tracer][0]
        # biases = []
        # aligned = []
        # offset = 0
        # for t in tracers:
        #     if t._is_tracer:
        #         assert isinstance(t, ShiftLinTracer)
        #         t_aligned = t.shift({dim: offset}, shape, lambda v: math.pad(v, {dim: (offset, full_size - offset - t.shape.get_size(dim))}, 0), lambda b: b)
        #         aligned.append(t_aligned)
        #         biases.append(expand(t._bias, t.shape[dim]))
        #     else:  # constant
        #         biases.append(expand(discard_constant_dims(t), t.shape[dim]))
        #     offset += t.shape.get_size(dim)
        # full_tracer = sum(aligned[1:], aligned[0])
        # full_bias = math.concat(biases, dim, expand_values=True)
        # return ShiftLinTracer(any_tracer._source, full_tracer.val, shape, full_bias, any_tracer._renamed)


def expand_tracer(tracer: Tensor, dims: Shape):
    if isinstance(tracer, GatherLinTracer):
        return GatherLinTracer(tracer._source, tracer._diag, tracer._bias, tracer.shape & dims, tracer._selection, tracer._renamed)
    if isinstance(tracer, ShiftLinTracer):
        return ShiftLinTracer(tracer._source, tracer.val, tracer.shape & dims, tracer._bias, tracer._renamed, tracer._nz_edge)
    assert isinstance(tracer, SparseLinTracer)
    return SparseLinTracer(tracer._source, tracer._matrix, tracer._bias, tracer.shape & dims)


class LinearTraceInProgress(Exception):

    def __init__(self, tracer: Tensor):
        self.tracer = tracer


def matrix_from_function(f: Callable,
                         *args,
                         auxiliary_args=None,
                         auto_compress=False,
                         sparsify_batch=None,
                         separate_independent=False,  # not fully implemented, requires auto_compress=False
                         _return_raw_output=False,
                         **kwargs) -> Tuple[Tensor, Tensor]:
    """
    Trace a linear function and construct a matrix.
    Depending on the functional form of `f`, the returned matrix may be dense or sparse.

    Args:
        f: Function to trace.
        *args: Arguments for `f`.
        auxiliary_args: Arguments in which the function is not linear.
            These parameters are not traced but passed on as given in `args` and `kwargs`.
        auto_compress: If `True`, returns a compressed matrix if supported by the backend.
        sparsify_batch: If `False`, the matrix will be batched.
            If `True`, will create dual dimensions for the involved batch dimensions.
            This will result in one large matrix instead of a batch of matrices.
        **kwargs: Keyword arguments for `f`.

    Returns:
        matrix: Matrix representing the linear dependency of the output `f` on the input of `f`.
            Input dimensions will be `dual` dimensions of the matrix while output dimensions will be regular.
        bias: Bias for affine functions or zero-vector if the function is purely linear.
    """
    assert isinstance(auxiliary_args, str) or auxiliary_args is None, f"auxiliary_args must be a comma-separated str but got {auxiliary_args}"
    from ._functional import function_parameters, f_name
    f_params = function_parameters(f)
    aux = set(s.strip() for s in auxiliary_args.split(',') if s.strip()) if isinstance(auxiliary_args, str) else f_params[1:]
    all_args = {**kwargs, **{f_params[i]: v for i, v in enumerate(args)}}
    aux_args = {k: v for k, v in all_args.items() if k in aux}
    trace_args = {k: v for k, v in all_args.items() if k not in aux}
    tree, tensors = disassemble_tree(trace_args, cache=False, attr_type=value_attributes)
    assert len(tensors) == 1, f"Only one input tensor can be traced bot got {tensors}"
    target_backend = backend_for(*tensors)
    # --- Trace function ---
    with NUMPY:
        src = TracerSource(tensors[0].shape, tensors[0].dtype, tuple(trace_args.keys())[0], 0)
        tracer = ShiftLinTracer(src, {Shift({}): math.ones()}, tensors[0].shape, bias=math.zeros(dtype=tensors[0].dtype), renamed={d: d for d in tensors[0].shape.names}, nz_edge={Shift({}): False})
        x_kwargs = assemble_tree(tree, [tracer] + tensors[1:], attr_type=value_attributes)
        result = f(**x_kwargs, **aux_args)
    out_tree, result_tensors = disassemble_tree(result, cache=False, attr_type=value_attributes)
    assert len(result_tensors) == 1, f"Linear function output must be or contain a single Tensor but got {result}"
    # for t in result_tensors[1:]:
    #     assert not t._is_tracer, f"Linear function must only return a single tracer at position 0 but got {result_tensors}"
    tracer = result_tensors[0]._simplify()
    assert tracer._is_tracer, f"Tracing linear function '{f_name(f)}' failed. Make sure only linear operations are used. Output: {tracer.shape}"
    # --- Convert to COO ---
    if sparsify_batch is None:
        if auto_compress:
            sparsify_batch = not target_backend.supports(Backend.csr_matrix_batched)
        else:
            sparsify_batch = not target_backend.supports(Backend.sparse_coo_tensor_batched)
    sparsify = tracer.shape if sparsify_batch else EMPTY_SHAPE
    if isinstance(tracer, SparseLinTracer):
        matrix, bias = tracer._get_matrix(sparsify), tracer._bias
    elif isinstance(tracer, GatherLinTracer):
        matrix, bias = to_sparse_tracer(tracer, None)._get_matrix(sparsify), tracer._bias
    else:
        matrix, bias = tracer_to_coo(tracer, sparsify, separate_independent)
    # --- Compress ---
    if auto_compress and matrix.backend.supports(Backend.mul_csr_dense) and target_backend.supports(Backend.mul_csr_dense) and isinstance(matrix, SparseCoordinateTensor):
        matrix = matrix.compress_rows()
    # elif backend.supports(Backend.mul_csc_dense):
    #     return matrix.compress_cols(), tracer._bias
    return (matrix, bias, (out_tree, result_tensors)) if _return_raw_output else (matrix, bias)
    

def tracer_to_coo(tracer: Tensor, sparsify: Shape, separate_independent: bool):
    if isinstance(tracer, TensorStack):  # This indicates separable solves
        matrices, biases = zip(*[tracer_to_coo(t, sparsify-tracer._stack_dim, separate_independent) for t in tracer._tensors])
        bias = stack(biases, tracer._stack_dim)
        if not separate_independent:
            indices = [math.concat_tensor([m._indices, expand(i, instance(m._indices), channel(sparse_idx=tracer._stack_dim.name))], 'sparse_idx') for i, m in enumerate(matrices)]
            indices = math.concat_tensor(indices, 'entries')
            values = math.concat_tensor([m._values for m in matrices], 'entries')
            # matrix = stack(matrices, tracer._stack_dim)
            dense_shape = matrices[0]._dense_shape + tracer._stack_dim
            matrix = SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries=False, indices_sorted=False, indices_constant=True)
        else:
            matrix = stack(matrices, tracer._stack_dim)
        return matrix, bias
    elif not tracer._is_tracer:  # This part of the output is independent of the input
        return expand(0, tracer.shape), tracer
    assert isinstance(tracer, ShiftLinTracer), f"Tracing linear function returned an unsupported construct: {type(tracer)}"
    pattern_dims = tracer._source.shape.only(pattern_dim_names(tracer))
    assert batch(pattern_dims).is_empty, f"Batch dimensions may not be sliced in linear operations but got pattern for {batch(pattern_dims)}"
    out_shape, src_shape, typed_src_shape, missing_dims, sliced_src_shape = matrix_dims_for_tracer(tracer, sparsify)
    original_out_names = [tracer._out_name_to_original.get(d, d) for d in out_shape.names]
    batch_val = merge_shapes(*tracer.val.values()).without(out_shape)
    if non_batch(out_shape).is_empty:
        assert len(tracer.val) == 1 and not non_batch(tracer.val[Shift({})])
        return tracer.val[Shift({})], tracer._bias
    out_indices = []
    src_indices = []
    values = []
    for shift_, shift_val in tracer.val.items():
        if shift_val.default_backend is NUMPY:  # sparsify stencil further
            native_shift_values = shift_val.native([batch_val, *out_shape])
            mask = np.sum(abs(native_shift_values), 0)  # only 0 where no batch entry has a non-zero value
            out_idx = numpy.nonzero(mask)
            src_idx = [(component + shift_[dim]) % typed_src_shape.get_size(dim) for component, dim in zip(out_idx, original_out_names)]
            values.append(native_shift_values[(slice(None), *out_idx)])
        else:  # add full stencil tensor
            out_idx = np.unravel_index(np.arange(out_shape.volume), out_shape.sizes) if out_shape else 0
            src_idx = [(component + shift_[dim]) % typed_src_shape.get_size(dim) for component, dim in zip(out_idx, original_out_names)]
            values.append(shift_val.native([batch_val, out_shape]))
        out_indices.append(out_idx)
        src_idx_all = []
        for dim in typed_src_shape:
            if dim in missing_dims:
                if not separate_independent:
                    offset = shift_.get_size(dim, default=0)
                    src_idx_all.append(np.zeros_like(src_idx[0]) + offset)
            else:
                out_dim = {v: k for k, v in tracer._renamed.items()}.get(dim.name, dim.name)
                src_idx_all.append(src_idx[out_shape.index(out_dim)])
        src_indices.append(src_idx_all)
    indices_np = np.concatenate([np.concatenate(src_indices, axis=1), np.concatenate(out_indices, axis=1)]).T
    indices = wrap(indices_np, instance('entries'), channel(sparse_idx=(sliced_src_shape if separate_independent else src_shape).names + out_shape.names))
    backend = choose_backend(*values)
    values = math.reshaped_tensor(backend.concat(values, axis=-1), [batch_val, instance('entries')], convert=False)
    dense_shape = (sliced_src_shape if separate_independent else src_shape) & out_shape
    max_rank = out_shape.volume - tracer.min_rank_deficiency()
    matrix = SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries=False, indices_sorted=False, indices_constant=True, m_rank=max_rank)
    return matrix, tracer._bias


def matrix_dims_for_tracer(tracer: Union[ShiftLinTracer, SparseLinTracer], sparsify: Shape):
    renamed_src_names = [o for n, o in tracer._out_name_to_original.items() if n != o]
    removed_dims = tracer._source.shape.without(tracer.shape).without(renamed_src_names)  # these were sliced off
    ignored_dims = tracer._source.shape - dependent_out_dims(tracer, sparsify) - removed_dims - renamed_src_names  # these will be parallelized and not added to the matrix
    out_shape = tracer.shape.without(ignored_dims)
    typed_src_shape = tracer._source.shape.without(ignored_dims)
    src_shape = typed_src_shape.as_dual()
    sliced_src_shape = src_shape.without(removed_dims.as_dual())
    return out_shape, src_shape, typed_src_shape, removed_dims, sliced_src_shape


def dependent_src_dims(tracer: Tensor) -> Shape:
    """
    Source dimensions relevant to the linear operation.
    This includes `pattern_dims` as well as dimensions along which only the values vary.
    These dimensions cannot be parallelized trivially with a non-batched matrix.

    Bias dimensions do not require a batched matrix but are subtracted from the right-hand-side vector.
    They are not included unless also relevant to the matrix.
    """
    if isinstance(tracer, ShiftLinTracer):
        bias_dims = set(variable_shape(tracer._bias).names)
        names = pattern_dim_names(tracer) | set(sum([t.shape.names for t in tracer.val.values()], ())) | bias_dims
        result = tracer._source.shape.only(names)
        assert len(result) == len(names)
        return result
    elif isinstance(tracer, GatherLinTracer):
        dims = set()
        if tracer._selection is not None:
            dims.update(set(channel(tracer._selection).item_names[0]))
        dims.update(set([tracer._renamed.get(d, d) for d in tracer._diag.shape.names]))
        return tracer._source.shape.only(dims)
    elif isinstance(tracer, SparseLinTracer):
        return tracer._source.shape.only(sparse_dims(tracer._matrix).names)


def dependent_out_dims(tracer: Tensor, sparsify=None) -> Shape:
    """
    Current dimensions relevant to the linear operation.
    This includes `pattern_dims` as well as dimensions along which only the values vary.
    These dimensions cannot be parallelized trivially with a non-batched matrix.

    Bias dimensions do not require a batched matrix but are subtracted from the right-hand-side vector.
    They are not included unless also relevant to the matrix.
    """
    if isinstance(tracer, ShiftLinTracer):
        bias_names = set(variable_shape(tracer._bias).names)
        pattern_names = pattern_dim_names(tracer)
        if sparsify is None:
            value_names = set(sum([t.shape.names for t in tracer.val.values()], ()))
        else:
            value_names = set([n for t in tracer.val.values() for n in t.shape.names if n in sparsify])
        names = bias_names | pattern_names | value_names
        result = tracer.shape.only(names)
        assert len(result) == len(names), f"Tracer was modified along {names} but the dimensions {names - set(result.names)} are not present anymore, probably due to slicing. Make sure the linear function output retains all dimensions relevant to the linear operation."
        return result
    elif isinstance(tracer, GatherLinTracer):
        result = tracer._diag.shape
        if tracer._selection is not None:
            result &= tracer._selection.shape.non_channel
        return result
    elif isinstance(tracer, SparseLinTracer):
        return tracer._matrix.sparse_dims.only(tracer.shape)


def pattern_dim_names(tracer) -> Set[str]:
    """
    Dimensions along which the sparse matrix contains off-diagonal elements.
    These dimensions must be part of the sparse matrix and cannot be parallelized.
    """
    if isinstance(tracer, ShiftLinTracer):
        return set().union(*[shift.by_dim.keys() for shift in tracer.val])
    raise NotImplementedError
    # elif isinstance(tracer, GatherLinTracer):
        # return set(dependent_src_dims(tracer).names)
    # elif isinstance(tracer, SparseLinTracer):
        # return set(dependent_src_dims(tracer).names)


def to_sparse_tracer(tracer: Tensor, ref: Optional[Tensor]) -> SparseLinTracer:
    assert tracer._is_tracer
    if isinstance(tracer, SparseLinTracer):
        return tracer
    if isinstance(tracer, ShiftLinTracer):
        matrix, bias = tracer_to_coo(tracer, sparsify=dependent_out_dims(ref), separate_independent=False)
        src_dims = dual(matrix) - set(tracer._renamed)
        matrix = rename_dims(matrix, src_dims, [f'~{n}_src' for n in src_dims.as_batch().names])
        return SparseLinTracer(tracer._source, matrix, bias, tracer.shape)
    assert isinstance(tracer, GatherLinTracer)
    if tracer._selection is None:
        if ref is not None:
            in_dims = dependent_src_dims(ref).as_dual()
            out_dims = dependent_out_dims(ref)
            cols = math.meshgrid(out_dims.as_instance())
        else:
            in_dims = dependent_src_dims(tracer).as_dual()
            out_dims = dependent_out_dims(tracer)
            cols = math.meshgrid(out_dims.as_instance())
    else:
        in_dims = dependent_src_dims(tracer).as_dual()
        out_dims = dependent_out_dims(tracer)
        cols = rename_dims(tracer._selection, non_channel, instance)
    in_dims = rename_dims(in_dims, in_dims, [n + "_src" for n in in_dims.names])
    cols = rename_dims(cols, channel, channel(sparse_idx=in_dims))
    gather_dims = cols.shape.non_channel
    rows = math.meshgrid(gather_dims, stack_dim=channel(sparse_idx=out_dims))
    indices = concat_tensor([rows, cols], 'sparse_idx')
    dense_shape = in_dims & out_dims
    matrix = sparse_tensor(indices, rename_dims(tracer._diag, indices.shape.non_channel.non_batch, instance), dense_shape, can_contain_double_entries=False, indices_sorted=False, format='coo', indices_constant=True)
    # ToDo check renaming
    return SparseLinTracer(tracer._source, matrix, tracer._bias, tracer._shape)


def to_gather_tracer(t: Tensor) -> GatherLinTracer:
    if isinstance(t, GatherLinTracer):
        return t
    if isinstance(t, SparseLinTracer):
        raise AssertionError
    assert isinstance(t, ShiftLinTracer)
    if len(t.val) > 1 or next(iter(t.val)):
        raise NotImplementedError(f"Converting off-diagonal elements to sparse tracer not supported")
    return GatherLinTracer(t._source, t.val[Shift({})], t._bias, t._shape, None, t._renamed)


def expand_matrix(matrix: Tensor, dims: Shape) -> Tensor:
    """Add missing dims as diagonals"""
    if dims in matrix.shape:
        return matrix
    missing = dims.without(matrix.shape)
    src_dims = [d for d in missing if d.name.startswith('~') and d.name.endswith('_src')]
    out_dims = [d.name[1:-4] for d in src_dims]
    out_dims = [missing[d] for d in out_dims]
    if isinstance(matrix, Dense):
        if len(src_dims) > 1:
            raise NotImplementedError
        for src_dim, out_dim in zip(src_dims, out_dims):
            assert src_dim.size == out_dim.size
            diagonal_idx = expand(math.arange(instance(diagonal_entries=src_dim.size)), channel(sparse_idx=[src_dim.name, out_dim.name]))
            values = math.ones(instance(diagonal_entries=src_dim.size))
            return sparse_tensor(diagonal_idx, values, missing, can_contain_double_entries=False, indices_sorted=True, indices_constant=True)
    raise NotImplementedError
