import dataclasses
import operator
from numbers import Number
import traceback
import warnings
from contextlib import contextmanager
from typing import Union, TypeVar, Sequence, Any, Literal

from dataclasses import dataclass
from typing import Tuple, Callable, List

import numpy
import numpy as np

from ._magic_ops import PhiTreeNodeType, variable_attributes, copy_with, stack, pack_dims, expand, slice_, flatten, rename_dims, unpack_dim, unstack, value_attributes, all_attributes
from ._shape import (Shape,
                     CHANNEL_DIM, BATCH_DIM, SPATIAL_DIM, EMPTY_SHAPE,
                     parse_dim_order, shape_stack, merge_shapes, channel, concat_shapes, primal,
                     SUPERSCRIPT, IncompatibleShapes, INSTANCE_DIM, batch, spatial, dual, instance, shape, shape as shape_, DimFilter, non_batch, DEBUG_CHECKS, parse_shape_spec,
                     prepare_renaming_gather, after_gather, concat_shapes_, Dim, PureShape, SHAPE_TYPES)
from ..backend import NoBackendFound, choose_backend, BACKENDS, get_precision, default_backend, convert as convert_, \
    Backend, ComputeDevice, OBJECTS, NUMPY, ML_LOGGER
from ..backend._backend import get_operator
from ..backend._dtype import DType, combine_types, BOOL, INT64, INT32
from .magic import BoundDim, PhiTreeNode, slicing_dict, Shaped, _BoundDims
from .magic import Shapable
from ..backend.xops import ExtraOperator


class Tensor:
    """
    Abstract base class to represent structured data of one data type.
    This class replaces the native tensor classes `numpy.ndarray`, `torch.Tensor`, `tensorflow.Tensor` or `jax.numpy.ndarray` as the main data container in Φ-ML.

    `Tensor` instances are different from native tensors in two important ways:

    * The dimensions of Tensors have *names* and *types*.
    * Tensors can have non-uniform shapes, meaning that the size of dimensions can vary along other dimensions.

    To check whether a value is a tensor, use `isinstance(value, Tensor)`.

    To construct a Tensor, use `phiml.math.tensor()`, `phiml.math.wrap()` or one of the basic tensor creation functions,
    see https://tum-pbs.github.io/PhiML/Tensors.html .

    Tensors are not editable.
    When backed by an editable native tensor, e.g. a `numpy.ndarray`, do not edit the underlying data structure.
    """

    def __init__(self):
        if DEBUG_CHECKS:
            self._init_stack = traceback.extract_stack()

    def native(self, order: Union[str, tuple, list, Shape] = None, force_expand=True):
        """
        Returns a native tensor object with the dimensions ordered according to `order`.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        Additionally, groups of dims can be specified for `order` to pack dims.
        To do this, pass a `tuple` or `list` of dims to be packed into one native axis. Each entry must be one of the following:

        * `str`: the name of one dimension that is present on `value`.
        * `Shape`: Dimensions to be packed. If `force_expand`, missing dimensions are first added, otherwise they are ignored.
        * Filter function: Packs all dimensions of this type that are present on `value`.
        * Ellipsis `...`: Packs all remaining dimensions into this slot. Can only be passed once.
        * `None` or `()`: Adds a singleton dimension.

        Collections of or comma-separated dims may also be used but only if all dims are present on `value`.

        Args:
            order: (Optional) Order of dimension names as comma-separated string, list or `Shape`.
            force_expand: If `False`, dimensions along which values are guaranteed to be constant will not be expanded to their true size but returned as singleton dimensions.
                If `True`, repeats the tensor along missing dimensions.
                If `False`, puts singleton dimensions where possible.
                If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.

        Returns:
            Native tensor representation, such as PyTorch tensor or NumPy array.

        Raises:
            `ValueError` if the tensor cannot be transposed to match target_shape
        """
        raise NotImplementedError

    def _reshaped_native(self, groups: Sequence[Shape]):
        """constant (collapsed) dims may be left out of groups to avoid expanding.
        All present dims will be expanded to match `groups`"""
        raise NotImplementedError

    def _transposed_native(self, order: Sequence[str], force_expand: bool):
        """Returns the native tensor, transposing it to match `order`.
        New names in `order` are added as singleton dims.
        Constant dims may be dropped to avoid expanding. If present, they will be included as singleton, not expanded to their full size."""
        raise NotImplementedError

    def numpy(self, order: Union[str, tuple, list, Shape] = None, force_expand=True) -> np.ndarray:
        """
        Converts this tensor to a `numpy.ndarray` with dimensions ordered according to `order`.
        
        *Note*: Using this function breaks the autograd chain. The returned tensor is not differentiable.
        To get a differentiable tensor, use `Tensor.native()` instead.
        
        Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
        If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

        If this `Tensor` is backed by a NumPy array, a reference to this array may be returned.

        See Also:
            `phiml.math.numpy()`

        Args:
            order: (Optional) Order of dimension names as comma-separated string, list or `Shape`.
            force_expand: If `False`, dimensions along which values are guaranteed to be constant will not be expanded to their true size but returned as singleton dimensions.

        Returns:
            NumPy representation

        Raises:
            ValueError if the tensor cannot be transposed to match target_shape
        """
        return self.backend.numpy(self.native(order, force_expand))

    def __array__(self, dtype=None):  # NumPy conversion
        if self.rank > 1:
            warnings.warn("Automatic conversion of Φ-ML tensors to NumPy can cause problems because the dimension order is not guaranteed.", SyntaxWarning, stacklevel=3)
        return self.numpy(self._shape)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # NumPy interface
        if len(inputs) != 2:
            return NotImplemented
        switch_args = self is inputs[1]
        other = inputs[0] if switch_args else inputs[1]
        if ufunc.__name__ == 'multiply':
            return self._op2(other, operator.mul, switch_args)
        if ufunc.__name__ == 'add':
            return self._op2(other, operator.add, switch_args)
        if ufunc.__name__ == 'subtract':
            return self._op2(other, operator.sub, switch_args)
        if ufunc.__name__ in ['divide', 'true_divide']:
            return self._op2(other, operator.truediv, switch_args)
        if ufunc.__name__ == 'floor_divide':
            return self._op2(other, operator.floordiv, switch_args)
        if ufunc.__name__ == 'remainder':
            return self._op2(other, operator.mod, switch_args)
        if ufunc.__name__ == 'power':
            return self._op2(other, operator.pow, switch_args)
        if ufunc.__name__ == 'equal':
            return self.__eq__(inputs[1] if self is inputs[0] else inputs[0])
        if ufunc.__name__ == 'not_equal':
            return self.__ne__(inputs[1] if self is inputs[0] else inputs[0])
        if ufunc.__name__ == 'greater':
            return self._op2(other, operator.gt, switch_args)
        if ufunc.__name__ == 'greater_equal':
            return self._op2(other, operator.ge, switch_args)
        if ufunc.__name__ == 'less':
            return self._op2(other, operator.gt, not switch_args)
        if ufunc.__name__ == 'less_equal':
            return self._op2(other, operator.ge, not switch_args)
        if ufunc.__name__ == 'left_shift':
            return self._op2(other, operator.lshift, switch_args)
        if ufunc.__name__ == 'right_shift':
            return self._op2(other, operator.rshift, switch_args)
        raise NotImplementedError(f"NumPy function '{ufunc.__name__}' is not compatible with Φ-ML tensors.")

    @property
    def dtype(self) -> DType:
        """ Data type of the elements of this `Tensor`. """
        raise NotImplementedError(self.__class__)

    @property
    def shape(self) -> Shape:
        """ The `Shape` lists the dimensions with their sizes, names and types. """
        raise NotImplementedError(self.__class__)

    @property
    def backend(self) -> Backend:
        raise NotImplementedError(self.__class__)

    @property
    def default_backend(self) -> Backend:
        return self.backend

    def _with_shape_replaced(self, new_shape: Shape):
        raise NotImplementedError(self.__class__)

    def _with_natives_replaced(self, natives: list):
        """ Replaces all n _natives() of this Tensor with the first n elements of the list and removes them from the list. """
        raise NotImplementedError(self.__class__)

    @property
    def rank(self) -> int:
        """
        Number of explicit dimensions of this `Tensor`. Equal to `tensor.shape.rank`.
        This replaces [`numpy.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html) /
        [`torch.Tensor.dim`](https://pytorch.org/docs/master/generated/torch.Tensor.dim.html) /
        [`tf.rank()`](https://www.tensorflow.org/api_docs/python/tf/rank) /
        [`jax.numpy.ndim()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndim.html).
        """
        return self.shape.rank

    @property
    def _is_tracer(self) -> bool:
        """
        Tracers store additional internal information.
        They should not be converted to `native()` in intermediate operations.
        
        TensorStack prevents performing the actual stack operation if one of its component tensors is special.
        """
        raise NotImplementedError(self.__class__)

    def _to_dict(self):
        return cached(self)._to_dict()

    def __len__(self):
        return self.shape.volume if self.rank == 1 else NotImplemented

    def __bool__(self):
        assert self.rank == 0, f"Cannot convert tensor with non-empty shape {self.shape} to bool. Use tensor.any or tensor.all instead."
        from ._ops import all_
        if not self.default_backend.supports(Backend.jit_compile):  # NumPy
            return bool(self.native()) if self.rank == 0 else bool(all_(self).native())
        else:
            # __bool__ does not work with TensorFlow tracing.
            # TensorFlow needs to see a tf.Tensor in loop conditions but won't allow bool() invocations.
            # However, this function must always return a Python bool.
            raise AssertionError("To evaluate the boolean value of a Tensor, use 'Tensor.all'.")

    @property
    def all(self):
        """ Whether all values of this `Tensor` are `True` as a native bool. """
        from ._ops import all_, cast
        if self.rank == 0:
            return cast(self, BOOL).native()
        else:
            return all_(self, dim=self.shape).native()

    @property
    def any(self):
        """ Whether this `Tensor` contains a `True` value as a native bool. """
        from ._ops import any_, cast
        if self.rank == 0:
            return cast(self, BOOL).native()
        else:
            return any_(self, dim=self.shape).native()

    @property
    def mean(self):
        """ Mean value of this `Tensor` as a native scalar. """
        from ._ops import mean
        return mean(self, dim=self.shape).native()

    @property
    def finite_mean(self):
        """ Mean value of all finite values in this `Tensor` as a native scalar. """
        from ._ops import finite_mean
        return finite_mean(self, dim=self.shape).native()

    @property
    def std(self):
        """ Standard deviation of this `Tensor` as a native scalar. """
        from ._ops import std
        return std(self, dim=self.shape).native()

    @property
    def sum(self):
        """ Sum of all values of this `Tensor` as a native scalar. """
        from ._ops import sum_
        return sum_(self, dim=self.shape).native()

    @property
    def finite_sum(self):
        """ Sum of all finite values of this `Tensor` as a native scalar. """
        from ._ops import finite_sum
        return finite_sum(self, dim=self.shape).native()

    @property
    def min(self):
        """ Minimum value of this `Tensor` as a native scalar. """
        from ._ops import min_
        return min_(self, dim=self.shape).native()

    @property
    def finite_min(self):
        """ Minimum finite value of this `Tensor` as a native scalar. """
        from ._ops import finite_min
        return finite_min(self, dim=self.shape).native()

    @property
    def max(self):
        """ Maximum value of this `Tensor` as a native scalar. """
        from ._ops import max_
        return max_(self, dim=self.shape).native()

    @property
    def finite_max(self):
        """ Maximum finite value of this `Tensor` as a native scalar. """
        from ._ops import finite_max
        return finite_max(self, dim=self.shape).native()

    @property
    def real(self) -> 'Tensor':
        """
        Returns the real part of this tensor.

        See Also:
            `phiml.math.real()`
        """
        from ._ops import real
        return real(self)

    @property
    def imag(self) -> 'Tensor':
        """
        Returns the imaginary part of this tensor.
        If this tensor does not store complex numbers, returns a zero tensor with the same shape and dtype as this tensor.

        See Also:
            `phiml.math.imag()`
        """
        from ._ops import imag
        return imag(self)

    @property
    def available(self) -> bool:
        """
        A tensor is available if it stores concrete values and these can currently be read.

        Tracers used inside jit compilation are typically not available.

        See Also:
            `phiml.math.jit_compile()`.
        """
        if self._is_tracer:
            return False
        natives = self._natives()
        natives_available = [choose_backend(native).is_available(native) for native in natives]
        return all(natives_available)

    @property
    def device(self) -> Union[ComputeDevice, None]:
        """
        Returns the `ComputeDevice` that this tensor is allocated on.
        The device belongs to this tensor's `default_backend`.

        See Also:
            `Tensor.default_backend`.
        """
        natives = self._natives()
        if not natives:
            return None
        return self.default_backend.get_device(natives[0])

    def __int__(self):
        return int(self.native()) if self.shape.volume == 1 else NotImplemented

    def __float__(self):
        return float(self.native()) if self.shape.volume == 1 else NotImplemented

    def __complex__(self):
        return complex(self.native()) if self.shape.volume == 1 else NotImplemented

    def __index__(self):
        assert self.shape.volume == 1, f"Only scalar tensors can be converted to index but has shape {self.shape}"
        assert self.dtype.kind == int, f"Only int tensors can be converted to index but dtype is {self.dtype}"
        return int(self.native())

    def __contains__(self, item):
        if isinstance(item, SHAPE_TYPES):
            return item in self.shape
        elif isinstance(item, BoundDim):
            return item.name in self.shape
        elif isinstance(item, _BoundDims):
            return item.dims in self.shape
        elif isinstance(item, str):
            assert self.dtype.kind != object, "str in Tensor not allowed for object-type Tensors"
            return item in self.shape
        raise ValueError(f"'dim in Tensor' requires dim to be a Shape or str but got {item}")

    def __repr__(self):
        return format_tensor(self, PrintOptions())

    def _repr_pretty_(self, printer, cycle):
        printer.text(format_tensor(self, PrintOptions(colors=DEFAULT_COLORS)))

    def print(self, layout='full', float_format=None, threshold=8, include_shape=None, include_dtype=None):
        print(format_tensor(self, PrintOptions(layout=layout, float_format=float_format, threshold=threshold, colors=DEFAULT_COLORS, include_shape=include_shape, include_dtype=include_dtype)))

    def __format__(self, format_spec: str):
        if BROADCAST_FORMATTER.values is not None:
            return BROADCAST_FORMATTER.register_formatted(self, format_spec)
        specs = format_spec.split(':')
        layout_ = 'auto'
        for possible_layout in ['summary', 'full', 'row', 'numpy']:
            if possible_layout in specs:
                assert layout_ == 'auto', f"Two layout identifiers encountered in '{format_spec}'"
                layout_ = possible_layout
        include_shape = 'shape' in specs or (False if 'no-shape' in specs else None)
        include_dtype = 'dtype' in specs or (False if 'no-dtype' in specs else None)
        color = 'color' in specs or (False if 'no-color' in specs else None)
        threshold = 8
        float_format = None
        for spec in specs:
            if spec.startswith('threshold='):
                threshold = int(spec[len('threshold='):])
            elif '.' in spec:
                float_format = spec
        result = format_tensor(self, PrintOptions(layout_, float_format, threshold, color, include_shape, include_dtype))
        return result

    def __getitem__(self, item) -> 'Tensor':
        if isinstance(item, Tensor):
            if item.dtype.kind == bool:
                from ._ops import boolean_mask
                return boolean_mask(self, item.shape.non_batch or item.shape, item)
            elif item.dtype.kind == int:
                from ._ops import gather
                return gather(self, item)
            else:
                raise AssertionError(f"Index tensor must be of dtype int (gather) or bool (boolean_mask) but got {item}")
        item = slicing_dict(self, item)
        selections = {}
        sliced = self
        for dim, selection in item.items():
            if dim not in self.shape:
                continue
            selection, new_dim = prepare_renaming_gather(self.shape, dim, selection)
            # Either handle slicing directly or add it to the dict
            if isinstance(selection, (tuple, list)):
                result = [sliced[{dim: i}] for i in selection]
                stack_dim = after_gather(sliced.shape[dim], {dim: selection})
                sliced = stack(result, stack_dim)
                if new_dim is not None:
                    sliced = rename_dims(sliced, dim, new_dim)
            elif isinstance(selection, Tensor) and selection.dtype.kind == bool:
                from ._ops import boolean_mask
                sliced = boolean_mask(sliced, dim, selection)
            elif isinstance(selection, Tensor) and selection.dtype.kind == int:
                from ._ops import gather
                sliced = gather(sliced, selection, dims=dim)
            elif isinstance(selection, slice):
                if selection.start in (0, None) and selection.stop is not None and isinstance(self.shape.get_size(dim), int) and selection.stop >= self.shape.get_size(dim):
                    continue
                if selection.start is not None and isinstance(self.shape.get_size(dim), int) and selection.start >= self.shape.get_size(dim) and (selection.step is None or selection.step > 0):
                    return Dense(0, (), self.shape.after_gather(item), self.backend)
                selections[dim] = selection
            else:
                selections[dim] = selection
        return sliced._getitem(selections) if selections else sliced

    def _getitem(self, selection: dict) -> 'Tensor':
        """
        Slice the tensor along specified dimensions.

        Args:
          selection: dim_name: str -> Union[int, slice]
        """
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise SyntaxError("Tensors are not editable to preserve the autodiff chain. This feature might be added in the future. To update part of a tensor, use math.where() or math.scatter()")

    def __unstack__(self, dims: Tuple[str, ...]) -> Tuple['Tensor', ...]:  # from phiml.math.magic.Sliceable
        if len(dims) == 1:
            return self._unstack(dims[0])
        else:
            return NotImplemented

    def _unstack(self, dim: str):
        """
        Splits this tensor along the specified dimension.
        The returned tensors have the same dimensions as this tensor save the unstacked dimension.

        Raises an error if the dimension is not part of the `Shape` of this `Tensor`.

        See Also:
            `TensorDim.unstack()`

        Args:
            dim: name of dimension to unstack

        Returns:
            tuple of tensors

        """
        raise NotImplementedError()

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        if any(isinstance(v, Layout) for v in values):
            layout_ = [v for v in values if isinstance(v, Layout)][0]
            return layout_.__stack__(values, dim, **_kwargs)
        from ._ops import stack_tensors
        return stack_tensors(values, dim)

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        return expand_tensor(self, dims)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Tensor':
        from ._ops import concat_tensor
        return concat_tensor(values, dim)

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Tensor':
        return self._with_shape_replaced(rename_dims(self.shape, dims, new_dims))

    def __unpack_dim__(self, dim: str, unpacked_dims: Shape, **kwargs) -> 'Tensor':
        if self.shape.is_uniform:
            native = self.native(shape_, True)
            new_shape = self.shape.replace(dim, unpacked_dims)
            if not new_shape.well_defined:
                assert new_shape.undefined.rank <= 1, f"At most one dim can have an undefined size to be inferred during un-packing but got {new_shape}"
                missing = int(self.shape.volume / new_shape.defined.volume)
                sizes = [missing if s is None else s for s in new_shape.sizes]
                new_shape = new_shape.with_sizes(sizes)
            if new_shape.is_uniform:
                native_reshaped = choose_backend(native).reshape(native, new_shape.sizes)
                return Dense(native_reshaped, new_shape.names, new_shape, self.backend)
            else:
                split_dim = new_shape.non_uniform_shape[-1]
                i = 0
                result = []
                for idx in split_dim.meshgrid():
                    s = after_gather(new_shape, idx).get_size(new_shape.non_uniform.name)
                    sliced = self[{dim: slice(i, i + s)}]
                    result.append(sliced._with_shape_replaced(sliced.shape.replace(dim, unpacked_dims - split_dim)))
                    i += s
                return stack(result, split_dim)
        else:
            tensors = self._tensors
            if dim == self._stack_dim.name:
                for udim in unpacked_dims:
                    tensors = [TensorStack(tensors[o::len(tensors)//udim.size], udim) for o in range(len(tensors)//udim.size)]
                assert len(tensors) == 1
                return tensors[0]
            raise NotImplementedError

    def __cast__(self, dtype: DType):
        return self._op1(lambda native: choose_backend(native).cast(native, dtype=dtype))

    def dimension(self, name: Union[str, Shape]) -> 'TensorDim':
        """
        Returns a reference to a specific dimension of this tensor.
        This is equivalent to the syntax `tensor.<name>`.

        The dimension need not be part of the `Tensor.shape` in which case its size is 1.

        Args:
            name: dimension name

        Returns:
            `TensorDim` corresponding to a dimension of this tensor
        """
        if isinstance(name, str):
            return TensorDim(self, name)
        elif isinstance(name, SHAPE_TYPES):
            return TensorDim(self, name.name)
        else:
            raise ValueError(name)

    def pack(self, dims, packed_dim):
        """ See `pack_dims()` """
        from ._ops import pack_dims
        return pack_dims(self, dims, packed_dim)

    def unpack(self, dim, unpacked_dims):
        """ See `unpack_dim()` """
        from ._ops import unpack_dim
        return unpack_dim(self, dim, unpacked_dims)

    @property
    def T(self):
        return self._with_shape_replaced(self.shape.transposed())

    @property
    def Ti(self):
        return self._with_shape_replaced(self.shape.transpose(INSTANCE_DIM))

    @property
    def Tc(self):
        return self._with_shape_replaced(self.shape.transpose(CHANNEL_DIM))

    @property
    def Ts(self):
        return self._with_shape_replaced(self.shape.transpose(SPATIAL_DIM))

    def map(self, function: Callable, dims=shape_, range=range, unwrap_scalars=True, **kwargs):
        from ._functional import map_
        return map_(function, self, dims=dims, range=range, unwrap_scalars=unwrap_scalars, **kwargs)

    def __getattr__(self, name):
        if name.startswith('__'):  # called by hasattr in magic ops
            raise AttributeError
        if name.startswith('_'):
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        if name == 'is_tensor_like':  # TensorFlow replaces abs() while tracing and checks for this attribute
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        assert name not in ('shape', '_shape', 'tensor'), name
        return TensorDim(self, name)

    def __add__(self, other):
        return self._op2(other, operator.add, False)

    def __radd__(self, other):
        return self._op2(other, operator.add, True)

    def __sub__(self, other):
        return self._op2(other, operator.sub, False)

    def __rsub__(self, other):
        return self._op2(other, operator.sub, True)

    def __and__(self, other):
        return self._op2(other, operator.and_, False)

    def __rand__(self, other):
        return self._op2(other, operator.and_, True)

    def __or__(self, other):
        return self._op2(other, operator.or_, False)

    def __ror__(self, other):
        return self._op2(other, operator.or_, True)

    def __xor__(self, other):
        return self._op2(other, operator.xor, False)

    def __rxor__(self, other):
        return self._op2(other, operator.xor, True)

    def __mul__(self, other):
        return self._op2(other, operator.mul, False)

    def __rmul__(self, other):
        return self._op2(other, operator.mul, True)

    def __truediv__(self, other):
        return self._op2(other, operator.truediv, False)

    def __rtruediv__(self, other):
        return self._op2(other, operator.truediv, True)

    def __divmod__(self, other):
        return self._op2(other, divmod, False)

    def __rdivmod__(self, other):
        return self._op2(other, divmod, True)

    def __floordiv__(self, other):
        return self._op2(other, operator.floordiv, False)

    def __rfloordiv__(self, other):
        return self._op2(other, operator.floordiv, True)

    def __pow__(self, power, modulo=None):
        assert modulo is None
        return self._op2(power, operator.pow, False)

    def __rpow__(self, other):
        return self._op2(other, operator.pow, True)

    def __mod__(self, other):
        return self._op2(other, operator.mod, False)

    def __rmod__(self, other):
        return self._op2(other, operator.mod, True)

    def __eq__(self, other) -> 'Tensor':
        if self is other:
            return expand(True, self.shape)
        if _EQUALITY_REDUCE[-1]['type'] == 'ref':
            return wrap(self is other)
        elif _EQUALITY_REDUCE[-1]['type'] == 'shape_and_value':
            if set(self.shape) != set(other.shape):
                return wrap(False)
            from ._ops import close
            return wrap(close(self, other, rel_tolerance=_EQUALITY_REDUCE[-1]['rel_tolerance'], abs_tolerance=_EQUALITY_REDUCE[-1]['abs_tolerance'], equal_nan=_EQUALITY_REDUCE[-1]['equal_nan']))
        if other is None:
            other = float('nan')
        if self.shape.is_compatible(shape(other)):
            return self._op2(other, operator.eq, False)
        else:
            return wrap(False)

    def __ne__(self, other) -> 'Tensor':
        if _EQUALITY_REDUCE[-1]['type'] == 'ref':
            return wrap(self is not other)
        elif _EQUALITY_REDUCE[-1]['type'] == 'shape_and_value':
            if set(self.shape) != set(other.shape):
                return wrap(True)
            from ._ops import close
            return wrap(not close(self, other, rel_tolerance=_EQUALITY_REDUCE[-1]['rel_tolerance'], abs_tolerance=_EQUALITY_REDUCE[-1]['abs_tolerance'], equal_nan=_EQUALITY_REDUCE[-1]['equal_nan']))
        if other is None:
            other = float('nan')
        if self.shape.is_compatible(shape(other)):
            return self._op2(other, operator.ne, False)
        else:
            return wrap(True)

    def __lt__(self, other):
        return self._op2(other, operator.gt, True)

    def __le__(self, other):
        return self._op2(other, operator.ge, True)

    def __gt__(self, other):
        return self._op2(other, operator.gt, False)

    def __ge__(self, other):
        return self._op2(other, operator.ge, False)

    def __lshift__(self, other):
        return self._op2(other, operator.lshift, False)

    def __rlshift__(self, other):
        return self._op2(other, operator.lshift, True)

    def __rshift__(self, other):
        return self._op2(other, operator.rshift, False)

    def __rrshift__(self, other):
        return self._op2(other, operator.rshift, True)

    def __abs__(self):
        return self._op1(lambda t: choose_backend(t).abs(t))

    def __round__(self, n=None):
        return self._op1(lambda t: choose_backend(t).round(t))

    def __copy__(self):
        return self._op1(lambda t: choose_backend(t).copy(t, only_mutable=True))

    def __deepcopy__(self, memodict={}):
        return self._op1(lambda t: choose_backend(t).copy(t, only_mutable=False))

    def __neg__(self) -> 'Tensor':
        return self._op1(operator.neg)

    def __invert__(self) -> 'Tensor':
        return self._op1(lambda t: choose_backend(t).invert(t))

    def __reversed__(self):
        assert self.shape.channel.rank == 1
        return self[::-1]

    def __iter__(self):
        if self.rank == 1:
            return iter(self.native())
        elif self.rank == 0:
            return iter([self.native()])
        else:
            native = self.native([self.shape])
            return iter(native)

    def __matmul__(self, other):
        from ._ops import dot
        assert isinstance(other, Tensor), f"Matmul '@' requires two Tensor arguments but got {type(other)}"
        if not self.shape.dual_rank and self.shape.channel_rank:
            match = self.shape.channel.only(other.shape.channel)
            if match:
                return dot(self, match, other, match)
        match_names = self.shape.dual.as_batch().names
        if not match_names:  # this is not a matrix
            assert self.shape.primal.only(other.shape).is_empty, f"Cannot compute matmul {self.shape} @ {other.shape}. First argument is not a matrix; it has no dual dimensions."
            return self * other
        match_primal = other.shape.only(match_names, reorder=True)
        if not match_primal:
            assert non_batch(other).non_dual.rank == 1, f"Cannot multiply {self.shape} @ {other.shape} because arg2 does not have appropriate non-dual dimensions"
            assert non_batch(other).non_dual.size == match_primal.volume, f"Cannot multiply {self.shape} @ {other.shape} because dual dims of arg1 have no match"
            match_primal = non_batch(other).non_dual
        match_dual = self.shape.dual.only(match_primal.as_dual(), reorder=True)
        left_arg = pack_dims(self, match_dual, dual('_reduce'))
        right_arg = pack_dims(other, match_primal, channel('_reduce'))
        return dot(left_arg, '~_reduce', right_arg, '_reduce')

    # def __rmatmul__(self, other):

    def _tensor(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return other
        elif isinstance(other, (tuple, list)) and any(isinstance(v, Tensor) for v in other):
            if 'vector' in self.shape:
                outer_dim = self.shape['vector']
            elif self.shape.channel_rank == 1:
                outer_dim = self.shape.channel
            else:
                raise ValueError(f"Cannot combine tensor of shape {self.shape} with tuple {tuple([type(v).__name__ for v in other])}")
            remaining_shape = self.shape.without(outer_dim)
            other_items = [v if isinstance(v, Tensor) else compatible_tensor(v, compat_shape=remaining_shape, compat_natives=self._natives(), convert=False) for v in other]
            other_stacked = stack(other_items, outer_dim, expand_values=True)
            return other_stacked
        else:
            return compatible_tensor(other, compat_shape=self.shape, compat_natives=self._natives(), convert=False)

    def _op1(self, native_function) -> 'Tensor':
        """
        Transform the values of this tensor given a function that can be applied to any native tensor.

        Args:
          native_function:

        Returns:

        """
        raise NotImplementedError(self.__class__)

    def _op2(self, other, op: Callable, switch_args: bool) -> 'Tensor':
        """
        Apply a broadcast operation on two tensors.

        Args:
            other: second argument
            op: Operator function (a, b) -> c, used to propagate the operation to children tensors to have Python choose the callee

        Returns:
            `Tensor`
        """
        raise NotImplementedError(self.__class__)

    def _natives(self) -> tuple:
        raise NotImplementedError(self.__class__)

    def _spec_dict(self) -> dict:
        raise NotImplementedError(self.__class__)

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        raise NotImplementedError(cls)

    def _simplify(self):
        """ Does not cache this value but if it is already cached, returns the cached version. """
        return self


TensorOrTree = TypeVar('TensorOrTree', Tensor, PhiTreeNode, Number, bool, tuple, list, dict, Any)


class TensorDim(BoundDim):
    """
    Reference to a specific dimension of a `Tensor`.

    To obtain a `TensorDim`, use `Tensor.dimension()` or the syntax `tensor.<dim>`.

    Indexing a `TensorDim` as `tdim[start:stop:step]` returns a sliced `Tensor`.

    See the documentation at https://tum-pbs.github.io/PhiML/Introduction.html#Slicing .
    """

    def __init__(self, tensor: Tensor, name: str):
        super().__init__(tensor, name)
        self.tensor = tensor

    def __len__(self):
        warnings.warn("Use Tensor.dim.size instead of len(Tensor.dim). len() only supports with integer sizes.", DeprecationWarning)
        return self.size

    @property
    def dual(self):
        return TensorDim(self.tensor, '~' + self.name)

    @property
    def index(self):
        return self.tensor.shape.index(self.name)

    def split(self, split_dimensions: Shape):
        """ See `phiml.math.unpack_dim()` """
        warnings.warn("dim.split() is deprecated. Use math.split_dims() instead.", stacklevel=2)
        return unpack_dim(self.tensor, self.name, split_dimensions)

    def __matmul__(self, other):
        from ._ops import dot
        if isinstance(other, BoundDim):
            return dot(self.obj, (self.name,), other.obj, (other.name,))
        if isinstance(other, (tuple, list)):
            other = wrap(other, self.obj.shape[self.name])
        if isinstance(other, Tensor):
            assert self.name in other.shape, f"Canno reduce '{self.name}' of tensor with shape {self.obj.shape} against tensor with shape {other.shape}. Dimension must be present on both tensors."
            return dot(self.tensor, (self.name,), other, (self.name,))
        else:
            return NotImplemented

    __rmul__ = __mul__ = __rmatmul__ = __matmul__

    def sum(self):
        from ._ops import sum_
        return sum_(self.tensor, self.name)

    def prod(self):
        from ._ops import prod
        return prod(self.tensor, self.name)


_EQUALITY_REDUCE = [{'type': 'elementwise'}]


@contextmanager
def equality_by_ref():
    """
    Enables Tensor.__bool__
    """
    _EQUALITY_REDUCE.append({'type': 'ref'})
    try:
        yield None
    finally:
        assert _EQUALITY_REDUCE.pop(-1) == {'type': 'ref'}


@contextmanager
def equality_by_shape_and_value(rel_tolerance=0., abs_tolerance=0., equal_nan=False):
    """
    Enables Tensor.__bool__
    """
    spec = {'type': 'shape_and_value', 'rel_tolerance': rel_tolerance, 'abs_tolerance': abs_tolerance, 'equal_nan': equal_nan}
    _EQUALITY_REDUCE.append(spec)
    try:
        yield None
    finally:
        assert _EQUALITY_REDUCE.pop(-1) == spec


class Layout(Tensor):
    """
    Tensor representation of a PyTree consisting of only lists, tuples and leaves.
    Leaves can be any Python object or primitive, including tuples and lists.
    The PyTree may be deeper but only the outer `shape.rank` levels are represented as a tensor.
    """

    def __init__(self, obj, stack_dim: Shape):
        super().__init__()
        self._obj = obj
        obj_shapes = Layout._recursive_get_shapes(obj, stack_dim)
        self._shape = shape_stack(stack_dim, *obj_shapes, stack_dim_first=True)
        self._stack_dim = stack_dim
        if DEBUG_CHECKS:
            if self._stack_dim:
                assert stack_dim == self._shape[:stack_dim.rank]
            elif isinstance(obj, Shapable) and obj is not None:
                warnings.warn(f"Empty stack_dim for Layout with value {obj}")

    @staticmethod
    def _recursive_get_shapes(obj, s: Shape) -> Tuple[Shape]:
        if not s:
            return shape(obj, allow_unshaped=True),
        elif isinstance(obj, (tuple, list)):
            return sum([Layout._recursive_get_shapes(o, after_gather(s, {s.names[0]: i})) for i, o in enumerate(obj)], ())
        elif isinstance(obj, dict):
            return sum([Layout._recursive_get_shapes(v, after_gather(s, {s.names[0]: i})) for i, (k, v) in enumerate(obj.items())], ())
        obj_shape = shape(obj, allow_unshaped=True)
        return (obj_shape,) * s.volume

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> DType:
        if isinstance(self._obj, bool):
            return BOOL
        if isinstance(self._obj, int):
            return INT64
        elif isinstance(self._obj, (float, complex)):
            return DType(type(self._obj), precision=64)
        else:
            return DType(object)

    @property
    def default_backend(self):
        return None

    def native(self, order: Union[str, tuple, list, Shape] = None, force_expand=True, to_numpy=False):
        if order is not None:
            order = parse_dim_order(order)
            assert order == self._stack_dim.names, "Layout.native() does not allow for changing the dimension order"
        native = self._obj
        return numpy.asarray(native) if to_numpy else native

    def _getitem(self, selection: dict) -> 'Tensor':
        selection_list = [selection.get(dim, None) for dim in self._stack_dim.names]
        native = self._getitem_recursive(self._obj, tuple(selection_list), selection)
        return Layout.wrap(native, after_gather(self._stack_dim, selection))

    @staticmethod
    def wrap(native, stack_dim: Shape):
        if isinstance(native, Tensor):
            return native
        if isinstance(native, Shapable) and native is not None and not isinstance(native, (tuple, list, dict)):
            # maybe allow class to configure whether to be unpacked
            return native
        if isinstance(native, (bool, Number)):
            return wrap(native)
        return Layout(native, stack_dim)

    def __repr__(self):
        return repr(self._obj)

    def __format__(self, format_spec):
        if BROADCAST_FORMATTER.values is not None:
            return BROADCAST_FORMATTER.register_formatted(self, format_spec)
        return repr(self._obj)

    def _unstack(self, dimension: str):
        if dimension == self._stack_dim.names[0]:
            native = tuple(self._obj.values()) if isinstance(self._obj, dict) else self._obj
            inner_stack_dim = self._stack_dim[1:]
            return tuple([Layout.wrap(n, inner_stack_dim) for n in native])
        else:
            raise NotImplementedError()

    @staticmethod
    def _getitem_recursive(native, selection: tuple, sel_dict: dict):
        if not selection:
            return native
        native = tuple(native.values()) if isinstance(native, dict) else native
        if len(selection) == 1:
            return slice_(native if selection[0] is None else native[selection[0]], sel_dict)
        else:
            if selection[0] is None:
                return type(native)([Layout._getitem_recursive(n, selection[1:], sel_dict) for n in native])
            if isinstance(selection[0], int):
                return Layout._getitem_recursive(native[selection[0]], selection[1:], sel_dict)
            elif isinstance(selection[0], slice):
                subset = native[selection[0]]
                return type(subset)([Layout._getitem_recursive(n, selection[1:], sel_dict) for n in subset])
            else:
                raise ValueError(f"Illegal selection: {selection}")

    def _as_list(self):
        return self._as_list_recursive(self._obj, self._stack_dim.rank, [])

    @staticmethod
    def _as_list_recursive(native, dims: int, result: list):
        if dims == 0:
            result.append(native)
        else:
            native = tuple(native.values()) if isinstance(native, dict) else native
            for n in native:
                Layout._as_list_recursive(n, dims - 1, result)
        return result

    @property
    def _is_tracer(self) -> bool:
        return False

    def __bool__(self):
        assert self.rank == 0, f"Cannot convert tensor with non-empty shape {self.shape} to bool. Use tensor.any or tensor.all instead."
        return bool(self._obj)

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'Layout':
        obj = [v.native(self._stack_dim) for v in values]
        new_stack_dim = dim + self._stack_dim
        return Layout(obj, new_stack_dim)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Shapable':
        return NotImplemented

    def __flatten__(self, flat_dim: Shape, flatten_batch: bool):
        return NotImplemented

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        new_stack_dims = dims.without(self._stack_dim)
        if not new_stack_dims:
            return self
        obj = self._obj
        for dim in reversed(new_stack_dims):
            assert isinstance(dim.size, int), "Can only expand layouts by integer-sized dimensions"
            obj = [obj] * dim.size
        return Layout(obj, new_stack_dims + self._stack_dim)

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Tensor':
        new_stack_dim = self._stack_dim.replace(dims, new_dims)
        def inner_replace(obj):
            return rename_dims(obj, dims, new_dims, **kwargs)
        obj = self._recursive_op1(self._obj, self._stack_dim, inner_replace)
        return Layout(obj, new_stack_dim)

    def __pack_dims__(self, dims: Shape, packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Layout':
        if dims.names == self._stack_dim.names:
            native = self._as_list()
            return Layout(native, packed_dim.with_size(len(native)))
        else:
            obj = []
            for i in dims.meshgrid():
                obj.append(self[i].native())
            return Layout(obj, packed_dim.with_size(dims.volume) + (self._stack_dim - dims))

    def __unpack_dim__(self, dim: str, unpacked_dims: Shape, **kwargs) -> 'Layout':
        return NotImplemented

    def __cast__(self, dtype: DType):
        obj = self._recursive_cast(self._obj, self._stack_dim, dtype)
        return Layout(obj, self._stack_dim)

    def __copy__(self):
        return Layout(self._obj, self._stack_dim)

    def __iter__(self):
        if self.rank == 1:
            return iter(self._obj)
        elif self.rank == 0:
            return iter([self._obj])
        else:
            return iter(self._as_list())

    def __eq__(self, other):
        if _EQUALITY_REDUCE[-1]['type'] != 'elementwise':
            return Tensor.__eq__(self, other)
        return self._op2(other, operator.eq, False)

    def __ne__(self, other):
        if _EQUALITY_REDUCE[-1]['type'] != 'elementwise':
            return Tensor.__ne__(self, other)
        return self._op2(other, operator.ne, False)

    def _assert_close(self, other: Tensor, rel_tolerance: float, abs_tolerance: float, msg: str, verbose: bool):
        from ._ops import assert_close
        inner_test = lambda x, y: assert_close(x, y, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, msg=msg, verbose=verbose)
        return self._op2(other, inner_test, False)

    def _op2(self, other, op: Callable, switch_args: bool) -> 'Tensor':
        obj = self._recursive_op2(self._obj, self._stack_dim, other, op)
        new_stack = self._stack_dim + (other._stack_dim - self._stack_dim) if isinstance(other, Layout) else self._stack_dim
        return Layout(obj, new_stack)

    @staticmethod
    def _recursive_op2(obj, shape: Shape, other: Tensor, operator: Callable):
        if shape:
            dim = shape.names[0]
            if isinstance(other, Tensor) and dim in other.shape:
                assert other.shape.get_size(dim) == len(obj), f"Shape mismatch during {operator.__name__}: '{dim}' has size {len(obj)} on layout but {other.shape.get_size(dim)} on other tensor."
                others = [other[{dim: i}] for i in range(len(obj))]
            else:
                others = [other] * len(obj)
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_op2(i, shape[1:], o, operator) for i, o in zip(obj, others)])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_op2(v, shape[1:], o, operator) for (k, v), o in zip(obj.items(), others)}
        else:  # leaf
            if isinstance(other, Layout) and not other.shape:
                return operator(obj, other.native())
            if isinstance(other, Tensor):
                return operator(obj, other)
            else:
                if obj is None or other is None:
                    return operator(obj, other)
                native_function = get_operator(operator, choose_backend(obj, other))
                return native_function(obj, other)

    def _op1(self, native_function):
        return Layout(self._recursive_op1(self._obj, self._stack_dim, native_function), self._stack_dim)

    @staticmethod
    def _recursive_op1(obj, shape: Shape, native_function):
        if shape:
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_op1(i, shape[1:], native_function) for i in obj])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_op1(v, shape[1:], native_function) for k, v in obj.items()}
            raise ValueError(obj)
        else:
            return native_function(obj)

    @staticmethod
    def _recursive_cast(obj, shape: Shape, dtype: DType):
        if shape:
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_cast(i, shape[1:], dtype) for i in obj])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_cast(v, shape[1:], dtype) for k, v in obj.items()}
            elif isinstance(obj, Tensor):
                assert obj.shape == shape
                from ._ops import cast
                return cast(obj, dtype)
            raise ValueError(obj)
        elif isinstance(obj, Tensor):
            from ._magic_ops import cast
            return cast(obj, dtype)
        else:
            return dtype.kind(obj)


class Dense(Tensor):
    """
    Tensor backed by a (possibly lower-rank) backend-specific tensor.
    The dimension names and types corresponding to the native tensor are stored in _native_shape.
    The property _shape can contain additional dimensions along which the tensor is constant.
    """

    def __init__(self, native_tensor, names: Sequence[str], expanded_shape: Shape, backend: Backend):
        super().__init__()
        self._native = native_tensor
        self._shape = expanded_shape
        self._names = names
        self._backend = backend
        if DEBUG_CHECKS:
            assert isinstance(names, (tuple, list)), f"names must be a tuple or list[str] but got {type(names)}"
            assert all(isinstance(n, str) for n in names), f"names must be a tuple or list[str] but got {names}"
            assert isinstance(backend, Backend)
            for dim in expanded_shape:
                if dim.size is not None and isinstance(dim.size, Tensor):
                    assert dim.size.rank > 0
                    for s_dim in dim.size.shape.names:
                        assert s_dim in expanded_shape.names, f"Dimension {dim} varies along {s_dim} but {s_dim} is not part of the Shape {self}"
            assert choose_backend(native_tensor) == backend
            assert expanded_shape.is_uniform, expanded_shape
            shape_sizes = [expanded_shape.get_size(n) for n in names]
            assert backend.staticshape(native_tensor) == tuple(shape_sizes), f"Shape {expanded_shape} at {names} does not match native tensor with shape {backend.staticshape(native_tensor)}"

    def native(self, order: Union[str, tuple, list, Shape] = None, force_expand=True):
        if order is None:
            assert len(self._shape) <= 1, f"When calling Tensor.native() or Tensor.numpy(), the dimension order must be specified for Tensors with more than one dimension, e.g. '{','.join(self._shape.names)}'. The listed default dimension order can vary depending on the chosen backend. Consider using math.reshaped_native(Tensor) instead."
            if len(self._names) == len(self._shape):
                return self.backend.auto_cast1(self._native)
            assert len(self._names) == 0  # shape.rank is 1
            return self.backend.tile(self.backend.expand_dims(self.backend.auto_cast(self._native)[0]), (self._shape.size,))
        if isinstance(order, str):
            return self._transposed_native(parse_dim_order(order), force_expand)
        if isinstance(order, (tuple, list)) and all(isinstance(o, str) for o in order):
            return self._transposed_native(parse_dim_order(order), force_expand)
        groups = process_groups_for(self._shape, order)
        return self._reshaped_native(groups)

    def _reshaped_native(self, groups: Sequence[Shape]):
        native = self.backend.auto_cast1(self._native)
        if not groups:
            return native
        perm = []
        slices = []
        tile = []
        for group in groups:
            for dim in group:
                if dim.name in self._names:
                    perm.append(self._names.index(dim.name))
                    slices.append(slice(None))
                    tile.append(1)
                else:
                    slices.append(None)
                    tile.append(dim.size)
        native = self._backend.transpose(native, perm)
        native = native[tuple(slices)]
        native = self._backend.tile(native, tile)
        native = self._backend.reshape(native, [g.volume if g.well_defined else -1 for g in groups])
        return native

    def _transposed_native(self, order: Sequence[str], force_expand: bool):
        backend = self._backend
        if order == self._names:
            if self.dtype.precision in [None, get_precision()]:
                return self._native
            else:
                return backend.cast(self._native, DType(self.dtype.kind, precision=get_precision()))
        # --- Transpose ---
        perm = [self._names.index(n) for n in order if n in self._names]
        if perm != list(range(len(perm))):
            transposed = backend.transpose(self._native, perm)  # this will cast automatically
        else:
            transposed = backend.as_tensor(self._native)
        if len(order) == len(perm):
            return transposed  # nothing to expand
        # --- Expand ---
        slices = [slice(None) if n in self._names else None for n in order]
        expanded = transposed[tuple(slices)]
        if force_expand:
            multiples = [self._shape.get_size(dim) if dim in self._shape and dim not in self._names else 1 for dim in order]
            expanded = backend.tile(expanded, multiples)
        return expanded

    def _contiguous(self):
        if len(self._names) == len(self._shape):
            return self
        expanded = self.native(order=self._shape)
        return Dense(expanded, self._shape, self._shape, self._backend)

    def _cached(self, dims: Shape = None) -> 'Dense':
        if len(self._names) == len(self._shape):  # nothing to expand
            return self
        elif dims is None or len(self._shape) == len(set(self._names) | set(dims.names)):  # expand all
            return Dense(self.native(order=self._shape), self._shape.names, self._shape, self._backend)
        else:  # expand specific dims
            new_native_shape = dims & self._shape[self._names]
            tmp_tensor = Dense(self._native, self._names, new_native_shape, self._backend)
            return Dense(tmp_tensor.native(new_native_shape), new_native_shape.names, self._shape, self._backend)

    def __pack_dims__(self, dims: Shape, packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Tensor':
        groups = [dim for dim in self.shape if dim not in dims]
        i0 = self.shape.index(dims[0])
        groups.insert(i0, dims)
        native = self._reshaped_native(groups)
        if pos is None:
            pos = min(self.shape.indices(dims.names))
        packed_dim = packed_dim.with_sizes([dims.volume])
        remaining = self.shape - dims
        new_shape = concat_shapes_(remaining[:pos], packed_dim, remaining[pos:])
        native = self.backend.reshape(native, new_shape.sizes)
        return Dense(native, new_shape.names, new_shape, self.backend)

    @property
    def collapsed_dims(self):
        return self._shape.without(self._names)

    @property
    def dtype(self):
        return choose_backend(self._native).dtype(self._native)

    @property
    def shape(self):
        return self._shape

    @property
    def backend(self) -> Backend:
        return self._backend

    default_backend = backend

    def _with_shape_replaced(self, new_shape):
        if new_shape.rank != self._shape.rank:
            raise IncompatibleShapes(f"Tensor {self} is not compatible with shape {new_shape}", self._shape, new_shape)
        new_shape = new_shape.with_sizes(self._shape.sizes)
        name_map = {old: new for old, new in zip(self._shape.names, new_shape.names)}
        names = [name_map[n] for n in self._names]
        return Dense(self._native, names, new_shape, self._backend)

    def _with_natives_replaced(self, natives: list):
        native = natives.pop(0)
        if DEBUG_CHECKS:
            sizes = choose_backend(native).shape(native)
            old_sizes = self._backend.shape(self._native)
            assert sizes == old_sizes or (not sizes and not old_sizes), f"Shape of new _native {sizes} does not match old one {self._backend.shape(self._native)}"  # TF sees empty shapes as non-equal
        # new_shape = self._shape.with_sizes(new_native_shape)
        return Dense(native, self._names, self._shape, self._backend)

    @property
    def _is_tracer(self) -> bool:
        return False

    def _to_dict(self):
        result = self.shape._to_dict(include_sizes=False)
        if self.rank == 0:
            result['data'] = self.numpy().item()
        else:
            result['data'] = self.numpy(self._shape).tolist()  # works for all 1+ dimensional arrays
        return result

    def _getitem(self, selection: dict):
        if not selection:
            return self
        selections = [slice(None)] * len(self._names)
        for name, sel in selection.items():
            assert isinstance(name, str)
            if name in self._names:
                selections[self._names.index(name)] = sel
            elif name not in self._shape:
                assert isinstance(sel, int), f"Attempting slice missing dimension {name} with {selection}"
        gathered = self.default_backend.multi_slice(self._native, tuple(selections)) if selections else self._native
        new_shape = after_gather(self._shape, selection)
        names = [n for n in self._names if n in new_shape]
        return Dense(gathered, names, new_shape, self._backend)

    def _unstack(self, dim: str):
        new_shape = self._shape.without(dim)
        if dim in self._names:
            tensors = self.default_backend.unstack(self._native, axis=self._names.index(dim))
            new_names = [n for n in self._names if n != dim]
            return tuple([Dense(t, new_names, new_shape, self._backend) for t in tensors])
        else:
            assert dim in self._shape, f"Cannot unstack tensor {self._shape} along non-existant dimension '{dim}'"
            return (Dense(self._native, self._names, new_shape, self._backend),) * self._shape.get_size(dim)

    def _op1(self, native_function):
        native = native_function(self._native)
        return Dense(native, self._names, self._shape, self._backend) if native is not None else self

    def _op2(self, other, op: Callable, switch_args: bool):
        if isinstance(other, (bool, int, float, complex)):
            n1, n2 = (other, self._backend.auto_cast1(self._native)) if switch_args else (self._backend.auto_cast1(self._native), other)
            try:
                result = op(n1, n2)
            except (ExtraOperator, TypeError):
                b_fun = get_operator(op, self._backend)
                result = b_fun(n1, n2)
            except Exception as err:
                warnings.warn(f"Direct op2 failed with arguments {type(n1)}, {type(n2)}: {err}")
                b_fun = get_operator(op, self._backend)
                result = b_fun(n1, n2)
            return Dense(result, self._names, self._shape, self._backend)
        was_other_tensor = isinstance(other, Tensor)
        if not was_other_tensor:
            try:
                other = self._tensor(other)
            except NoBackendFound:
                return NotImplemented
        if not isinstance(other, Dense) and was_other_tensor:
            return NotImplemented
        if not isinstance(other, Dense):
            other = Dense(other.native(other.shape), other.shape.names, other.shape, self._backend)
        backend = backend_for(self, other)
        backend_op = get_operator(op, backend)
        if backend_op is None:
            return op(self, other) if switch_args else op(other, self)
        if other._names == self._names:
            nat1, nat2 = self._native, other._native
            names = self._names
            if other._shape == self._shape:
                r_shape = self._shape
            else:
                r_shape = self._shape & other._shape
        else:
            first_names = [n for n in self._names if n not in other._names]
            names = first_names + list(other._names)
            nat1 = self._transposed_native(names, False)
            nat2 = other._native
            r_shape = self._shape & other._shape
        result_nat = backend_op(nat2, nat1) if switch_args else backend_op(nat1, nat2)
        return Dense(result_nat, names, r_shape, backend)

    def _natives(self) -> tuple:
        return self._native,

    def _spec_dict(self) -> dict:
        return {'type': Dense, 'names': self._names, 'shape': self._shape}

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        names: Sequence[str] = spec['names']
        expanded_shape: Shape = spec['shape']
        native = natives.pop(0)
        # --- update sizes in case JIT compilation gave None or outdated size ---
        sizes = {n: s for n, s in zip(names, choose_backend(native).staticshape(native))}
        expanded_shape = expanded_shape.with_sizes([sizes.get(n, s) for n, s in zip(expanded_shape.names, expanded_shape.sizes)])
        return Dense(native, names, expanded_shape, choose_backend(native))


class TensorStack(Tensor):
    """
    Implicit stack of multiple tensors.
    List of tensors, does not store stacked tensor in memory.

    Args:

    Returns:

    """

    def __init__(self, components: Union[tuple, list], stack_dim: Shape):
        assert isinstance(stack_dim, SHAPE_TYPES) and stack_dim.rank == 1, f"stack_dim must be a single-dimension Shape object but got {type(stack_dim)}"
        # assert len(components) > 1, "Use a CollapsedTensor instead"
        for t in components:
            assert isinstance(t, Tensor)
            assert stack_dim.name not in t.shape, f"Cannot stack along '{stack_dim.name}' because the dimension already exists."
        self._tensors = tuple(components)
        self._stack_dim = stack_dim.with_sizes([len(components)], keep_item_names=True)
        try:
            merge_shapes(*self._tensors)
            self._varying_shapes = False
        except IncompatibleShapes:
            self._varying_shapes = True
        self._shape = shape_stack(self._stack_dim, *[t.shape for t in self._tensors])
        self._backend = backend_for(*components)

    @property
    def _is_tracer(self) -> bool:
        return any([t._is_tracer for t in self._tensors])

    @property
    def requires_broadcast(self):
        if self._varying_shapes or not self._shape.well_defined or self._is_tracer or self._tensors[0].shape.is_non_uniform:
            return True
        from ._sparse import is_sparse
        return is_sparse(self)
    
    @property
    def stack_dim(self):
        warnings.warn("TensorStack.stack_dim is deprecated. Use Shape.non_uniform instead.", DeprecationWarning, stacklevel=2)
        return self._stack_dim

    @property
    def dtype(self):
        return combine_types(*[t.dtype for t in self._tensors])

    @property
    def shape(self):
        return self._shape

    @property
    def backend(self) -> Backend:
        return self._backend

    def native(self, order: Union[str, tuple, list, Shape] = None, force_expand=True):
        if force_expand:
            return self._reshaped_native(process_groups_for(self._shape, order))
        # ToDo
        return self._reshaped_native(process_groups_for(self._shape, order))

    def _transposed_native(self, order: Sequence[str], force_expand: bool):
        assert not self.requires_broadcast, f"_transposed_native requires uniform Tensor but has shape {self._shape}"
        return self._contiguous()._transposed_native(order, force_expand)

    def _reshaped_native(self, groups: Sequence[Shape]):
        if not self.requires_broadcast:
            return self._contiguous()._reshaped_native(groups)
        stack_group = [g for g in groups if self._stack_dim in g]
        assert len(stack_group) == 1
        stack_group = stack_group[0]
        stack_index = groups.index(stack_group)
        stack_dim = stack_group.only(self._stack_dim, reorder=True)
        # ToDo assert self._stack_dim are first dims of stack_group
        other_packed = stack_group[len(stack_dim):]
        assert other_packed, f"Getting native of non-uniform tensors requires non-uniform dims {self._shape.non_uniform_shape} to be grouped with stack dim {self._stack_dim} but got {groups}"
        inner_groups = list(groups)
        if len(other_packed) > 1:
            raise NotImplementedError
            # tensors = [t.__pack_dims__() for t in self._tensors]
        natives = []
        for idx, t in zip(self._stack_dim.meshgrid(), self._tensors):
            inner_groups[stack_index] = other_packed.after_gather(idx)
            natives.append(t._reshaped_native(inner_groups))
        native = self.backend.concat(natives, stack_index)
        return native

        # # Is only the stack dimension shifted?
        # if self._shape.without(self._stack_dim).names == tuple(filter(lambda name: name != self._stack_dim.name, order)):
        #     inner_order = [dim for dim in order if dim != self._stack_dim.name]
        #     natives = [t.native(inner_order) for t in self._tensors]
        #     assert self._stack_dim.name in order, f"Dimension {self._stack_dim} missing from 'order'. Got {order} but tensor has shape {self.shape}."
        #     native = choose_backend(*natives).stack(natives, axis=order.index(self._stack_dim.name))
        #     return native
        # assert not self.shape.is_non_uniform, f"Cannot convert non-uniform tensor with shape {self.shape} to native tensor."
        # return self._contiguous().native(order=order, force_expand=force_expand)

    def _contiguous(self) -> Tensor:
        assert not self.requires_broadcast
        if all([t.shape.is_uniform for t in self._tensors]):
            natives = [t.native(order=self._shape.names) for t in self._tensors]
            native = self.backend.concat(natives, axis=self.shape.index(self._stack_dim.name))
            return Dense(native, self._shape.names, self._shape, self.backend)
        else:  # cache stack_dim on inner tensors
            non_uniform_dim = self._tensors[0].shape.non_uniform_shape
            if len(non_uniform_dim) > 1:
                raise NotImplementedError
            unstacked = [t._unstack(non_uniform_dim.name) for t in self._tensors]
            stacked = []
            for to_stack in zip(*unstacked):
                tensor = TensorStack(to_stack, self._stack_dim)._contiguous()
                stacked.append(tensor)
            return TensorStack(stacked, non_uniform_dim)

    def __pack_dims__(self, dims: Shape, packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Tensor':
        from ._ops import concat_tensor
        value = cached(self)
        assert isinstance(value, TensorStack)
        inner_packed = [pack_dims(t, dims, packed_dim) for t in value._tensors]
        return concat_tensor(inner_packed, packed_dim.name)

    def _with_shape_replaced(self, new_shape: Shape):
        new_stack_dim = new_shape[self._shape.index(self._stack_dim.name)]
        new_tensors = []
        for t in self._tensors:
            inner_indices = [self.shape.index(d) for d in t.shape.names]
            new_inner_shape = new_shape[inner_indices]
            new_tensors.append(t._with_shape_replaced(new_inner_shape))
        return TensorStack(new_tensors, new_stack_dim)

    def _getitem(self, selection: dict):
        if (self._stack_dim.name not in selection or len(selection) != 1) and not self.requires_broadcast:
            return self._contiguous()._getitem(selection)
        # --- Inner dims ---
        inner_dict = {dim: sel for dim, sel in selection.items() if dim != self._stack_dim.name}
        tensors = self._tensors
        if len(inner_dict) > 0:
            tensors = [t[inner_dict] for t in tensors]
        # --- stack dimension ---
        if self._stack_dim.name in selection:
            selection = selection[self._stack_dim.name]
            if isinstance(selection, slice):
                return TensorStack(tensors[selection], after_gather(self._stack_dim, {self._stack_dim.name: selection}))
            else:
                selection = int(selection)
                return tensors[selection]
        else:
            return TensorStack(tensors, self._stack_dim)

    def _unstack(self, dim: str):
        if dim == self._stack_dim.name:
            return self._tensors
        else:
            if self.requires_broadcast:
                unstacked = [t._unstack(dim) for t in self._tensors]
                return tuple([TensorStack(items, self._stack_dim) for items in zip(*unstacked)])
            else:
                return self._contiguous()._unstack(dim)

    def _op1(self, native_function):
        if self.requires_broadcast:
            tensors = [t._op1(native_function) for t in self._tensors]
            return TensorStack(tensors, self._stack_dim)
        else:
            return self._contiguous()._op1(native_function)

    def _op2(self, other, op, switch_args):
        other = self._tensor(other)
        if self.requires_broadcast:
            if self._stack_dim.name in other.shape:
                other_slices = other._unstack(self._stack_dim.name)
                tensors = [custom_op2(t1, t2, op, switch_args) for t1, t2 in zip(self._tensors, other_slices)]
            else:
                tensors = [custom_op2(t, other, op, switch_args) for t in self._tensors]
            return TensorStack(tensors, self._stack_dim)
        elif isinstance(other, Dense) or (isinstance(other, TensorStack) and not other.requires_broadcast):
            names, new_shape, (n1, n2) = broadcastable_native_tensors(self, other)  # ToDo we don't have to expand all
            if switch_args:
                n1, n2 = n2, n1
            native_function = get_operator(op, self.backend)
            native_result = native_function(n1, n2)
            return Dense(native_result, names, new_shape, backend_for(self, other))
        elif isinstance(other, TensorStack) and other.requires_broadcast:
            if other._stack_dim.name in self.shape:
                self_slices = self._unstack(other._stack_dim.name)
                tensors = [custom_op2(t1, t2, op, switch_args) for t1, t2 in zip(self_slices, other._tensors)]
            else:
                tensors = [custom_op2(self, t, op, switch_args) for t in other._tensors]
            return TensorStack(tensors, self._stack_dim)
        else:
            return NotImplemented

    def _natives(self) -> tuple:
        return sum([t._natives() for t in self._tensors], ())

    def _spec_dict(self) -> dict:
        return {'type': TensorStack, 'stack_dim': self._stack_dim, 'tensors': [t._spec_dict() for t in self._tensors]}

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        tensors = [t['type']._from_spec_and_natives(t, natives) for t in spec['tensors']]
        return TensorStack(tensors, spec['stack_dim'])

    def _with_natives_replaced(self, natives: list):
        tensors = [t._with_natives_replaced(natives) for t in self._tensors]
        return TensorStack(tensors, self._stack_dim)

    @property
    def is_cached(self):
        return False

    def _simplify(self):
        return self


def tensor(data,
           *shape: Union[Shape, str, list],
           convert: bool = True,
           default_list_dim=channel('vector')) -> Tensor:  # TODO assume convert_unsupported, add convert_external=False for constants
    """
    Create a Tensor from the specified `data`.
    If `convert=True`, converts `data` to the preferred format of the default backend.

    `data` must be one of the following:
    
    * Number: returns a dimensionless Tensor.
    * Native tensor such as NumPy array, TensorFlow tensor or PyTorch tensor.
    * `tuple` or `list` of numbers: backs the Tensor with native tensor.
    * `tuple` or `list` of non-numbers: creates tensors for the items and stacks them.
    * Tensor: renames dimensions and dimension types if `names` is specified. Converts all internal native values of the tensor if `convert=True`.
    * Shape: creates a 1D tensor listing the dimension sizes.
    
    While specifying `names` is optional in some cases, it is recommended to always specify them.
    
    Dimension types are always inferred from the dimension names if specified.

    Implementations:

    * NumPy: [`numpy.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
    * PyTorch: [`torch.tensor`](https://pytorch.org/docs/stable/generated/torch.tensor.html), [`torch.from_numpy`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html)
    * TensorFlow: [`tf.convert_to_tensor`](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor)
    * Jax: [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)

    See Also:
        `phiml.math.wrap()` which uses `convert=False`, `layout()`.

    Args:
        data: native tensor, sparse COO / CSR / CSC matrix, scalar, sequence, `Shape` or `Tensor`
        shape: Ordered dimensions and types. If sizes are defined, they will be checked against `data`.`
            You may also pass a single `str` specifying dimension in the format `name:t` or `name:t=(item_names)` where `t` refers to the type letter, one of s,i,c,d,b.
            Alternatively, you can pass a `list` of shapes which will call `reshaped_tensor`.
        convert: If True, converts the data to the native format of the current default backend.
            If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Raises:
        AssertionError: if dimension names are not provided and cannot automatically be inferred
        ValueError: if `data` is not tensor-like

    Returns:
        Tensor containing same values as data

    Examples:
        >>> tensor([1, 2, 3], channel(vector='x,y,z'))
        (x=1, y=2, z=3)

        >>> tensor([1., 2, 3], channel(vector='x,y,z'))
        (x=1.000, y=2.000, z=3.000) float64

        >>> tensor(numpy.zeros([10, 8, 6, 2]), batch('batch'), spatial('x,y'), channel(vector='x,y'))
        (batchᵇ=10, xˢ=8, yˢ=6, vectorᶜ=x,y) float64 const 0.0

        >>> tensor([(0, 1), (0, 2), (1, 3)], instance('particles'), channel(vector='x,y'))
        (x=0, y=1); (x=0, y=2); (x=1, y=3) (particlesⁱ=3, vectorᶜ=x,y)

        >>> tensor(numpy.random.randn(10))
        (vectorᶜ=10) float64 -0.128 ± 1.197 (-2e+00...2e+00)
    """
    if len(shape) == 1 and isinstance(shape[0], list):
        return reshaped_tensor(data, shape[0], convert=convert)
    shape = [parse_shape_spec(s) if isinstance(s, str) else s for s in shape]
    shape = None if len(shape) == 0 else concat_shapes_(*shape)
    if isinstance(data, SHAPE_TYPES):
        if shape is None:
            shape = channel('dims')
            shape = shape.with_size(data.names)
            data = data.sizes
        elif not shape:
            assert data.rank == 1, f"When wrapping a Shape as a scalar tensor, it must be a rank-1 shape but got {data}"
            data = data.size
        else:
            assert shape.rank == 1, "Can only convert 1D shapes to Tensors"
            shape = shape.with_size(data.names)
            data = data.sizes
    if isinstance(data, Tensor):
        if convert:
            backend = data.backend
            if backend != default_backend():
                data = data._from_spec_and_natives(data._spec_dict(), [convert_(n, use_dlpack=False) for n in data._natives()])
        if shape is None:
            return data
        else:
            if None in shape.sizes:
                shape = shape.with_sizes(data.shape.sizes)
            return data._with_shape_replaced(shape)
    elif isinstance(data, str) or data is None:
        return layout(data)
    elif isinstance(data, (Number, bool)):
        assert not shape, f"Trying to create a zero-dimensional Tensor from value '{data}' but shape={shape}"
        if convert:
            data = default_backend().as_tensor(data, convert_external=True)
        return Dense(data, (), EMPTY_SHAPE, default_backend() if convert else NUMPY)
    if isinstance(data, (tuple, list)):
        if all(isinstance(d, (bool, int, float, complex, np.generic)) for d in data):
            data = np.array(data)
            assert data.dtype != object
            data = NUMPY.auto_cast1(data)
        elif all(isinstance(d, str) for d in data):
            return layout(data, shape or default_list_dim)
        else:
            try:
                inner_shape = [] if shape is None else [shape[1:]]
                tensors = [d if isinstance(d, Tensor) else tensor(d, *inner_shape, convert=convert) for d in data]
                return stack(tensors, default_list_dim if shape is None else shape[0].with_sizes([len(tensors)]), expand_values=True)
            except IncompatibleShapes:
                assert not convert, f"Cannot convert {data} to tensor given shape {shape}"
                return layout(data, shape or default_list_dim)
            except ValueError:
                assert not convert, f"Cannot convert {data} to tensor"
                return layout(data, shape or default_list_dim)
    try:
        backend = choose_backend(data)
        sizes = backend.staticshape(data)
        if shape is None:
            assert backend.ndims(data) <= 1, "Specify dimension names for tensors with more than 1 dimension"
            shape = default_list_dim if backend.ndims(data) == 1 else EMPTY_SHAPE
            shape = shape.with_sizes(sizes)
        elif 0 not in sizes:
            # fill in sizes or check them
            if len(sizes) != len(shape):
                raise IncompatibleShapes(f"Rank of given shape {shape} does not match data with sizes {sizes}")
            for size, s in zip(sizes, shape.sizes):
                if s is not None:
                    assert s == size, f"Given shape {shape} does not match data with sizes {sizes}. Consider leaving the sizes undefined."
            shape = shape.with_sizes(sizes, keep_item_names=True)
        if backend.is_sparse(data):
            from ._sparse import from_sparse_native
            return from_sparse_native(data, shape, indices_constant=backend == NUMPY, convert=convert)
        elif convert:
            data = convert_(data, use_dlpack=False)
            backend = default_backend()
        if 0 in sizes:
            present_shape = shape[:len(sizes)].with_sizes(sizes)
            return Dense(data, present_shape.names, shape.with_sizes(shape.undefined.with_sizes(0)).with_sizes(present_shape), backend)
        return Dense(data, shape.names, shape, backend)
    except NoBackendFound:
        raise ValueError(f"{type(data)} is not supported. Only (Tensor, tuple, list, np.ndarray, native tensors) are allowed.\nCurrent backends: {BACKENDS}")


def wrap(data, *shape: Union[Shape, str, list], default_list_dim=channel('vector')) -> Tensor:
    """ Short for `phiml.math.tensor()` with `convert=False`. """
    return tensor(data, *shape, convert=False, default_list_dim=default_list_dim)


def layout(objects, *shape: Union[Shape, str]) -> Tensor:
    """
    Wraps a Python tree in a `Tensor`, allowing elements to be accessed via dimensions.
    A python tree is a structure of nested `tuple`, `list`, `dict` and *leaf* objects where leaves can be any Python object.

    All keys of `dict` containers must be of type `str`.
    The keys are automatically assigned as item names along that dimension unless conflicting with other elements.

    Strings may also be used as containers.

    Example:
    >>> t = layout({'a': 'text', 'b': [0, 1]}, channel('dict,inner'))
    >>> t.inner[1].dict['a'].native()
    'e'

    See Also:
        `tensor()`, `wrap()`.

    Args:
        objects: PyTree of `list` or `tuple`.
        *shape: Tensor dimensions

    Returns:
        `Tensor`.
        Calling `Tensor.native()` on the returned tensor will return `objects`.
    """
    shape = [parse_shape_spec(s) if isinstance(s, str) else s for s in shape]
    assert all(isinstance(s, SHAPE_TYPES) for s in shape), f"shape needs to be one or multiple Shape instances but got {shape}"
    shape = EMPTY_SHAPE if len(shape) == 0 else concat_shapes_(*shape)
    if isinstance(objects, Layout):
        assert objects.shape == shape
        return objects

    if not shape.well_defined:

        def recursive_determine_shape(native, shape: Shape):
            if not shape:
                return shape
            if isinstance(native, dict):
                assert all([isinstance(k, str) for k in native.keys()]), f"All dict keys in PyTrees must be str but got {tuple(native.keys())}"
                shape = shape.replace(shape[0], shape[0].with_size(tuple(native.keys())))
            if shape.rank == 1:
                return shape.with_sizes((len(native),))
            inner_shape = shape[1:]
            if isinstance(native, (tuple, list)):
                inner_shapes = [recursive_determine_shape(n, inner_shape) for n in native]
            elif isinstance(native, dict):
                inner_shapes = [recursive_determine_shape(n, inner_shape) for n in native.values()]
            else:
                raise ValueError(native)
            return shape_stack(shape[0], *inner_shapes)

        shape = recursive_determine_shape(objects, shape)

    return Layout(objects, shape)
    # if shape.volume == 1:
    #     objects = np.asarray(objects, dtype=object)
    #
    # if isinstance(objects, (tuple, list)):
    #     objects = np.asarray(objects, dtype=object)
    # if isinstance(objects, np.ndarray) and objects.dtype == object:
    #     return Layout(objects, shape)
    # else:
    #     assert shape.volume == 1, f"Cannot layout object of type {objects} along {shape}, a tuple, list or object array is required."


def compatible_tensor(data, compat_shape: Shape = None, compat_natives=(), convert=False):
    if isinstance(data, Tensor):
        return data
    elif isinstance(data, SHAPE_TYPES):
        if data.spatial.rank == 1:
            return wrap(data.spatial.size)
        assert compat_shape.channel.rank == 1, "Only single-channel tensors support implicit casting from Shape to tensor"
        assert data.rank == compat_shape.channel.volume
        return wrap(data.spatial.sizes, *compat_shape.channel.with_size(data.names))
    else:
        data_type = type(data)
        backend = choose_backend(*compat_natives, data)
        try:
            data = backend.as_tensor(data, convert_external=convert)
            shape = backend.staticshape(data)
        except ValueError as e:
            raise ValueError(e)
        if len(shape) == 0:
            return Dense(data, (), EMPTY_SHAPE, backend if convert else choose_backend(data))
        elif isinstance(data, (tuple, list)):  # always channel, add vector if not available
            data = backend.as_tensor(data)
        if len(shape) == compat_shape.channel_rank:
            other_tensor = wrap(data, compat_shape.channel)
            return other_tensor
        if compat_shape.channel_rank > 1 and len(shape) == 1 and 'vector' in compat_shape.channel:
            return wrap(data, compat_shape['vector'].without_sizes())
        elif len(shape) == compat_shape.rank:
            if len(shape) > 1:
                warnings.warn(f"Combining a phiml.math.Tensor with a {data_type} of same shape is not invariant under shape permutations. Please convert the {data_type} to a phiml.math.Tensor first. Shapes: {shape} and {compat_shape}", SyntaxWarning, stacklevel=5)
            return Dense(data, compat_shape.names, compat_shape.with_sizes(shape), backend)
        else:
            raise ValueError(f"Cannot combine tensor of shape {shape} with tensor of shape {compat_shape}")


def broadcastable_native_tensors(*tensors) -> Tuple[Sequence[str], Shape, Sequence]:
    """ Adds missing dims but does not expand to full size. """
    from ._sparse import SparseCoordinateTensor, CompressedSparseMatrix, dense
    if any(isinstance(t, (SparseCoordinateTensor, CompressedSparseMatrix)) for t in tensors) and not all(isinstance(t, (SparseCoordinateTensor, CompressedSparseMatrix)) for t in tensors):
        tensors = [dense(t) for t in tensors]
    var_names = tuple(set.union(*[set(variable_dim_names(t)) for t in tensors]))
    natives = [t._transposed_native(var_names, False) if t.rank > 0 else t.native(None, False) for t in tensors]
    broadcast_shape = merge_shapes(*tensors)
    return var_names, broadcast_shape, natives


def custom_op2(x: Union[Tensor, float], y: Union[Tensor, float], op: Callable, switch_args=False) -> Tensor:
    """
    Perform a custom operator on two tensors.
    This method first tries calling _op2() on the first tensor and if that fails, tries it on the second tensor.

    Args:
      x: Left argument
      y: Right argument
      op: Operator function taking two arguments. Should be contained in the Backend operator mapping.

    Returns:
        `Tensor`
    """
    if switch_args:
        x, y = y, x
    if isinstance(x, Tensor):
        result = x._op2(y, op, False)
        if result is not NotImplemented:
            return result
    elif isinstance(y, Tensor):
        result = y._op2(x, op, True)
        if result is not NotImplemented:
            return result
    x = wrap(x)
    y = wrap(y)
    result = x._op2(y, op, False)
    if result is NotImplemented:
        result = y._op2(x, op, True)
        if result is NotImplemented:
            raise NotImplementedError(f"Operation not supported between {type(x)} and {type(y)}")
    return result


def disassemble_tensors(tensors: Sequence[Tensor], expand: bool) -> Tuple[tuple, Tuple[Shape], tuple]:
    """
    Args:
        tensors: Tuple or list of Tensors.
        expand: Whether to add collapsed dimensions to the native tensors.

    Returns:
        natives: tuple of native tensors
        specs: Identification primitives from which the tensor can be reconstructed given the natives.
            One per tensor.
    """
    tensors = [cached(t) if isinstance(t, TensorStack) or expand else t for t in tensors]
    natives = sum([t._natives() for t in tensors], ())
    shapes = tuple([t.shape for t in tensors])
    specs = tuple([t._spec_dict() for t in tensors])
    return natives, shapes, specs


def assemble_tensors(natives: Union[tuple, list], specs: Union[Tuple[dict, ...], List[dict]]):
    natives = list(natives)
    result = []
    for spec in specs:
        t = spec['type']._from_spec_and_natives(spec, natives)
        result.append(t)
    return result


MISSING_TENSOR = '__missing__'
NATIVE_TENSOR = '__native__'


def disassemble_tree(obj: PhiTreeNodeType, cache: bool, attr_type=variable_attributes) -> Tuple[PhiTreeNodeType, List[Tensor]]:
    """
    Splits a nested structure of Tensors into the structure without the tensors and an ordered list of tensors.
    Native tensors will be wrapped in phiml.math.Tensors with default dimension names and dimension types `None`.

    See Also:
        `assemble_tree()`

    Args:
        obj: Nested structure of `Tensor` objects.
            Nested structures include: `tuple`, `list`, `dict`, `phiml.math.magic.PhiTreeNode`.
        cache: Whether to return cached versions of the tensors. This may reduce the number of native tensors required.

    Returns:
        empty structure: Same structure as `obj` but with the tensors replaced by `None`.
        tensors: Ordered `list` of all contained `Tensor` objects.
    """
    if obj is None:
        return MISSING_TENSOR, []
    elif isinstance(obj, Layout):
        keys, values = disassemble_tree(obj._obj, cache, attr_type)
        return {'__layout__': 1, 'stack_dim': obj._stack_dim._to_dict(False), 'obj': keys}, values
    elif isinstance(obj, Tensor):
        return None, [cached(obj) if cache else obj]
    elif isinstance(obj, (tuple, list)):
        keys = []
        values = []
        for item in obj:
            key, value = disassemble_tree(item, cache, attr_type)
            keys.append(key)
            values.extend(value)
        return (tuple(keys) if isinstance(obj, tuple) else keys), values
    elif isinstance(obj, dict):
        keys = {}
        values = []
        for name, item in obj.items():
            key, value = disassemble_tree(item, cache, attr_type)
            keys[name] = key
            values.extend(value)
        return keys, values
    elif dataclasses.is_dataclass(obj):
        from ..dataclasses._dataclasses import disassemble
        container, values = disassemble(obj, attr_type=attr_type)
        if cache:
            values = [cached(v) for v in values]
        return container, values
    elif isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        keys = {}
        values = []
        for attr in attributes:
            key, value = disassemble_tree(getattr(obj, attr), cache, attr_type)
            keys[attr] = key
            values.extend(value)
        return copy_with(obj, **keys), values
    else:  # native tensor?
        try:
            backend = choose_backend(obj)
            if backend == OBJECTS:
                return obj, []
            sizes = backend.staticshape(obj)
            dims = [Dim(f"dim{i}", s, CHANNEL_DIM, None) for i, s in enumerate(sizes)]
            shape = PureShape(CHANNEL_DIM, {dim.name: dim for dim in dims})
            return NATIVE_TENSOR, [Dense(obj, shape.names, shape, backend)]
        except NoBackendFound:
            return obj, []


def assemble_tree(obj: PhiTreeNodeType, values: List[Tensor], attr_type=variable_attributes) -> PhiTreeNodeType:
    """ Reverses `disassemble_tree()` given an empty nested structure and a list of tensors. """
    if isinstance(obj, str) and obj == MISSING_TENSOR:
        return None
    elif isinstance(obj, str) and obj == NATIVE_TENSOR:
        value = values.pop(0)
        assert isinstance(value, Dense), f"Failed to assemble tree structure. Encountered {value}"
        if isinstance(value._native, np.ndarray) and value.shape == EMPTY_SHAPE:  # this can be represented as a Python scalar, which leads to less conversion errors
            return value._native.item()
        return value._native
    elif obj is None:
        value = values.pop(0)
        assert isinstance(value, Tensor)
        return value
    elif isinstance(obj, list):
        return [assemble_tree(item, values, attr_type) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([assemble_tree(item, values, attr_type) for item in obj])
    elif isinstance(obj, dict) and '__layout__' in obj:
        content = assemble_tree(obj['obj'], values, attr_type)
        return Layout(content, Shape._from_dict(obj['stack_dim']))
    elif isinstance(obj, dict):
        return {name: assemble_tree(val, values, attr_type) for name, val in obj.items()}
    elif isinstance(obj, Tensor):
        return obj
    elif dataclasses.is_dataclass(obj):
        from ..dataclasses._dataclasses import DataclassTreeNode, assemble
        if isinstance(obj, DataclassTreeNode):
            return assemble(obj, values)
    if isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        values = {a: assemble_tree(getattr(obj, a), values, attr_type) for a in attributes}
        return copy_with(obj, **values)
    return obj


def attr_paths(obj: PhiTreeNodeType, attr_type: Callable, root: str) -> List[str]:
    if obj is None:
        return []
    elif isinstance(obj, Layout):
        return attr_paths(obj._obj, attr_type, f'{root}._obj')
    elif isinstance(obj, Tensor):
        return [root]
    elif isinstance(obj, (tuple, list)):
        paths = []
        for i, item in enumerate(obj):
            path = attr_paths(item, attr_type, f'{root}[{i}]')
            paths.extend(path)
        return paths
    elif isinstance(obj, dict):
        paths = []
        for name, item in obj.items():
            path = attr_paths(item, attr_type, f'{root}[{name}]')
            paths.extend(path)
        return paths
    elif isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        paths = []
        for attr in attributes:
            path = attr_paths(getattr(obj, attr), attr_type, f'{root}.{attr}')
            paths.extend(path)
        return paths
    else:  # native tensor?
        try:
            return [] if choose_backend(obj) == OBJECTS else [root]
        except NoBackendFound:
            return []


def attr_paths_from_container(obj: PhiTreeNodeType, attr_type: Callable, root: str) -> List[str]:
    if isinstance(obj, str) and obj == MISSING_TENSOR:
        return []
    elif isinstance(obj, str) and obj == NATIVE_TENSOR:
        return [root]
    elif obj is None:
        return [root]
    elif isinstance(obj, (tuple, list)):
        return sum([attr_paths_from_container(v, attr_type, f'{root}[{i}]') for i, v in enumerate(obj)], [])
    elif isinstance(obj, dict) and '__layout__' in obj:
        return attr_paths_from_container(obj['obj'], attr_type, f'{root}._obj')
    elif isinstance(obj, dict):
        return sum([attr_paths_from_container(v, attr_type, f'{root}[{k}]') for k, v in obj.items()], [])
    elif isinstance(obj, Tensor):
        raise RuntimeError("Tensor found in container. This should have been set to None by disassemble_tree()")
    elif dataclasses.is_dataclass(obj):
        from ..dataclasses._dataclasses import DataclassTreeNode
        if isinstance(obj, DataclassTreeNode):
            assert attr_type == obj.attr_type
            return sum([attr_paths_from_container(v, attr_type, f'{root}.{k}') for k, v in obj.extracted.items()], [])
    if isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        return sum([attr_paths_from_container(getattr(obj, k), attr_type, f'{root}.{k}') for k in attributes], [])
    return []


def cached(t: TensorOrTree) -> TensorOrTree:
    from ._sparse import SparseCoordinateTensor, CompressedSparseMatrix, CompactSparseTensor
    assert isinstance(t, (Tensor, PhiTreeNode)), f"All arguments must be Tensors but got {type(t)}"
    if isinstance(t, Dense):
        return t._cached()
    elif isinstance(t, TensorStack):
        inners = cached(t._tensors)
        if t.requires_broadcast:
            return TensorStack(inners, t._stack_dim)
        else:
            natives = [t.native(order=t.shape.names) for t in inners]
            b = choose_backend(*natives)
            native = b.stack(natives, axis=t.shape.index(t._stack_dim.name))
            return Dense(native, t.shape.names, t.shape, b)
    elif isinstance(t, SparseCoordinateTensor):
        return SparseCoordinateTensor(cached(t._indices), cached(t._values), t._dense_shape, t._can_contain_double_entries, t._indices_sorted, t._indices_constant, t._matrix_rank)
    elif isinstance(t, CompressedSparseMatrix):
        return CompressedSparseMatrix(cached(t._indices), cached(t._pointers), cached(t._values), t._uncompressed_dims, t._compressed_dims, t._indices_constant, t._uncompressed_offset, t._uncompressed_indices, t._uncompressed_indices_perm, t._matrix_rank)
    elif isinstance(t, CompactSparseTensor):
        return CompactSparseTensor(cached(t._indices), cached(t._values), t._compressed_dims, t._indices_constant, t._matrix_rank)
    elif isinstance(t, Layout):
        return t
    elif isinstance(t, PhiTreeNode):
        tree, tensors = disassemble_tree(t, cache=True)
        return assemble_tree(tree, tensors)
    else:
        raise AssertionError(f"Cannot cache {type(t)} {t}")


def expand_tensor(value: Tensor, dims: Shape):
    if not dims:
        return value
    assert dims.well_defined
    if isinstance(value, Dense):
        if dims.is_uniform:
            return Dense(value._native, value._names, dims & value._shape, value._backend)
        else:
            stack_dim = dims.non_uniform_shape
            if stack_dim.rank > 1:
                raise NotImplementedError(f"Higher-order non-uniform expand() not yet supported. Tried expanding {value.shape} by {dims}")
            unstacked_dims = [after_gather(dims, i) for i in stack_dim.meshgrid()]
            if stack_dim in value.shape:
                unstacked = unstack(value, stack_dim)
                components = [Dense(inner._native, inner._names, inner_shape & inner._shape[inner._names], value._backend) for inner_shape, inner in zip(unstacked_dims, unstacked)]
            else:
                components = [Dense(value._native, value._names, inner_shape & value._shape[value._names], value._backend) for inner_shape in unstacked_dims]
            return TensorStack(components, stack_dim)
    if isinstance(value, TensorStack):
        expanded = [expand_tensor(v, after_gather(dims, {value._stack_dim.name: i})) for i, v in enumerate(value._tensors)]
        return TensorStack(expanded, value.stack_dim)
    if value._is_tracer:
        from ._trace import expand_tracer
        return expand_tracer(value, dims)
    raise NotImplementedError


class Dict(dict):
    """
    Dictionary of `Tensor` or `phiml.math.magic.PhiTreeNode` values.
    Dicts are not themselves tensors and do not have a shape.
    Use `layout()` to treat `dict` instances like tensors.

    In addition to dictionary functions, supports mathematical operators with other `Dict`s and lookup via `.key` syntax.
    `Dict` implements `phiml.math.magic.PhiTreeNode` so instances can be passed to math operations like `sin`.
    """

    def __value_attrs__(self):
        return tuple(self.keys())
    
    # --- Dict[key] ---

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)
        
    # --- operators ---
    
    def __neg__(self):
        return Dict({k: -v for k, v in self.items()})
    
    def __invert__(self):
        return Dict({k: ~v for k, v in self.items()})
    
    def __abs__(self):
        return Dict({k: abs(v) for k, v in self.items()})
    
    def __round__(self, n=None):
        return Dict({k: round(v) for k, v in self.items()})

    def __add__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val + other[key] for key, val in self.items()})
        else:
            return Dict({key: val + other for key, val in self.items()})

    def __radd__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] + val for key, val in self.items()})
        else:
            return Dict({key: other + val for key, val in self.items()})

    def __sub__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val - other[key] for key, val in self.items()})
        else:
            return Dict({key: val - other for key, val in self.items()})

    def __rsub__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] - val for key, val in self.items()})
        else:
            return Dict({key: other - val for key, val in self.items()})

    def __mul__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val * other[key] for key, val in self.items()})
        else:
            return Dict({key: val * other for key, val in self.items()})

    def __rmul__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] * val for key, val in self.items()})
        else:
            return Dict({key: other * val for key, val in self.items()})

    def __truediv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val / other[key] for key, val in self.items()})
        else:
            return Dict({key: val / other for key, val in self.items()})

    def __rtruediv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] / val for key, val in self.items()})
        else:
            return Dict({key: other / val for key, val in self.items()})

    def __floordiv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val // other[key] for key, val in self.items()})
        else:
            return Dict({key: val // other for key, val in self.items()})

    def __rfloordiv__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] // val for key, val in self.items()})
        else:
            return Dict({key: other // val for key, val in self.items()})

    def __pow__(self, power, modulo=None):
        assert modulo is None
        if isinstance(power, Dict):
            return Dict({key: val ** power[key] for key, val in self.items()})
        else:
            return Dict({key: val ** power for key, val in self.items()})

    def __rpow__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] ** val for key, val in self.items()})
        else:
            return Dict({key: other ** val for key, val in self.items()})

    def __mod__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val % other[key] for key, val in self.items()})
        else:
            return Dict({key: val % other for key, val in self.items()})

    def __rmod__(self, other):
        if isinstance(other, Dict):
            return Dict({key: other[key] % val for key, val in self.items()})
        else:
            return Dict({key: other % val for key, val in self.items()})

    def __eq__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val == other[key] for key, val in self.items()})
        else:
            return Dict({key: val == other for key, val in self.items()})

    def __ne__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val != other[key] for key, val in self.items()})
        else:
            return Dict({key: val != other for key, val in self.items()})

    def __lt__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val < other[key] for key, val in self.items()})
        else:
            return Dict({key: val < other for key, val in self.items()})

    def __le__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val <= other[key] for key, val in self.items()})
        else:
            return Dict({key: val <= other for key, val in self.items()})

    def __gt__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val > other[key] for key, val in self.items()})
        else:
            return Dict({key: val > other for key, val in self.items()})

    def __ge__(self, other):
        if isinstance(other, Dict):
            return Dict({key: val >= other[key] for key, val in self.items()})
        else:
            return Dict({key: val >= other for key, val in self.items()})

    # --- overridden methods ---

    def copy(self):
        return Dict(self)

def native(value: Union[Tensor, Number, tuple, list, Any]):
    """
    Returns the native tensor representation of `value`.
    If `value` is a `phiml.math.Tensor`, this is equal to calling `phiml.math.Tensor.native()`.
    Otherwise, checks that `value` is a valid tensor object and returns it.

    Args:
        value: `Tensor` or native tensor or tensor-like.

    Returns:
        Native tensor representation

    Raises:
        ValueError if the tensor cannot be transposed to match target_shape
    """
    if isinstance(value, Tensor):
        return value.native()
    else:
        choose_backend(value)  # check that value is a native tensor
        return value


def numpy_(value: Union[Tensor, Number, tuple, list, Any]):
    """
    Converts `value` to a `numpy.ndarray` where value must be a `Tensor`, backend tensor or tensor-like.
    If `value` is a `phiml.math.Tensor`, this is equal to calling `phiml.math.Tensor.numpy()`.

    *Note*: Using this function breaks the autograd chain. The returned tensor is not differentiable.
    To get a differentiable tensor, use `Tensor.native()` instead.

    Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
    If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

    If `value` is a NumPy array, it may be returned directly.

    Returns:
        NumPy representation of `value`

    Raises:
        ValueError if the tensor cannot be transposed to match target_shape
    """
    if isinstance(value, Tensor):
        return value.numpy()
    else:
        backend = choose_backend(value)
        return backend.numpy(value)


def reshaped_native(value: Tensor,
                    groups: Union[tuple, list],
                    force_expand: Any = True,
                    to_numpy=False):
    """
    Returns a native representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `native()`, `pack_dims()`.

    Args:
        value: `Tensor`
        groups: `tuple` or `list` of dimensions to be packed into one native dimension. Each entry must be one of the following:

            * `str`: the name of one dimension that is present on `value`.
            * `Shape`: Dimensions to be packed. If `force_expand`, missing dimensions are first added, otherwise they are ignored.
            * Filter function: Packs all dimensions of this type that are present on `value`.
            * Ellipsis `...`: Packs all remaining dimensions into this slot. Can only be passed once.
            * `None` or `()`: Adds a singleton dimension.

            Collections of or comma-separated dims may also be used but only if all dims are present on `value`.

        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.
        to_numpy: If True, converts the native tensor to a `numpy.ndarray`.

    Returns:
        Native tensor with dimensions matching `groups`.
    """
    warnings.warn("phiml.math.reshaped_native() is deprecated. Use Tensor.native() instead.", DeprecationWarning, stacklevel=2)
    assert isinstance(value, Tensor), f"value must be a Tensor but got {value} {type(value)}"
    assert not value._is_tracer, f"Failed accessing native values because tensor {value.shape} is a tracer"
    return value.numpy(groups, force_expand=force_expand) if to_numpy else value.native(groups, force_expand=force_expand)


def process_groups_for(shape: Shape, groups: Any) -> List[Shape]:
    if callable(groups):
        return list(groups(shape))
    def process_group(g):  # returns Shape or Ellipsis
        if g is None or (isinstance(g, tuple) and len(g) == 0):
            return EMPTY_SHAPE
        if isinstance(g, SHAPE_TYPES):
            return g
        if g is Ellipsis:
            return g
        if callable(g):
            return g(shape)
        g = parse_dim_order(g)
        if len(g) > 1:
            for name in g:
                assert name in shape, f"When specifying a group by dim names, all dims must be present but {name} is not part of {shape}"
        return shape.only(g, reorder=True)
    groups = [process_group(g) for g in groups]
    if Ellipsis in groups:
        ellipsis_dims = shape.without([g for g in groups if g is not Ellipsis])
        groups = [ellipsis_dims if g is Ellipsis else g for g in groups]
    return groups


def reshaped_numpy(value: Tensor, groups: Union[tuple, list], force_expand: Any = True) -> np.ndarray:
    """
    Returns the NumPy representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `numpy()`, `reshaped_native()`, `pack_dims()`, `reshaped_tensor()`.

    Args:
        value: `Tensor`
        groups: Sequence of dimension names as `str` or groups of dimensions to be packed_dim as `Shape`.
        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.

    Returns:
        NumPy `ndarray` with dimensions matching `groups`.
    """
    warnings.warn("phiml.math.reshaped_numpy() is deprecated. Use Tensor.numpy() instead.", DeprecationWarning, stacklevel=2)
    return reshaped_native(value, groups, force_expand=force_expand, to_numpy=True)


def reshaped_tensor(value: Any,
                    groups: Union[tuple, list],
                    check_sizes=False,
                    convert=True):
    """
    Creates a `Tensor` from a native tensor or tensor-like whereby the dimensions of `value` are split according to `groups`.

    See Also:
        `phiml.math.tensor()`, `reshaped_native()`, `unpack_dim()`.

    Args:
        value: Native tensor or tensor-like.
        groups: Sequence of dimension groups to be packed_dim as `tuple[Shape]` or `list[Shape]`.
        check_sizes: If True, group sizes must match the sizes of `value` exactly. Otherwise, allows singleton dimensions.
        convert: If True, converts the data to the native format of the current default backend.
            If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Returns:
        `Tensor` with all dimensions from `groups`
    """
    assert all(isinstance(g, SHAPE_TYPES) for g in groups), "groups must be a sequence of Shapes"
    v_shape = choose_backend(value).staticshape(value)
    dims = [batch(f'group{i}') if group.rank != 1 else (group if check_sizes else group.with_size(v_shape[i])) for i, group in enumerate(groups)]
    try:
        value = tensor(value, *dims, convert=convert)
    except IncompatibleShapes:
        raise IncompatibleShapes(f"Cannot reshape native tensor {type(value)} with sizes {value.shape} given groups {groups}")
    for i, group in enumerate(groups):
        if group.rank != 1:
            if value.shape.get_size(f'group{i}') == group.volume:
                value = unpack_dim(value, f'group{i}', group)
            elif check_sizes:
                raise AssertionError(f"Group {group} does not match dimension {i} of value {value.shape}")
            else:
                value = unpack_dim(value, f'group{i}', group)
    return value


def to_dict(value: Union[Tensor, Shape]):
    """
    Returns a serializable form of a `Tensor` or `Shape`.
    The result can be written to a JSON file, for example.

    See Also:
        `from_dict()`.

    Args:
        value: `Tensor` or `Shape`

    Returns:
        Serializable Python tree of primitives
    """
    if isinstance(value, SHAPE_TYPES):
        return value._to_dict(include_sizes=True)
    elif isinstance(value, Tensor):
        return value._to_dict()
    raise ValueError(f"Cannot convert {value} to a dict")


def from_dict(dict_: dict, convert=False):
    """
    Loads a `Tensor` or `Shape` from a serialized form.

    See Also:
        `to_dict()`.

    Args:
        dict_: Serialized tensor properties.
        convert: Whether to convert the data to the current backend format or keep it as a Numpy array.

    Returns:
        `Tensor` or `Shape`.
    """
    shape = Shape._from_dict(dict_)
    if 'data' in dict_:
        return tensor(dict_['data'], shape, convert=convert)
    else:
        return shape




class BroadcastFormatter:
    """
    Usage documented in math.__init__.

    How it works:
    * -f calls __neg__ which tells tensors to call register_formatted() instead of formatting normally.
    * Then __sub__ is called which maps the actual string formatting.
    """

    def __init__(self):
        self.values: List[Tensor] = None

    def register_formatted(self, value: Tensor, format_spec: str):
        self.values.append(value)
        return "{" + f"{len(self.values) - 1}:{format_spec}" + "}"

    def format(self, other: str):
        assert isinstance(other, str), "math.f must be used on a string"
        from ._functional import map_
        if self.values is None:
            raise SyntaxError("Use the syntax -f-f\"{tensor}\". Leading '-' is missing.")
        result = map_(other.format, *self.values)
        self.values = None
        return result

    def __sub__(self, other):
        return self.format(other)

    def __neg__(self):
        if self.values is not None:
            raise SyntaxError("-f called twice without formatting string.")
        self.values = []
        return self


BROADCAST_FORMATTER = BroadcastFormatter()


@dataclass
class Color:
    name: str
    console_foreground_begin: str

    def __call__(self, obj, **kwargs):
        text = str(obj).replace(CONSOLE_END, self.console_foreground_begin)
        return f"{self.console_foreground_begin}{text}{CONSOLE_END if self.console_foreground_begin else ''}"


DEFAULT = Color("Default", '')
BLUE = Color("Blue", '\033[94m')
GREEN = Color("Green", '\033[92m')
YELLOW = Color("Yellow", '\033[93m')
GREY = Color("Grey", '\033[37m')
CONSOLE_END = '\033[0m'


@dataclass
class ColorScheme:
    value: Color
    shape: Color
    dtype: Color
    fine: Color


DEFAULT_COLORS = ColorScheme(BLUE, GREEN, YELLOW, GREY)
NO_COLORS = ColorScheme(DEFAULT, DEFAULT, DEFAULT, DEFAULT)


@dataclass
class PrintOptions:
    layout: str = 'auto'
    float_format: str = None
    threshold: int = 8
    colors: ColorScheme = None
    include_shape: bool = None
    include_dtype: bool = None

    def get_colors(self):
        if self.colors is True:
            return DEFAULT_COLORS
        elif self.colors is False:
            return NO_COLORS
        elif self.colors is not None:
            return self.colors
        else:  # None
            return DEFAULT_COLORS if check_is_printing() else NO_COLORS


def check_is_printing():
    import traceback, sys
    stack = traceback.extract_stack()
    for frame in stack:
        if "_pydevd_bundle\\pydevd_xml.py" in frame.filename or "_pydevd_bundle/pydevd_xml.py" in frame.filename:
            return False
    for frame in stack:
        if frame.line.strip().startswith('print('):
            return True
    if 'ipykernel' in sys.modules:
        return True
    return False


def format_summary(self: Tensor, options: PrintOptions) -> str:
    """
    Returns shape + dtype + content summary

    * `bool`: n / N True
    * `float`: mean ± std (min...max)
    """
    if not self.available:
        return format_tracer(self, options)
    from ._sparse import SparseCoordinateTensor, CompressedSparseMatrix
    if isinstance(self, (SparseCoordinateTensor, CompressedSparseMatrix)):
        return sparse_summary(self, options)
    colors = options.get_colors()
    tokens = []
    if self.shape if options.include_shape is None else options.include_shape:
        tokens.append(f"{colors.shape(self.shape)}")
    if is_unexpected_dtype(self.dtype) if options.include_dtype is None else options.include_dtype:
        tokens.append(f"{colors.dtype(self.dtype)}")
    try:
        if self.rank == 0:
            tokens.append(colors.value(self.numpy()))
        elif self.dtype.kind == bool:
            tokens.append(colors.value(f"{self.sum} / {self.shape.volume} True"))
        elif self.dtype.kind in (float, int):
            min_val, max_val, mean, max_val_nan = [float(self.default_backend.numpy(f)) for f in [self.finite_min, self.finite_max, self.finite_mean, self.max]]
            if min_val == max_val:
                if max_val_nan == max_val:
                    tokens.append(colors.value(f"const {mean:{options.float_format or ''}}"))
                else:
                    tokens.append(colors.value(f"const {mean:{options.float_format or ''}} / nan"))
            else:
                std = float(self.default_backend.numpy(self.std))
                if any([abs(val) < 0.001 or abs(val) > 1000 for val in [mean, std]]):
                    tokens.append(colors.value(f"{mean:{options.float_format or '.2e'}} ± {std:{options.float_format or '.1e'}}"))
                else:
                    tokens.append(colors.value(f"{mean:{options.float_format or '.3f'}} ± {std:{options.float_format or '.3f'}}"))
                tokens.append(colors.fine(f"({min_val:{options.float_format or '.0e'}}...{max_val:{options.float_format or '.0e'}})"))
        elif self.dtype.kind == complex:
            tokens.append(colors.value(f"|...| < {abs(self).max}"))
    except BaseException as err:
        tokens.append(f"failed to fetch values: {err}")
    return " ".join(tokens)


def sparse_summary(value: Tensor, options: PrintOptions) -> str:
    colors = options.get_colors()
    from ._sparse import get_format, CompressedSparseMatrix
    tokens = []
    if is_unexpected_dtype(value.dtype) if options.include_dtype is None else options.include_dtype:
        tokens.append(f"{colors.dtype(value.dtype)}")
    tokens.append("sparse " + get_format(value))
    if options.include_shape is not False:
        tokens.append(f"{colors.shape(value.shape)}")
    if isinstance(value, CompressedSparseMatrix) and value._uncompressed_offset is not None:
        num_valid = value._valid_mask().sum
        tokens.append(f"with {instance(value._values).volume} entries ({num_valid} valid):")
    else:
        tokens.append(f"with {instance(value._values).volume} entries:")
    tokens.append(format_summary(value._values, options))
    return " ".join(tokens)


def is_unexpected_dtype(dtype: DType):
    if dtype in [BOOL, INT32]:
        return False
    if dtype.kind == float and dtype.precision == get_precision():
        return False
    return True


def format_tracer(self: Tensor, options: PrintOptions) -> str:
    colors = options.get_colors()
    if self._is_tracer:
        return f"{colors.shape(self.shape)} {colors.dtype(self.dtype)} {colors.value(f'linear tracer for {self.default_backend}')}"
    else:
        return f"{colors.shape(self.shape)} {colors.dtype(self.dtype)} {colors.value(f'{self.default_backend} tracer')}"


def format_full(value: Tensor, options: PrintOptions) -> str:  # multi-line content
    if not value.available:
        return format_tracer(value, options)
    from ._sparse import is_sparse, dense
    if is_sparse(value):
        try:
            return format_full_sparse(value, options)
        except NotImplementedError:
            value = dense(value)
    import re
    colors = options.get_colors()
    dim_order = tuple(sorted(value.shape.spatial.names, reverse=True))
    lines = []
    formatter = {}
    if options.float_format:
        formatter['float_kind'] = ('{:' + options.float_format + '}').format
    with numpy.printoptions(threshold=np.inf, formatter=formatter):
        if value.shape.dual_rank > 0:  # matrix
            if options.include_shape is not None:
                lines.append(colors.shape(value.shape))
            if value.shape.dual_rank > 1:
                corresponding_primal = value.shape.only(spatial(','.join(dual(value).names)).names, reorder=True)
                if corresponding_primal:
                    value = pack_dims(value, corresponding_primal, corresponding_primal[0].type('&'.join(corresponding_primal.names)))
                value = pack_dims(value, dual, dual('&'.join(value.shape.dual.names)))
            dual_dim = dual(value).name
            primal = dual(value).as_spatial().name
            if primal not in value.shape:
                primal = non_batch(value).non_dual
            for b in batch(value).meshgrid(names=True):
                text = " " + np.array2string(value[b].numpy([primal, dual_dim]), separator=', ', max_line_width=np.inf) + " "
                text = re.sub('[\\[\\]]', '', text).replace(',', ' ')
                prefixes, prefix_len = prefix_indices(non_batch(value).non_dual, colors)
                if options.include_shape is not False:
                    for line, prefix in zip(text.split("\n"), prefixes):
                        lines.append(f"{prefix}  {colors.value(line)} along {colors.shape(dual_dim)}")
                else:
                    lines.append(colors.value(text))
        elif value.shape.spatial_rank == 0:  # no spatial or dual dimensions
            if options.include_shape is not None:
                lines.append(colors.shape(value.shape))
            if value.shape.rank <= 1:
                text = np.array2string(value.numpy(), separator=', ', max_line_width=np.inf)
                lines.append(' ' + re.sub('[\\[\\]]', '', text))
            else:
                text = np.array2string(value.numpy(value.shape), separator=', ', max_line_width=np.inf)
                lines.append(text)
        elif value.shape.spatial_rank in (1, 2):
            if value.shape.non_spatial.volume > 1:
                indices = [f"{colors.shape(', '.join(f'{name}={idx}' for name, idx in index_dict.items()))}" for index_dict in value.shape.non_spatial.meshgrid(names=True)]
                max_index_length = max(len(index) for index in indices)
            for i, index_dict in enumerate(value.shape.non_spatial.meshgrid(names=True)):
                row = ""
                if value.shape.non_spatial.volume > 1:
                    row += indices[i] + " " * (max_index_length - len(indices[i]) + 2)
                    if value.shape.spatial_rank == 2:
                        row += "\n"
                if value.shape.spatial_rank == 1:
                    text = np.array2string(value[index_dict].numpy(dim_order), separator=', ', max_line_width=np.inf)
                else:
                    text = " " + np.array2string(value[index_dict].numpy(dim_order)[::-1], separator=', ', max_line_width=np.inf)
                lines.append(row + colors.value(re.sub('[\\[\\]]', '', text)) + (f"  along {colors.shape(spatial(value))}" if options.include_shape is not False else ""))
        else:
            raise NotImplementedError('Can only print tensors with up to 2 spatial dimensions.')
    return "\n".join(lines)


def format_full_sparse(value: Tensor, options: PrintOptions) -> str:
    from ._ops import log10, ravel_index
    from ._sparse import stored_indices, stored_values
    colors = options.get_colors()
    right = dual(value) if dual(value) else non_batch(value)[-1]
    down = non_batch(value) - right
    kind = value.dtype.kind
    if kind == int:
        str_max = 2 + int(log10(value.max))
    elif kind == bool:
        str_max = 3 + int(log10(right.volume))
    else:
        raise NotImplementedError
    lines = []
    if value.shape.dual_rank > 0:  # matrix
        if options.include_shape is not None:
            lines.append(colors.shape(value.shape))
    data = [[" "] * (str_max * right.volume) for _ in range(down.volume)]
    for b in batch(value).meshgrid(names=True):
        idx = stored_indices(value[b])
        vals = stored_values(value[b]).numpy()
        cols = ravel_index(idx[right.name_list], right).numpy()
        rows = ravel_index(idx[down.name_list], down).numpy()
        for val, col, row in zip(vals, cols, rows):
            if kind == bool:
                val_str = f"[{col}]" if val else "[ ]"
            elif kind == int:
                val_str = str(val)
            else:
                raise NotImplementedError
            col *= str_max
            data[row][col:col+len(val_str)] = val_str
    data = ["".join(line) for line in data]
    prefixes, prefix_len = prefix_indices(down, colors)
    if kind != bool:  # add col index header
        header = [" "] * (str_max * right.volume)
        for i in range(right.volume):
            i_str = str(i)
            header[i*str_max:i*str_max+len(i_str)] = i_str
        if options.include_shape is not False:
            header += " " + colors.shape(down)
        lines.append(" "*prefix_len + "".join(header))
    if options.include_shape is not False:
        for line, prefix in zip(data, prefixes):
            lines.append(f"{prefix}{colors.value(line)} along {colors.shape(right)}")
    else:
        for line in data:
            lines.append(colors.value(line))
    return "\n".join(lines)


def prefix_indices(index_shape, colors: ColorScheme, pad=2):
    prefix_texts = [f"{', '.join(f'{name}={idx}' for name, idx in index_dict.items())}" for index_dict in index_shape.meshgrid(names=True)]
    prefixes = [f"{colors.shape(text)}" for text in prefix_texts]
    max_len = max(len(p) for p in prefix_texts) + pad
    prefixes = [p + " " * (max_len - len(t)) for p, t in zip(prefixes, prefix_texts)]
    return prefixes, max_len


def format_row(self: Tensor, options: PrintOptions) -> str:  # all values in a single line
    """
    Including shape:  (x=5, y=4) along vector
    Without shape: (5, 4)
    Auto: don't show if 'vector' but show item names

    Args:
        self:
        options:

    Returns:

    """
    if not self.available:
        return format_tracer(self, options)
    from ..backend import NUMPY
    from ._sparse import dense
    from ._ops import convert
    self = convert(self, NUMPY)
    with NUMPY:
        self = dense(self)
    colors = options.get_colors()
    if self.shape.rank == 1:
        content = _format_vector(self, options)
        is_vector = self.shape.name == 'vector' and self.shape.channel_rank == 1
        is_dual_vector = self.shape.name == '~vector'
        if (not is_vector and not is_dual_vector) if options.include_shape is None else options.include_shape:
            content += f" along {colors.shape(f'{self.shape.name}{SUPERSCRIPT[self.shape.dim_type]}')}"
        elif is_dual_vector:
            content = "~" + content
    else:
        if channel(self):
            rows = [_format_vector(self[b], options) for b in self.shape.non_channel.meshgrid()]
        else:
            rows = [_format_number(self[b].numpy(), options, self.dtype) for b in self.shape.non_channel.meshgrid()]
        content = "; ".join(rows)
        if options.include_shape is not False:
            content += " " + colors.shape(self.shape)
    if is_unexpected_dtype(self.dtype) if options.include_dtype is None else options.include_dtype:
        content += f" {colors.dtype(self.dtype)}"
    return content


def format_numpy(self: Tensor, options: PrintOptions) -> str:
    from ._sparse import dense
    self = dense(self)
    header = []
    colors = options.get_colors()
    if options.include_shape:
        header.append(colors.shape(self.shape))
    if options.include_dtype:
        header.append(colors.dtype(self.dtype))
    numpy_array = self.numpy(self.shape)
    formatter = {}
    if options.float_format:
        formatter['float_kind'] = ('{:' + options.float_format + '}').format
    with numpy.printoptions(threshold=options.threshold, formatter=formatter):
        content = colors.value(numpy_array)
    return " ".join(header) + "\n" + content if header else content


def _format_vector(self: Tensor, options: PrintOptions) -> str:
    colors = options.get_colors()
    if self.shape.rank > 1:
        self = flatten(self, channel('flat'))
    if self.shape.item_names[0] is not None and options.include_shape is not False:
        content = ", ".join([f"{item}={_format_number(number, options, self.dtype)}" for number, item in zip(self, self.shape.item_names[0])])
    else:
        content = ", ".join([_format_number(num, options, self.dtype) for num in self])
    return colors.value(f"({content})")


def _format_number(num, options: PrintOptions, dtype: DType):
    if options.float_format is not None:
        return format(num, options.float_format)
    if dtype.kind == int:
        return format(num, 'd')
    if dtype.kind == bool:
        return str(bool(num))
    if dtype.kind == float:
        return format(num, options.float_format or '.3f')
    return str(num)


def format_tensor(self: Tensor, options: PrintOptions) -> str:
    if not self.available:
        return format_tracer(self, options)
    if self.shape.is_non_uniform:
        return format_summary(self, options)
    if options.layout == 'auto':
        if not self.shape:
            return format_summary(self, options)
        if self.shape.volume is not None and self.shape.volume < options.threshold:
            return format_row(self, options)
        else:
            return format_summary(self, options)
    elif options.layout == 'summary':
        return format_summary(self, options)
    elif options.layout == 'full':
        return format_full(self, options)
    elif options.layout == 'row':
        return format_row(self, options)
    elif options.layout == 'numpy':
        return format_numpy(self, options)
    else:
        raise NotImplementedError(f"Layout '{options.layout}' is not supported.")


def is_scalar(value) -> bool:
    """
    Checks whether `value` has no dimensions.

    Args:
        value: `Tensor` or Python primitive or native tensor.

    Returns:
        `bool`
    """
    if isinstance(value, Tensor):
        return value.shape.rank == 0
    elif isinstance(value, Number):
        return True
    else:
        return len(choose_backend(value).staticshape(value)) == 0


def variable_shape(value):
    return value._shape.only(value._names) if isinstance(value, Dense) else shape(value)


def variable_dim_names(value):
    return value._names if isinstance(value, Dense) else shape(value).names


def may_vary_along(value: Tensor, dims: DimFilter):
    return variable_shape(value).only(dims).volume > 1


def object_dims(value):
    if isinstance(value, Layout):
        return value._stack_dim
    return EMPTY_SHAPE


def discard_constant_dims(value: Tensor):
    non_variable = value.shape.without(variable_shape(value))
    return value[{dim: 0 for dim in non_variable.names}]


def specs_equal(spec1, spec2):
    if isinstance(spec1, Tensor) or isinstance(spec2, Tensor):
        if isinstance(spec1, Tensor) and isinstance(spec2, Tensor):
            if not spec1.shape.is_compatible(spec2.shape):
                return False
            from ._ops import equal
            return equal(spec1, spec2, equal_nan=True)
        return False
    if isinstance(spec1, dict):
        return set(spec1) == set(spec2) and all([key in spec2 and specs_equal(spec1[key], spec2[key]) for key in spec1.keys()])
    if isinstance(spec1, (tuple, list)):
        return len(spec1) == len(spec2) and all([specs_equal(s1, s2) for s1, s2 in zip(spec1, spec2)])
    return spec1 == spec2


def save(file: str, obj):
    """
    Saves a `Tensor` or tree using NumPy.
    This function converts all tensors contained in `obj` to NumPy tensors before storing.
    Each tensor is given a name corresponding to its path within `obj`, allowing reading only specific arrays from the file later on.
    Pickle is used for structures, but no reference to `Tensor` or its sub-classes is included.

    See Also:
        `load()`.

    Args:
        file: Target file, will be stored as `.npz`.
        obj: `Tensor` or tree to store.
    """
    tree, tensors = disassemble_tree(obj, False, all_attributes)
    paths = attr_paths(obj, all_attributes, 'root')
    assert len(paths) == len(tensors)
    natives = [t._natives() for t in tensors]
    specs = [serialize_spec(t._spec_dict()) for t in tensors]
    native_paths = [[f'{p}:{i}' for i in range(len(ns))] for p, ns in zip(paths, natives)]
    all_natives = sum(natives, ())
    all_paths = sum(native_paths, [])
    all_np = [choose_backend(n).numpy(n) for n in all_natives]
    np.savez(file, tree=np.asarray(tree, dtype=object), specs=specs, paths=paths, **{p: n for p, n in zip(all_paths, all_np)})


def load(file: str):
    """
    Loads a `Tensor` or tree from a file previously written using `save`.

    All tensors are restored as NumPy arrays, not the backend-specific tensors they may have been written as.
    Use `convert()` to convert all or some of the tensors to a different backend.

    Args:
        file: File to read.

    Returns:
        Same type as what was written.
    """
    data = np.load(file, allow_pickle=True)
    all_np = {k: data[k] for k in data if k not in ['tree', 'specs', 'paths']}
    specs = [unserialize_spec(spec) for spec in data['specs'].tolist()]
    tensors = assemble_tensors(list(all_np.values()), specs)
    tree = data['tree'].tolist()  # this may require outside classes via pickle
    stored_paths = data['paths'].tolist()
    new_paths = attr_paths_from_container(tree, all_attributes, 'root')
    if tuple(stored_paths) != tuple(new_paths):
        lookup = {path: t for path, t in zip(stored_paths, tensors)}
        tensors = [lookup[p] for p in new_paths]
    return assemble_tree(tree, tensors, attr_type=all_attributes)


def serialize_spec(spec: dict):
    from ._sparse import SparseCoordinateTensor, CompactSparseTensor, CompressedSparseMatrix
    type_names = {Dense: 'dense', TensorStack: 'stack', CompressedSparseMatrix: 'compressed', SparseCoordinateTensor: 'coo', CompactSparseTensor: 'compact'}
    result = {}
    for k, v in spec.items():
        if k == 'type':
            result[k] = type_names[v]
        elif isinstance(v, dict):
            result[k] = serialize_spec(v)
        else:
            assert not isinstance(v, type)
            result[k] = v
    return result


def unserialize_spec(spec: dict):
    from ._sparse import SparseCoordinateTensor, CompactSparseTensor, CompressedSparseMatrix
    type_names = {Dense: 'dense', TensorStack: 'stack', CompressedSparseMatrix: 'compressed', SparseCoordinateTensor: 'coo', CompactSparseTensor: 'compact'}
    lookup = {v: k for k, v in type_names.items()}
    result = {}
    for k, v in spec.items():
        if k == 'type':
            result[k] = lookup[v]
        elif isinstance(v, dict):
            result[k] = unserialize_spec(v)
        else:
            result[k] = v
    return result


_BACKEND_RULES = {}  # Tuple[Backend...] -> Backend


def backend_for(*values: Tensor) -> Backend:
    """
    Chooses an appropriate backend based on the backends of `values`.

    Args:
        *values: Input tensors to some operation.

    Returns:
        `Backend` that is compatible with all `values´.

    Raises:
        `NoBackendFound`: If no backend exists that can handle all `values`.

    """
    backends = tuple([v.backend for v in values])
    result = _BACKEND_RULES.get(backends, None)
    if result is not None:
        return result
    natives = sum([v._natives() if isinstance(v, Tensor) else (v,) for v in values], ())
    result = _BACKEND_RULES[backends] = choose_backend(*natives)
    ML_LOGGER.debug(f"Caching new backend combination: {backends} -> {result}")
    return result


def preferred_backend_for(*values: Tensor) -> Backend:
    """
    Choose backend for given `Tensor` or native tensor values, preferring the default backend when possible.
    Backends need to be registered to be available, e.g. via `init()` or `use()`.

    Args:
        *values: Sequence of `Tensor`s, native tensors or constants.
        prefer_default: Whether to always select the default backend if it can work with `values`, see `default_backend()`.

    Returns:
        The selected `phiml.math.backend.Backend`
    """
    natives = sum([v._natives() if isinstance(v, Tensor) else (v,) for v in values], ())
    return choose_backend(*natives, prefer_default=True)
