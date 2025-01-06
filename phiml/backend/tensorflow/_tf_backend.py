import numbers
from functools import wraps, partial
from typing import List, Callable, Tuple, Union, Optional, Sequence

import keras
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.framework.errors_impl import NotFoundError

from .._backend import combined_dim, TensorType
from .._dtype import DType, to_numpy_dtype, from_numpy_dtype, BOOL
from .. import Backend, ComputeDevice, NUMPY
from ._tf_cuda_resample import resample_cuda, use_cuda


class TFBackend(Backend):

    def __init__(self):
        devices = [ComputeDevice(self, device.name, simple_device_type(device.device_type), device.memory_limit, -1, str(device), device.name) for device in device_lib.list_local_devices()]
        # Example refs: '/device:CPU:0'
        default_device_ref = '/' + os.path.basename(tf.zeros(()).device)
        default_device = None
        for device in devices:
            if device.ref == default_device_ref:
                default_device = device
        assert default_device is not None
        Backend.__init__(self, 'tensorflow', devices, default_device)

    def prefers_channels_last(self) -> bool:
        return True

    def _device_for(self, *values):
        devices = set(v.device for v in values if hasattr(v, 'device'))
        if len(devices) == 0:
            return tf.device(self._default_device.ref)
        elif len(devices) == 1:
            return tf.device(next(iter(devices)))
        else:
            return tf.device(self._default_device.ref)

    def nn_library(self):
        from . import nets
        return nets

    def seed(self, seed: int):
        tf.random.set_seed(seed)

    def is_module(self, obj):
        return isinstance(obj, keras.Model)

    def is_tensor(self, x, only_native=False):
        is_tf_tensor = isinstance(x, (tf.Tensor, tf.SparseTensor, tf.RaggedTensor, tf.Variable))
        # is_tf_tensor = tf.is_tensor(x) is True  # tf.is_tensor() also returns True for CenteredGrid..., tf.is_tensor() can return non-bool values which indicates not a Tensor
        if only_native:
            return is_tf_tensor
        else:
            return is_tf_tensor or NUMPY.is_tensor(x, only_native=False)

    def is_sparse(self, x) -> bool:
        return isinstance(x, tf.SparseTensor)

    def get_sparse_format(self, x) -> str:
        return 'coo' if isinstance(x, tf.SparseTensor) else 'dense'

    def as_tensor(self, x, convert_external=True):
        with self._device_for(x):
            if self.is_tensor(x, only_native=convert_external):
                return tf.identity(x)
            tensor = tf.convert_to_tensor(x)
            # --- Enforce Precision ---
            if not isinstance(tensor, numbers.Number):
                if isinstance(tensor, np.ndarray):
                    tensor = NUMPY.as_tensor(tensor)
                elif tensor.dtype.is_floating:
                    tensor = self.to_float(tensor)
            return tensor

    def is_available(self, tensor) -> bool:
        if self.is_tensor(tensor, only_native=True):
            return tf.executing_eagerly()
        else:
            return True

    def numpy(self, tensor):
        if self.is_sparse(tensor):
            assemble, parts = self.disassemble(tensor)
            return assemble(NUMPY, *[self.numpy(t) for t in parts])
        if tf.is_tensor(tensor):
            return tensor.numpy()
        return NUMPY.numpy(tensor)

    def disassemble(self, x) -> Tuple[Callable, Sequence[TensorType]]:
        if self.is_sparse(x):
            return lambda b, i, v: b.sparse_coo_tensor(i, v, x.shape), (x.indices, x.values)
        else:
            return lambda b, t: t, (x,)

    def device_of(self, tensor: TensorType):
        if self.is_sparse(tensor):
            return self.device_of(tensor.values)
        return tf.device(tensor.device)

    def to_dlpack(self, tensor):
        from tensorflow import experimental
        return experimental.dlpack.to_dlpack(tensor)

    def from_dlpack(self, capsule):
        from tensorflow import experimental
        with tf.device(self._default_device.ref):
            return experimental.dlpack.from_dlpack(capsule)

    def copy(self, tensor, only_mutable=False):
        if not only_mutable or tf.executing_eagerly():
            with self.device_of(tensor):
                return tf.identity(tensor)
        else:
            return tensor

    def get_device(self, tensor: TensorType) -> ComputeDevice:
        device_name = '/' + os.path.basename(tensor.device)
        return self.get_device_by_ref(device_name)

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        with tf.device(device.ref):
            result = tf.identity(tensor)
            assert self.get_device(result) == device
            return result

    def vectorized_call(self, f, *args, output_dtypes=None, **aux_args):
        with self.device_of(args[0]):
            batch_size = self.determine_size(args, 0)
            args = [self.tile_to(t, 0, batch_size) for t in args]
            if output_dtypes is None:
                output0 = f(*[t[0] for t in args], **aux_args)  # Call f to determine its output signature.
                output_dtypes = tf.nest.map_structure(lambda x: x.dtype, output0)
            else:
                output_dtypes = tf.nest.map_structure(lambda dtype: to_numpy_dtype(dtype), output_dtypes)
            return tf.map_fn(lambda vals: f(*vals, **aux_args), tuple(args), fn_output_signature=output_dtypes)

    def numpy_call(self, f, output_shapes, output_dtypes, *args, **aux_args):
        def aux_f(*args):
            args = [self.numpy(a) for a in args]
            return f(*args, **aux_args)
        with self._device_for(*args):
            if output_dtypes is None:
                output0 = f(*[t[0] for t in args], **aux_args)  # Call f to determine its output signature.
                output_dtypes = tf.nest.map_structure(lambda x: x.dtype, output0)
            else:
                output_dtypes = tf.nest.map_structure(lambda dtype: to_numpy_dtype(dtype), output_dtypes)
            result = tf.py_function(aux_f, args, output_dtypes)
            self.set_shapes_tree(result, output_shapes)
            return result

    def jit_compile(self, f: Callable) -> Callable:
        compiled = tf.function(f)
        return lambda *args: self.as_registered.call(compiled, *args, name=f"run jit-compiled '{f.__name__}'")

    def custom_gradient(self, f: Callable, gradient: Callable, get_external_cache: Callable = None, on_call_skipped: Callable = None) -> Callable:
        @tf.custom_gradient
        def tf_function(*args, **kwargs):
            def grad(*grad_args):
                return gradient(args, y, grad_args)
            y = f(*args, **kwargs)
            return y, grad
        return tf_function

    def transpose(self, tensor, axes):
        with self.device_of(tensor):
            return tf.transpose(tensor, perm=axes)

    def equal(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return tf.equal(x, y)

    def divide_no_nan(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return tf.math.divide_no_nan(x, y)

    def random_uniform(self, shape, low, high, dtype: Union[DType, None]):
        dtype = dtype or self.float_type
        tdt = to_numpy_dtype(dtype)
        with tf.device(self._default_device.ref):
            if dtype.kind != complex:
                return tf.random.uniform(shape, low, high, dtype=tdt)
            else:
                real = tf.cast(tf.random.uniform(shape, low.real, high.real, dtype=to_numpy_dtype(DType(float, dtype.precision))), tdt)
                imag = tf.cast(tf.random.uniform(shape, low.imag, high.imag, dtype=to_numpy_dtype(DType(float, dtype.precision))), tdt)
                return real + 1j * imag

    def random_normal(self, shape, dtype: DType):
        with tf.device(self._default_device.ref):
            return tf.random.normal(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def random_permutations(self, permutations: int, n: int):
        with tf.device(self._default_device.ref):
            ordered = tf.range(0, n)
            return tf.stack([tf.random.shuffle(ordered) for _ in range(permutations)])

    def range(self, start, limit=None, delta=1, dtype: DType = DType(int, 32)):
        with tf.device(self._default_device.ref):
            return tf.range(start, limit, delta, to_numpy_dtype(dtype))

    def tile(self, value, multiples):
        with self.device_of(value):
            if isinstance(multiples, (tuple, list)) and self.ndims(value) < len(multiples):
                value = self.expand_dims(value, axis=0, number=len(multiples) - self.ndims(value))
            return tf.tile(value, multiples)

    def repeat(self, x, repeats, axis: int, new_length=None):
        x = self.as_tensor(x)
        with self.device_of(x):
            return tf.repeat(x, repeats, axis)
        # new_length can be larger than the actual required size. So if we use it, we have to pad the data.
        # if new_length is not None and self.staticshape(result)[axis] is None:
        #     shape = list(self.staticshape(result))
        #     shape[axis] = new_length
        #     result = tf.ensure_shape(result, shape)

    def ravel_multi_index(self, multi_index, shape, mode: Union[str, int] = 'undefined'):
        if self.is_tensor(shape, only_native=True):
            with self.device_of(shape):
                shape = self.unstack(shape)
        return Backend.ravel_multi_index(self, multi_index, shape, mode=mode)

    def stack(self, values, axis=0):
        with self._device_for(*values):
            return tf.stack(values, axis=axis)

    def concat(self, values, axis):
        with self._device_for(*values):
            values = self.auto_cast(*values)
            return tf.concat(values, axis)

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        if mode == 'boundary' and np.all(np.array(pad_width) <= 1):
            mode = 'symmetric'
        if mode in ('constant', 'symmetric', 'reflect'):
            with self.device_of(value):
                constant_values = tf.cast(constant_values, value.dtype)
                return tf.pad(value, pad_width, mode.upper(), constant_values=constant_values)
        else:
            return NotImplemented

    def reshape(self, value, shape):
        with self.device_of(value):
            return tf.reshape(value, shape)

    def sum(self, value, axis=None, keepdims=False):
        with self.device_of(value):
            if self.dtype(value).kind == bool:
                value = self.to_int32(value)
            if axis is not None:
                if not isinstance(axis, int):
                    axis = list(axis)
            if isinstance(value, tf.SparseTensor):
                return tf.sparse.reduce_sum(value, axis=axis, keepdims=keepdims, output_is_sparse=False)
            if isinstance(value, (tuple, list)) and any([isinstance(x, tf.SparseTensor) for x in value]):
                result = value[0]
                for v in value[1:]:
                    result = tf.sparse.add(result, v, threshold=0)
                return result
            return tf.reduce_sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        with self.device_of(value):
            if axis is not None:
                if not isinstance(axis, int):
                    axis = list(axis)
            if value.dtype == bool:
                return tf.reduce_all(value, axis=axis)
            return tf.reduce_prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        with self._device_for(condition, x, y):
            x, y = self.auto_cast(x, y)
            condition = tf.cast(condition, tf.bool)
            return tf.where(condition, x, y)

    def nonzero(self, values, length=None, fill_value=-1):
        with self.device_of(values):
            result = tf.where(tf.not_equal(values, 0))
            if length is not None:
                result = self.pad_to(result, 0, length, fill_value)
            return result

    def mean(self, value, axis=None, keepdims=False):
        with self.device_of(value):
            if self.dtype(value).kind not in (float, complex):
                value = self.to_float(value)
            if axis is not None:
                if not isinstance(axis, int):
                    axis = list(axis)
            return tf.reduce_mean(value, axis, keepdims=keepdims)

    def grid_sample(self, grid, coordinates, extrapolation: str):
        assert extrapolation in ('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect'), extrapolation
        if use_cuda(grid):
            if self.staticshape(grid)[0] > self.staticshape(coordinates)[0]:
                assert self.staticshape(coordinates)[0] == 1
                coordinates = self.tile(coordinates, [self.staticshape(grid)[0], *[1] * (self.ndims(coordinates) - 1)])
            return resample_cuda(grid, coordinates, extrapolation)
        else:
            return NotImplemented

    def zeros(self, shape, dtype: DType = None):
        with tf.device(self._default_device.ref):
            return tf.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def zeros_like(self, tensor):
        with tf.device(self._default_device.ref):
            return tf.zeros_like(tensor)

    def ones(self, shape, dtype: DType = None):
        with tf.device(self._default_device.ref):
            return tf.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones_like(self, tensor):
        with tf.device(self._default_device.ref):
            return tf.ones_like(tensor)

    def meshgrid(self, *coordinates):
        with tf.device(self._default_device.ref):
            result = tf.meshgrid(*coordinates, indexing='ij')
            return result

    def linspace(self, start, stop, number):
        with tf.device(self._default_device.ref):
            return self.to_float(tf.linspace(start, stop, number))

    def tensordot(self, a, a_axes: Union[tuple, list], b, b_axes: Union[tuple, list]):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b, bool_to_int=True)
            return tf.tensordot(a, b, (a_axes, b_axes))

    def mul_matrix_batched_vector(self, A, b):
        with self._device_for(A, b):
            if isinstance(A, tf.SparseTensor):
                result_T = tf.sparse.sparse_dense_matmul(A, tf.transpose(b))  # result shape contains unknown size
                result = tf.transpose(result_T)
                result.set_shape(tf.TensorShape([b.shape[0], A.shape[0]]))
                return result
            else:
                return tf.transpose(tf.matmul(A, b, transpose_b=True))

    def einsum(self, equation, *tensors):
        with self._device_for(*tensors):
            return tf.einsum(equation, *tensors)

    def cumsum(self, x, axis: int):
        with self.device_of(x):
            return tf.cumsum(x, axis=axis, exclusive=False)

    def while_loop(self, loop: Callable, values: tuple, max_iter: Union[int, Tuple[int, ...], List[int]]):
        with self._device_for(*values):
            if isinstance(max_iter, (tuple, list)):  # stack traced trajectory, unroll until max_iter
                values = self.stop_gradient_tree(values)
                trj = [values] if 0 in max_iter else []
                for i in range(1, max(max_iter) + 1):
                    values = loop(*values)
                    if i in max_iter:
                        trj.append(values)  # values are not mutable so no need to copy
                    condition = values[0]
                    if self.is_available(condition) and not self.any(values[0]):
                        break
                trj.extend([trj[-1]] * (len(max_iter) - len(trj)))  # fill trj with final values
                return self.stop_gradient_tree(self.stack_leaves(trj))
            else:
                cond = lambda c, *vals: tf.reduce_any(tf.cast(c, tf.bool))
                return self.stop_gradient_tree(tf.while_loop(cond, loop, values, maximum_iterations=max_iter))

    def stop_gradient_tree(self, tree):
        return tf.nest.map_structure(tf.stop_gradient, tree)

    def set_shapes_tree(self, values, shapes):
        if isinstance(values, (tuple, list)):
            for e, s in zip(values, shapes):
                self.set_shapes_tree(e, s)
        elif self.is_tensor(values, only_native=False):
            if self.is_tensor(values, only_native=True):
                values.set_shape(shapes)
        else:
            raise NotImplementedError(type(values))

    def abs(self, x):
        with self.device_of(x):
            return tf.abs(x)

    def sign(self, x):
        with self.device_of(x):
            return tf.sign(x)

    def round(self, x):
        with self.device_of(x):
            return tf.round(x)

    def ceil(self, x):
        with self.device_of(x):
            return tf.math.ceil(x)

    def floor(self, x):
        with self.device_of(x):
            return tf.floor(x)

    def max(self, x, axis=None, keepdims=False):
        with self.device_of(x):
            if isinstance(x, (tuple, list)):
                x = tf.stack(x)
            if x.dtype == tf.bool:
                return tf.cast(tf.reduce_max(tf.cast(x, tf.uint8), axis=axis, keepdims=keepdims), tf.bool)  # reduce_max allows no bool
            return tf.reduce_max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        with self.device_of(x):
            if isinstance(x, (tuple, list)):
                x = tf.stack(x)
            if x.dtype == tf.bool:
                return tf.cast(tf.reduce_min(tf.cast(x, tf.uint8), axis=axis, keepdims=keepdims), tf.bool)  # reduce_min allows no bool
            return tf.reduce_min(x, axis=axis, keepdims=keepdims)

    def maximum(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return tf.maximum(a, b)

    def minimum(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return tf.minimum(a, b)

    def clip(self, x, minimum, maximum):
        with self._device_for(x, minimum, maximum):
            x, minimum, maximum = self.auto_cast(x, minimum, maximum)
            return tf.clip_by_value(x, minimum, maximum)

    def argmax(self, x, axis: int, keepdims=False):
        with self._device_for(x):
            result = tf.argmax(x, axis)
            if keepdims:
                result = self.expand_dims(result, axis)
            return result

    def argmin(self, x, axis: int, keepdims=False):
        with self._device_for(x):
            result = tf.argmin(x, axis)
            if keepdims:
                result = self.expand_dims(result, axis)
            return result

    def sqrt(self, x):
        with self.device_of(x):
            return tf.sqrt(x)

    def exp(self, x):
        with self.device_of(x):
            return tf.exp(x)

    def erf(self, x):
        with self.device_of(x):
            return tf.math.erf(x)

    def softplus(self, x):
        with self.device_of(x):
            return tf.math.softplus(x)

    def log_gamma(self, x):
        with self.device_of(x):
            return tf.math.lgamma(self.to_float(x))

    def gamma_inc_l(self, a, x):
        with self._device_for(a, x):
            return tf.math.igamma(a, x)

    def gamma_inc_u(self, a, x):
        with self._device_for(a, x):
            return tf.math.igammac(a, x)

    def conv(self, value, kernel, zero_padding=True):
        with self._device_for(value, kernel):
            value = self.to_float(value)
            kernel = self.to_float(kernel)  # should use auto_cast but TensorFlow only supports DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32
            if zero_padding:
                value_padding = [[0, 0]] * 2 + [[s // 2, (s - 1) // 2] for s in kernel.shape[3:]]
                value = tf.pad(value, value_padding)
            convf = {3: partial(tf.nn.conv1d, stride=1),
                     4: partial(tf.nn.conv2d, strides=[1, 1, 1, 1]),
                     5: partial(tf.nn.conv3d, strides=[1, 1, 1, 1, 1])}[len(value.shape)]
            value = tf.transpose(value, [0, *range(2, self.ndims(value)), 1])  # could use data_format='NC...' but it's supported neither on CPU and for int tensors
            kernel = tf.transpose(kernel, [0, *range(3, self.ndims(kernel)), 2, 1])
            if kernel.shape[0] == 1:
                result = convf(value, kernel[0, ...], padding='VALID')
            else:
                result = []
                for b in range(kernel.shape[0]):
                    result.append(convf(value[b:b+1, ...], kernel[b], padding='VALID'))
                result = tf.concat(result, 0)
            result = tf.transpose(result, [0, self.ndims(result) - 1, *range(1, self.ndims(result) - 1)])
            return result

    def expand_dims(self, a, axis=0, number=1):
        a = self.as_tensor(a)
        with self.device_of(a):
            if number == 0:
                return a
            for _i in range(number):
                a = tf.expand_dims(a, axis)
            return a

    def shape(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor.shape
        else:
            with self.device_of(tensor):
                return tf.shape(tensor)

    def staticshape(self, tensor):
        if self.is_tensor(tensor, only_native=True):
            return tuple(tensor.shape.as_list())
        else:
            return np.shape(tensor)

    def gather(self, values, indices, axis: int):
        with self._device_for(values, indices):
            indices = indices % self.cast(self.shape(values)[axis], self.dtype(indices))
            return tf.gather(values, indices, axis=axis)

    def gather_by_component_indices(self, values, *component_indices):
        indices = self.stack(component_indices, -1)
        return tf.gather_nd(values, indices)

    def batched_gather_nd(self, values, indices):
        with self._device_for(values, indices):
            values_shape = self.staticshape(values)
            # --- tile values/indices as needed ---
            if values_shape[0] is not None and self.staticshape(indices)[0] is not None:  # does not work without knowing the batch dims
                if values_shape[0] == 1 and self.staticshape(indices)[0] > 1:
                    result = tf.gather_nd(values[0, ...], indices, batch_dims=0)
                    return result
                if values_shape[0] > 1 and self.staticshape(indices)[0] == 1:
                    indices = tf.tile(indices, [values_shape[0]] + [1] * (len(values_shape) - 1))
            return tf.gather_nd(values, indices, batch_dims=1)

    def unstack(self, tensor, axis=0, keepdims=False):
        with self.device_of(tensor):
            unstacked = tf.unstack(tensor, axis=axis)
            if keepdims:
                unstacked = [self.expand_dims(c, axis=axis) for c in unstacked]
            return unstacked

    def std(self, x, axis=None, keepdims=False):
        with self.device_of(x):
            if self.dtype(x).kind not in (float, complex):
                x = self.to_float(x)
            _mean, var = tf.nn.moments(x, axis, keepdims=keepdims)
            return tf.sqrt(var)

    def boolean_mask(self, x, mask, axis=0, new_length=None, fill_value=0):
        with self._device_for(x, mask):
            return tf.boolean_mask(x, mask, axis=axis)

    def isfinite(self, x):
        if self.dtype(x).kind in (bool, int):
            return self.ones(self.shape(x), dtype=BOOL)
        with self.device_of(x):
            return tf.math.is_finite(x)

    def isnan(self, x):
        if self.dtype(x).kind in (bool, int):
            return self.zeros(self.shape(x), dtype=BOOL)
        with self.device_of(x):
            return tf.math.is_nan(x)

    def isinf(self, x):
        if self.dtype(x).kind in (bool, int):
            return self.zeros(self.shape(x), dtype=BOOL)
        with self.device_of(x):
            return tf.math.is_inf(x)

    def any(self, boolean_tensor, axis=None, keepdims=False):
        with self.device_of(boolean_tensor):
            if self.dtype(boolean_tensor).kind != bool:
                boolean_tensor = tf.not_equal(boolean_tensor, 0)
            return tf.reduce_any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        with self.device_of(boolean_tensor):
            if self.dtype(boolean_tensor).kind != bool:
                boolean_tensor = tf.not_equal(boolean_tensor, 0)
            return tf.reduce_all(boolean_tensor, axis=axis, keepdims=keepdims)

    def quantile(self, x, quantiles):
        try:
            import tensorflow_probability as tfp
        except ModuleNotFoundError:
            return NotImplemented
        with self.device_of(x):
            x = self.to_float(x)
            result = tfp.stats.percentile(x, quantiles * 100, axis=-1, interpolation='linear')
            return result

    def argsort(self, x, axis=-1):
        with self.device_of(x):
            return tf.argsort(x, axis)

    def sort(self, x, axis=-1):
        with self.device_of(x):
            return tf.sort(x, axis)

    def searchsorted(self, sorted_sequence, search_values, side: str, dtype=DType(int, 32)):
        with self._device_for(sorted_sequence, search_values):
            return tf.searchsorted(sorted_sequence, search_values, side=side, out_type=to_numpy_dtype(dtype))

    def scatter(self, base_grid, indices, values, mode: str):
        out_kind = self.combine_types(self.dtype(base_grid), self.dtype(values)).kind
        with self._device_for(base_grid, indices, values):
            base_grid, values = self.auto_cast(base_grid, values, bool_to_int=mode == 'add')
            indices = self.as_tensor(indices)
            _batch_size = combined_dim(combined_dim(indices.shape[0], values.shape[0]), base_grid.shape[0])
            scatter = {'add': tf.tensor_scatter_nd_add, 'update': tf.tensor_scatter_nd_update, 'max': tf.tensor_scatter_nd_max, 'min': tf.tensor_scatter_nd_min}[mode]
            def scatter_single(b_grid, b_indices, b_values):
                return scatter(b_grid, b_indices, b_values)
            result = self.vectorized_call(scatter_single, base_grid, indices, values, output_dtypes=self.dtype(base_grid))
            if self.dtype(result).kind != out_kind:
                if out_kind == bool:
                    result = self.cast(result, BOOL)
            return result

    def histogram1d(self, values, weights, bin_edges):
        with self._device_for(values, weights, bin_edges):
            bin_count = self.staticshape(bin_edges)[-1] - 1
            bin_indices = tf.minimum(tf.searchsorted(bin_edges, values, side='right') - 1, bin_count - 1)  # ToDo this includes values outside
            hist = tf.math.bincount(bin_indices, weights=weights, minlength=bin_count, maxlength=bin_count, axis=-1)
            return hist

    def bincount(self, x, weights: Optional[TensorType], bins: int, x_sorted=False):
        # if x_sorted:
        #     return tf.math.segment_sum(weights or 1, x)
        # else:
        with self._device_for(x, weights):
            x = tf.cast(x, tf.int32)
            return tf.math.bincount(x, weights=weights, minlength=bins, maxlength=bins)

    def unique(self, x: TensorType, return_inverse: bool, return_counts: bool, axis: int) -> Tuple[TensorType, ...]:
        with self.device_of(x):
            if self.ndims(x) > 1:
                if return_counts:
                    raise NotImplementedError("TensorFlow multidimensional unique does not support counts")
                ux, ui = tf.raw_ops.UniqueV2(x=x, axis=[0])
                return (ux, ui) if return_inverse else ux
            if return_counts:
                ux, ui, uc = tf.unique_with_counts(x)
                return (ux, ui, uc) if return_inverse else (ux, uc)
            else:
                ux, ui = tf.unique(x)
                return (ux, ui) if return_inverse else ux

    def fft(self, x, axes: Union[tuple, list]):
        with self.device_of(x):
            if not axes:
                return x
            x = self.to_complex(x)
            perm = (*[i for i in range(self.ndims(x)) if i not in axes], *axes)
            iperm = np.argsort(perm)
            with self.device_of(x):
                if len(axes) == 1:
                    return tf.transpose(tf.signal.fft(tf.transpose(x, perm)), iperm)
                elif len(axes) == 2:
                    return tf.transpose(tf.signal.fft2d(tf.transpose(x, perm)), iperm)
                elif len(axes) == 3:
                    return tf.transpose(tf.signal.fft3d(tf.transpose(x, perm)), iperm)
                else:
                    for axis in axes:
                        x = self.fft(x, [axis])
                    return x

    def ifft(self, k, axes: Union[tuple, list]):
        with self.device_of(k):
            if not axes:
                return k
            k = self.to_complex(k)
            perm = (*[i for i in range(self.ndims(k)) if i not in axes], *axes)
            iperm = np.argsort(perm)
            with self.device_of(k):
                if len(axes) == 1:
                    return tf.transpose(tf.signal.ifft(tf.transpose(k, perm)), iperm)
                elif len(axes) == 2:
                    return tf.transpose(tf.signal.ifft2d(tf.transpose(k, perm)), iperm)
                elif len(axes) == 3:
                    return tf.transpose(tf.signal.ifft3d(tf.transpose(k, perm)), iperm)
                else:
                    for axis in axes:
                        k = self.ifft(k, [axis])
                    return k

    def imag(self, x):
        with self.device_of(x):
            return tf.math.imag(x)

    def real(self, x):
        with self.device_of(x):
            return tf.math.real(x)

    def conj(self, x):
        with self.device_of(x):
            return tf.math.conj(x)

    def cast(self, x, dtype: DType):
        if not self.is_tensor(x, only_native=True):
            x = self.as_tensor(x, convert_external=True)
        if self.dtype(x) == dtype:
            return x
        else:
            with self.device_of(x):
                return tf.cast(x, to_numpy_dtype(dtype))

    def unravel_index(self, flat_index, shape):
        idx_first = tf.unravel_index(flat_index, shape)
        return tf.transpose(idx_first, perm=tuple(range(1, self.ndims(flat_index)+1)) + (0,))

    def sin(self, x):
        with self.device_of(x):
            return tf.math.sin(x)

    def arcsin(self, x):
        with self.device_of(x):
            return tf.math.asin(x)

    def cos(self, x):
        with self.device_of(x):
            return tf.math.cos(x)

    def arccos(self, x):
        with self.device_of(x):
            return tf.math.acos(x)

    def tan(self, x):
        with self.device_of(x):
            return tf.math.tan(x)

    def arctan(self, x):
        with self.device_of(x):
            return tf.math.atan(x)

    def arctan2(self, y, x):
        y, x = self.auto_cast(y, x)
        with self.device_of(x):
            return tf.math.atan2(y, x)

    def sinh(self, x):
        with self.device_of(x):
            return tf.math.sinh(x)

    def arcsinh(self, x):
        with self.device_of(x):
            return tf.math.asinh(x)

    def cosh(self, x):
        with self.device_of(x):
            return tf.math.cosh(x)

    def arccosh(self, x):
        with self.device_of(x):
            return tf.math.acosh(x)

    def tanh(self, x):
        with self.device_of(x):
            return tf.math.tanh(x)

    def arctanh(self, x):
        with self.device_of(x):
            return tf.math.atanh(x)

    def log(self, x):
        with self.device_of(x):
            return tf.math.log(x)

    def sigmoid(self, x):
        with self.device_of(x):
            return tf.math.sigmoid(x)

    def log2(self, x):
        with self.device_of(x):
            return tf.math.log(x) / 0.6931471805599453094  # log(x) / log(2)

    def log10(self, x):
        with self.device_of(x):
            return tf.math.log(x) / 2.3025850929940456840  # log(x) / log(10)

    def dtype(self, array) -> DType:
        if tf.is_tensor(array):
            dt = array.dtype.as_numpy_dtype
            return from_numpy_dtype(dt)
        else:
            return NUMPY.dtype(array)

    def sparse_coo_tensor(self, indices, values, shape):
        with self._device_for(indices, values):
            return tf.SparseTensor(indices=self.to_int64(indices), values=values, dense_shape=shape)

    def mul_coo_dense(self, indices, values, shape, dense):
        values, dense = self.auto_cast(values, dense)
        batch_size, nnz, channel_count = self.staticshape(values)
        if batch_size > 1:
            return Backend.mul_coo_dense(self, indices, values, shape, dense)
        indices = tf.cast(indices, np.int64)
        result = []
        for b in range(batch_size):
            b_result = []
            for c in range(channel_count):
                matrix = tf.SparseTensor(indices=indices[b], values=values[b, :, c], dense_shape=shape)
                try:
                    b_result.append(tf.sparse.sparse_dense_matmul(matrix, dense[b, :, c, :]))
                except NotFoundError:  # These data types are probably not supported by TensorFlow
                    return Backend.mul_coo_dense(self, indices, values, shape, dense)
            result.append(tf.stack(b_result, 1))
        return tf.stack(result)

    def not_equal(self, x, y):
        with self._device_for(x, y):
            return ~self.equal(x, y)

    def greater_than(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return x > y

    def greater_or_equal(self, x, y):
        with self._device_for(x, y):
            x, y = self.auto_cast(x, y)
            return x >= y

    def add(self, a, b):
        with self._device_for(a, b):
            if isinstance(a, tf.SparseTensor) or isinstance(b, tf.SparseTensor):
                return tf.sparse.add(a, b, threshold=1e-5)
            else:
                return Backend.add(self, a, b)

    def sub(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a - b

    def mul(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a * b

    def div(self, numerator, denominator):
        with self._device_for(numerator, denominator):
            numerator, denominator = self.auto_cast(numerator, denominator)
            return numerator / denominator

    def pow(self, base, exp):
        with self._device_for(base, exp):
            base, exp = self.auto_cast(base, exp)
            return base ** exp

    def mod(self, dividend, divisor):
        with self._device_for(divisor, dividend):
            dividend, divisor = self.auto_cast(dividend, divisor)
            return dividend % divisor

    def and_(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a & b

    def or_(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a | b

    def xor(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a ^ b

    def floordiv(self, a, b):
        with self._device_for(a, b):
            a, b = self.auto_cast(a, b)
            return a // b

    def jacobian(self, f, wrt: Union[tuple, list], get_output: bool, is_f_scalar: bool):
        @wraps(f)
        def eval_grad(*args):
            args = [self.as_tensor(arg, True) if i in wrt else arg for i, arg in enumerate(args)]
            args = [self.to_float(arg) if self.dtype(arg).kind in (bool, int) and i in wrt else arg for i, arg in enumerate(args)]
            wrt_args = [arg for i, arg in enumerate(args) if i in wrt]
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                for arg in wrt_args:
                    assert arg.dtype in (tf.float16, tf.float32, tf.float64, tf.complex64, tf.complex128), f"Gradients can only be computed for float or complex tensors but got {arg.dtype} for argument with shape {arg.shape}"
                    tape.watch(arg)
                loss, output = f(*args)
            if self.prod(tf.shape(loss)) == 1:
                grads = list(self.as_registered.call(tape.gradient, loss, wrt_args, name=f"Backpropagation"))
            else:
                grads = list(self.as_registered.call(tape.jacobian, loss, wrt_args, name=f"Backpropagation"))
            assert None not in grads, f"Gradient could not be computed for wrt argument {grads.index(None)} (argument {wrt[grads.index(None)]}) with shape {wrt_args[grads.index(None)].shape}. TensorFlow returned gradient=None."
            return (*output, *grads) if get_output else grads
        return eval_grad

    def stop_gradient(self, value):
        with self._device_for(value):
            return tf.stop_gradient(value)

    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        with self._device_for(matrix, rhs):
            solution = tf.linalg.lstsq(matrix, rhs)
        return solution, None, None, None

    def solve_triangular_dense(self, matrix, rhs, lower: bool, unit_diagonal: bool):
        with self._device_for(matrix, rhs):
            matrix, rhs = self.auto_cast(matrix, rhs, int_to_float=True, bool_to_int=True)
            rhs = self.expand_dims(rhs, -1)
            if unit_diagonal:
                diag = np.diag(np.ones((self.staticshape(matrix)[-1],)))
                matrix = self.where(diag, diag, matrix)
            result = tf.linalg.triangular_solve(matrix, rhs, lower=lower)
            return result[..., 0]

    def matrix_rank_dense(self, matrix, hermitian=False):
        with self._device_for(matrix):
            matrix, = self.auto_cast(matrix, bool_to_int=True, int_to_float=True)
            return tf.linalg.matrix_rank(matrix)

    def eigvals(self, matrix: TensorType) -> TensorType:
        with self._device_for(matrix):
            return tf.linalg.eigval(matrix)

    def eig(self, matrix: TensorType) -> TensorType:
        with self._device_for(matrix):
            return tf.linalg.eig(matrix)

    def svd(self, matrix: TensorType, full_matrices=True) -> Tuple[TensorType, TensorType, TensorType]:
        with self._device_for(matrix):
            s, u, v = tf.linalg.svd(matrix, full_matrices=full_matrices)
            vh = tf.einsum('...ij -> ...ji', v)  # tf.transpose(v, perm=[*range(v.ndim-2), -1, -2]) doesn't work
            return u, s, vh

    def get_diagonal(self, matrices, offset=0):
        with self._device_for(matrices):
            matrices = tf.transpose(matrices, [0, 3, 1, 2])
            result = tf.linalg.diag_part(matrices, k=offset)
            return tf.transpose(result, [0, 2, 1])


_TAPES = []


def simple_device_type(t: str):
    return t[len('XLA_'):] if t.startswith('XLA_') else t