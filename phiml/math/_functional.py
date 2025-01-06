import inspect
import time
import types
import warnings
from functools import wraps, partial
from numbers import Number
from typing import Tuple, Callable, Dict, Generic, List, TypeVar, Any, Set, Union, Optional, Sequence

import numpy as np

from . import _ops as math, all_available, stop_gradient
from ._magic_ops import stack, slice_, find_differences, rename_dims, all_attributes
from ._shape import Shape, spatial, instance, batch, channel, merge_shapes, DimFilter, shape, dual
from ._sparse import SparseCoordinateTensor
from ._tensors import Tensor, disassemble_tree, assemble_tree, disassemble_tensors, assemble_tensors, variable_attributes, wrap, specs_equal, equality_by_shape_and_value, \
    object_dims, backend_for
from ._trace import ShiftLinTracer, matrix_from_function, LinearTraceInProgress
from .magic import PhiTreeNode, Shapable
from ..backend import Backend
from ..backend._backend import get_spatial_derivative_order, functional_derivative_evaluation, ML_LOGGER
from ..backend._buffer import set_buffer_config, get_buffer_config, get_required_buffer_sizes, wasted_memory
from ..backend._dtype import DType, FLOAT64

X = TypeVar('X')
Y = TypeVar('Y')


class SignatureKey:

    def __init__(self,
                 source_function: Union[Callable, None],
                 tree: Dict[str, Any],
                 shapes: Union[Shape, Tuple[Shape]],
                 specs: Union[Tuple[Shape], None],
                 backend: Backend,
                 tracing: bool,
                 auxiliary_kwargs: Any = None,
                 buffer_config: Dict[str, int] = None,
                 traced_aux: Dict[str, Any] = None):
        if source_function is None:  # this is an input signature
            assert isinstance(shapes, tuple)
        self.source_function = source_function
        self.tree: Dict[str, Any] = tree
        self.shapes = shapes
        self.backend = backend
        self.tracing = tracing
        self.specs = specs
        self.auxiliary_kwargs = auxiliary_kwargs
        self.spatial_derivative_order = get_spatial_derivative_order()
        self.buffer_config = buffer_config
        self.traced_aux = traced_aux

    def __repr__(self):
        return f"{self.tree} with shapes {self.shapes}"

    def __eq__(self, other: 'SignatureKey'):
        assert isinstance(other, SignatureKey)
        with equality_by_shape_and_value():
            diff = find_differences(self.auxiliary_kwargs, other.auxiliary_kwargs)
            cond_equal = not diff
        if isinstance(cond_equal, Tensor):
            cond_equal = cond_equal.all
        # shapes need not be compared because they are included in specs
        specs = specs_equal(self.specs, other.specs)
        return self.tree == other.tree and specs and self.backend == other.backend and self.spatial_derivative_order == other.spatial_derivative_order and cond_equal

    def __hash__(self):
        return hash(self.shapes) + hash(self.backend)

    def matches_structure_and_names(self, other: 'SignatureKey'):
        assert isinstance(other, SignatureKey)
        with equality_by_shape_and_value():
            cond_equal = self.auxiliary_kwargs == other.auxiliary_kwargs
        if isinstance(cond_equal, Tensor):
            cond_equal = cond_equal.all
        return self.tree == other.tree and all(s1.names == s2.names for s1, s2 in zip(self.shapes, other.shapes)) and self.backend == other.backend and cond_equal

    def extrapolate(self, rec_in: 'SignatureKey', new_in: 'SignatureKey') -> 'SignatureKey':
        assert self.source_function is not None, "extrapolate() must be called on output keys"
        shapes = [self._extrapolate_shape(s, rec_in, new_in) for s in self.shapes]
        return SignatureKey(self.source_function, self.tree, shapes, self.specs, self.backend, self.tracing, self.auxiliary_kwargs)

    @staticmethod
    def _extrapolate_shape(shape_: Shape, rec_in: 'SignatureKey', new_in: 'SignatureKey') -> Shape:
        sizes = []
        for dim in shape_:
            for p_in, n_in in zip(rec_in.shapes, new_in.shapes):
                if dim.name in p_in and dim.size == p_in.get_size(dim):
                    sizes.append(n_in.get_size(dim))
                    break
            else:
                raise ValueError(shape_, rec_in, new_in)
        return shape_.with_sizes(sizes)


def match_output_signature(new_in: SignatureKey, recorded_mappings: Dict[SignatureKey, SignatureKey], source) -> SignatureKey:
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in == new_in:  # exact match
            return rec_out
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in.matches_structure_and_names(new_in):
            return rec_out.extrapolate(rec_in, new_in)
    transforms_str = ''.join([f'\n* {i} -> {o}' for i, o in recorded_mappings.items()])
    raise RuntimeError(f"{source}: no output shape found for input shapes {new_in}.\n"
                       f"Maybe the backend extrapolated the concrete function from another trace?\n"
                       f"Registered transforms:\n{transforms_str}")  # KeyError does not support \n


def match_output_signature_list(new_in: SignatureKey, recorded_mappings: Dict[SignatureKey, List[SignatureKey]], source) -> List[SignatureKey]:
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in == new_in:  # exact match
            return rec_out
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in.matches_structure_and_names(new_in):
            return [r.extrapolate(rec_in, new_in) for r in rec_out]
    transforms_str = ''.join([f'\n* {i} -> {o}' for i, o in recorded_mappings.items()])
    raise RuntimeError(f"{source}: no output shape found for input shapes {new_in}.\n"
                       f"Maybe the backend extrapolated the concrete function from another trace?\n"
                       f"Registered transforms:\n{transforms_str}")  # KeyError does not support \n


def key_from_args(args: tuple,
                  kwargs: Dict[str, Any],
                  parameters: Tuple[str, ...],
                  cache=False,
                  aux: Set[str] = (),
                  attr_type=variable_attributes,
                  for_jit=False) -> Tuple[SignatureKey, List[Tensor], tuple, Dict[str, Any], Dict[str, Any]]:
    kwargs = {**kwargs, **{parameters[i]: v for i, v in enumerate(args)}}
    attached_aux_kwargs = {}
    detached_aux_kwargs = {}
    if aux:
        for param in aux:
            if param in kwargs:
                attached_aux_kwargs[param] = kwargs[param]
                detached_aux_kwargs[param] = stop_gradient(kwargs[param])
                del kwargs[param]
    tree, tensors = disassemble_tree(kwargs, cache=cache, attr_type=attr_type)
    _, aux_tensors = disassemble_tree(detached_aux_kwargs, cache=cache, attr_type=variable_attributes)
    tracing = not math.all_available(*tensors, *aux_tensors)
    backend = backend_for(*tensors, *aux_tensors)
    natives, shapes, specs = disassemble_tensors(tensors, expand=cache)
    if for_jit and backend.name == 'torch':  # for PyTorch, add tracers from aux to natives, but keep aux. PyTorch does not support using tensors with grad inside jit otherwise.
        _, aux_tensors = disassemble_tree(attached_aux_kwargs, cache=cache, attr_type=variable_attributes)
        aux_natives, aux_shapes, aux_specs = disassemble_tensors(aux_tensors, expand=False)
        from torch import Tensor
        grad_aux_indices = [i for i, n in enumerate(aux_natives) if isinstance(n, Tensor) and n.requires_grad]
        natives += tuple(aux_natives[i] for i in grad_aux_indices)
        traced_aux = {'indices': grad_aux_indices, 'shapes': aux_shapes, 'specs': aux_specs}
    else:
        traced_aux = None
    key = SignatureKey(None, tree, shapes, specs, backend, tracing, detached_aux_kwargs, traced_aux=traced_aux)
    return key, tensors, natives, kwargs, attached_aux_kwargs


def function_parameters(f) -> Tuple[str]:
    return tuple(get_function_parameters(f).keys())


def get_function_parameters(f) -> Dict[str, inspect.Parameter]:
    if isinstance(f, (JitFunction, GradientFunction, HessianFunction, CustomGradientFunction, LinearFunction)):
        return get_function_parameters(f.f)
    elif hasattr(f, '__wrapped__') and f.__wrapped__ is not None:
        inner_params = get_function_parameters(f.__wrapped__)
        outer_parameters = dict(inspect.signature(f, follow_wrapped=False).parameters)
        args_param = [name for name, param in outer_parameters.items() if param.kind == inspect.Parameter.VAR_POSITIONAL]
        assert args_param, f"Wrapping function {f.__name__} must have a varargs parameter"
        kwargs_param = [name for name, param in outer_parameters.items() if param.kind == inspect.Parameter.VAR_KEYWORD]
        outer_names: List[str] = list(outer_parameters.keys())
        if kwargs_param:
            outer_names.remove(kwargs_param[0])
        index = outer_names.index(args_param[0])
        return dict(**{n: outer_parameters[n] for n in outer_names[:index]}, **inner_params, **{n: outer_parameters[n] for n in outer_names[index + 1:]})
    else:
        params = dict(inspect.signature(f).parameters)
        assert 'args' not in params, f"Failed to determine signature of {f}. If it wraps another function, decorate it with @functools.wraps(func_with_signature)"
        return params


def f_name(f):
    assert callable(f), f
    if hasattr(f, '__name__'):
        return f.__name__
    if isinstance(f, partial):
        return f"partial({f.func})"
    else:
        return "unknown"


def default_determine_config(traced_configs: Sequence[Dict[str, int]], requirement_history: Dict[str, Sequence[int]]) -> Tuple[Optional[int], Dict[str, int]]:
    """
    Decides on the buffer configuration to use for the next invocation of a JIT-compiled function.

    Args:
        traced_configs: Configurations for which traces already exist. These should be preferred where possible.
        requirement_history: Required buffer sizes from the most recent function evaluations, at least 1.
            This includes failed evaluations due to too-small buffers.

    Returns:
        trace_index: If we should use an existing trace, returns the index into `traced_configs`. For a new trace, returns `None`.
        config: Configuration parameters to use, both for existing and new traces.
    """
    last_required = {k: hist[-1] for k, hist in requirement_history.items()}
    possible_configs = [c for c in traced_configs if wasted_memory(c, last_required) >= 0]
    if possible_configs:
        best = min(possible_configs, key=lambda c: wasted_memory(c, last_required))
        # ToDo if too much memory wasted, re-trace
        # wasted_fraction = wasted_memory(best, last_required) /
        return traced_configs.index(best), best
    # --- no previously traced config works ---
    def with_margin(size: int, margin=0.25, multiple=32):
        size = int(size * (1 + margin))
        r = size % multiple
        return size + multiple - r if r != 0 else size
    new_config = {k: with_margin(v) for k, v in last_required.items()}
    return None, new_config


class JitFunction:

    def __init__(self, f: Callable, auxiliary_args: Set[str], forget_traces: bool):
        self.f = f
        self.f_params = function_parameters(f)
        self.auxiliary_args = auxiliary_args
        self.forget_traces = forget_traces
        self.traces: Dict[SignatureKey, List[Callable]] = {}
        self.recorded_mappings: Dict[SignatureKey, List[SignatureKey]] = {}
        self.req_buffer_history: Dict[SignatureKey, Dict[str, List[int]]] = {}
        self.grad_jit = GradientFunction(f.f, self.f_params, f.wrt, f.get_output, f.is_f_scalar, jit=True) if isinstance(f, GradientFunction) else None
        self._extract_tensors: List[Tuple[Tensor]] = []
        self._post_call: List[Callable] = []
        self._tracing_in_key: SignatureKey = None
        self._buffer_manager = default_determine_config

    def _jit_compile(self, in_key: SignatureKey, buffer_config: Dict[str, int], attached_aux_args: dict):
        def jit_f_native(*natives):
            ML_LOGGER.debug(f"Φ-ML-jit: Tracing '{f_name(self.f)}'")
            _TRACING_JIT.append(self)
            set_buffer_config(buffer_config)
            try:
                self._tracing_in_key = in_key
                in_tensors = assemble_tensors(natives, in_key.specs)
                kwargs = assemble_tree(in_key.tree, in_tensors)
                attached_aux_args_ = attached_aux_args
                if in_key.traced_aux:
                    aux_tree, aux_tensors = disassemble_tree(attached_aux_args_, cache=False)
                    aux_natives, aux_shapes, aux_specs = disassemble_tensors(aux_tensors, expand=False)
                    aux_natives = list(aux_natives)
                    indices = in_key.traced_aux['indices']
                    traced_aux_natives = natives[-len(indices):]
                    for i, n in zip(indices, traced_aux_natives):
                        aux_natives[i] = n
                    aux_tensors = assemble_tensors(aux_natives, aux_specs)
                    attached_aux_args_ = assemble_tree(aux_tree, aux_tensors)
                f_output = self.f(**kwargs, **attached_aux_args_)  # Tensor or tuple/list of Tensors
                tree, out_tensors = disassemble_tree((f_output, self._extract_tensors), cache=True)
                result_natives, result_shapes, specs = disassemble_tensors(out_tensors, expand=True)
                tracers = get_required_buffer_sizes()
                out_key = SignatureKey(jit_f_native, tree, result_shapes, specs, in_key.backend, in_key.tracing, buffer_config=get_buffer_config())
                if not buffer_config and get_buffer_config():
                    ML_LOGGER.info(f"Tracing {self} with default buffer sizes: {get_buffer_config()}")
                self.recorded_mappings.setdefault(in_key, []).append(out_key)
                set_buffer_config({})
            finally:
                assert _TRACING_JIT.pop(-1) is self
                self._tracing_in_key = None
            result_natives = tuple([in_key.backend.as_tensor(n) for n in result_natives])  # PyTorch only allows Tensor outputs
            return result_natives + tuple(tracers.values())

        jit_f_native.__name__ = f"native({f_name(self.f) if isinstance(self.f, types.FunctionType) else str(self.f)})"
        return in_key.backend.jit_compile(jit_f_native)

    def __call__(self, *args, **kwargs):
        try:
            key, _, natives, _, aux_kwargs = key_from_args(args, kwargs, self.f_params, cache=True, aux=self.auxiliary_args, for_jit=True)
        except LinearTraceInProgress:
            return self.f(*args, **kwargs)
        if isinstance(self.f, GradientFunction) and key.backend.supports(Backend.jit_compile_grad):
            return self.grad_jit(*args, **kwargs)
        if not key.backend.supports(Backend.jit_compile):
            warnings.warn(f"jit_copmile() not supported by {key.backend}. Running function '{f_name(self.f)}' as-is.", RuntimeWarning)
            return self.f(*args, **kwargs)
        # --- do we need to trace? ---
        if key in self.traces:
            traced_configs = [k.buffer_config for k in self.recorded_mappings[key]]
            trace_index, buffer_config = self._buffer_manager(traced_configs, self.req_buffer_history[key])
            if trace_index is None:
                last_required = {k: hist[-1] for k, hist in self.req_buffer_history[key].items()}
                ML_LOGGER.info(f"Re-tracing {self} due to larger buffer requirements: {last_required}. Setting {buffer_config}")
            out_key = match_output_signature_list(key, self.recorded_mappings, self)[trace_index] if trace_index is not None else None
            # out_key = self.recorded_mappings[key][trace_index] if trace_index is not None else None
        else:  # called with new arguments
            buffer_config = {}
            out_key = None
            trace_index = None
        # --- trace if necessary ---
        if out_key is None:
            ML_LOGGER.debug(f"Φ-ML-jit: '{f_name(self.f)}' called with new key.\nshapes={[s.volume for s in key.shapes]}, args={key.tree}\n", trace_check(self, *args, **kwargs))
            if self.forget_traces:
                self.traces.clear()
                self.recorded_mappings.clear()
            native_jit_function = self._jit_compile(key, buffer_config, aux_kwargs)
            self.traces.setdefault(key, []).append(native_jit_function)
            if len(self.traces) >= 10:
                warnings.warn(f"""Φ-ML: The jit-compiled function '{f_name(self.f)}' was traced {len(self.traces)} times.
Performing many traces may be slow and cause memory leaks.
Re-tracing occurs when the number or types of arguments vary, tensor shapes vary between calls or different auxiliary arguments are given (compared by reference).
Set forget_traces=True to avoid memory leaks when many traces are required. Tracing reason: {trace_check(self, *args, **kwargs)[1]}""", RuntimeWarning)
            all_natives = native_jit_function(*natives)  # this appends out_key to recorded_mappings
            out_key = self.recorded_mappings[key][-1]
            buffer_config = out_key.buffer_config
        else:
            all_natives = self.traces[key][trace_index](*natives)  # call compiled function
        # --- record buffers ---
        if buffer_config:
            native_result = all_natives[:-len(buffer_config)]
            req_buffers = all_natives[-len(buffer_config):]
            req_buffers = {k: int(size) for k, size in zip(buffer_config.keys(), req_buffers)}
            history = self.req_buffer_history.setdefault(key, {})
            for k, size in req_buffers.items():
                history.setdefault(k, []).append(size)
            # --- re-run function if any buffer was too small ---
            re_run = any([req > buffer_config[k] for k, req in req_buffers.items()])
            if re_run:
                return self.__call__(*args, **kwargs)
        else:
            native_result = all_natives
            self.req_buffer_history.setdefault(key, {})
        # --- assemble result ---
        output_tensors = assemble_tensors(native_result, out_key.specs)
        output, extracted_tensor_lists = assemble_tree(out_key.tree, output_tensors)
        for extracted_tensors, runnable in zip(extracted_tensor_lists, self._post_call):
            runnable(*extracted_tensors)
        return output

    def extract_and_call(self, tensors: Tuple[Tensor], runnable: Callable):
        self._extract_tensors.append(tensors)
        self._post_call.append(runnable)

    def set_buffer_manager(self, manager: Callable):
        assert callable(manager)
        self._buffer_manager = manager

    def __repr__(self):
        return f"jit({f_name(self.f)})"

    @property
    def __name__(self):
        return f_name(self.f)


def jit_compile(f: Callable = None, auxiliary_args: str = '', forget_traces: bool = None) -> Callable:
    """
    Compiles a graph based on the function `f`.
    The graph compilation is performed just-in-time (jit), e.g. when the returned function is called for the first time.

    The traced function will compute the same result as `f` but may run much faster.
    Some checks may be disabled in the compiled function.

    Can be used as a decorator:
    ```python
    @math.jit_compile
    def my_function(x: math.Tensor) -> math.Tensor:
    ```

    Invoking the returned function may invoke re-tracing / re-compiling `f` after the first call if either

    * it is called with a different number of arguments,
    * the tensor arguments have different dimension names or types (the dimension order also counts),
    * any `Tensor` arguments require a different backend than previous invocations,
    * `phiml.math.magic.PhiTreeNode` positional arguments do not match in non-variable properties.

    Compilation is implemented for the following backends:

    * PyTorch: [`torch.jit.trace`](https://pytorch.org/docs/stable/jit.html)
    * TensorFlow: [`tf.function`](https://www.tensorflow.org/guide/function)
    * Jax: [`jax.jit`](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions)

    Jit-compilations cannot be nested, i.e. you cannot call `jit_compile()` while another function is being compiled.
    An exception to this is `jit_compile_linear()` which can be called from within a jit-compiled function.

    See Also:
        `jit_compile_linear()`

    Args:
        f: Function to be traced.
            All positional arguments must be of type `Tensor` or `phiml.math.magic.PhiTreeNode` returning a single `Tensor` or `phiml.math.magic.PhiTreeNode`.
        auxiliary_args: Comma-separated parameter names of arguments that are not relevant to backpropagation.
        forget_traces: If `True`, only remembers the most recent compiled instance of this function.
            Upon tracing with new instance (due to changed shapes or auxiliary args), deletes the previous traces.

    Returns:
        Function with similar signature and return values as `f`.
    """
    if f is None:
        kwargs = {k: v for k, v in locals().items() if v is not None}
        return partial(jit_compile, **kwargs)
    auxiliary_args = set(s.strip() for s in auxiliary_args.split(',') if s.strip())
    return f if isinstance(f, (JitFunction, LinearFunction)) and f.auxiliary_args == auxiliary_args else JitFunction(f, auxiliary_args, forget_traces or False)


class LinearFunction(Generic[X, Y], Callable[[X], Y]):
    """
    Just-in-time compiled linear function of `Tensor` arguments and return values.

    Use `jit_compile_linear()` to create a linear function representation.
    """

    def __init__(self, f, auxiliary_args: Set[str], forget_traces: bool):
        self.f = f
        self.f_params = function_parameters(f)
        self.auxiliary_args = auxiliary_args
        self.forget_traces = forget_traces
        self.matrices_and_biases: Dict[SignatureKey, Tuple[SparseCoordinateTensor, Tensor, Tuple]] = {}
        self.nl_jit = JitFunction(f, self.auxiliary_args, forget_traces)  # for backends that do not support sparse matrices

    def _get_or_trace(self, key: SignatureKey, args: tuple, f_kwargs: dict):
        if not key.tracing and key in self.matrices_and_biases:
            return self.matrices_and_biases[key]
        else:
            if self.forget_traces:
                self.matrices_and_biases.clear()
            _TRACING_LINEAR.append(self)
            try:
                matrix, bias, raw_out = matrix_from_function(self.f, *args, **f_kwargs, auto_compress=True, _return_raw_output=True)
            finally:
                assert _TRACING_LINEAR.pop(-1) is self
            if not key.tracing:
                self.matrices_and_biases[key] = matrix, bias, raw_out
                if len(self.matrices_and_biases) >= 4:
                    warnings.warn(f"""Φ-ML-lin: The compiled linear function '{f_name(self.f)}' was traced {len(self.matrices_and_biases)} times.
Performing many traces may be slow and cause memory leaks.
Tensors in auxiliary arguments (all except the first parameter unless specified otherwise) are compared by reference, not by tensor values.
Auxiliary arguments: {key.auxiliary_kwargs}
Multiple linear traces can be avoided by jit-compiling the code that calls the linear function or setting forget_traces=True.""", RuntimeWarning, stacklevel=3)
            return matrix, bias, raw_out

    def __call__(self, *args: X, **kwargs) -> Y:
        try:
            key, tensors, natives, x, aux_kwargs = key_from_args(args, kwargs, self.f_params, cache=False, aux=self.auxiliary_args)
        except LinearTraceInProgress:
            return self.f(*args, **kwargs)
        assert tensors, "Linear function requires at least one argument"
        if any(isinstance(t, ShiftLinTracer) for t in tensors):
            # TODO: if t is identity, use cached ShiftLinTracer, otherwise multiply two ShiftLinTracers
            return self.f(*args, **kwargs)
        if not key.backend.supports(Backend.sparse_coo_tensor):  # This might be called inside a Jax linear solve
            # warnings.warn(f"Sparse matrices are not supported by {backend}. Falling back to regular jit compilation.", RuntimeWarning)
            if not math.all_available(*tensors):  # avoid nested tracing, Typical case jax.scipy.sparse.cg(LinearFunction). Nested traces cannot be reused which results in lots of traces per cg.
                ML_LOGGER.debug(f"Φ-ML-lin: Running '{f_name(self.f)}' as-is with {key.backend} because it is being traced.")
                return self.f(*args, **kwargs)
            else:
                return self.nl_jit(*args, **kwargs)
        matrix, bias, (out_tree, out_tensors) = self._get_or_trace(key, args, aux_kwargs)
        result = matrix @ tensors[0] + bias
        out_tensors = list(out_tensors)
        out_tensors[0] = result
        return assemble_tree(out_tree, out_tensors)

    def sparse_matrix(self, *args, **kwargs):
        """
        Create an explicit representation of this linear function as a sparse matrix.

        See Also:
            `sparse_matrix_and_bias()`.

        Args:
            *args: Function arguments. This determines the size of the matrix.
            **kwargs: Additional keyword arguments for the linear function.

        Returns:
            Sparse matrix representation with `values` property and `native()` method.
        """
        key, *_, aux_kwargs = key_from_args(args, kwargs, self.f_params, cache=False, aux=self.auxiliary_args)
        matrix, bias, *_ = self._get_or_trace(key, args, aux_kwargs)
        assert math.close(bias, 0), "This is an affine function and cannot be represented by a single matrix. Use sparse_matrix_and_bias() instead."
        return matrix

    def sparse_matrix_and_bias(self, *args, **kwargs):
        """
        Create an explicit representation of this affine function as a sparse matrix and a bias vector.

        Args:
            *args: Positional arguments to the linear function.
                This determines the size of the matrix.
            **kwargs: Additional keyword arguments for the linear function.

        Returns:
            matrix: Sparse matrix representation with `values` property and `native()` method.
            bias: `Tensor`
        """
        key, *_, aux_kwargs = key_from_args(args, kwargs, self.f_params, cache=False, aux=self.auxiliary_args)
        return self._get_or_trace(key, args, aux_kwargs)[:2]

    def __repr__(self):
        return f"lin({f_name(self.f)})"


def jit_compile_linear(f: Callable[[X], Y] = None, auxiliary_args: str = None, forget_traces: bool = None) -> 'LinearFunction[X, Y]':
    """
    Compile an optimized representation of the linear function `f`.
    For backends that support sparse tensors, a sparse matrix will be constructed for `f`.

    Can be used as a decorator:
    ```python
    @math.jit_compile_linear
    def my_linear_function(x: math.Tensor) -> math.Tensor:
    ```

    Unlike `jit_compile()`, `jit_compile_linear()` can be called during a regular jit compilation.

    See Also:
        `jit_compile()`

    Args:
        f: Function that is linear in its positional arguments.
            All positional arguments must be of type `Tensor` and `f` must return a `Tensor`.
        auxiliary_args: Which parameters `f` is not linear in. These arguments are treated as conditioning arguments and will cause re-tracing on change.
        forget_traces: If `True`, only remembers the most recent compiled instance of this function.
            Upon tracing with new instance (due to changed shapes or auxiliary args), deletes the previous traces.

    Returns:
        `LinearFunction` with similar signature and return values as `f`.
    """
    if f is None:
        kwargs = {k: v for k, v in locals().items() if v is not None}
        return partial(jit_compile_linear, **kwargs)
    if isinstance(f, JitFunction):
        f = f.f  # cannot trace linear function from jitted version
    if isinstance(auxiliary_args, str):
        auxiliary_args = set(s.strip() for s in auxiliary_args.split(',') if s.strip())
    else:
        assert auxiliary_args is None
        f_params = function_parameters(f)
        auxiliary_args = f_params[1:]
    return f if isinstance(f, LinearFunction) and f.auxiliary_args == auxiliary_args else LinearFunction(f, auxiliary_args, forget_traces or False)


_TRACING_JIT: List[JitFunction] = []
_TRACING_LINEAR: List[LinearFunction] = []


def when_available(runnable: Callable, *tensor_args: Tensor):
    """
    Calls `runnable(*tensor_args)` once the concrete values of all tensors are available.
    In eager mode, `runnable` is called immediately.
    When jit-compiled, `runnable` is called after the jit-compiled function has returned.

    Args:
        runnable: Function to call as `runnable(*tensor_args)`. This can be a `lambda` function.
        *tensor_args: `Tensor` values to pass to `runnable` with concrete values.
    """
    if _TRACING_LINEAR:
        raise RuntimeError(f"when_available() cannot be called inside a function marked as @jit_compile_linear")
    if all_available(*tensor_args):  # eager or NumPy
        runnable(*tensor_args)
    else:
        assert _TRACING_JIT, f"tensors are not available but no JIT function is being traced. Maybe you are using external jit?"
        for jit_f in _TRACING_JIT:
            jit_f.extract_and_call(tensor_args, runnable)


def simplify_wrt(f, wrt: Union[str, int, tuple, list]):
    f_params = function_parameters(f)
    if wrt is None:  # Old default
        wrt = f_params[0],
    elif isinstance(wrt, (tuple, list)) and all(isinstance(i, str) for i in wrt):
        wrt = tuple(wrt)
    elif isinstance(wrt, str) and ',' in wrt:
        wrt = tuple(i.strip() for i in wrt.split(',') if i.strip())
    elif isinstance(wrt, str):
        wrt = wrt
    else:  # int or tuple or list
        if isinstance(wrt, int):
            wrt = f_params[wrt]
        elif isinstance(wrt, (tuple, list)) and all(isinstance(i, int) for i in wrt):
            wrt = tuple(f_params[i] for i in wrt)
        else:
            raise ValueError(f"Invalid value given as wrt: {wrt}. Please pass a comma-separated string of parameter names.")
        warnings.warn("Specifying wrt by position is deprecated in phiml.math.funcitonal_gradient() and phiml.math.jacobian(). Please pass a list or comma-separated string of parameter names.",
                      SyntaxWarning, stacklevel=4)
    return f_params, wrt


class GradientFunction:
    """ Jacobian or Gradient of a function. """

    def __init__(self, f: Callable, f_params, wrt: Union[str, Tuple[str, ...]], get_output: bool, is_f_scalar: bool, jit=False):
        self.f = f
        self.f_params = f_params
        self.wrt = wrt
        self._wrt_tuple = wrt if isinstance(wrt, tuple) else (wrt,)
        self.get_output = get_output
        self.is_f_scalar = is_f_scalar
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}
        self.jit = jit

    def _trace_grad(self, in_key: SignatureKey, wrt_natives):
        def f_native(*natives):
            ML_LOGGER.debug(f"Φ-ML-grad: Evaluating gradient of {f_name(self.f)}")
            in_tensors = assemble_tensors(natives, in_key.specs)
            kwargs = assemble_tree(in_key.tree, in_tensors, attr_type=variable_attributes)
            with functional_derivative_evaluation(order=1):
                result = self.f(**kwargs)  # Tensor or tuple/list of Tensors
            loss = result[0] if isinstance(result, (tuple, list)) else result
            if isinstance(loss, Tensor):
                loss_reduced = math.sum_(loss, batch)
                loss_native = loss_reduced.native(loss_reduced.shape.names)
            else:
                loss_native = loss
                loss_shape = in_key.backend.staticshape(loss_native)
                assert len(
                    loss_shape) == 0, f"Only scalar losses are allowed when returning a native tensor but {f_name(self.f)} returned {type(loss_native).__name__} of shape {loss_shape}. For higher-dimensional values, use Φ-ML tensors instead."
            nest, out_tensors = disassemble_tree(result, cache=True, attr_type=all_attributes)
            result_natives, result_shapes, specs = disassemble_tensors(out_tensors, expand=True)
            self.recorded_mappings[in_key] = SignatureKey(f_native, nest, result_shapes, specs, in_key.backend, in_key.tracing)
            return loss_native, result_natives

        if self.jit:
            return in_key.backend.jit_compile_grad(f_native, wrt=wrt_natives, get_output=self.get_output, is_f_scalar=self.is_f_scalar)
        else:
            return in_key.backend.jacobian(f_native, wrt=wrt_natives, get_output=self.get_output, is_f_scalar=self.is_f_scalar)

    def __call__(self, *args, **kwargs):
        key, tensors, natives, kwargs, _ = key_from_args(args, kwargs, self.f_params, cache=True, attr_type=variable_attributes)
        if not key.backend.supports(Backend.jacobian):
            if math.default_backend().supports(Backend.jacobian):
                warnings.warn(f"Using {math.default_backend()} for gradient computation because {key.backend} does not support jacobian()", RuntimeWarning)
                key.backend = math.default_backend()
            else:
                raise AssertionError(f"jacobian() not supported by {key.backend}.")
        wrt_tensors = self._track_wrt(kwargs)
        assert wrt_tensors, f"Cannot compute gradient of {self.__name__} because the input contains no variable tensors"
        wrt_natives = self._track_wrt_natives(wrt_tensors, disassemble_tree(kwargs, cache=True, attr_type=variable_attributes)[1])
        if key not in self.traces:
            self.traces[key] = self._trace_grad(key, wrt_natives)
        native_result = self.traces[key](*natives)
        output_key = match_output_signature(key, self.recorded_mappings, self)
        jac_shape = output_key.shapes[0].non_batch  # ToDo prepend this to all wrt shapes
        wrt_specs = [key.specs[i] for i in wrt_tensors]
        if self.get_output:
            output_tensors = assemble_tensors(native_result, list(output_key.specs) + wrt_specs)
            output_structure, grad_tuple = assemble_tree((output_key.tree, [key.tree[i] for i in self._wrt_tuple]), output_tensors, attr_type=all_attributes)
            return output_structure, grad_tuple if isinstance(self.wrt, tuple) else grad_tuple[0]
        else:
            output_tensors = assemble_tensors(native_result, wrt_specs)
            grad_tuple = assemble_tree([key.tree[i] for i in self._wrt_tuple], output_tensors, attr_type=all_attributes)
            return grad_tuple if isinstance(self.wrt, tuple) else grad_tuple[0]

    def __repr__(self):
        return f"grad({f_name(self.f)})"

    @property
    def __name__(self):
        return f_name(self.f)

    def _track_wrt(self, kwargs: dict):
        wrt_tensors = []
        for name, arg in kwargs.items():
            _, tensors = disassemble_tree(arg, cache=True, attr_type=variable_attributes)
            wrt_tensors.extend([name] * len(tensors))
        return [t_i for t_i, name in enumerate(wrt_tensors) if name in self._wrt_tuple]

    @staticmethod
    def _track_wrt_natives(wrt_tensors, values):
        wrt_natives = []
        for i, value in enumerate(values):
            wrt_natives.extend([i] * len(value._natives()))
        return [n_i for n_i, t_i in enumerate(wrt_natives) if t_i in wrt_tensors]


def jacobian(f: Callable, wrt: str = None, get_output=True) -> Callable:
    """
    Creates a function which computes the Jacobian matrix of `f`.
    For scalar functions, consider using `gradient()` instead.

    Example:
    ```python
    def f(x, y):
        prediction = f(x)
        loss = math.l2_loss(prediction - y)
        return loss, prediction

    dx = jacobian(loss_function, wrt='x', get_output=False)(x, y)

    (loss, prediction), (dx, dy) = jacobian(loss_function,
                                        wrt='x,y', get_output=True)(x, y)
    ```

    Functional gradients are implemented for the following backends:

    * PyTorch: [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad) / [`torch.autograd.backward`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward)
    * TensorFlow: [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
    * Jax: [`jax.grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad)

    When the gradient function is invoked, `f` is called with tensors that track the gradient.
    For PyTorch, `arg.requires_grad = True` for all positional arguments of `f`.

    Args:
        f: Function to be differentiated.
            `f` must return a floating point `Tensor` with rank zero.
            It can return additional tensors which are treated as auxiliary data and will be returned by the gradient function if `return_values=True`.
            All arguments for which the gradient is computed must be of dtype float or complex.
        get_output: Whether the gradient function should also return the return values of `f`.
        wrt: Comma-separated parameter names of `f` with respect to which the gradient should be computed.
            If not specified, the gradient will be computed w.r.t. the first positional argument (highly discouraged).

    Returns:
        Function with the same arguments as `f` that returns the value of `f`, auxiliary data and Jacobian of `f` if `get_output=True`, else just the Jacobian of `f`.
    """
    f_params, wrt = simplify_wrt(f, wrt)
    return GradientFunction(f, f_params, wrt, get_output, is_f_scalar=False)


def gradient(f: Callable, wrt: str = None, get_output=True) -> Callable:
    """
    Creates a function which computes the gradient of `f`.

    Example:
    ```python
    def loss_function(x, y):
        prediction = f(x)
        loss = math.l2_loss(prediction - y)
        return loss, prediction

    dx = gradient(loss_function, 'x', get_output=False)(x, y)

    (loss, prediction), (dx, dy) = gradient(loss_function,
                                            'x,y', get_output=True)(x, y)
    ```

    Functional gradients are implemented for the following backends:

    * PyTorch: [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad) / [`torch.autograd.backward`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward)
    * TensorFlow: [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
    * Jax: [`jax.grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad)

    When the gradient function is invoked, `f` is called with tensors that track the gradient.
    For PyTorch, `arg.requires_grad = True` for all positional arguments of `f`.

    Args:
        f: Function to be differentiated.
            `f` must return a floating point `Tensor` with rank zero.
            It can return additional tensors which are treated as auxiliary data and will be returned by the gradient function if `return_values=True`.
            All arguments for which the gradient is computed must be of dtype float or complex.
        get_output: Whether the gradient function should also return the return values of `f`.
        wrt: Comma-separated parameter names of `f` with respect to which the gradient should be computed.
            If not specified, the gradient will be computed w.r.t. the first positional argument (highly discouraged).

    Returns:
        Function with the same arguments as `f` that returns the value of `f`, auxiliary data and gradient of `f` if `get_output=True`, else just the gradient of `f`.
    """
    f_params, wrt = simplify_wrt(f, wrt)
    return GradientFunction(f, f_params, wrt, get_output, is_f_scalar=True)


class HessianFunction:

    def __init__(self, f: Callable, f_params, wrt: tuple, get_output: bool, get_gradient: bool, dim_suffixes: tuple, jit=False):
        assert isinstance(dim_suffixes, tuple) and len(dim_suffixes) == 2
        self.f = f
        self.f_params = f_params
        self.wrt = wrt
        self._wrt_tuple = wrt if isinstance(wrt, tuple) else (wrt,)
        self.get_output = get_output
        self.get_gradient = get_gradient
        self.dim_suffixes = dim_suffixes
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}
        self.jit = jit
#
#     def _trace_hessian(self, in_key: SignatureKey, wrt_natives):
#         def f_native(*natives):
#             PHI_LOGGER.debug(f"Φ-ML-grad: Evaluating gradient of {f_name(self.f)}")
#             in_tensors = assemble_tensors(natives, in_key.specs)
#             kwargs = assemble_tree(in_key.tree, in_tensors)
#             with functional_derivative_evaluation(order=2):
#                 result = self.f(**kwargs)
#             nest, out_tensors = disassemble_tree(result, cache=True)
#             result_natives, result_shapes, specs = disassemble_tensors(out_tensors, expand=True)
#             self.recorded_mappings[in_key] = SignatureKey(f_native, nest, result_shapes, specs, in_key.backend, in_key.tracing)
#             return result_natives
#
#         hessian_generator = in_key.backend.jit_compile_hessian if self.jit else in_key.backend.hessian
#         return hessian_generator(f_native, wrt=wrt_natives, get_output=self.get_output, get_gradient=self.get_gradient)
#
#     def __call__(self, *args, **kwargs):
#         key, tensors, natives, kwargs, batch_shape = key_from_args_pack_batch(args, kwargs, self.f_params, cache=True, attr_type=variable_attributes)
#         if not key.backend.supports(Backend.jacobian):
#             if math.default_backend().supports(Backend.jacobian):
#                 warnings.warn(f"Using {math.default_backend()} for gradient computation because {key.backend} does not support jacobian()", RuntimeWarning)
#                 key.backend = math.default_backend()
#             else:
#                 raise AssertionError(f"jacobian() not supported by {key.backend}.")
#         wrt_tensors: List[int] = self._track_wrt(kwargs)
#         wrt_natives: List[int] = self._track_wrt_natives(wrt_tensors, disassemble_tree(kwargs)[1])
#         if key not in self.traces:
#             self.traces[key] = self._trace_hessian(key, wrt_natives)
#         native_result = self.traces[key](*natives)
#         assert len(native_result) == 1 + int(self.get_output) + int(self.get_gradient)
#         output_key = match_output_signature(key, self.recorded_mappings, self)
#         result = ()
#         if self.get_output:
#             output_tensors = assemble_tensors(native_result[0], output_key.specs)
#             output_tensors = [unpack_dim(t, 'batch', batch_shape) for t in output_tensors]
#             # output_tensors = [math.reshaped_tensor(n, [batch_shape, *shape.non_batch]) for n, shape in zip(native_result[0], output_key.shapes)]
#             result += assemble_tree(output_key.tree, output_tensors),
#         if self.get_gradient:
#             grad_tensors = assemble_tensors(native_result[int(self.get_output)], [key.specs[i] for i in wrt_tensors])
#             grad_tensors = [unpack_dim(t, 'batch', batch_shape) for t in grad_tensors]
#             grads = assemble_tree([key.tree[i] for i in self._wrt_tuple], grad_tensors)
#             if not isinstance(self.wrt, tuple):
#                 grads = grads[0]
#             result += grads,
#         if len(wrt_natives) == 1:
#             native_hessian = native_result[-1][0][0]
#             hessian_tensor = math.reshaped_tensor(native_hessian, [batch_shape, *self.shape_with_suffixes(key.shapes[0].non_batch, self.dim_suffixes[0]),
#                                                                    *self.shape_with_suffixes(key.shapes[0].non_batch, self.dim_suffixes[1])], check_sizes=True)
#             hessian_tree = assemble_tree(key.tree[self.wrt[0] if isinstance(self.wrt, tuple) else self.wrt], [hessian_tensor])
#             result += [hessian_tree] if isinstance(self.wrt, tuple) else hessian_tree,
#         else:
#             assert all([t is None for t in key.tree]), "When computing the Hessian w.r.t. multiple tensors, all inputs must be Tensors."
#             raise NotImplementedError()
#             hessian_tree = [[] for _ in self.wrt]
#             for i in range(len(self.wrt)):
#                 for j in range(len(self.wrt)):
#                     native_hessian_ij = native_result[-1][i][j]
#                     hessian_tensor_ij = math.reshaped_tensor(native_hessian_ij, [batch_shape, *key.shapes[i].non_batch, *self.dupli_shape(key.shapes[j].non_batch)], check_sizes=True)
#                     hessian_tree[i].append(hessian_tensor_ij)
#             result += tuple([tuple(col) for col in hessian_tree]),
#         return result
#
#     def shape_with_suffixes(self, shape: Shape, suffix: str):
#         return shape._with_names([n + suffix for n in shape.names])
#
#     def __repr__(self):
#         return f"grad({f_name(self.f)})"
#
#     @property
#     def __name__(self):
#         return f_name(self.f)
#
#     def _track_wrt(self, kwargs: dict):
#         wrt_tensors = []
#         for name, arg in kwargs.items():
#             _, tensors = disassemble_tree(arg)
#             wrt_tensors.extend([name] * len(tensors))
#         return [t_i for t_i, name in enumerate(wrt_tensors) if name in self._wrt_tuple]
#
#     @staticmethod
#     def _track_wrt_natives(wrt_tensors, values):
#         wrt_natives = []
#         for i, value in enumerate(values):
#             wrt_natives.extend([i] * len(value._natives()))
#         return [n_i for n_i, t_i in enumerate(wrt_natives) if t_i in wrt_tensors]
#
#
# def hessian(f: Callable, wrt: str, get_output=True, get_gradient=True, dim_suffixes=('', '_')) -> Callable:
#     """
#     *Experimental. This function currently only supports PyTorch and the Hessian can only be computed w.r.t. one argument.*
#
#     Creates a function which computes the Hessian (second derivative) of `f`.
#
#     Example:
#     ```python
#     def loss_function(x, y):
#         prediction = f(x)
#         loss = math.l2_loss(prediction - y)
#         return loss, prediction
#
#     hess, = hessian(loss_function, 'x', get_output=False, get_gradient=False)(x, y)
#
#     (loss, prediction), (dx, dy), ((dx_dx, dx_dy), (dy_dx, dy_dy)) = hessian(loss_function,
#                                         wrt='x,y', get_output=True)(x, y)
#     ```
#
#     When the gradient function is invoked, `f` is called with tensors that track the gradient.
#     For PyTorch, `arg.requires_grad = True` for all positional arguments of `f`.
#
#     Args:
#         f: Function to be differentiated.
#             `f` must return a floating point `Tensor` with rank zero.
#             It can return additional tensors which are treated as auxiliary data and will be returned by the gradient function if `return_values=True`.
#             All arguments for which the gradient is computed must be of dtype float or complex.
#         wrt: Comma-separated parameter names of `f` with respect to which the gradient should be computed.
#             If not specified, the gradient will be computed w.r.t. the first positional argument (highly discouraged).
#         get_output: Whether the Hessian function should also return the return values of `f`.
#         get_gradient: Whether the Hessian function should also return the gradient of `f`.
#         dim_suffixes: `tuple` containing two strings.
#             All Non-batch dimensions of the parameters occur twice in the corresponding Hessian.
#             To avoid duplicate names, suffixes are added to non-batch dimensions.
#             The dimensions from the first derivative computation are appended with `dim_suffixes[0]` and the second ones with `dim_suffixes[1]`.
#             This argument has no effect on the dimension names of the gradient if `get_gradient=True`.
#
#     Returns:
#         Function with the same arguments as `f` that returns `(f(x), g(x), H(x))` or less depending on `get_output` and `get_gradient`.
#     """
#     f_params, wrt = simplify_wrt(f, wrt)
#     return HessianFunction(f, f_params, wrt, get_output, get_gradient, dim_suffixes)


class CustomGradientFunction:

    def __init__(self, f: Callable, gradient: Callable, auxiliary_args: Set[str]):
        self.f = f
        self.f_params = function_parameters(f)
        self.gradient = gradient
        self.auxiliary_args = auxiliary_args
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}

    def _trace(self, in_key: SignatureKey):
        def forward_native(*natives):
            in_tensors = assemble_tensors(natives, in_key.specs)
            kwargs = assemble_tree(in_key.tree, in_tensors, attr_type=variable_attributes)
            ML_LOGGER.debug(f"Running forward pass of custom op {forward_native.__name__} given args {tuple(kwargs.keys())} containing {len(natives)} native tensors")
            result = self.f(**kwargs, **in_key.auxiliary_kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_tree(result, cache=True, attr_type=variable_attributes)
            result_natives, result_shapes, specs = disassemble_tensors(out_tensors, expand=True)
            self.recorded_mappings[in_key] = SignatureKey(forward_native, nest, result_shapes, specs, in_key.backend, in_key.tracing)
            return result_natives

        def backward_native(x_natives, y_natives, dy_natives):
            ML_LOGGER.debug(f"Running backward pass of custom op {backward_native.__name__}")
            out_key = self.recorded_mappings[in_key]
            # del self.recorded_mappings[in_key]  # this may be required multiple times
            x_tensors = assemble_tensors(x_natives, in_key.specs)
            y_tensors = assemble_tensors(y_natives, out_key.specs)
            dy_tensors = assemble_tensors(dy_natives, out_key.specs)
            kwargs = assemble_tree(in_key.tree, x_tensors, attr_type=variable_attributes)
            if in_key.auxiliary_kwargs:
                kwargs = {**kwargs, **in_key.auxiliary_kwargs}
            y = assemble_tree(out_key.tree, y_tensors, attr_type=variable_attributes)
            dy = assemble_tree(out_key.tree, dy_tensors, attr_type=variable_attributes)
            result = self.gradient(kwargs, y, dy)
            assert isinstance(result, dict) and all(key in kwargs for key in result.keys()), f"gradient function must return a dict containing only parameter names of the forward function. Forward '{f_name(self.f)}' has arguments {kwargs}."
            full_result = tuple(result.get(name, None) for name in in_key.tree.keys())
            result_natives = self.incomplete_tree_to_natives(full_result, tuple(in_key.tree.values()), list(in_key.shapes))
            ML_LOGGER.debug(f"Backward pass of custom op {backward_native.__name__} returned gradients for {tuple(result.keys())} out of {tuple(in_key.tree.keys())} containing {len(result_natives)} native tensors")
            return result_natives

        forward_native.__name__ = f"forward '{f_name(self.f) if isinstance(self.f, types.FunctionType) else str(self.f)}'"
        backward_native.__name__ = f"{self.gradient.__name__ if isinstance(self.gradient, types.FunctionType) else str(self.gradient)} (of '{f_name(self.f) if isinstance(self.f, types.FunctionType) else str(self.f)}')"

        return in_key.backend.custom_gradient(forward_native, backward_native, get_external_cache=lambda: self.recorded_mappings[in_key], on_call_skipped=partial(self.recorded_mappings.__setitem__, in_key))

    def __call__(self, *args, **kwargs):
        key, _, natives, _, _ = key_from_args(args, kwargs, self.f_params, cache=False, aux=self.auxiliary_args, attr_type=variable_attributes)
        if not key.backend.supports(Backend.jacobian) and not key.backend.supports(Backend.jacobian):
            return self.f(*args, **kwargs)  # no need to use custom gradient if gradients aren't supported anyway
        elif not key.backend.supports(Backend.custom_gradient):
            warnings.warn(f"custom_gradient() not supported by {key.backend}. Running function '{f_name(self.f)}' as-is.", RuntimeWarning)
            return self.f(*args, **kwargs)
        if key not in self.traces:
            self.traces[key] = self._trace(key)
            if len(self.traces) >= 8:
                warnings.warn(f"""{self.__name__} has been traced {len(self.traces)} times.
To avoid memory leaks, call {f_name(self.f)}.traces.clear(), {f_name(self.f)}.recorded_mappings.clear().
Traces can be avoided by jit-compiling the code that calls custom gradient functions.
""", RuntimeWarning, stacklevel=2)
        native_result = self.traces[key](*natives)  # With PyTorch + jit, this does not call forward_native every time
        output_key = match_output_signature(key, self.recorded_mappings, self)
        output_tensors = assemble_tensors(native_result, output_key.specs)
        return assemble_tree(output_key.tree, output_tensors, attr_type=variable_attributes)

    def __repr__(self):
        return f"custom_gradient(forward={f_name(self.f)}, backward={self.gradient.__name__}, id={id(self)})"

    @property
    def __name__(self):
        return f"custom_grad({f_name(self.f)})"

    @staticmethod
    def incomplete_tree_to_natives(incomplete, tree, complete_shapes: List[Shape]) -> list:
        """ None in nest means there is a tensor. """
        if tree is None:
            c_shape = complete_shapes.pop(0)
            if incomplete is None:
                return [None] * c_shape.non_uniform_shape.volume
            else:
                assert isinstance(incomplete, Tensor)
                return list(incomplete._natives())
        elif isinstance(tree, (tuple, list)):
            if incomplete is None:
                return sum([CustomGradientFunction.incomplete_tree_to_natives(None, item, complete_shapes) for item in tree], [])
            else:
                assert type(tree) == type(incomplete) and len(tree) == len(incomplete)
                return sum([CustomGradientFunction.incomplete_tree_to_natives(i_item, c_item, complete_shapes) for i_item, c_item in zip(incomplete, tree)], [])
        elif isinstance(tree, dict):
            if incomplete is None:
                return sum([CustomGradientFunction.incomplete_tree_to_natives(None, item, complete_shapes) for item in tree.values()], [])
            else:
                assert type(tree) == type(incomplete) and len(tree) == len(incomplete) and set(tree.keys()) == set(incomplete.keys())
                return sum([CustomGradientFunction.incomplete_tree_to_natives(incomplete[key], c_item, complete_shapes) for key, c_item in tree.items()], [])
        elif isinstance(tree, PhiTreeNode):
            attributes = variable_attributes(tree)
            natives = []
            for attr in attributes:
                n_val = getattr(tree, attr)
                i_val = getattr(incomplete, attr) if incomplete is not None else None
                natives_item = CustomGradientFunction.incomplete_tree_to_natives(i_val, n_val, complete_shapes)
                natives.extend(natives_item)
            return natives
        else:
            assert incomplete is None
            return []


def custom_gradient(f: Callable, gradient: Callable, auxiliary_args: str = ''):
    """
    Creates a function based on `f` that uses a custom gradient for the backpropagation pass.

    *Warning* This method can lead to memory leaks if the gradient function is not called.
    Make sure to pass tensors without gradients if the gradient is not required, see `stop_gradient()`.

    Args:
        f: Forward function mapping `Tensor` arguments `x` to a single `Tensor` output or sequence of tensors `y`.
        gradient: Function to compute the vector-Jacobian product for backpropagation.
            Will be called as `gradient(input_dict, *y, *dy) -> output_dict` where `input_dict` contains all named arguments passed to the forward function
            and `output_dict` contains only those parameters for which a gradient is defined.
        auxiliary_args: Comma-separated parameter names of arguments that are not relevant to backpropagation.

    Returns:
        Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
    """
    auxiliary_args = set(s.strip() for s in auxiliary_args.split(',') if s.strip())
    return CustomGradientFunction(f, gradient, auxiliary_args)


def print_gradient(value: Tensor, name="", detailed=False) -> Tensor:
    """
    Prints the gradient vector of `value` when computed.
    The gradient at `value` is the vector-Jacobian product of all operations between the output of this function and the loss value.

    The gradient is not printed in jit mode, see `jit_compile()`.

    Example:
        ```python
        def f(x):
            x = math.print_gradient(x, 'dx')
            return math.l1_loss(x)

        math.jacobian(f)(math.ones(x=6))
        ```

    Args:
        value: `Tensor` for which the gradient may be computed later.
        name: (Optional) Name to print along with the gradient values
        detailed: If `False`, prints a short summary of the gradient tensor.

    Returns:
        `identity(value)` which when differentiated, prints the gradient vector.
    """

    def print_grad(params: dict, _y, dx):
        param_name, x = next(iter(params.items()))
        if math.all_available(x, dx):
            if detailed:
                math.print_(dx, name=name)
            else:
                print(f"{name}:  \t{dx}")
        else:
            print(f"Cannot print gradient for {param_name}, data not available.")
        return {param_name: dx}

    identity = custom_gradient(lambda x: x, print_grad)
    return identity(value)


def safe_mul(x: Union[Number, Tensor], y: Union[Number, Tensor]):
    """Multiplication for tensors with non-finite values.
    Computes *x·y* in the forward pass but drops gradient contributions from infinite and `NaN` values."""
    return _safe_mul(x, y)


def _safe_mul_fwd(x, y):
    return x * y


def _safe_mul_bwd(input_dict, prod, adj):
    x, y = input_dict['x'], input_dict['y']
    x = math.where(math.is_finite(x), x, 0)
    y = math.where(math.is_finite(y), y, 0)
    return {
        'x': adj * y,
        'y': adj * x,
    }


_safe_mul = custom_gradient(_safe_mul_fwd, _safe_mul_bwd)


def trace_check(traced_function, *args, **kwargs) -> Tuple[bool, str]:
    """
    Tests if `f(*args, **kwargs)` has already been traced for arguments compatible with `args` and `kwargs`.
    If true, jit-compiled functions are very fast since the Python function is not actually called anymore.

    Args:
        traced_function: Transformed Function, e.g. jit-compiled or linear function.
        *args: Hypothetical arguments to be passed to `f`
        **kwargs: Hypothetical keyword arguments to be passed to `f`

    Returns:
        result: `True` if there is an existing trace that can be used, `False` if `f` would have to be re-traced.
        message: A `str` that, if `result == False`, gives hints as to why `f` needs to be re-traced given `args` and `kwargs`.
    """
    assert args or kwargs, f"Please pass the hypothetical function arguments to trace_check()"
    f = traced_function
    if isinstance(f, (JitFunction, GradientFunction, HessianFunction, CustomGradientFunction)):
        keys = f.traces.keys()
    elif isinstance(f, LinearFunction):
        keys = f.matrices_and_biases.keys()
    else:
        raise ValueError(f"{f_name(f)} is not a traceable function. Only supports jit_compile, jit_compile_linear, gradient, custom_gradient, jacobian, hessian")
    key, *_ = key_from_args(args, kwargs, f.f_params, aux=f.auxiliary_args)
    if not keys:
        return False, "Function has not yet been traced"
    if key in keys:
        return True, ""
    traced_key = next(iter(keys))  # ToDo compare against all
    with equality_by_shape_and_value():
        cond_equal = key.auxiliary_kwargs == traced_key.auxiliary_kwargs
    if isinstance(cond_equal, Tensor):
        cond_equal = cond_equal.all
    if not cond_equal:
        return False, "Auxiliary arguments do not match"
    # shapes need not be compared because they are included in specs
    if traced_key.tree.keys() != key.tree.keys():
        return False, f"Different primary arguments passed: {set(traced_key.tree.keys())} vs {set(key.tree.keys())}"
    for name in traced_key.tree.keys():
        if traced_key.tree[name] != key.tree[name]:
            return False, f"Primary argument '{name}' differs in non-traced variables: {traced_key.tree[name]} vs {key.tree[name]}. Make sure the corresponding class overrides __eq__()."
    with equality_by_shape_and_value():
        if traced_key.specs != key.specs:
            return False, "Traced variables differ in shape"
    if traced_key.backend != key.backend:
        return False, f"Function was not traced with backend {key.backend}"
    if traced_key.spatial_derivative_order != key.spatial_derivative_order:
        return False, f"Different in spatial_derivative_order. This is likely an internal problem."
    return True, ""


def map_types(f: Callable, dims: Union[Shape, tuple, list, str, Callable], dim_type: Union[Callable, str]) -> Callable:
    """
    Wraps a function to change the dimension types of its `Tensor` and `phiml.math.magic.PhiTreeNode` arguments.

    Args:
        f: Function to wrap.
        dims: Concrete dimensions or dimension type, such as `spatial` or `batch`.
            These dimensions will be mapped to `dim_type` for all positional function arguments.
        dim_type: Dimension type, such as `spatial` or `batch`.
            `f` will be called with dimensions remapped to this type.

    Returns:
        Function with signature matching `f`.
    """

    def forward_retype(obj, input_types: Dict[str, Callable]):
        tree, tensors = disassemble_tree(obj, False, all_attributes)
        retyped = []
        for t in tensors:
            originals = t.shape.only(dims)
            new_dims = originals.as_type(dim_type)
            for o, n in zip(originals, new_dims):
                input_types[n.name] = o.type
            retyped.append(rename_dims(t, originals, new_dims))
        return assemble_tree(tree, retyped), input_types

    def reverse_retype(obj, input_types: Dict[str, Callable]):
        tree, tensors = disassemble_tree(obj, False, all_attributes)
        retyped = []
        for t in tensors:
            output = t.shape.only(set(input_types))
            to_dims = [o.as_type(input_types[o.name]) for o in output]
            retyped.append(rename_dims(t, output, to_dims))
        return assemble_tree(tree, retyped)

    @wraps(f)
    def retyped_f(*args, **kwargs):
        input_types = {}
        retyped_args = []
        for arg in args:
            retyped_arg, input_types = forward_retype(arg, input_types)
            retyped_args.append(retyped_arg)
        output = f(*retyped_args, **kwargs)
        restored_output = reverse_retype(output, input_types)
        return restored_output

    return retyped_f


map_s2b = partial(map_types, dims=spatial, dim_type=batch)
map_s2b.__doc__ = "Map spatial dims to type batch. Short for `map_types(f, spatial, batch)`. "
map_i2b = partial(map_types, dims=instance, dim_type=batch)
map_i2b.__doc__ = "Map instance dims to type batch. Short for `map_types(f, spatial, batch)`. "
map_c2b = partial(map_types, dims=channel, dim_type=batch)
map_c2b.__doc__ = "Map channel dims to type batch. Short for `map_types(f, spatial, batch)`. "
map_d2b = partial(map_types, dims=dual, dim_type=channel)
map_d2b.__doc__ = "Map dual dims to type batch. Short for `map_types(f, spatial, batch)`. "

map_d2c = partial(map_types, dims=dual, dim_type=channel)
map_d2c.__doc__ = "Map dual dims to type channel. Short for `map_types(f, spatial, batch)`."
map_c2d = partial(map_types, dims=channel, dim_type=dual)
map_c2d.__doc__ = "Map channel dims to type dual. Short for `map_types(f, spatial, batch)`."


def broadcast(function=None, dims=shape, range=range, unwrap_scalars=True, simplify=False):
    """
    Function decorator for non-vectorized functions.
    When passing `Tensor` arguments to a broadcast function, the function is called once for each slice of the tensor.
    How tensors are sliced is determined by `dims`.
    Decorating a function with `broadcast` is equal to passing the function to `phi.math.map()`.

    See Also:
        `phiml.math.map`

    Args:
        function: Function to broadcast.
        dims: Dimensions which should be sliced.
            `function` is called once for each element in `dims`, i.e. `dims.volume` times.
            If `dims` is not specified, all dimensions from the `phiml.math.magic.Sliceable` values in `args` and `kwargs` will be mapped.
        range: Optional range function. Can be used to generate `tqdm` output by passing `trange`.
        unwrap_scalars: If `True`, passes the contents of scalar `Tensor`s instead of the tensor objects.
        simplify: If `True`, reduces constant dims of output tensors that don't vary across broadcast slices.

    Returns:
        Broadcast function
    """
    if function is None:
        kwargs = {k: v for k, v in locals().items() if v is not None}
        return partial(broadcast, **kwargs)
    @wraps(function)
    def broadcast_(*args, **kwargs):
        return map_(function, *args, dims=dims, range=range, unwrap_scalars=unwrap_scalars, simplify=simplify, **kwargs)
    return broadcast_


builtin_range = range


def iterate(map_function: Callable,
            iterations: Union[int, Shape],
            *x0,
            f_kwargs: dict = None,
            range: Callable = range,
            measure: Callable = None,
            substeps: int = 1,
            **f_kwargs_):
    """
    Repeatedly call `function`, passing the previous output as the next input.

    If the function outputs more values than the number of arguments in `x0`, only the first `len(x0)` ones are passed to `map_function`.
    However, all outputs will be returned by `iterate`.

    Args:
        map_function: Function to call. Must be callable as `f(x0, **f_kwargs)` and `f(f(x0, **f_kwargs), **f_kwargs)`.
        iterations: Number of iterations as `int` or single-dimension `Shape`.
            If `int`, returns the final output of `map_function`.
            If `Shape`, returns the trajectory (`x0` and all outputs of `map_function`), stacking the values along this dimension.
        x0: Initial positional arguments for `map_function`.
            Values that are initially `None` are not stacked with the other values if `iterations` is a `Shape`.
        range: Range function. Can be used to generate tqdm output by passing `trange`.
        measure: Function without arguments to call at the start and end (and in between if `isinstance(iterations, Shape)`) calls to `map_function`.
            The measure of each call to `map_function` is `measure()` after minus `measure()` before the call.
        substeps: If > 1, iterates the function multiple times for each recorded step.
            The returned trajectories as well as measurements only record the large steps, not the sub-steps.
            The `range` is also only used on large steps, not sub-steps.
        f_kwargs: Additional keyword arguments to be passed to `map_function`.
            These arguments can be of any type.
        f_kwargs_: More keyword arguments.

    Returns:
        final_or_trajectory: Stacked trajectory or final output of `map_function`, depending on `iterations`.
        measured: Only if `measure` was specified, returns the measured value or trajectory tensor.
    """
    if f_kwargs is None:
        f_kwargs = {}
    f_kwargs.update(f_kwargs_)
    assert isinstance(substeps, int), f"substeps must be an int but got {type(substeps)}"
    assert substeps >= 1, f"substeps must be >= 1"
    x = x0
    if isinstance(iterations, int):
        start_time = measure() if measure else None
        for _i in range(iterations):
            for _sub_i in builtin_range(substeps):
                x = map_function(*x[:len(x0)], **f_kwargs)
                x = x if isinstance(x, tuple) else (x,)
                if len(x) < len(x0):
                    raise AssertionError(f"Function to iterate must return at least {len(x0)} outputs to match input but got {x}")
        result = x[0] if len(x) == 1 else x
        return (result, measure() - start_time) if measure else result
    elif isinstance(iterations, Shape):
        xs = [x0]
        ts = [measure()] if measure else None
        for _i in range(iterations.size):
            for _sub_i in builtin_range(substeps):
                x = map_function(*x[:len(x0)], **f_kwargs)
                x = x if isinstance(x, tuple) else (x,)
                if len(x) < len(x0):
                    raise AssertionError(f"Function to iterate must return at least {len(x0)} outputs to match input but got {x}")
                elif len(x) > len(x0):
                    xs[0] = xs[0] + (None,) * (len(x) - len(x0))
            xs.append(x)
            if measure:
                ts.append(measure())
        xs = [stack(item[1:] if item[0] is None else item, iterations.with_size(None)) for item in zip(*xs)]
        result = xs[0] if len(xs) == 1 else xs
        ts = np.asarray(ts)
        return (result, wrap(ts[1:] - ts[:-1], iterations.with_size(None))) if measure else result
    else:
        raise ValueError(f"iterations must be an int or Shape but got {type(iterations)}")


def map_(function: Callable[..., Y], *args, dims: DimFilter = shape, range=range, unwrap_scalars=True, expand_results=False, simplify=False, **kwargs) -> Union[None, Tensor, Y]:
    """
    Calls `function` on slices of the arguments and returns the stacked result.

    Args:
        function: Function to be called on slices of `args` and `kwargs`.
            Must return one or multiple values that can be stacked.
            `None` may be returned but if any return value is `None`, all calls to `function` must return `None` in that position.
        *args: Positional arguments for `function`.
            Values that are `phiml.math.magic.Sliceable` will be sliced along `dims`.
        **kwargs: Keyword arguments for `function`.
            Values that are `phiml.math.magic.Sliceable` will be sliced along `dims`.
        dims: Dimensions which should be sliced.
            `function` is called once for each element in `dims`, i.e. `dims.volume` times.
            If `dims` is not specified, all dimensions from the `phiml.math.magic.Sliceable` values in `args` and `kwargs` will be mapped.
            Pass `object` to map only objects, not tensors of primitives (`dtype.kind == object`). This will select only `layout`-type dimensions.
        range: Optional range function. Can be used to generate `tqdm` output by passing `trange`.
        unwrap_scalars: If `True`, passes the contents of scalar `Tensor`s instead of the tensor objects.
        simplify: If `True`, reduces constant dims of output tensors that don't vary across mapped slices.

    Returns:
        `Tensor` of same shape as `value`.
    """
    sliceable_args = [v for v in args if isinstance(v, Shapable)]
    sliceable_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, Shapable)}
    extra_args = [v for v in args if not isinstance(v, Shapable)]
    extra_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Shapable)}
    if dims is object:
        dims_ = merge_shapes(*[object_dims(a) for a in sliceable_args], *[object_dims(a) for a in sliceable_kwargs.values()], allow_varying_sizes=True)
    else:
        dims_ = merge_shapes(*sliceable_args, *sliceable_kwargs.values(), allow_varying_sizes=True).only(dims)
    assert dims_.well_defined, f"All arguments must have consistent sizes for all mapped dimensions. Trying to map along {dims} but some have varying sizes (marked as None)."
    assert dims_.volume > 0, f"map dims must have volume > 0 but got {dims_}"
    results = []
    for _, idx in zip(range(dims_.volume), dims_.meshgrid()):
        idx_args = [slice_(v, idx) for v in sliceable_args]
        idx_kwargs = {k: slice_(v, idx) for k, v in sliceable_kwargs.items()}
        if unwrap_scalars:
            idx_args = [v.native() if isinstance(v, Tensor) and v.rank == 0 else v for v in idx_args]
            idx_kwargs = {k: v.native() if isinstance(v, Tensor) and v.rank == 0 else v for k, v in idx_kwargs.items()}
        idx_extra_args = list(extra_args)
        idx_all_args = [idx_args.pop(0) if isinstance(v, Shapable) else idx_extra_args.pop(0) for v in args]
        f_output = function(*idx_all_args, **idx_kwargs, **extra_kwargs)
        results.append(f_output)
    if isinstance(results[0], tuple):
        stacked: List[Optional[Tensor]] = []
        for i in range(len(results[0])):
            if any(r[i] is None for r in results):
                assert all(r[i] is None for r in results), f"map function returned None for some elements, {results}"
                stacked.append(None)
            else:
                stacked.append(math.stack([r[i] for r in results], dims_, expand_values=expand_results))
        return tuple(stacked)
    else:
        if any(r is None for r in results):
            assert all(r is None for r in results), f"map function returned None for some elements, {results}"
            return None
        return stack(results, dims_, expand_values=expand_results, simplify=simplify)


def identity(x):
    """
    Identity function for one argument.
    Vararg functions cannot be transformed as the argument names are unknown.

    Args:
        x: Positional argument.

    Returns:
        `x`
    """
    return x


def perf_counter(wait_for_tensor, *wait_for_tensors: Tensor) -> Tensor:
    """
    Get the time (`time.perf_counter()`) at which all `wait_for_tensors` are computed.
    If all tensors are already available, returns the current `time.perf_counter()`.

    Args:
        wait_for_tensor: `Tensor` that need to be computed before the time is measured.
        *wait_for_tensors: Additional tensors that need to be computed before the time is measured.

    Returns:
        Time at which all `wait_for_tensors` are ready as a scalar `Tensor`.
    """
    assert not _TRACING_LINEAR, f"Cannot use perf_counter inside a function decorated with @jit_compile_linear"
    if not _TRACING_JIT:
        return wrap(time.perf_counter())
    else:  # jit
        backend = _TRACING_JIT[0]._tracing_in_key.backend
        natives, _, _ = disassemble_tensors([wait_for_tensor, *wait_for_tensors], expand=False)
        natives = [n for n in natives if backend.is_tensor(n, only_native=True)]
        assert natives, f"in jit mode, perf_counter must be given at least one traced tensor, as the current time is evaluated after all tensors are computed."
        def perf_counter(*_wait_for_natives):
            return np.asarray(time.perf_counter())
        return wrap(backend.numpy_call(perf_counter, (), FLOAT64, *natives))
