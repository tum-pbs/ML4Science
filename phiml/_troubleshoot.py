import gc
from contextlib import contextmanager
from os.path import dirname

import packaging.version


def assert_minimal_config():  # raises AssertionError
    import sys
    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "Φ-ML requires Python 3.6 or newer to run"

    try:
        import numpy
    except ImportError:
        raise AssertionError("Φ-ML is unable to run because NumPy is not installed.")
    try:
        import scipy
    except ImportError:
        raise AssertionError("Φ-ML is unable to run because SciPy is not installed.")
    from . import math
    with math.NUMPY:
        a = math.ones()
        math.assert_close(a + a, 2)


def troubleshoot():
    from . import __version__
    return f"Φ-ML {__version__} at {dirname(__file__)}\n"\
           f"PyTorch: {troubleshoot_torch()}\n"\
           f"Jax: {troubleshoot_jax()}\n"\
           f"TensorFlow: {troubleshoot_tensorflow()}\n"  # TF last so avoid VRAM issues


def troubleshoot_tensorflow():
    from . import math
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors
    try:
        import tensorflow
    except ImportError:
        return "Not installed."
    tf_version = f"{tensorflow.__version__} at {dirname(tensorflow.__file__)}"
    try:
        import tensorflow_probability
        tf_prob = f"tensorflow_probability {tensorflow_probability.__version__}"
    except ImportError:
        tf_prob = "'tensorflow_probability' missing. Using fallback implementations instead. To install it, run  $ pip install tensorflow-probability"
    try:
        from .backend import tensorflow as tf
    except BaseException as err:
        return f"Installed ({tf_version}) but not available due to internal error: {err}\n{tf_prob}"
    try:
        gpu_count = len(tf.TENSORFLOW.list_devices('GPU'))
    except BaseException as err:
        return f"Installed ({tf_version}) but device initialization failed with error: {err}\n{tf_prob}"
    with tf.TENSORFLOW:
        try:
            math.assert_close(math.ones() + math.ones(), 2)
            # TODO cuDNN math.convolve(math.ones(batch=8, x=64), math.ones(x=4))
        except BaseException as err:
            return f"Installed ({tf_version}) but tests failed with error: {err}\n{tf_prob}"
    if gpu_count == 0:
        return f"Installed ({tf_version}), {gpu_count} GPUs available.\n{tf_prob}"
    else:
        from .backend.tensorflow._tf_cuda_resample import librariesLoaded
        if librariesLoaded:
            cuda_str = 'CUDA kernels available.'
        else:
            import platform
            if platform.system().lower() != 'linux':
                cuda_str = f"Optional TensorFlow CUDA kernels not available and compilation not recommended on {platform.system()}. GPU will be used nevertheless."
            else:
                cuda_str = f"Optional TensorFlow CUDA kernels not available. GPU will be used nevertheless. Clone the Φ-ML source from GitHub and run 'python setup.py tf_cuda' to compile them. See https://tum-pbs.github.io/PhiML/Installation_Instructions.html"
        return f"Installed ({tf_version}), {gpu_count} GPUs available.\n{cuda_str}\n{tf_prob}"


def troubleshoot_torch():
    from . import math
    try:
        import torch
    except ImportError:
        return "Not installed."
    torch_version = f"{torch.__version__} at {dirname(torch.__file__)}"
    try:
        from .backend import torch as torch_
    except BaseException as err:
        return f"Installed ({torch_version}) but not available due to internal error: {err}"
    try:
        gpu_count = len(torch_.TORCH.list_devices('GPU'))
    except BaseException as err:
        return f"Installed ({torch_version}) but device initialization failed with error: {err}"
    with torch_.TORCH:
        try:
            math.assert_close(math.ones() + math.ones(), 2)
        except BaseException as err:
            return f"Installed ({torch_version}) but tests failed with error: {err}"
    if torch_version.startswith('1.10.'):
        return f"Installed ({torch_version}), {gpu_count} GPUs available. This version has known bugs with JIT compilation. Recommended: 1.11+ or 1.8.2 LTS"
    if torch_version.startswith('1.9.'):
        return f"Installed ({torch_version}), {gpu_count} GPUs available. You may encounter problems importing torch.fft. Recommended: 1.11+ or 1.8.2 LTS"
    return f"Installed ({torch_version}), {gpu_count} GPUs available."


def troubleshoot_jax():
    from . import math
    try:
        import jax
        import jaxlib
    except ImportError:
        return "Not installed."
    version = f"jax {jax.__version__} at {dirname(jax.__file__)}, jaxlib {jaxlib.__version__}"
    try:
        from .backend import jax as jax_
    except BaseException as err:
        return f"Installed ({version}) but not available due to internal error: {err}"
    try:
        gpu_count = len(jax_.JAX.list_devices('GPU'))
    except BaseException as err:
        return f"Installed ({version}) but device initialization failed with error: {err}"
    with jax_.JAX:
        try:
            math.assert_close(math.ones() + math.ones(), 2)
        except BaseException as err:
            return f"Installed ({version}) but tests failed with error: {err}"
    if packaging.version.parse(jax.__version__) < packaging.version.parse('0.2.20'):
        return f"Installed ({version}), {gpu_count} GPUs available. This is an old version of Jax that may not support all required features, e.g. sparse matrices."
    return f"Installed ({version}), {gpu_count} GPUs available."


@contextmanager
def plot_solves():
    """
    While `plot_solves()` is active, certain performance optimizations and algorithm implementations may be disabled.
    """
    from . import math
    import pylab
    cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']
    with math.SolveTape(record_trajectories=True) as solves:
        try:
            yield solves
        finally:
            for i, result in enumerate(solves):
                assert isinstance(result, math.SolveInfo)
                from .math._tensors import disassemble_tree
                from phiml.math._magic_ops import value_attributes
                _, (residual,) = disassemble_tree(result.residual, cache=False, attr_type=value_attributes)
                residual_mse = math.mean(math.sqrt(math.sum(residual ** 2)), residual.shape.without('trajectory'))
                residual_mse_max = math.max(math.sqrt(math.sum(residual ** 2)), residual.shape.without('trajectory'))
                # residual_mean = math.mean(math.abs(residual), residual.shape.without('trajectory'))
                residual_max = math.max(math.abs(residual), residual.shape.without('trajectory'))
                pylab.plot(residual_mse.numpy(), label=f"{i}: {result.method}", color=cycle[i % len(cycle)])
                pylab.plot(residual_max.numpy(), '--', alpha=0.2, color=cycle[i % len(cycle)])
                pylab.plot(residual_mse_max.numpy(), alpha=0.2, color=cycle[i % len(cycle)])
                print(f"Solve {i}: {result.method} ({1000 * result.solve_time:.1f} ms)\n"
                      f"\t{result.solve}\n"
                      f"\t{result.msg}\n"
                      f"\tConverged: {result.converged.trajectory[-1]}\n"
                      f"\tDiverged: {result.diverged.trajectory[-1]}\n"
                      f"\tIterations: {result.iterations.trajectory[-1]}\n"
                      f"\tFunction evaulations: {result.function_evaluations.trajectory[-1]}")
            pylab.yscale('log')
            pylab.ylabel("Residual: MSE / max / individual max")
            pylab.xlabel("Iteration")
            pylab.title(f"Solve Convergence")
            pylab.legend(loc='upper right')
            pylab.savefig(f"pressure-solvers-FP32.png")
            pylab.show()


def count_tensors_in_memory(min_print_size: int = None):
    import sys
    import gc
    from .math import Tensor

    gc.collect()
    total = 0
    bytes = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Tensor):
                total += 1
                size = obj.shape.volume * obj.dtype.itemsize
                bytes += size
                if isinstance(min_print_size, int) and size >= min_print_size:
                    print(f"Tensor '{obj}' ({sys.getrefcount(obj)} references)")
                    # referrers = gc.get_referrers(obj)
                    # print([type(r) for r in referrers])
        except Exception:
            pass
    print(f"There are {total} Φ-ML Tensors with a total size of {bytes / 1024 / 1024:.1f} MB")


def cache_all_tensors(print=print):
    import gc
    gc.collect()
    from .math._tensors import Dense, TensorStack
    for obj in gc.get_objects():
        if isinstance(obj, Dense):
            if len(obj._shape) > len(obj._names):
                if print is not None:
                    print(f"Expanding tensor with shape {obj._shape} from {obj._names} {type(obj._native).__name__} {obj._native}")
                _check_for_tracers(obj)
        elif isinstance(obj, TensorStack):
            if not obj.requires_broadcast:
                if print is not None:
                    print(f"Caching tensor stack with shape {obj._shape} along {obj._stack_dim}. Contents: {obj._tensors}")
                _check_for_tracers(obj)


def _check_for_tracers(tensor):
    try:
        tensor._contiguous()
    except BaseException as exc:
        print(f"ERROR    Expansion failed. {exc}")
        path = find_variable_reference(tensor)
        print(f"Reference path to tensor: {path}")
        if hasattr(tensor, '_init_stack'):
            print("Tensor creation stack trace:")
            for frame in tensor._init_stack:
                filename, line_number, function_name, line = frame
                print(f"{filename}:{line_number} ({function_name})    {line}")
        else:
            print("Enable debug checks to obtain the stack trace for the Tensor's creation.")


def find_jax_tracers(root=None, closures=True):
    from jax.interpreters.ad import JVPTracer
    from jax.core import Tracer
    import gc
    gc.collect()
    if root is None:
        for obj in gc.get_objects():
            if isinstance(obj, JVPTracer):
                print(f"Found JVPTracer {obj} at {find_variable_reference(obj, closures=closures)}")
            elif isinstance(obj, Tracer):
                print(f"Found JIT Tracer {obj} at {find_variable_reference(obj, closures=closures)}")
    else:
        _find_jax_tracers(root, "", set(), 0)


def _find_jax_tracers(obj, prefix, indexed: set, depth: int, max_depth=50):
    if id(obj) in indexed:
        return None
    if depth >= max_depth:
        print(f"max depth reached at {prefix}")
    indexed.add(id(obj))
    from jax.interpreters.ad import JVPTracer
    from jax.core import Tracer
    if isinstance(obj, (JVPTracer, Tracer)):
        print(prefix, ":", type(obj).__name__)
        return
    if isinstance(obj, (tuple, list)):
        for i, e in enumerate(obj):
            _find_jax_tracers(e, prefix + f"[{i}]", indexed, depth+1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _find_jax_tracers(v, prefix + f"[{k}]", indexed, depth+1)
            _find_jax_tracers(k, prefix + f".keys<{k}>", indexed, depth+1)
    else:
        if hasattr(obj, '__dict__'):
            attrs = obj.__dict__
            for k, v in attrs.items():
                _find_jax_tracers(v, prefix + f".{k}", indexed, depth+1)
    return None


def find_variable_reference(obj, closures=True):
    gc.collect()
    existing_ids = set(id(o) for o in gc.get_objects())
    indexed = set()
    return _find_variable_reference(obj, existing_ids, indexed, closures)[0]


def _find_variable_reference(obj, existing_ids: set, indexed: set, closures: bool):
    referrers = gc.get_referrers(obj)
    for ref in reversed(referrers):
        if id(ref) in existing_ids and id(ref) not in indexed:
            indexed.add(id(ref))
            if isinstance(ref, dict) and '__package__' in ref and '__name__' in ref:
                return f"{ref['__name__']}", ref
            else:
                path, parent = _find_variable_reference(ref, existing_ids, indexed, closures)
                if parent is not None:
                    if isinstance(parent, dict):
                        keys = [k for k, v in parent.items() if v is ref]
                        name = keys[0] if keys else f"<unknown>"
                        name = f".{name}" if '__name__' in parent and '__doc__' in parent else f"[{name}]"
                    elif isinstance(parent, (tuple, list)):
                        name = f"[{[i for i, v in enumerate(parent) if v is ref][0]}]"
                    else:
                        for a in dir(parent):
                            try:
                                if getattr(parent, a) is ref:
                                    name = f".{a}"
                                    break
                            except Exception:
                                pass  # failed to get value of property
                        else:
                            name = ".<unknown>"
                    if not closures and callable(parent) and name == '.__closure__':
                        return None, None
                    return path + name, ref
    return None, None
