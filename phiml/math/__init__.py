"""
Vectorized operations, tensors with named dimensions.

This package provides a common interface for tensor operations.
Is internally uses NumPy, TensorFlow or PyTorch.

Main classes: `Tensor`, `Shape`, `DType`, `Extrapolation`.

The provided operations are not implemented directly.
Instead, they delegate the actual computation to either NumPy, TensorFlow or PyTorch, depending on the configuration.
This allows the user to write simulation code once and have it run with various computation backends.

See the documentation at https://tum-pbs.github.io/PhiML/
"""

from ..backend._dtype import DType
from ..backend import NUMPY, precision, set_global_precision, get_precision, set_global_default_backend as use

from ._shape import (
    shape, Shape, EMPTY_SHAPE, DimFilter,
    spatial, channel, batch, instance, dual,
    non_batch, non_spatial, non_instance, non_channel, non_dual, non_primal, primal,
    merge_shapes, concat_shapes, IncompatibleShapes,
    enable_debug_checks,
)

from ._magic_ops import (
    slice_ as slice, unstack,
    stack, concat, ncat, tcat, ccat, scat, icat, dcat, expand,
    rename_dims, rename_dims as replace_dims, pack_dims, dpack, ipack, spack, cpack, unpack_dim, flatten, squeeze,
    b2i, c2b, c2d, i2b, s2b, si2d, p2d, d2i, d2s,
    copy_with, replace, find_differences
)

from ._tensors import Tensor, wrap, tensor, layout, native, numpy_ as numpy, reshaped_native, reshaped_numpy, Dict, to_dict, from_dict, is_scalar, BROADCAST_FORMATTER as f, save, load

from ._sparse import dense, get_sparsity, get_format, to_format, is_sparse, sparse_tensor, stored_indices, stored_values, tensor_like, matrix_rank

from .extrapolation import Extrapolation, as_extrapolation

from ._ops import (
    backend_for as choose_backend, all_available, convert, seed, to_device,
    reshaped_tensor, copy, native_call,
    print_ as print,
    slice_off,
    zeros, ones, fftfreq, random_normal, random_normal as randn, random_uniform, random_uniform as rand,
    meshgrid, linspace, arange, arange as range, range_tensor,  # creation operators (use default backend)
    zeros_like, ones_like,
    pad,
    swap_axes,  # reshape operations
    sort, dsort, psort, isort, ssort, csort,
    safe_div,
    where, nonzero, ravel_index, unravel_index,
    sum_ as sum, finite_sum, dsum, psum, isum, ssum, csum,
    mean, finite_mean, dmean, pmean, imean, smean, cmean,
    std,
    prod, dprod, pprod, sprod, iprod, cprod,
    max_ as max, dmax, pmax, smax, imax, cmax, finite_max,
    min_ as min, dmin, pmin, smin, imin, cmin, finite_min,
    any_ as any, all_ as all, quantile, median,  # reduce
    at_max, at_min, argmax, argmin,
    dot,
    abs_ as abs, sign,
    round_ as round, ceil, floor,
    maximum, minimum, clip,
    sqrt, exp, erf, log, log2, log10, sigmoid, soft_plus, softmax,
    sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, log_gamma, factorial, incomplete_gamma,
    to_float, to_int32, to_int64, to_complex, imag, real, conjugate, angle,
    radians_to_degrees, degrees_to_radians,
    boolean_mask,
    is_finite, is_nan, is_inf, nan_to_0,
    closest_grid_values, grid_sample, scatter, gather,
    histogram,
    fft, ifft, convolve, cumulative_sum,
    dtype, cast,
    close, always_close, assert_close, equal,
    stop_gradient,
    pairwise_differences, pairwise_differences as pairwise_distances, map_pairs,
    with_diagonal,
    eigenvalues, svd,
    contains, count_occurrences, count_intersections,
)

from ._nd import (
    shift, index_shift,
    vec, const_vec, norm, squared_norm, normalize, normalize as vec_normalize,
    dim_mask,
    normalize_to,
    l1_loss, l2_loss, frequency_loss,
    spatial_gradient, laplace,
    neighbor_reduce, neighbor_mean, neighbor_sum, neighbor_max, neighbor_min, at_min_neighbor, at_max_neighbor,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, sample_subgrid,
    masked_fill, finite_fill,
    find_closest,
)

from ._trace import matrix_from_function

from ._functional import (
    LinearFunction, jit_compile_linear, jit_compile,
    jacobian, gradient, custom_gradient, print_gradient,
    safe_mul,
    map_types, map_s2b, map_i2b, map_c2b, map_d2b, map_d2c, map_c2d,
    broadcast,
    iterate,
    identity,
    trace_check,
    map_ as map,
    when_available,
    perf_counter,
)

from ._optimize import solve_linear, solve_nonlinear, minimize, Solve, SolveInfo, ConvergenceException, NotConverged, Diverged, SolveTape, factor_ilu

from ._deprecated import clip_length, cross_product, cross_product as cross, rotate_vector, rotation_matrix, length, length as vec_length, vec_squared

import sys as _sys
math = _sys.modules[__name__]
"""Convenience alias for the module `phiml.math`.
This way, you can import the module and contained items in one line.
```
from phiml.math import math, Tensor, wrap, extrapolation, l2_loss
```"""

PI = 3.14159265358979323846
"""Value of π to double precision """
pi = PI  # intentionally undocumented, use PI instead. Exists only as an anlog to numpy.pi

INF = float("inf")
""" Floating-point representation of positive infinity. """
inf = INF  # intentionally undocumented, use INF instead. Exists only as an anlog to numpy.inf


NAN = float("nan")
""" Floating-point representation of NaN (not a number). """
nan = NAN  # intentionally undocumented, use NAN instead. Exists only as an anlog to numpy.nan

NUMPY = NUMPY  # to show up in pdoc
"""Default backend for NumPy arrays and SciPy objects."""

f = f
"""
Automatic mapper for broadcast string formatting of tensors, resulting in tensors of strings.
Used with the special `-f-` syntax.

Examples:
    >>> from phiml.math import f
    >>> -f-f'String containing {tensor1} and {tensor2:.1f}'
    # Result is a str tensor containing all dims of tensor1 and tensor2
"""

__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'Extrapolation': False,
    'Shape.__init__': False,
    'SolveInfo.__init__': False,
    'TensorDim.__init__': False,
    'ConvergenceException.__init__': False,
    'Diverged.__init__': False,
    'NotConverged.__init__': False,
    'LinearFunction.__init__': False,
}
