import warnings
from typing import Union, Optional

from ._shape import shape
from ._nd import Tensor, DimFilter, channel, tensor, math, wrap, normalize, rename_dims, norm, squared_norm
from ._ops import clip, safe_div, stack_tensors


def length(*args, **kwargs):
    """Deprecated. Use `norm` instead."""
    warnings.warn("phiml.math.length is deprecated in favor of phiml.math.norm", DeprecationWarning, stacklevel=2)
    return norm(*args, **kwargs)


def vec_squared(*args, **kwargs):
    """Deprecated. Use `squared_norm` instead."""
    warnings.warn("phiml.math.vec_squared is deprecated in favor of phiml.math.squared_norm", DeprecationWarning, stacklevel=2)
    return squared_norm(*args, **kwargs)


def clip_length(vec: Tensor, min_len=0, max_len=1, vec_dim: DimFilter = channel, eps: Union[float, Tensor] = None):
    """
    Clips the length of a vector to the interval `[min_len, max_len]` while keeping the direction.
    Zero-vectors remain zero-vectors.

    Args:
        vec: `Tensor`
        min_len: Lower clipping threshold.
        max_len: Upper clipping threshold.
        vec_dim: Dimensions to compute the length over. By default, all channel dimensions are used to compute the vector length.
        eps: Minimum vector length. Use to avoid `inf` gradients for zero-length vectors.

    Returns:
        `Tensor` with same shape as `vec`.
    """
    warnings.warn("phiml.math.clip_length() is deprecated. Use PhiFlow's geometry functions instead.", DeprecationWarning)
    le = length(vec, vec_dim, eps)
    new_length = clip(le, min_len, max_len)
    return vec * safe_div(new_length, le)


def cross_product(vec1: Tensor, vec2: Tensor) -> Tensor:
    """
    Computes the cross product of two vectors in 2D.

    Args:
        vec1: `Tensor` with a single channel dimension called `'vector'`
        vec2: `Tensor` with a single channel dimension called `'vector'`

    Returns:
        `Tensor`
    """
    warnings.warn("phiml.math.cross_product() is deprecated. Use PhiFlow's geometry functions instead.", DeprecationWarning)
    vec1 = tensor(vec1)
    vec2 = tensor(vec2)
    spatial_rank = vec1.vector.size if 'vector' in vec1.shape else vec2.vector.size
    if spatial_rank == 2:  # Curl in 2D
        assert 'vector' in vec2.shape
        if 'vector' in vec1.shape:
            v1_x, v1_y = vec1.vector
            v2_x, v2_y = vec2.vector
            return v1_x * v2_y - v1_y * v2_x
        else:
            v2_x, v2_y = vec2.vector
            return vec1 * stack_tensors([-v2_y, v2_x], channel(vec2))
    elif spatial_rank == 3:  # Curl in 3D
        assert 'vector' in vec1.shape and 'vector' in vec2.shape, f"Both vectors must have a 'vector' dimension but got shapes {vec1.shape}, {vec2.shape}"
        v1_x, v1_y, v1_z = vec1.vector
        v2_x, v2_y, v2_z = vec2.vector
        return stack_tensors([
            v1_y * v2_z - v1_z * v2_y,
            v1_z * v2_x - v1_x * v2_z,
            v1_x * v2_y - v1_y * v2_x,
        ], vec1.shape['vector'])
    else:
        raise AssertionError(f'dims = {spatial_rank}. Vector product not available in > 3 dimensions')



def rotate_vector(vector: math.Tensor, angle: Optional[Union[float, math.Tensor]], invert=False, dim='vector') -> Tensor:
    """
    Rotates `vector` around the origin.

    Args:
        vector: n-dimensional vector with exactly one channel dimension
        angle: Euler angle(s) or rotation matrix.
            `None` is interpreted as no rotation.
        invert: Whether to apply the inverse rotation.

    Returns:
        Rotated vector as `Tensor`
    """
    warnings.warn("phiml.math.rotate_vector() is deprecated. Use PhiFlow's geometry functions instead.", DeprecationWarning)
    assert 'vector' in vector.shape, f"vector must have exactly a channel dimension named 'vector'"
    if angle is None:
        return vector
    matrix = rotation_matrix(angle, matrix_dim=channel(vector))
    if invert:
        matrix = rename_dims(matrix, '~vector,vector', matrix.shape['vector'] + matrix.shape['~vector'])
    assert matrix.vector.dual.size == vector.vector.size, f"Rotation matrix from {shape(angle)} is {matrix.vector.dual.size}D but vector {vector.shape} is {vector.vector.size}D."
    dim = vector.shape.only(dim)
    return math.dot(matrix, dim.as_dual(), vector, dim)


def rotation_matrix(x: Union[float, math.Tensor, None], matrix_dim=channel('vector')) -> Optional[Tensor]:
    """
    Create a 2D or 3D rotation matrix from the corresponding angle(s).

    Args:
        x:
            2D: scalar angle
            3D: Either vector pointing along the rotation axis with rotation angle as length or Euler angles.
            Euler angles need to be laid out along a `angle` channel dimension with dimension names listing the spatial dimensions.
            E.g. a 90° rotation about the z-axis is represented by `vec('angles', x=0, y=0, z=PI/2)`.
            If a rotation matrix is passed for `angle`, it is returned without modification.
        matrix_dim: Matrix dimension for 2D rotations. In 3D, the channel dimension of angle is used.

    Returns:
        Matrix containing `matrix_dim` in primal and dual form as well as all non-channel dimensions of `x`.
    """
    warnings.warn("phiml.math.rotation_matrix() is deprecated. Use PhiFlow's geometry functions instead.", DeprecationWarning)
    if x is None:
        return None
    if isinstance(x, Tensor) and '~vector' in x.shape and 'vector' in x.shape.channel and x.shape.get_size('~vector') == x.shape.get_size('vector'):
        return x  # already a rotation matrix
    elif 'angle' in shape(x) and shape(x).get_size('angle') == 3:  # 3D Euler angles
        assert channel(x).rank == 1 and channel(x).size == 3, f"x for 3D rotations needs to be a 3-vector but got {x}"
        s1, s2, s3 = math.sin(x).angle  # x, y, z
        c1, c2, c3 = math.cos(x).angle
        matrix_dim = matrix_dim.with_size(shape(x).get_item_names('angle'))
        return wrap([[c3 * c2, c3 * s2 * s1 - s3 * c1, c3 * s2 * c1 + s3 * s1],
                     [s3 * c2, s3 * s2 * s1 + c3 * c1, s3 * s2 * c1 - c3 * s1],
                     [-s2, c2 * s1, c2 * c1]], matrix_dim, matrix_dim.as_dual())  # Rz * Ry * Rx  (1. rotate about X by first angle)
    elif 'vector' in shape(x) and shape(x).get_size('vector') == 3:  # 3D axis + x
        angle = length(x)
        s, c = math.sin(angle), math.cos(angle)
        t = 1 - c
        k1, k2, k3 = normalize(x, epsilon=1e-12).vector
        matrix_dim = matrix_dim.with_size(shape(x).get_item_names('vector'))
        return wrap([[c + k1**2 * t, k1 * k2 * t - k3 * s, k1 * k3 * t + k2 * s],
                     [k2 * k1 * t + k3 * s, c + k2**2 * t, k2 * k3 * t - k1 * s],
                     [k3 * k1 * t - k2 * s, k3 * k2 * t + k1 * s, c + k3**2 * t]], matrix_dim, matrix_dim.as_dual())
    else:  # 2D rotation
        sin = wrap(math.sin(x))
        cos = wrap(math.cos(x))
        return wrap([[cos, -sin], [sin, cos]], matrix_dim, matrix_dim.as_dual())
