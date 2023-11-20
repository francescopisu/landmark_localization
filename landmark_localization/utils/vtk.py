import random
import os, sys
import numpy as np
from typing import Union
import SimpleITK as sitk
import numpy.typing as npt
from vtkmodules.vtkCommonMath import vtkMatrix3x3, vtkMatrix4x4
from typing import Union, List, Tuple, Annotated, Literal, TypeVar

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Union[np.ndarray, List[List[float]]]

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

Point3D = Annotated[npt.NDArray[DType], Literal[3, 3]]
Point4D = Annotated[npt.NDArray[DType], Literal[4, 4]]

Seed: TypeAlias = Union[int, random.seed, np.random.seed]


def update_VTK_matrix_from_array(vtk_matrix, np_arr) -> Union[vtkMatrix3x3, vtkMatrix4x4]:
    """Update VTK matrix values from a numpy array.
    :param vtk_matrix: Empty VTK matrix (vtkMatrix4x4 or vtkMatrix3x3) that will be updated
    :param np_arr: input numpy array
    To set numpy array from VTK matrix, use :py:meth:`arrayFromVTKMatrix`.
    """
    if isinstance(vtk_matrix, vtkMatrix4x4):
        matrix_size = 4
    elif isinstance(vtk_matrix, vtkMatrix3x3):
        matrix_size = 3
    else:
        raise RuntimeError("Output vmatrix must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    if np_arr.shape != (matrix_size, matrix_size):
        raise RuntimeError("Input narray size must match output vmatrix size ({0}x{0})".format(matrix_size))

    vtk_matrix.DeepCopy(np_arr.ravel()) # copy content from np_arr into vtk_matrix

    return vtk_matrix


def array_from_VTK_matrix(vtk_matrix: Union[vtkMatrix3x3, vtkMatrix4x4]) -> np.array:
    """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
    The returned array is just a copy and so any modification in the array will not affect the input matrix.
    To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
    :py:meth:`updateVTKMatrixFromArray`.
    """

    if isinstance(vtk_matrix, vtkMatrix4x4):
        matrix_size = 4
    elif isinstance(vtk_matrix, vtkMatrix3x3):
        matrix_size = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")

    np_arr = np.eye(matrix_size)
    vtk_matrix.DeepCopy(np_arr.ravel(), vtk_matrix) # copy content of vtk_matrix into np_arr

    return np_arr

def get_ijk_to_LPS_matrix(origin, spacing, direction, image: sitk.Image = None) -> vtkMatrix4x4:
    """
    """
    if image is not None:
        direction = image.GetDirection() 
        origin = list(image.GetOrigin())
        spacing = list(image.GetSpacing() + (1,))
        
    ijk_to_LPS_matrix = np.diag(spacing)
    ijk_to_LPS_matrix[:-1, -1] = origin

    # should we multiply by -1 the first two values of the origin ?
    # we currently do it in ijk_to_RAS 
    
    ijk_to_LPS_matrix_vtk = vtkMatrix4x4()
    update_VTK_matrix_from_array(ijk_to_LPS_matrix_vtk, ijk_to_LPS_matrix)

    return ijk_to_LPS_matrix_vtk


def get_ijk_to_RAS_matrix(origin, spacing, direction, image: sitk.Image = None) -> vtkMatrix4x4:
    """
    DICOM images follow the LPS coordinate system.
    RAS inverts the first two axes. 
    In order to get from image space to RAS world/patient space,
    we need to invert the first two elements of the origin (O_x -> -O_x, O_y -> -O_y)
    and apply the following formula:
    coord_x = (coord_i - O_x) / spacing_x
    coord_y = (coord_i - O_y) / spacing_y
    coord_z = (coord_i - O_z) / spacing_z
    """
    if image is not None:
        direction = image.GetDirection() 
        origin = list(image.GetOrigin())
        spacing = list(image.GetSpacing() + (1.,))
    
    # spacing[0] = spacing[0] * -1
    # spacing[1] = spacing[1] * -1
    
    ijk_to_RAS_matrix = np.diag(spacing)

    # added
    direction_matrix = np.array(direction).reshape(3,3)
    direction_matrix[0, 0] = direction_matrix[0, 0] * -1
    direction_matrix[1, 1] = direction_matrix[1, 1] * -1
    ijk_to_RAS_matrix[:3, :3] = ijk_to_RAS_matrix[:3, :3] * direction_matrix
    
    origin[0] = origin[0] * -1
    origin[1] = origin[1] * -1
    ijk_to_RAS_matrix[:-1, -1] = origin

    ijk_to_RAS_matrix_vtk = vtkMatrix4x4()
    update_VTK_matrix_from_array(ijk_to_RAS_matrix_vtk, ijk_to_RAS_matrix)

    return ijk_to_RAS_matrix_vtk


def get_RAS_to_ijk_matrix(image: sitk.Image, ijk_to_RAS_matrix: vtkMatrix4x4) -> vtkMatrix4x4:
    """Retrieve ijk to RAS matrix in vtk format from a sitk Image.
    """
    if image is None and ijk_to_RAS_matrix is None:
        raise ValueError("Either a SimpleITK image or an ijk to RAS matrix must be provided.")
    
    if not ijk_to_RAS_matrix and image is not None:
        ijk_to_RAS_matrix = get_ijk_to_RAS_matrix(image)
    
    ijk_to_RAS = array_from_VTK_matrix(ijk_to_RAS_matrix)
    RAS_to_ijk_arr = np.linalg.inv(ijk_to_RAS)
    RAS_to_ijk_matrix = vtkMatrix4x4()
    update_VTK_matrix_from_array(RAS_to_ijk_matrix, RAS_to_ijk_arr)

    return RAS_to_ijk_matrix


def invert_matrix(matrix: vtkMatrix4x4) -> vtkMatrix4x4:
    """Invert a vtkMatrix

    Parameters
    ----------
    matrix : vtkMatrix4x4
        The vtkMatrix to be inverted

    Returns
    -------
    vtkMatrix4x4
        The inverted matrix
    """
    np_matrix = array_from_VTK_matrix(matrix)
    inv_matrix = np.linalg.inv(np_matrix)
    vtk_inv_matrix = vtkMatrix4x4()
    update_VTK_matrix_from_array(vtk_inv_matrix, inv_matrix)

    return vtk_inv_matrix    
    
def get_LPS_to_ijk_matrix(origin=None, spacing=None, direction=None, 
                          image: 'sitk.Image'=None, 
                          ijk_to_LPS_matrix: 'vtkMatrix4x4'=None) -> 'vtkMatrix4x4':
    """
    Retrieve the IJK to RAS matrix in VTK format from a SimpleITK Image or from geometric information.
    
    Parameters:
    - origin: The origin of the image volume.
    - spacing: The spacing of the pixels/voxels in the image volume.
    - direction: The direction cosines for the image volume.
    - image: A SimpleITK Image object.
    - ijk_to_LPS_matrix: A precomputed IJK to LPS matrix in VTK format.
    
    Returns:
    - A VTK matrix representing the LPS to IJK transformation.
    """

    if ijk_to_LPS_matrix is not None:
        ijk_to_LPS = array_from_VTK_matrix(ijk_to_LPS_matrix)
    elif image is not None:
        ijk_to_LPS = array_from_VTK_matrix(get_ijk_to_LPS_matrix(image))
    elif all(param is not None for param in [origin, spacing, direction]):
        ijk_to_LPS = array_from_VTK_matrix(get_ijk_to_LPS_matrix(origin=origin, 
                                                                 spacing=spacing + (1.,), 
                                                                 direction=direction))
    else:
        raise ValueError("Either an image, an ijk to LPS matrix or geometric information must be provided.")
    
    # Inverting the matrix to go from LPS to IJK.
    LPS_to_ijk_arr = np.linalg.inv(ijk_to_LPS)
    LPS_to_ijk = vtkMatrix4x4()
    update_VTK_matrix_from_array(LPS_to_ijk, LPS_to_ijk_arr)

    return LPS_to_ijk


def ijk_to_RAS(point_ijk: Point4D, ijk_to_RAS_matrix: vtkMatrix4x4) -> Point3D:
    return ijk_to_RAS_matrix.MultiplyPoint(point_ijk)[0:3]

def RAS_to_ijk(point_RAS: Point4D, RAS_to_ijk_matrix: vtkMatrix4x4) -> Point3D:
    return RAS_to_ijk_matrix.MultiplyPoint(point_RAS)[0:3]

