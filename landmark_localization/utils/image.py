import vtk
import copy
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from typing import List, Tuple, Dict

from landmark_localization.utils.vtk import (
    get_ijk_to_RAS_matrix, 
    get_RAS_to_ijk_matrix,
    get_ijk_to_LPS_matrix, 
    get_LPS_to_ijk_matrix,
    array_from_VTK_matrix, 
    vtkMatrix4x4
)

from landmark_localization.utils.vtk import Point3D

class Geometry:
    """A class describing the geometric properties of a medical image.
    It holds information about origin, spacing and direction, as well as
    the matrices needed to convert between world coordinates (LPS) and 
    image indexes (ijk).

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    origin: tuple
    spacing: tuple
    direction: tuple
    desc: str
    ijk_to_RAS: vtkMatrix4x4
    RAS_to_ijk: vtkMatrix4x4
    ijk_to_LPS: vtkMatrix4x4
    LPS_to_ijk: vtkMatrix4x4
    
    def __init__(self, origin, spacing, direction, desc):
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.desc = desc
        
        self.ijk_to_RAS_matrix = get_ijk_to_RAS_matrix(origin=list(self.origin), 
                                            spacing=list(tuple(self.spacing) + (1.,)), 
                                            direction=list(self.direction))
        self.RAS_to_ijk_matrix = get_RAS_to_ijk_matrix(image=None,
                                                ijk_to_RAS_matrix=self.ijk_to_RAS_matrix)
        
        self.ijk_to_LPS_matrix = get_ijk_to_LPS_matrix(origin=list(self.origin), 
                                            spacing=list(tuple(self.spacing) + (1.,)), 
                                            direction=list(self.direction))
        self.LPS_to_ijk_matrix = get_LPS_to_ijk_matrix(image=None,
                                                ijk_to_LPS_matrix=self.ijk_to_LPS_matrix)

    def convert(self, 
                points: np.ndarray,
                new_spacing: np.ndarray,
                mode: str = "ijk -> LPS") -> np.ndarray:
        entities = ["ijk", "LPS", "RAS"]
        mode_norm = mode.replace(" ", "")
        from_, to_ = mode_norm.split("->")
        if from_ == to_:
            raise ValueError("Starting and destiantion coordinate systems can't be the same.")
        
        if from_ not in entities or to_ not in entities:
            raise ValueError("Starting and/or destination coordinate systems are not supported.")
        
        # respace points if a new spacing is specified
        if new_spacing:
            respaced_points = np.zeros_like(points)
            for i, point in enumerate(points):
                point_resampled = get_resized_coords_no_cast(point, orig_spacing, new_spacing) # typically new_spacing is np.flip(resize_factor)
                respaced_points[i, :] = point_resampled
            respaced_points = [[float(c) for c in arr] for arr in respaced_points] 

        if mode_norm == "ijk->LPS":
            LPS_points = []
            for point in points:
                # convert ijk to RAS
                RAS_point = self.ijk_to_RAS_matrix.MultiplyPoint(point+[1.0])[0:3]
                LPS_coords = [RAS_point[0] * -1, RAS_point[1] * -1, RAS_point[2]]
                LPS_points.append(LPS_coords)
            to_return = LPS_points     
        elif mode_norm == "LPS->ijk":
            ijk_points = []
            for point in points:
                # convert ijk to RAS
                RAS_point = [point[0] * -1, point[1] * -1, point[2]]
                ijk_coords = self.RAS_to_ijk_matrix.MultiplyPoint(RAS_point+[1.0])[0:3]
                ijk_points.append(ijk_coords)
            to_return = ijk_points
        
        return to_return
    
    def __repr__(self):
        return f"Geometry(origin={self.origin}, spacing={self.spacing}, " \
                f"direction={self.direction})\nDescription: {self.desc}"
                
    def __str__(self):
        return f"\nGeometry data - {self.desc}\nOrigin: {self.origin}\n" \
            f"Spacing: {self.spacing}\nDirection: {self.direction}"

def resample(imgs, spacing, new_spacing, order=2):
    """
    Resample 3D or 4D image to new spacing using PyTorch.
    
    :param imgs: Original image array (numpy array).
    :param spacing: spacing of the original image (numpy array).
    :param new_spacing: new spacing (numpy array).
    :param order: interpolation order, 2 corresponds to bilinear.
    :return: tuple of new image, true spacing, and resize factor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert images to PyTorch tensor and move to GPU if available
    imgs_tensor = torch.from_numpy(imgs).float().to(device)
    
    # Calculate new shape
    new_shape = np.round(imgs.shape * (spacing / new_spacing))
    true_spacing = spacing * (imgs.shape / new_shape)
    resize_factor = new_shape / imgs.shape

    # Add batch and channel dimensions if they don't exist
    if len(imgs_tensor.shape) == 3:
        imgs_tensor = imgs_tensor[None, None, :, :, :]
    elif len(imgs_tensor.shape) == 4:
        imgs_tensor = imgs_tensor[None, :, :, :, :]

    # Determine the interpolation mode
    if order == 2:
        mode = 'trilinear'  # for 3D images
    else:
        raise ValueError('Currently only order=2 (trilinear) is supported.')

    # Perform the resampling
    imgs_resampled = F.interpolate(imgs_tensor, size=tuple(int(dim) for dim in new_shape), mode=mode, align_corners=False)
    
    # Remove batch and channel dimensions and move back to CPU
    imgs_resampled = imgs_resampled[0, 0].cpu().numpy()
    
    return imgs_resampled, true_spacing, resize_factor



def extract_patch(image: np.ndarray, coords: Point3D, cut_size: int = 9) -> np.ndarray:
    """Extract a cubic patch with side `cut_size*2+1` center at `coords` from `image`.

    Parameters
    ----------
    image : np.ndarray
        The image to extract the patch from.
    coords : Point3D
        Center of the patch.
    cut_size : int, optional
        Size of cubic patch side, by default 9

    Returns
    -------
    np.ndarray
        The cubic patch
    """
    coord_x, coord_y, coord_z = coords
    patch_size = cut_size * 2 + 1

    # Define start and end indices for the patch
    start_x, end_x = max(coord_x - cut_size, 0), min(coord_x + cut_size, image.shape[2] - 1)
    start_y, end_y = max(coord_y - cut_size, 0), min(coord_y + cut_size, image.shape[1] - 1)
    start_z, end_z = max(coord_z - cut_size, 0), min(coord_z + cut_size, image.shape[0] - 1)

    # Initialize the patch with a fill value
    fill_value = -50  # TODO: Adjust based on your data
    patch = np.full((patch_size, patch_size, patch_size), fill_value, dtype=image.dtype)


    # Extract the valid region and place it into the patch
    patch[start_z - coord_z + cut_size:end_z - coord_z + cut_size + 1,
          start_y - coord_y + cut_size:end_y - coord_y + cut_size + 1,
          start_x - coord_x + cut_size:end_x - coord_x + cut_size + 1] = \
          image[start_z:end_z + 1, start_y:end_y + 1, start_x:end_x + 1]

    return patch
                                    
        
    return patch

def get_resized_coords(coords_orig: np.ndarray, 
                       resize_factor: float) -> np.ndarray:
    """
    Calculates the resized coordinates of an array of points based on 
    their original coordinates and the resize factor needed to obtain the resized image.
    It rounds the coordinates to 0 decimal places and applies an int cast
    to obtain actual image indexes.
    
    Parameters
    ----------
    coords_orig: np.ndarray
        The coordinates of points on the original image.
    resize_factor: float
        The resize factor between the original and the resized image.
    
    Returns
    -------
    np.ndarray
        The resampled coordinates.
    """
    return np.round(coords_orig * resize_factor).astype("int")


def get_resized_coords_no_cast(coords_orig: np.ndarray, 
                               spacing_orig: tuple, 
                               spacing_new: tuple):
    """
    Calculates the resized coordinates of an array of points based on 
    their original coordinates and the old and new spacings.
    No rounding nor cast to int.
    
    Parameters
    ----------
    coords_orig: np.ndarray
        The coordinates of points on the original image.
    spacing_orig: tuple
        Spacing of unresampled (unresized) image.
    spacing_new: tuple
        Spacing of the resampled (resized) image.
    
    Returns
    -------
    np.ndarray
        The resampled coordinates.
    """    
    return (x / spacing_old) * spacing_new
