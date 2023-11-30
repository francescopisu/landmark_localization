import sys
import vtk
import cv2
import copy
import torch
import torch.nn.functional as F
from skimage import measure
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from typing import List, Tuple, Dict, Union

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
                points: Union[List, List[List], List[np.ndarray], np.ndarray], 
                new_spacing: np.ndarray = None,
                mode: str = "ijk -> LPS") -> np.ndarray:
        """Convert a one or more points in the 3d space between IJK and LPS

        Parameters
        ----------
        points : Union[List, List[List], List[np.ndarray], np.ndarray]
            The points to be converted
        new_spacing : np.ndarray, optional
            Spacing for respace coordinates, by default None
        mode : str, optional
            Direction of conversion, by default "ijk -> LPS"

        Returns
        -------
        np.ndarray
            The point coordinates after conversion.

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if isinstance(points, list):
            points = np.array(points).reshape(-1, 3)
        
        if len(points) == 1:
            # single point (i.e., 3d coords of a single point)
            points = np.expand_dims(points, axis=0)
        
        assert type(points) == np.ndarray
        
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
                point_respaced = get_resized_coords_no_cast(point, orig_spacing, new_spacing) # typically new_spacing is np.flip(resize_factor)
                respaced_points[i, :] = point_respaced
            points = [[float(c) for c in arr] for arr in respaced_points] 
        
        if points.shape[1] == 3:
            print(f"Points shape before: {points.shape}")
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            print(f"Points shape after: {points.shape}")
        
        if mode_norm == "ijk->LPS":
            LPS_points = []
            for point in points:
                # convert ijk to RAS
                ijk_point = point.reshape(-1)
                RAS_point = self.ijk_to_RAS_matrix.MultiplyPoint(ijk_point)[0:3]
                LPS_coords = [RAS_point[0] * -1, RAS_point[1] * -1, RAS_point[2]]
                LPS_points.append(LPS_coords)
            to_return = LPS_points     
        elif mode_norm == "LPS->ijk":
            ijk_points = []
            for point in points:
                # convert ijk to RAS
                # RAS_point = [point[0] * -1, point[1] * -1, point[2]]
                RAS_point = point.reshape(-1,) * np.array([-1, -1, 1, 1])
                print(RAS_point, RAS_point.shape)
                ijk_coords = self.RAS_to_ijk_matrix.MultiplyPoint(RAS_point)[0:3]
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

def crop_heart(input_arr):
    """
    Improved function to segment the heart region using thresholding and morphological operations.
    :param input_arr: 3D array representing the CCTA scan.
    :return: Cropped heart data and bounding box coordinates.
    """
    print(f"Shape of input array is: {input_arr.shape}")

    src_array = input_arr.astype(np.float32)
    z, w, h = src_array.shape
    new_arr = np.full_like(src_array, -1000)  # Use np.full_like for efficiency

    # Initialize variables for calculating mean bounding box
    #sum_minr, sum_minc, sum_maxr, sum_maxc = 0, 0, 0, 0
    minrs, mincs, maxrs, maxcs = [], [], [], []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for k in range(z):
        image = src_array[k]
        ret, thresh = cv2.threshold(image, 20, 650, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

        label_opening = measure.label(opening)
        regionprops = measure.regionprops(label_opening)

        if regionprops:
            # Find the largest region, assuming it's the heart
            largest_region = max(regionprops, key=lambda x: x.area)
            minr, minc, maxr, maxc = largest_region.bbox

            # Update the new array and bounding box sums
            new_arr[k, minr:maxr, minc:maxc] = src_array[k, minr:maxr, minc:maxc]
            # sum_minr += minr
            # sum_minc += minc
            # sum_maxr += maxr
            # sum_maxc += maxc
            minrs.append(minr)
            mincs.append(minc)
            maxrs.append(maxr)
            maxcs.append(maxc)

    # Calculate mean bounding box coordinates
    # mean_minr = sum_minr // z
    # mean_minc = sum_minc // z
    # mean_maxr = sum_maxr // z
    # mean_maxc = sum_maxc // z

    # Calculate the index for the upper third
    upper_third_index = new_arr.shape[0] // 3
    upper_third_index = int(upper_third_index - (upper_third_index*0.15))

    # Extract the upper third of the CCTA scan
    upper_third_new_arr = new_arr[upper_third_index*2:]
    
    # discard measures associated to slices in z axis we just discarded
    
    
    min_minr = np.array(minrs)[upper_third_index*2:].min()
    min_minc = np.array(mincs)[upper_third_index*2:].min()
    max_maxr = np.array(maxrs)[upper_third_index*2:].max()
    max_maxc = np.array(maxcs)[upper_third_index*2:].max()
    minz = upper_third_index
    maxz = new_arr.shape[0]
    
    return upper_third_new_arr, min_minc, min_minr, max_maxc, max_maxr, minz, maxz