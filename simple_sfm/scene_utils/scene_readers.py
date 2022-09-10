from typing import Tuple
import os
import math
from glob import glob

import numpy as np
from PIL import Image, ImageFile
import torch

from simple_sfm.utils.coord_conversion import coords_pixel_to_film


def relative_intrinsic_to_absolute(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
    scaling = torch.tensor([width, height, 1.]).view(-1, 1)
    return intrinsic * scaling


def absolute_intrinsic_to_relative(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
    scaling = torch.tensor([width, height, 1.]).view(-1, 1)
    return intrinsic / scaling


def crop_data(
        image: torch.Tensor,
        intrinsic: torch.Tensor,
        crop_size,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Central crop the image and transform the absolute intrinsics

    Args:
        image: C x H x W
        intrinsic: 3 x 3
        crop_size: [h, w]

    Returns:
        cropped_image: C x H_crop x W_crop
        cropped_intrinsic: 3 x 3
    """
    height, width = image.shape[1:]
    crop_height, crop_width = crop_size
    assert (crop_height <= height) and (crop_width <= width), f"Crop size {crop_height, crop_width} " \
                                                              f"must be less then image size {height, width}! "

    crop_x = math.floor((width - crop_width) / 2)
    crop_y = math.floor((height - crop_height) / 2)

    cropped_image = image[..., crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    pixel_coords = torch.tensor([crop_x, crop_y], dtype=torch.float).view(1, 1, -1)
    film_coords = coords_pixel_to_film(pixel_coords, intrinsic.unsqueeze(0))[0, 0]
    new_principal_point = - film_coords * torch.diagonal(intrinsic[:-1, :-1], dim1=0, dim2=1)
    cropped_intrinsic = intrinsic.clone()
    cropped_intrinsic[:-1, -1] = new_principal_point

    return cropped_image, cropped_intrinsic


def read_image(image_path,
               intrinsic,
               image_resize_size,
               image_crop_size=None,
               relative_intrinsics=True
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        with Image.open(image_path) as img:
            if image_resize_size is not None:
                if isinstance(image_resize_size, int):
                    img_size = np.array(img.size)
                    shortest_idx = np.argsort(img_size)[0]
                    resize_ratio = image_resize_size / img_size[shortest_idx]
                    image_resize_size = [int(resize_ratio * img_size[1]), int(resize_ratio * img_size[0])]
                current_image = img.resize(image_resize_size)
                image_size = image_resize_size
            else:
                w, h = img.size
                image_size = [h, w]
                current_image = np.array(img)
                current_image = (current_image / 255) * 2 - 1
                current_image = current_image.transpose(2, 0, 1)
    except OSError as e:
        logger.error(f'Possibly, image file is broken: {image_path}')
        raise e

    if not relative_intrinsics or image_crop_size is not None:
        absolute_intr = relative_intrinsic_to_absolute(*image_size, intrinsic)

    if image_crop_size is not None:
        current_image, absolute_intr = crop_data(image=current_image,
                                                 intrinsic=absolute_intr,
                                                 crop_size=image_crop_size)

    if relative_intrinsics:
        if image_crop_size is not None:
            current_intr = absolute_intrinsic_to_relative(*image_crop_size, absolute_intr)
        else:
            current_intr = intrinsic
    else:
        current_intr = absolute_intr

    return current_image, current_intr


def read_re10k_views(views_file_path,
                     frames_path,
                     frames_resize_size=None,
                     frames_crop_size=None,
                     translation_scale=1.0,
                     images_ext='jpg'
                     ):
    frames_paths = glob(os.path.join(frames_path, '*.' + images_ext))
    frames_names = [el.split('/')[-1].split('.')[0] for el in frames_paths]
    frames = dict(zip(frames_names, frames_paths))

    with open(views_file_path) as f:
        text = f.read()

    rows = text.split('\n')[1:-1]
    intrinsics = []
    extrinsics = []
    images = []

    for cam_idx, row in enumerate(rows):
        row_name = row.split(' ')[0]
        if row_name in frames:
            (focal_length_x,
             focal_length_y,
             principal_point_x,
             principal_point_y) = np.array(row.split(' ')[1:5]).astype(np.float32)

            intrinsic = np.array([[focal_length_x, 0, principal_point_x],
                                  [0, focal_length_y, principal_point_y],
                                  [0, 0, 1]])
            intrinsic = torch.from_numpy(intrinsic).float()

            extrinsic = np.array(row.split(' ')[7:19]).astype(np.float32).reshape((3, 4), order='C')
            extrinsic[:, -1] = extrinsic[:, -1] / translation_scale
            extrinsic = torch.from_numpy(extrinsic).float()
            extrinsics.append(extrinsic)

            current_frame, intrinsic = read_image(image_path=frames[row_name],
                                                  intrinsic=intrinsic,
                                                  image_resize_size=frames_resize_size,
                                                  image_crop_size=frames_crop_size,
                                                  relative_intrinsics=True
                                                  )
            intrinsics.append(intrinsic)
            current_frame = torch.from_numpy(current_frame).float()
            images.append(current_frame)

    return intrinsics, extrinsics, images
