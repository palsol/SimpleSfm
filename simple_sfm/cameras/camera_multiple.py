__all__ = ['CameraMultiple']

import math
import json
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .camera_pinhole import CameraPinhole
from simple_sfm.utils.coord_conversion import coords_pixel_to_film
from simple_sfm.utils.geometry import qvec2rotmat
from simple_sfm.utils.io import load_krt_data

class CameraMultiple(CameraPinhole):
    """
    The instance of this class contains a 'tensor' of cameras: D1 x ... x Dn cameras instead of just B cameras in
    CameraPinhole class.
    Arguments of the methods should start with the same dimensions.
    """

    def __init__(self,
                 extrinsics: torch.Tensor,
                 intrinsics: torch.Tensor,
                 images_sizes: Union[Tuple, List] = None,
                 cameras_ids: List = None,
                 cameras_names: List = None,
                 ):
        """
         Args:
            extrinsics (torch.Tensor): Bc x 3 x 4 or Bc x 4 x 4, cameras extrinsics matrices
            intrinsics (torch.Tensor): Bc x 3 x 3, cameras intrinsics matrices
            images_sizes (torch.Tensor): Bc x 2 or 1 x 2 or 2, camera image plane size in pixels,
                needed for compute camera frustums.
            cameras_ids: list of size Bc with cameras ids
            cameras_ids: list of size Bc with cameras names
        """

        assert extrinsics.shape[:-2] == intrinsics.shape[:-2], \
            f'{extrinsics.shape} vs {intrinsics.shape}'

        super().__init__(
            extrinsics=extrinsics.contiguous().view(-1, *extrinsics.shape[-2:]),
            intrinsics=intrinsics.contiguous().view(-1, *intrinsics.shape[-2:]),
            images_sizes=images_sizes if images_sizes is None else torch.tensor(images_sizes)
                .expand(*extrinsics.shape[:-2], -1).contiguous().view(-1, 2),
            cameras_ids=cameras_ids if cameras_ids is None else np.array(cameras_ids).reshape(-1, 1),
            cameras_names=cameras_names if cameras_names is None else np.array(cameras_names).reshape(-1, 1),
        )
        self.cameras_shape = extrinsics.shape[:-2]
        self.cameras_numel = torch.tensor(self.cameras_shape).prod().item()
        self.cameras_ndim = len(self.cameras_shape)
        self.images_size = images_sizes

    def __len__(self):
        return self.cameras_shape[0]

    def __getitem__(self, key):
        """
        Camera Multiple tensor-like indexing
        Args:
            key: slice indexing
        Returns:
            Selection of Camera Multiple
        """
        selected_extrinsics = self._unflatten_tensor(self.extrinsics)[key]
        selected_intrinsics = self._unflatten_tensor(self.intrinsics)[key]
        image_sizes = None if not hasattr(self, 'images_sizes') else self._unflatten_tensor(self.images_sizes)[key]
        cameras_ids = None if not hasattr(self, 'cameras_ids') else self._unflatten_nparray(self.cameras_ids)[key]
        cameras_names = None if not hasattr(self, 'cameras_names') else self._unflatten_nparray(self.cameras_names)[key]

        return CameraMultiple(selected_extrinsics, selected_intrinsics, image_sizes, cameras_ids, cameras_names)

    def _flatten_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[:self.cameras_ndim] == self.cameras_shape, \
            f'Expected {self.cameras_shape} but got {tensor.shape[:self.cameras_ndim]}'
        return tensor.contiguous().view(-1, *tensor.shape[self.cameras_ndim:])

    def _unflatten_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[0] == self.cameras_numel, \
            f'Expected length {self.cameras_numel} but got {tensor.shape[0]}'
        return tensor.contiguous().view(*self.cameras_shape, *tensor.shape[1:])

    def _unflatten_nparray(self, array: np.array) -> torch.Tensor:
        assert array.shape[0] == self.cameras_numel, \
            f'Expected length {self.cameras_numel} but got {array.shape[0]}'
        return array.reshape(*self.cameras_shape, *array.shape[1:])

    def get_extrinsics(self):
        return self.extrinsics.view(*self.cameras_shape, *self.extrinsics.shape[-2:])

    def get_intrinsics(self):
        return self.intrinsics.view(*self.cameras_shape, *self.intrinsics.shape[-2:])

    def _select_from_flat(self, keys):
        selected_extrinsics = self.extrinsics[keys]
        selected_intrinsics = self.intrinsics[keys]
        image_sizes = None if not hasattr(self, 'images_sizes') else self.images_sizes[keys]
        cameras_ids = None if not hasattr(self, 'cameras_ids') else self.cameras_ids[keys]
        cameras_names = None if not hasattr(self, 'cameras_names') else self.cameras_names[keys]

        return CameraMultiple(selected_extrinsics, selected_intrinsics, image_sizes, cameras_ids, cameras_names)

    def get_cams_with_cams_index(self, cams_index):
        """
        Return 1D CameraMultiple selected with cameras index, it is index for cameras_ids.
        """
        ids = [np.where(self.cameras_ids == el)[0][0] for el in cams_index]
        return self._select_from_flat(ids)

    @classmethod
    def from_cameras(cls, cameras):
        raise NotImplementedError

    @classmethod
    def from_colmap(cls, images: dict, cameras: dict):
        """
        Init :class:`~Cameras` instance from colmap_utils scene data

        Args:
            images: colmap_utils information about every view, usually stored in images.bin/txt
            cameras: colmap_utils cameras information,usually stored in cameras.bin/txt
        Returns:
            CameraPinhole: class:`~Cameras`
        """

        images = {k: v for k, v in sorted(images.items(), key=lambda item: item[1].name)}

        extrinsics = []
        intrinsics = []
        images_sizes = []
        cameras_ids = []
        cameras_names = []
        for item in images.items():
            camera = cameras[item[1].camera_id]
            rotation = qvec2rotmat(item[1].qvec)
            extrinsic = np.append(rotation, item[1].tvec[..., np.newaxis], axis=1)
            intrinsic = [[camera.params[0] / camera.width, 0, camera.params[1] / camera.width],
                         [0, camera.params[0] / camera.height, camera.params[2] / camera.height],
                         [0, 0, 1]]
            images_size = [camera.width, camera.height]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            images_sizes.append(images_size)
            cameras_ids.append(item[1].id)
            cameras_names.append(item[1].name)

        return cls(extrinsics=torch.tensor(extrinsics),
                   intrinsics=torch.tensor(intrinsics),
                   images_sizes=torch.tensor(images_sizes),
                   cameras_ids=cameras_ids,
                   cameras_names=cameras_names)

    @classmethod
    def from_simple_sfm_json(cls, path):
        """
        Init :class:`~Cameras` instance from simpleSfm json

        Args:
            path: path to simpleSfm json
        Returns:
            CameraPinhole: class:`~Cameras`
        """

        with open(path) as f:
            views = json.load(f)

        views = sorted(views['frames'], key=lambda item: item['id'])

        extrinsics = []
        intrinsics = []
        images_sizes = []
        cameras_ids = []
        cameras_names = []

        for view in views:
            camera_intrinsic_data = view['intrinsic']
            c2w = np.array(view['transform_matrix'])
            c2w[2, :] *= -1
            c2w = c2w[[1, 0, 2, 3], :]
            c2w[0:3, 1] *= -1
            c2w[0:3, 2] *= -1
            w2c = np.linalg.inv(c2w)

            extrinsic = np.array(w2c)
            intrinsic = [[camera_intrinsic_data['f_x'], 0, camera_intrinsic_data['c_x']],
                         [0, camera_intrinsic_data['f_y'], camera_intrinsic_data['c_x']],
                         [0, 0, 1]]
            images_size = [camera_intrinsic_data['original_resolution_x'], camera_intrinsic_data['original_resolution_y']]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            images_sizes.append(images_size)
            cameras_ids.append(view['id'])
            cameras_names.append(view['file_path'])

        return cls(extrinsics=torch.tensor(extrinsics),
                   intrinsics=torch.tensor(intrinsics),
                   images_sizes=torch.tensor(images_sizes),
                   cameras_ids=cameras_ids,
                   cameras_names=cameras_names)

    @classmethod
    def from_KRT_dataset(cls, path):
        """
        Init :class:`~Cameras` instance from KRT dataset

        |---- KRT.txt
        |---- pose.txt
        |---- image0.jpg
        |---- depth0.jpg
        |---- image1.jpg
        |---- depth1.jpg


        Args:
            images: colmap_utils information about every view, usually stored in images.bin/txt
            cameras: colmap_utils cameras information,usually stored in cameras.bin/txt
        Returns:
            CameraPinhole: class:`~Cameras`
        """

        krt_path = Path(path, 'KRT.txt')
        cameras_info = load_krt_data(krt_path)

        extrinsics = []
        intrinsics = []
        images_sizes = []
        cameras_ids = []
        cameras_names = []

        for i, camera_info in enumerate(cameras_info):
            w2c = np.array(camera_info['extrinsic'])
            # c2w[2, :] *= -1
            # c2w = c2w[[1, 0, 2, 3], :]
            # c2w[0:3, 1] *= -1
            # c2w[0:3, 2] *= -1
            # w2c = np.linalg.inv(c2w)

            extrinsic = np.array(w2c)
            intrinsic = camera_info['intrinsic']
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            cameras_ids.append(i)
            cameras_names.append(Path(path, f'image{i}.jpg'))
            w, h = Image.open(cameras_names[-1]).size
            images_sizes.append([w, h])

        return cls(extrinsics=torch.tensor(extrinsics),
                   intrinsics=torch.tensor(intrinsics),
                   images_sizes=torch.tensor(images_sizes),
                   cameras_ids=cameras_ids,
                   cameras_names=cameras_names)

    @classmethod
    def broadcast_cameras(cls, broadcasted_camera, source_camera):
        camera_extrinsics_broadcasted = broadcasted_camera.get_extrinsics() \
            .expand(*source_camera.cameras_shape, -1, -1)
        camera_intrinsics_broadcasted = broadcasted_camera.get_intrinsics() \
            .expand(*source_camera.cameras_shape, -1, -1)
        camera_broadcasted = CameraMultiple(extrinsics=camera_extrinsics_broadcasted,
                                            intrinsics=camera_intrinsics_broadcasted,
                                            )
        return camera_broadcasted

    def cam_to_film(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().cam_to_film(points)
        return self._unflatten_tensor(out)

    def film_to_cam(self,
                    points: torch.Tensor,
                    depth: Union[torch.Tensor, float, int]
                    ) -> torch.Tensor:
        if isinstance(depth, torch.Tensor):
            depth = self._flatten_tensor(depth)
        points = self._flatten_tensor(points)
        out = super().film_to_cam(points, depth)
        return self._unflatten_tensor(out)

    def film_to_pixel(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().film_to_pixel(points)
        return self._unflatten_tensor(out)

    def pixel_to_film(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().pixel_to_film(points)
        return self._unflatten_tensor(out)

    def cam_to_pixel(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().cam_to_pixel(points)
        return self._unflatten_tensor(out)

    def pixel_to_cam(self,
                     points: torch.Tensor,
                     depth: Union[torch.Tensor, float, int]
                     ) -> torch.Tensor:
        if isinstance(depth, torch.Tensor):
            depth = self._flatten_tensor(depth)
        points = self._flatten_tensor(points)
        out = super().pixel_to_cam(points, depth)
        return self._unflatten_tensor(out)

    def cam_to_world(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().cam_to_world(points)
        return self._unflatten_tensor(out)

    def world_to_cam(self, points: torch.Tensor) -> torch.Tensor:
        points = self._flatten_tensor(points)
        out = super().world_to_cam(points)
        return self._unflatten_tensor(out)

    @property
    def world_position(self) -> torch.Tensor:
        out = super().world_position
        return self._unflatten_tensor(out)

    def world_view_direction_unflatten(self, axis: str = 'z') -> torch.Tensor:
        out = super().world_view_direction(axis)
        return self._unflatten_tensor(out)

    def pixel_to_another_cam_ray_direction(self,
                                           points: torch.Tensor,
                                           another_camera: 'CameraMultiple',
                                           ) -> torch.Tensor:
        """
        Compute direction for rays that start at camera center and intersect the image plane at given pixel coordinates.
        Output in camera system of another_camera.

        Args:
            points: points

        Returns:
            torch.Tensor: ray directions
        """
        another_cam_cam_positions = another_camera.world_to_cam(self.world_position.unsqueeze(-2)).squeeze(-2)

        world_points_coords = self.pixel_to_world(points, 1.)
        another_cam_points_coords = another_camera.world_to_cam(world_points_coords)

        ndim_with_points = another_cam_points_coords.dim()
        ndim_without_points = another_cam_cam_positions.dim()
        shape = (another_cam_cam_positions.shape[:-1]
                 + (1,) * (ndim_with_points - ndim_without_points)
                 + another_cam_cam_positions.shape[-1:]
                 )
        another_cam_cam_positions = another_cam_cam_positions.view(*shape)

        ray_direction = another_cam_points_coords - another_cam_cam_positions
        ray_direction = F.normalize(ray_direction, dim=-1)

        return ray_direction

    def crop_center(self,
                    crop_size
                    ):
        """
        Central crop the  camera intrinsics

        Args:
            crop_size: [h, w]
        """
        height, width = self.images_size
        scaling = torch.tensor([width, height, 1.], device=self.intrinsics.device).view(1, 3, 1)
        absolute_intrinsics = self.intrinsics * scaling

        crop_height, crop_width = crop_size

        crop_x = math.floor((width - crop_width) / 2)
        crop_y = math.floor((height - crop_height) / 2)

        pixel_coords = torch.tensor([crop_x, crop_y], dtype=torch.float, device=self.intrinsics.device).view(1, 1, -1)
        film_coords = coords_pixel_to_film(pixel_coords, absolute_intrinsics)[:, 0]
        new_principal_point = - film_coords * torch.diagonal(absolute_intrinsics[:, :-1, :-1], dim1=1, dim2=2)
        cropped_intrinsic = absolute_intrinsics.clone()
        cropped_intrinsic[:, :-1, -1] = new_principal_point

        self.intrinsics = cropped_intrinsic / scaling
        self.images_size = crop_size
