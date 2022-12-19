__all__ = ['CameraMultiple']

import os
import math
import json
from pathlib import Path
from typing import Union, Tuple, List, Dict
import logging
import glob

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .camera_pinhole import CameraPinhole
from simple_sfm.utils.coord_conversion import coords_pixel_to_film
from simple_sfm.utils.geometry import qvec2rotmat, rotmat2qvec
from simple_sfm.utils.io import load_krt_data
from simple_sfm.colmap_utils import read_write_colmap_data

logger = logging.getLogger(__name__)


class CameraMultiple(CameraPinhole):
    """
    The instance of this class contains a 'tensor' of cameras: D1 x ... x Dn cameras instead of just B cameras in
    CameraPinhole class.
    Arguments of the methods should start with the same dimensions.
    """

    def __init__(self,
                 extrinsics: torch.Tensor,
                 intrinsics: torch.Tensor,
                 cameras_physical_id: List = None,
                 images_sizes: Union[Tuple, List] = None,
                 cameras_ids: List = None,
                 cameras_names: List = None,
                 cameras_meta: Dict[str, List] = None,
                 ):
        """
         Args:
            extrinsics (torch.Tensor): Bc x 3 x 4 or Bc x 4 x 4, cameras extrinsics matrices
            intrinsics (torch.Tensor): Bc x 3 x 3, cameras intrinsics matrices
            images_sizes (torch.Tensor): Bc x 2 or 1 x 2 or 2, camera image plane size in pixels,
                needed for compute camera frustums.
            cameras_ids: list of size Bc with cameras ids
            cameras_names: list of size Bc with cameras names
            cameras_physical_id: In this class, each camera has its own intrinsic data.
                However, some cameras share the same data and represent the same physical camera.
                It is important for camera optimization and is used in COLMAP.
                So, these physical cameras ids are stored here.
            cameras_meta: dict of lists of size Bc with cameras meta information
        """

        assert extrinsics.shape[:-2] == intrinsics.shape[:-2], \
            f'{extrinsics.shape} vs {intrinsics.shape}'

        self.cameras_shape = extrinsics.shape[:-2]

        if cameras_physical_id is None:
            cameras_physical_id = list(range(self.cameras_shape[0]))
        self.cameras_physical_id = np.array(cameras_physical_id).reshape(-1, 1)

        super().__init__(
            extrinsics=extrinsics.contiguous().view(-1, *extrinsics.shape[-2:]),
            intrinsics=intrinsics.contiguous().view(-1, *intrinsics.shape[-2:]),
            images_sizes=images_sizes if images_sizes is None else torch.tensor(images_sizes)
                .expand(*extrinsics.shape[:-2], -1).contiguous().view(-1, 2),
            cameras_ids=cameras_ids if cameras_ids is None else np.array(cameras_ids).reshape(-1, 1),
            cameras_names=cameras_names if cameras_names is None else np.array(cameras_names).reshape(-1, 1),
        )

        if cameras_meta is not None:
            self.cameras_meta = {}
            for key, item in cameras_meta.items():
                self.cameras_meta[key] = np.array(item).reshape(-1, 1)

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
        if isinstance(key, int):
            key = [key]
        selected_extrinsics = self._unflatten_tensor(self.extrinsics)[key]
        selected_intrinsics = self._unflatten_tensor(self.intrinsics)[key]
        selected_cameras_physical_id = self._unflatten_nparray(self.cameras_physical_id)[key]
        image_sizes = None if not hasattr(self, 'images_sizes') else self._unflatten_tensor(self.images_sizes)[key]
        cameras_ids = None if not hasattr(self, 'cameras_ids') else self._unflatten_nparray(self.cameras_ids)[key]
        cameras_names = None if not hasattr(self, 'cameras_names') else self._unflatten_nparray(self.cameras_names)[key]
        new_cameras_meta = None
        if hasattr(self, 'cameras_meta'):
            new_cameras_meta = {}
            for item_key, item in self.cameras_meta.items():
                new_cameras_meta[item_key] = self._unflatten_nparray(item)[key]

        return CameraMultiple(extrinsics=selected_extrinsics,
                              intrinsics=selected_intrinsics,
                              cameras_physical_id=selected_cameras_physical_id,
                              images_sizes=image_sizes,
                              cameras_ids=cameras_ids,
                              cameras_names=cameras_names,
                              cameras_meta=new_cameras_meta)

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
        selected_cameras_physical_id = self.cameras_physical_id[keys]
        image_sizes = None if not hasattr(self, 'images_sizes') else self.images_sizes[keys]
        cameras_ids = None if not hasattr(self, 'cameras_ids') else self.cameras_ids[keys]
        cameras_names = None if not hasattr(self, 'cameras_names') else self.cameras_names[keys]

        new_cameras_meta = None
        if hasattr(self, 'cameras_meta'):
            new_cameras_meta = {}
            for item_key, item in self.cameras_meta.items():
                new_cameras_meta[item_key] = item[keys]

        return CameraMultiple(extrinsics=selected_extrinsics,
                              intrinsics=selected_intrinsics,
                              cameras_physical_id=selected_cameras_physical_id,
                              images_sizes=image_sizes,
                              cameras_ids=cameras_ids,
                              cameras_names=cameras_names,
                              cameras_meta=new_cameras_meta)

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
            camera_physical_id = item[1].camera_id
            camera = cameras[camera_physical_id]
            rotation = qvec2rotmat(item[1].qvec)
            extrinsic = np.append(rotation, item[1].tvec[..., np.newaxis], axis=1)
            intrinsic = [[camera.params[0] / camera.width, 0, camera.params[1] / camera.width],
                         [0, camera.params[0] / camera.height, camera.params[2] / camera.height],
                         [0, 0, 1]]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            cameras_physical_id.append(camera_physical_id)
            images_sizes.append([camera.width, camera.height])
            cameras_ids.append(int(item[1].id))
            cameras_names.append(item[1].name)

        return cls(extrinsics=torch.tensor(extrinsics),
                   intrinsics=torch.tensor(intrinsics),
                   cameras_physical_id=cameras_physical_id,
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
        cameras_physical_id = []

        meta_fields_names = list(views[0].keys() - ['id', 'file_path', 'sharpness',
                                                    'transform_matrix', 'intrinsic', 'physical_camera_id'])
        meta_info = {}
        for meta_field_name in meta_fields_names:
            meta_info[meta_field_name] = []

        for i, view in enumerate(views):
            camera_intrinsic_data = view['intrinsic']
            c2w = np.array(view['transform_matrix'])
            c2w[2, :] *= 1
            c2w = c2w[[1, 0, 2, 3], :]
            c2w[0:3, 1] *= -1
            c2w[0:3, 2] *= -1
            w2c = np.linalg.inv(c2w)

            extrinsic = np.array(w2c)
            intrinsic = [[camera_intrinsic_data['f_x'], 0, camera_intrinsic_data['c_x']],
                         [0, camera_intrinsic_data['f_y'], camera_intrinsic_data['c_x']],
                         [0, 0, 1]]
            images_size = [camera_intrinsic_data['original_resolution_x'],
                           camera_intrinsic_data['original_resolution_y']]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            images_sizes.append(images_size)
            cameras_ids.append(int(view['id']))
            cameras_names.append(view['file_path'])

            if 'physical_camera_id' in view:
                cameras_physical_id.append(view['physical_camera_id'])
            else:
                cameras_physical_id.append(i)

            for meta_field_name in meta_fields_names:
                meta_info[meta_field_name].append(view[meta_field_name])

        return cls(extrinsics=torch.tensor(extrinsics),
                   intrinsics=torch.tensor(intrinsics),
                   cameras_physical_id=cameras_physical_id,
                   images_sizes=torch.tensor(images_sizes),
                   cameras_ids=cameras_ids,
                   cameras_names=cameras_names,
                   cameras_meta=meta_info)

    @classmethod
    def from_KRT_dataset(cls, path):
        """
        Init :class:`~Cameras` instance from KRT dataset

        |---- KRT.txt
        |---- pose.txt
        |---- image
            |---- image_0.jpg
            |---- image_1.jpg
            ...
        |---- segmentation
            |---- segmentation_0.jpg
            |---- segmentation_1.jpg
            ...

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
        cameras_physical_id = []
        cameras_meta = {}

        if os.path.exists(Path(path, 'segmentation')):
            cameras_meta['multi_label_segmentation'] = []

        camera_physical_id = 0
        for key, camera_info in cameras_info.items():
            w2c = np.array(camera_info['extrinsic'])
            w2c[0, :] *= -1
            w2c[1, :] *= -1

            extrinsic = np.array(w2c)
            intrinsic = camera_info['intrinsic']
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            cameras_ids.append(int(key))
            cur_image_name = '/'.join(glob.glob(os.path.join(path, f'image/image_{key}.*'))[0].split('/')[-2:])
            cameras_names.append(cur_image_name)
            w, h = Image.open(Path(path, cameras_names[-1])).size
            images_sizes.append([w, h])
            cameras_physical_id.append(camera_physical_id)
            camera_physical_id += 1

            if 'multi_label_segmentation' in cameras_meta:
                cur_segmentation_name = \
                    '/'.join(glob.glob(os.path.join(path, f'segmentation/segmentation_{key}.*'))[0].split('/')[-2:])
                cameras_meta['multi_label_segmentation'].append(cur_segmentation_name)
                seg_w, seg_h = Image.open(Path(path, cameras_meta['multi_label_segmentation'][-1])).size
                assert (seg_w == w) and (seg_h == h), f"Wrong segmentation size {seg_w} x {seg_h} must be {w} x {h}"

        if not cameras_meta:
            cameras_meta = None

        return cls(extrinsics=torch.tensor(extrinsics),
                   intrinsics=torch.tensor(intrinsics),
                   cameras_physical_id=cameras_physical_id,
                   images_sizes=torch.tensor(images_sizes),
                   cameras_ids=cameras_ids,
                   cameras_names=cameras_names,
                   cameras_meta=cameras_meta)

    @classmethod
    def broadcast_cameras(cls, broadcasted_camera, source_camera):
        ## TODO save other field during broadcasting
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

    def to_simple_sfm_json(self, output_path: str):
        frames = []
        scene_scale = 1
        for camera in self:
            c2w = np.linalg.inv(camera.extrinsics[0].numpy())
            c2w[0:3, 2] *= -1  # flip the y and z axis
            c2w[0:3, 1] *= 1
            c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
            c2w[2, :] *= 1  # flip whole world upside down
            c2w[0:3, 0] *= -1

            camera_intrinsic_data = dict()
            intrinsic = camera.intrinsics[0].numpy().tolist()

            camera_intrinsic_data['original_resolution_x'] = camera.images_size[0][0].item()
            camera_intrinsic_data['original_resolution_y'] = camera.images_size[0][1].item()

            camera_intrinsic_data['f_x'] = intrinsic[0][0] / camera_intrinsic_data['original_resolution_x']
            camera_intrinsic_data['f_y'] = intrinsic[1][1] / camera_intrinsic_data['original_resolution_y']
            camera_intrinsic_data['c_x'] = intrinsic[0][2] / camera_intrinsic_data['original_resolution_x']
            camera_intrinsic_data['c_y'] = intrinsic[1][2] / camera_intrinsic_data['original_resolution_y']

            frame = {
                "id": int(camera.cameras_ids[0][0]),
                "physical_camera_id": int(camera.cameras_physical_id[0][0]),
                "file_path": str(camera.cameras_names[0][0]),
                "sharpness": 100,
                "transform_matrix": c2w,
                "intrinsic": camera_intrinsic_data
            }

            if hasattr(camera, 'cameras_meta'):
                for key, item in camera.cameras_meta.items():
                    frame[key] = item[0][0]

            frames.append(frame)

        for f in frames:
            f["transform_matrix"] = f["transform_matrix"].tolist()

        out = {
            "frames": frames,
            "scene_scale": scene_scale,
        }

        logger.info(f"[INFO] writing {len(frames)} frames to {output_path}")

        with open(output_path, "w") as outfile:
            json.dump(out, outfile, indent=2)

    def to_colmap_cameras(self, output_path: str, format: str = 'txt'):

        colmap_images = dict()
        physical_cameras = {}
        for i, camera in enumerate(self):
            qvec = rotmat2qvec(camera.extrinsics[0][:3, :3].cpu().numpy())
            tvec = camera.extrinsics[0][:3, 3].cpu().numpy()
            image_name = camera.cameras_names[0][0].split('/')[-1]
            image_id = camera.cameras_ids[0][0]
            camera_physical_id = camera.cameras_physical_id[0][0].item()

            colmap_images[i] = read_write_colmap_data.Image(id=int(image_id) + 1, qvec=qvec, tvec=tvec,
                                                            camera_id=camera_physical_id, name=image_name,
                                                            xys=[], point3D_ids=[])

            if camera_physical_id not in physical_cameras:
                physical_cameras[camera_physical_id] = {'images_size': camera.images_sizes[0],
                                                        'intrinsic': camera.intrinsics[0]}

        points3D = dict()
        physical_cameras_colmap = {}
        for physical_camera_id, physical_camera in physical_cameras.items():
            cam_params = (physical_camera['intrinsic'][:2].float() *
                          physical_camera['images_size'][None].T.repeat(1, 3).float()).reshape(-1)
            cam_params = cam_params[[0, 2, 5]].cpu().numpy()
            physical_cameras_colmap[physical_camera_id] = \
                read_write_colmap_data.Camera(id=physical_camera_id,
                                              model="SIMPLE_PINHOLE",
                                              width=physical_camera['images_size'][0].item(),
                                              height=physical_camera['images_size'][1].item(),
                                              params=cam_params)

        read_write_colmap_data.write_model(cameras=physical_cameras_colmap,
                                           images=colmap_images,
                                           points3D=points3D,
                                           path=output_path,
                                           ext=format)
