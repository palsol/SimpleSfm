from typing import Tuple
import os
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
import numpy as np

from simple_sfm.cameras.camera_multiple import CameraMultiple
from simple_sfm.models.modnet import MODNETModel
from simple_sfm.colmap_utils import read_write_colmap_data
from simple_sfm.utils import geometry
from simple_sfm.utils.image import check_and_get_new_side


def generate_modnet_masks(
        dataset_path: str,
        modnet_weigths_path: str):
    """
    Method for generating modnet masks for simple_sfm_dataset format.
    The paths to mask are stored in 'object_of_interest_mask_path' field of every frame data.

    :param dataset_path: path to dataset (folder with 'views_data.json' file)
    :param modnet_weigths_path: path to MODNet weights.
    :return:
    """

    views_data_json_path = Path(dataset_path, "views_data.json")
    save_masks_path = Path(dataset_path, 'masks')
    os.makedirs(save_masks_path, exist_ok=True)

    with open(views_data_json_path, encoding="UTF-8") as file:
        views_data = json.load(file)

    transform = [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5),
                                      (0.5, 0.5, 0.5))
                 ]
    transform = transforms.Compose(transform)
    modnet_model = MODNETModel(modnet_weigths_path, batch_size=1)

    mask_path_dict = {}

    for item in views_data['frames']:
        image = Image.open(Path(dataset_path, item['file_path']))

        image_name = item['file_path'].split('/')[-1]
        image_id = item['id']
        camera_name_index = image_name + '_' + str(image_id)

        image_tensor = transform(image).cuda()[None]

        image_mask = modnet_model.segment_tensor(image_tensor)
        image_mask = Image.fromarray(
            ((torch.round(image_mask[0]).repeat(3, 1, 1).permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8))
        image_mask_path = Path('masks', f'mask_{camera_name_index}.png')
        image_mask.save(image_mask_path)

        mask_path_dict[image_id] = image_mask_path

    for i in range(len(views_data['frames'])):
        views_data['frames'][i]['object_of_interest_mask_path'] = str(mask_path_dict[views_data['frames'][i]['id']])

    with open(views_data_json_path, "w") as outfile:
        json.dump(views_data, outfile, indent=2)


def generate_sparse_depth_from_colmap(
        dataset_path: str,
        sparse_colmap_output_path: str = None,
        scale: float = 1):
    """
    Generate binary files with sparse depth based on the point cloud,
    which is restored during COLMAP SFM (sparse reconstruction).
    For correct result, it is necessary that a simple_sfm_dataset has
    the same coordinate system that sparse reconstruction.
    If you use simple_sfm_dataset generated from sparse COLMAP reconstruction,
    you must use the dataset without scaling or rotation.
    However, in the case of scaling, you can pass the scale factor to this method.

    Output binary fiels are dicts with two items:
        'sparse_depth' - sparse depth with float values
        'sparse_depth_mask' - binary mask of pixels where sparse depth has values.

    :param dataset_path: path to dataset (folder with 'views_data.json' file)
    :param sparse_colmap_output_path: path to sparse COLMAP reconstruction
    :param scale: scale factor
    :return:
    """

    if sparse_colmap_output_path is None:
        colmap_path = Path(dataset_path, 'colmap/sparse')
    else:
        colmap_path = sparse_colmap_output_path

    views_data_json_path = Path(dataset_path, "views_data.json")

    _, _, points3D_colmap = read_write_colmap_data.read_model(colmap_path)
    cameras = CameraMultiple.from_simple_sfm_json(views_data_json_path)

    save_sparse_depth_path = Path(dataset_path, 'sparse_depth')
    os.makedirs(save_sparse_depth_path, exist_ok=True)

    save_sparse_depth_vis_path = Path(save_sparse_depth_path, 'sparse_depth_vis')
    os.makedirs(save_sparse_depth_vis_path, exist_ok=True)

    sparse_depth_path_dict = {}

    for camera in cameras:
        camera_id = camera.cameras_ids[0][0]
        W, H = camera.images_sizes[0]

        H = int(H // scale)
        W = int(W // scale)

        camera_points = []
        for key, point_item in points3D_colmap.items():
            """colmap uses 1 as index of the first camera"""
            if camera_id + 1 in point_item.image_ids:
                camera_points.append(point_item.xyz)

        camera_points = torch.tensor(camera_points, device=cameras.device)
        camera_points = camera.world_to_cam(camera_points)

        cameras_points_depth = camera_points[..., [-1]]

        cameras_pixels = camera.cam_to_pixel(camera_points)
        cameras_pixels = torch.cat([cameras_pixels, cameras_points_depth], axis=1)

        cameras_pixels *= torch.tensor([[W, H, 1]], device=cameras_pixels.device)
        cameras_pixels_mask = ((cameras_pixels[:, [0]] >= 0) & (cameras_pixels[:, [0]] <= W) &
                               (cameras_pixels[:, [1]] >= 0) & (cameras_pixels[:, [1]] <= H) &
                               (cameras_pixels[:, [2]] >= 0))
        cameras_pixels_mask = cameras_pixels_mask.reshape(-1)

        cameras_pixels = cameras_pixels[cameras_pixels_mask, :]
        camera_points_xy = cameras_pixels[:, :2].long()
        camera_points_depth = cameras_pixels[:, 2:]

        sparse_depth = torch.zeros([W, H, 1], device=camera_points_xy.device, dtype=camera_points_depth.dtype) - 1
        sparse_depth[camera_points_xy[:, 0], camera_points_xy[:, 1]] = camera_points_depth[:]
        depth_mask = (sparse_depth != -1)
        points_bounds = [camera_points_depth.min(), camera_points_depth.max()]

        camera_name_index = camera.cameras_names[0][0].split('/')[-1] + '_' + str(camera.cameras_ids[0][0])

        img_depth = (sparse_depth.permute(1, 0, 2).repeat(1, 1, 3) - points_bounds[0]) / (
                points_bounds[1] - points_bounds[0] + 10e-6)
        sparse_depth_vis = Image.fromarray((img_depth * 255).cpu().numpy().astype(np.uint8))
        sparse_depth_vis_file_path = Path(save_sparse_depth_vis_path, f'sparse_depth_vis_{camera_name_index}.jpg')
        sparse_depth_vis.save(sparse_depth_vis_file_path)

        view_sparse_depth_dict = {'sparse_depth': sparse_depth.permute(2, 0, 1).cpu().numpy(),
                                  'sparse_depth_mask': depth_mask.permute(2, 0, 1).cpu().numpy()}
        parse_depth_file_path = Path(save_sparse_depth_path, f'sparse_depth_{camera_name_index}')
        np.save(parse_depth_file_path, view_sparse_depth_dict)

        sparse_depth_path_dict[camera_id] = Path('sparse_depth', f'sparse_depth_{camera_name_index}.npy')

    with open(views_data_json_path, encoding="UTF-8") as file:
        views_data_json = json.load(file)

    for i in range(len(views_data_json['frames'])):
        views_data_json['frames'][i]['sparse_depth_path'] = str(
            sparse_depth_path_dict[views_data_json['frames'][i]['id']])

    with open(views_data_json_path, "w") as outfile:
        json.dump(views_data_json, outfile, indent=2)


def generate_ptf_masks(
        dataset_path: str,
        model_weigths_path: str,
        segmentation_size_wh: Tuple[int, int] = (960, 540)
):
    views_data_json_path = Path(dataset_path, "views_data.json")
    save_masks_path = Path(dataset_path, 'masks')
    os.makedirs(save_masks_path, exist_ok=True)

    with open(views_data_json_path, encoding="UTF-8") as file:
        views_data = json.load(file)

    transform = [transforms.ToTensor(),
                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225))
                 ]
    transform = transforms.Compose(transform)
    model = torch.jit.load(model_weigths_path).cuda()

    mask_path_dict = {}
    for item in views_data['frames']:
        image = Image.open(Path(dataset_path, item['file_path']))

        image_name = item['file_path'].split('/')[-1]
        image_id = item['id']
        camera_name_index = image_name + '_' + str(image_id)

        w, h = image.size
        image = image.resize([check_and_get_new_side(segmentation_size_wh[0]),
                              check_and_get_new_side(segmentation_size_wh[1])])
        image_tensor = transform(image).float().cuda()[None]

        with torch.no_grad():
            image_mask = model(image_tensor)
            image_mask = torch.nn.functional.interpolate(image_mask, [h, w])
            image_mask = image_mask.argmax(dim=1).unsqueeze(1)
            ## Geting only the class number 1 - humans
            image_mask = (image_mask == 1)
            image_mask = (image_mask[0].float().repeat(3, 1, 1)) * 255
            image_mask = Image.fromarray(image_mask.permute(1, 2, 0).cpu().numpy().astype(np.uint8))

        image_mask_path = Path('masks', f'mask_{camera_name_index}.png')
        image_mask.save(Path(dataset_path, image_mask_path))

        mask_path_dict[image_id] = image_mask_path

    for i in range(len(views_data['frames'])):
        views_data['frames'][i]['object_of_interest_mask_path'] = str(mask_path_dict[views_data['frames'][i]['id']])

    with open(views_data_json_path, "w") as outfile:
        json.dump(views_data, outfile, indent=2)

def center_and_orient(
        input_view_json_path: str,
        output_view_json_path: str,
        orient_method: str = None):
    """
    Scale to [-1, 1] range and rotate simple_sfm_dataset.

    :param orient_method: orient_method could be 'up', 'pca' and 'none'
    :param input_view_json_path:
    :param output_view_json_path:
    :return:
    """
    with open(input_view_json_path, encoding="UTF-8") as file:
        views_data = json.load(file)

    poses = []
    for view in views_data['frames']:
        poses.append(np.array(view["transform_matrix"]))

    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    poses[:, :3, :], scene_scale_factor = geometry.auto_orient_and_center_poses(poses, method=orient_method)

    for i in range(len(views_data['frames'])):
        views_data['frames'][i]["transform_matrix"] = poses[i].numpy().tolist()

    views_data['scene_scale'] = scene_scale_factor.item()

    with open(output_view_json_path, "w") as outfile:
        json.dump(views_data, outfile, indent=2)
