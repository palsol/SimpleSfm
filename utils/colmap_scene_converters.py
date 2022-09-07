import os
import sys
from collections import OrderedDict

import numpy as np
import torch

from utils.read_write_colmap_data import qvec2rotmat, read_model
from utils.camera_pinhole import CameraPinhole


def get_info_from_colmap_scene(path_sparse, device='cuda'):
    """
    Parse info from colmap bin files, return dict with:
    Scene params:
        num_points - number of 3d points in colmap scene
        num_views - number of views
        p_x - depth for x percentile, calculated via all scene views
        error_mean - mean of colmap 3d point error
        error_std - std of colmap 3d point error

    Intrinsic (relative):
        f_x - focal distance x
        f_y - focal distance y
        c_x - central point x
        c_y - central point y
        original_resolution_x - original image x resolution (which used for colmap)
        original_resolution_y - original image y resolution (which used for colmap)

    undistorted (if dense on):
        f_x_undistorted - focal distance x for undistorted images
        f_y_undistorted - focal distance y for undistorted images
        c_x_undistorted - central point x for undistorted images
        c_y_undistorted - central point y for undistorted images
        undistorted_resolution_x - undistorted image x resolution
        undistorted_resolution_y - undistorted image y resolution


    :param path_sparse: path to colmap sparse reconstruction
    :return:
    """

    cameras_colmap, images_colmap, points3D_colmap = read_model(path_sparse)
    images_colmap = OrderedDict({k: v for k, v in sorted(images_colmap.items(), key=lambda item: item[1].name)})
    info = {}

    cameras = CameraPinhole.from_colmap(images_colmap, cameras_colmap)

    # Get 3d points
    points = [el.xyz for el in points3D_colmap.values()]
    points = torch.tensor(points, device=device)

    # project points on cameras
    cameras.to(device=device)
    camera_points = cameras.world_to_cam(points.repeat(len(images_colmap), 1, 1))

    cameras_points_depth = camera_points[..., [-1]]

    cameras_pixels = cameras.cam_to_pixel(camera_points)
    cameras_pixels = torch.cat([cameras_pixels, cameras_points_depth], axis=2)

    cameras_pixels_mask = ((cameras_pixels[:, :, [0]] < 0) | (cameras_pixels[:, :, [0]] > 1) |
                           (cameras_pixels[:, :, [1]] < 0) | (cameras_pixels[:, :, [1]] > 1))

    depth_numpy = cameras_pixels[:, :, [-1]][~cameras_pixels_mask].cpu().numpy()
    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    for el in percentiles:
        info['p_' + str(el)] = np.percentile(depth_numpy, el)

    error = np.array([el.error for el in points3D_colmap.values()])
    info['error_mean'] = error.mean()
    info['error_std'] = error.std()

    info['num_points'] = len(points3D_colmap)
    info['num_views'] = len(images_colmap)

    camera_colmap = cameras_colmap[1]
    if camera_colmap.model == 'SIMPLE_RADIAL' or camera_colmap.model == 'RADIAL':
        info['f_x'] = camera_colmap.params[0] / camera_colmap.width
        info['f_y'] = camera_colmap.params[0] / camera_colmap.height
        info['c_x'] = camera_colmap.params[1] / camera_colmap.width
        info['c_y'] = camera_colmap.params[2] / camera_colmap.height
    elif camera_colmap.model == 'OPENCV':
        info['f_x'] = camera_colmap.params[0] / camera_colmap.width
        info['f_y'] = camera_colmap.params[1] / camera_colmap.height
        info['c_x'] = camera_colmap.params[2] / camera_colmap.width
        info['c_y'] = camera_colmap.params[3] / camera_colmap.height
    else:
        print(f'Camera type is not supported, {camera_colmap}')
        return None

    info['original_resolution_x'] = camera_colmap.width
    info['original_resolution_y'] = camera_colmap.height

    images_colmap_undistorted = None

    return info, images_colmap, images_colmap_undistorted


def write_view_params_file(
        images_colmap,
        scene_name,
        f_x, f_y, c_x, c_y,
        views_output_file_path):
    """
    Write information about scene views (intrinsics, extrinsics) to RealEstate10K like txt file.

    :param images_colmap:
    :param info:
    :param views_output_file_path:
    :return:
    """
    f = open(views_output_file_path, "w+")

    f.write(scene_name + "\n")

    images = {k: v for k, v in sorted(images_colmap.items(), key=lambda item: item[1].name)}
    for item in images.items():
        rotation = qvec2rotmat(item[1].qvec)
        extrinsic = np.append(rotation, item[1].tvec[..., np.newaxis], axis=1)
        extrinsic_str = np.array2string(extrinsic.flatten(),
                                        precision=3,
                                        separator=' ',
                                        formatter={'float_kind': '{:9}'.format},
                                        max_line_width=9999)[1:-1]
        intrinsic_str = f'{f_x} {f_y} {c_x} {c_y} 0 0'
        frame_id = item[1].name.split('.')[0]
        f.write(frame_id + " " + intrinsic_str + " " + extrinsic_str + "\n")
    f.close()


def colmap_sparse_to_re10k_like_views(
        scene_colmap_sparse_path,
        views_file_output_path,
        device,
        scene_name='scene',
    ):
    """
    Generate file with views in RE10k style from colmap sparse data.

    :param scene_colmap_sparse_path: path to colmap sparse dir
    :param views_file_output_path: path to dir where views.txt will be stored
    :param device: device for processing
    :param scene_name:
    :return:
    """
    scene_views_file_path = os.path.join(views_file_output_path, 'views.txt')
    info, images_colmap, images_colmap_undistorted = get_info_from_colmap_scene(scene_colmap_sparse_path,
                                                                                device=device)
    write_view_params_file(images_colmap,
                           scene_name=scene_name,
                           f_x=info['f_x'],
                           f_y=info['f_y'],
                           c_x=info['c_x'],
                           c_y=info['c_y'],
                           views_output_file_path=scene_views_file_path)

