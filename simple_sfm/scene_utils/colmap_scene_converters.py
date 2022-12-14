import json
import os
from collections import OrderedDict

import math
import numpy as np
import torch
import yaml

from simple_sfm.cameras.camera_pinhole import CameraPinhole
from simple_sfm.colmap_utils.read_write_colmap_data import read_model
from simple_sfm.utils.geometry import qvec2rotmat


def get_intrinsic_info(camera_colmap):
    info = {'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0}

    if camera_colmap.model == 'PINHOLE':
        info['f_x'] = float(camera_colmap.params[0] / camera_colmap.width)
        info['f_y'] = float(camera_colmap.params[0] / camera_colmap.height)
        info['c_x'] = float(camera_colmap.params[1] / camera_colmap.width)
        info['c_y'] = float(camera_colmap.params[2] / camera_colmap.height)
    elif camera_colmap.model == 'SIMPLE_RADIAL':
        info['f_x'] = float(camera_colmap.params[0] / camera_colmap.width)
        info['f_y'] = float(camera_colmap.params[0] / camera_colmap.height)
        info['c_x'] = float(camera_colmap.params[1] / camera_colmap.width)
        info['c_y'] = float(camera_colmap.params[2] / camera_colmap.height)
        info['k1'] = float(camera_colmap.params[3])
    elif camera_colmap.model == 'RADIAL':
        info['f_x'] = float(camera_colmap.params[0] / camera_colmap.width)
        info['f_y'] = float(camera_colmap.params[1] / camera_colmap.height)
        info['c_x'] = float(camera_colmap.params[2] / camera_colmap.width)
        info['c_y'] = float(camera_colmap.params[3] / camera_colmap.height)
        info['k1'] = float(camera_colmap.params[4])
        info['k1'] = float(camera_colmap.params[5])
    elif camera_colmap.model == 'OPENCV':
        info['f_x'] = float(camera_colmap.params[0] / camera_colmap.width)
        info['f_y'] = float(camera_colmap.params[1] / camera_colmap.height)
        info['c_x'] = float(camera_colmap.params[2] / camera_colmap.width)
        info['c_y'] = float(camera_colmap.params[3] / camera_colmap.height)
        info['k1'] = float(camera_colmap.params[4])
        info['k2'] = float(camera_colmap.params[5])
        info['p1'] = float(camera_colmap.params[6])
        info['p2'] = float(camera_colmap.params[7])
    else:
        print(f'Camera type is not supported, {camera_colmap}')
        return None

    info['angle_x'] = float(math.atan(1.0 / (info['f_x'] * 2)) * 2)
    info['angle_y'] = float(math.atan(1.0 / (info['f_y'] * 2)) * 2)
    info['fov_x'] = float(info['angle_x'] * 180 / math.pi)
    info['fov_y'] = float(info['angle_y'] * 180 / math.pi)

    info['original_resolution_x'] = int(camera_colmap.width)
    info['original_resolution_y'] = int(camera_colmap.height)

    return info


def get_info_from_colmap_multi_camera_scene(path_sparse):
    """
    Parse info from colmap_utils bin files for a scene with multi cameras, return cameras_intrinsics and colmap images

    :param path_sparse: path to colmap_utils sparse reconstruction
    :return:
    """

    cameras_colmap, images_colmap, _ = read_model(path_sparse)
    images_colmap = OrderedDict({k: v for k, v in sorted(images_colmap.items(), key=lambda item: item[1].name)})
    cameras_intrinsics = OrderedDict({k: get_intrinsic_info(v) for k, v in cameras_colmap.items()})
    return cameras_intrinsics, images_colmap


def get_info_from_colmap_single_camera_scene(path_sparse, device='cuda'):
    """
    Parse info from colmap_utils bin files for a scene with single camera, return dict with:
    Scene params:
        num_points - number of 3d points in colmap_utils scene
        num_views - number of views
        p_x - depth for x percentile, calculated via all scene views
        error_mean - mean of colmap_utils 3d point error
        error_std - std of colmap_utils 3d point error

    Intrinsic (relative):
        f_x - focal distance x
        f_y - focal distance y
        c_x - central point x
        c_y - central point y
        original_resolution_x - original image x resolution (which used for colmap_utils)
        original_resolution_y - original image y resolution (which used for colmap_utils)

    undistorted (if dense on):
        f_x_undistorted - focal distance x for undistorted images
        f_y_undistorted - focal distance y for undistorted images
        c_x_undistorted - central point x for undistorted images
        c_y_undistorted - central point y for undistorted images
        undistorted_resolution_x - undistorted image x resolution
        undistorted_resolution_y - undistorted image y resolution


    :param path_sparse: path to colmap_utils sparse reconstruction
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
        info['p_' + str(el)] = float(np.percentile(depth_numpy, el))

    error = np.array([el.error for el in points3D_colmap.values()])
    info['error_mean'] = float(error.mean())
    info['error_std'] = float(error.std())

    info['num_points'] = int(len(points3D_colmap))
    info['num_views'] = int(len(images_colmap))

    camera_colmap = cameras_colmap[1]
    info.update(get_intrinsic_info(camera_colmap))

    images_colmap_undistorted = None

    return info, images_colmap, images_colmap_undistorted


def write_view_params_file_re10k_like(
        images_colmap,
        scene_info,
        scene_name,
        views_output_file_path,
        scene_meta_file_output_path=None,
        permute_axis=[0, 1, 2],
):
    """
    Write information about scene views (intrinsics, extrinsics) to RealEstate10K like txt file.

    :param images_colmap:
    :param info:
    :param views_output_file_path:
    :return:
    """

    f_x = scene_info['f_x']
    f_y = scene_info['f_y']
    c_x = scene_info['c_x']
    c_y = scene_info['c_y']

    f = open(views_output_file_path, "w+")

    f.write(scene_name + "\n")

    images = {k: v for k, v in sorted(images_colmap.items(), key=lambda item: item[1].name)}
    for item in images.items():
        rotation = qvec2rotmat(item[1].qvec)
        extrinsic = np.append(rotation[permute_axis], item[1].tvec[permute_axis][..., np.newaxis], axis=1)
        extrinsic_str = np.array2string(extrinsic.flatten(),
                                        precision=3,
                                        separator=' ',
                                        formatter={'float_kind': '{:9}'.format},
                                        max_line_width=9999)[1:-1]
        intrinsic_str = f'{f_x} {f_y} {c_x} {c_y} 0 0'
        frame_id = item[1].name.split('.')[0]
        f.write(frame_id + " " + intrinsic_str + " " + extrinsic_str + "\n")
    f.close()

    if scene_meta_file_output_path is not None:
        with open(os.path.join(scene_meta_file_output_path, 'scene_meta.yaml'), 'w+') as outfile:
            scene_info['scene_name'] = scene_name
            yaml.dump(scene_info, outfile, default_flow_style=False)


def closest_point_2_lines(oa, da, ob, db):
    """
    returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def write_view_params_file_nerf_like(
        images_colmap,
        scene_info,
        work_dir,
        relative_frames_path,
        up_vec=None,
        split_idx_list=None,
        scene_rescale=False,
):
    """
    Write information about scene views (intrinsics, extrinsics) to RealEstate10K like txt file.

    :param images_colmap:
    :param info:
    :param views_output_file_path:
    :param split_idx_list list of indexes for splitting transform.json on seceral parts.
    :return:
    """

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

    frames = []

    scene_scale = 1

    images = {k: v for k, v in sorted(images_colmap.items(), key=lambda item: item[1].name)}
    for item in images.items():
        rel_name = os.path.join(relative_frames_path, item[1].name)

        qvec = item[1].qvec
        tvec = item[1].tvec
        R = qvec2rotmat(qvec)
        t = tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)

        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
        c2w[2, :] *= -1  # flip whole world upside down

        frame = {
            "file_path": rel_name,
            "sharpness": 100,
            "transform_matrix": c2w,
            "id": int(item[1].id) - 1,
        }

        frames.append(frame)

    frames_dict = {el['id']: el for el in frames}

    if split_idx_list is None:
        split_idx_list = [len(frames)]

    idx_shift = 0
    for i, split_idx in enumerate(split_idx_list):
        curr_frames = [frames_dict[j] for j in range(idx_shift, split_idx)]
        for f in curr_frames:
            f["transform_matrix"] = f["transform_matrix"].tolist()

        out = {
            "camera_angle_x": scene_info['angle_x'],
            "camera_angle_y": scene_info['angle_y'],
            "fl_x": scene_info['f_x'] * scene_info['original_resolution_x'],
            "fl_y": scene_info['f_y'] * scene_info['original_resolution_y'],
            "k1": scene_info['k1'],
            "k2": scene_info['k2'],
            "p1": scene_info['p1'],
            "p2": scene_info['p2'],
            "cx": scene_info['c_x'] * scene_info['original_resolution_x'],
            "cy": scene_info['c_y'] * scene_info['original_resolution_y'],
            "w": scene_info['original_resolution_x'],
            "h": scene_info['original_resolution_y'],
            "frames": curr_frames,
            "scene_scale": scene_scale,
        }

        if i == 0:
            transforms_file_name = 'transforms.json'
        else:
            transforms_file_name = f'transforms_{i}.json'

        output_path = os.path.join(work_dir, transforms_file_name)
        print(f"[INFO] writing {len(curr_frames)} frames to {output_path}")
        with open(output_path, "w") as outfile:
            json.dump(out, outfile, indent=2)


def write_view_params_to_simple_sfm_json_file(
        images_colmap,
        cameras_intrinsics,
        work_dir,
        relative_frames_path,
):
    """
    Write information about scene views (intrinsics, extrinsics) to json.

    :param images_colmap:
    :param cameras_intrinsics:
    :param views_output_file_path:
    :return:
    """

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    frames = []
    scene_scale = 1

    images = {k: v for k, v in sorted(images_colmap.items(), key=lambda item: item[1].name)}
    for item in images.items():
        rel_name = os.path.join(relative_frames_path, item[1].name)

        qvec = item[1].qvec
        tvec = item[1].tvec
        R = qvec2rotmat(qvec)
        t = tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)

        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
        c2w[2, :] *= -1  # flip whole world upside down

        frame = {
            "id": int(item[1].id) - 1,
            "file_path": rel_name,
            "sharpness": 100,
            "transform_matrix": c2w,
            "intrinsic": cameras_intrinsics[item[1].camera_id]
        }
        frames.append(frame)

    for f in frames:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    out = {
        "frames": frames,
        "scene_scale": scene_scale,
    }

    transforms_file_name = 'views_data.json'

    output_path = os.path.join(work_dir, transforms_file_name)
    print(f"[INFO] writing {len(frames)} frames to {output_path}")
    with open(output_path, "w") as outfile:
        json.dump(out, outfile, indent=2)


def colmap_sparse_to_re10k_like_views(
        scene_colmap_sparse_path,
        views_file_output_path,
        device='cpu',
        scene_name='scene',
        scene_meta_file_output_path=None,
        permute_axis=[0, 1, 2],
):
    """
    Generate file with views in RE10k style from colmap_utils sparse data.

    :param scene_colmap_sparse_path: path to colmap_utils sparse dir
    :param views_file_output_path: path to dir where views.txt will be stored
    :param device: device for processing
    :param scene_name:
    :param scene_meta_file_output_path: path to store scene meta yaml
    :return:
    """
    scene_views_file_path = os.path.join(views_file_output_path, 'views.txt')
    info, images_colmap, images_colmap_undistorted = get_info_from_colmap_single_camera_scene(scene_colmap_sparse_path,
                                                                                              device=device)
    write_view_params_file_re10k_like(
        images_colmap,
        scene_name=scene_name,
        scene_info=info,
        views_output_file_path=scene_views_file_path,
        scene_meta_file_output_path=scene_meta_file_output_path,
        permute_axis=permute_axis,
    )


def colmap_sparse_to_nerf_like_views(
        scene_colmap_sparse_path,
        work_dir_path,
        relative_frames_path,
        device='cpu',
):
    info, images_colmap, images_colmap_undistorted = get_info_from_colmap_single_camera_scene(scene_colmap_sparse_path,
                                                                                              device=device)
    write_view_params_file_nerf_like(
        images_colmap,
        scene_info=info,
        work_dir=work_dir_path,
        relative_frames_path=relative_frames_path,
        split_idx_list=None,
    )


def colmap_sparse_to_simple_sfm_json_views(
        scene_colmap_sparse_path,
        work_dir_path,
        relative_frames_path,
):
    cameras_intrinsics, images_colmap = get_info_from_colmap_multi_camera_scene(scene_colmap_sparse_path)

    write_view_params_to_simple_sfm_json_file(
        images_colmap=images_colmap,
        cameras_intrinsics=cameras_intrinsics,
        work_dir=work_dir_path,
        relative_frames_path=relative_frames_path
    )
