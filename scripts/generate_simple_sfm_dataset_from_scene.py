import sys
import os
import logging
import argparse
from pathlib import Path
import shutil

import torch
import numpy as np

from simple_sfm.cameras import CameraMultiple
from simple_sfm.colmap_utils import ColmapBdManager
from simple_sfm.dataset import simple_sfm_dataset_utils
from simple_sfm.matchers import Matcher
from simple_sfm.scene_utils import OneVideoSceneProcesser
from simple_sfm.scene_utils.colmap_scene_converters import colmap_sparse_to_simple_sfm_json_views
from simple_sfm.utils.io import is_video_file
from simple_sfm.utils.video_streamer import VideoStreamer
from simple_sfm.utils.visualise import plotly_plot_cameras_to_images

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def main():
    # TODO add mode in which render use folder with images

    # logger = logging.getLogger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=None,
                        help='Path to the scene input data. It can be as a folder with images as a video file.')
    parser.add_argument('--output-dir-path', type=str, default=None,
                        help='Path to output dir.')
    parser.add_argument('--scene-name', type=str, default=None,
                        help='Name of scene dataset that will be generated.')

    parser.add_argument('--max-num-frames', type=int, default=30,
                        help='Max number of frames')
    parser.add_argument('--num-skip-frames', type=int, default=None,
                        help='Number frames that will be skipped before frame will be sampled.')
    parser.add_argument('--image-scale-factor', type=float, default=1,
                        help='Image rescale factor. If equal to 0.5  - will resize image in to half of it size.')
    parser.add_argument('--do-center-crop', type=bool, default=False,
                        help='Images will be central cropped before processing.')

    parser.add_argument('--nms-radius', type=int, default=4,
                        help='Non Maximum Suppression (NMS) radius for matching.')
    parser.add_argument('--keypoint-threshold', type=float, default=0.001,
                        help='Detector confidence threshold.')
    parser.add_argument('--match-threshold', type=float, default=0.9,
                        help='Threshold value for matching.')
    parser.add_argument('--match-nearest-radius', type=int, default=None,
                        help='Radius in which frames will be matched with each other. '
                             'It makes sense only for sequential data.')
    parser.add_argument('--sinkhorn-iterations', type=int, default=20,
                        help='Num sinkhorn iterations for super glue matching.')
    parser.add_argument('--super-glue-batch', type=int, default=5,
                        help='Batch size for super glue inference.')

    opts = parser.parse_args()

    input_data_path = opts.input_path
    output_dir_path = opts.output_dir_path
    scene_name = opts.scene_name

    modnet_weigths_path = '../weights/modnet_photographic_portrait_matting.torchscript'
    ptf_segm_model_weigths_path = '/home/palsol/projects/SimpleSfm/notebooks/data/mma_multi_211122_l_cuda_0.torchscript.pt'

    capture_work_dir = Path(output_dir_path, scene_name)
    frames_path = Path(capture_work_dir, 'frames')

    is_video = is_video_file(opts.input_path)
    if is_video:
        raw_video_dir_path = Path(capture_work_dir, 'raw_video')
        video_name = input_data_path.split('/')[-1]
        os.makedirs(raw_video_dir_path, exist_ok=True)
        raw_video_path = Path(raw_video_dir_path, video_name)
        shutil.copy(input_data_path, raw_video_path)

        video_to_frames = OneVideoSceneProcesser(
            video_path=str(raw_video_path),
            dataset_output_path=str(capture_work_dir),
            skip=opts.num_skip_frames,
            center_crop=opts.do_center_crop,
            scale_factor=opts.image_scale_factor,
            max_len=opts.max_num_frames,
            img_prefix='jpg',
            filter_with_sharpness=True,
            #     force_rotate_code=cv2.ROTATE_180
        )

        video_to_frames.run()

    else:
        os.makedirs(capture_work_dir, exist_ok=True)
        shutil.copytree(input_data_path, frames_path)

    matcher = Matcher(
        super_point_extractor_weights_path='../weights/superpoint_v1.pth',
        super_glue_weights_path='../weights/superglue_indoor.pth',
        nms_radius=opts.nms_radius,
        keypoint_threshold=opts.keypoint_threshold,
        matcher_type='super_glue',
        match_threshold=opts.match_threshold,
        sinkhorn_iterations=opts.sinkhorn_iterations,
        super_glue_batch=int(opts.super_glue_batch)
    )

    vs = VideoStreamer(
        str(frames_path),
        height=None,
        width=None,
        max_len=None,
        img_glob='*.jpg'
    )

    camera_size = vs.get_resolution()

    colmap = ColmapBdManager(
        db_dir=str(Path(capture_work_dir, 'colmap')),
        images_folder_path=str(frames_path),
        camera_type='OPENCV',
        camera_params=None,
        camera_size=camera_size
    )

    if opts.match_nearest_radius is not None:
        pairs_filter_func = lambda x: np.abs(x[0] - x[1]) < opts.match_nearest_radius
    else:
        pairs_filter_func = lambda x: True

    processed_frames, match_table, images_names = matcher.match_video_stream(vs, pairs_filter_func)
    torch.cuda.empty_cache()

    colmap.replace_images_data(images_names)
    colmap.replace_keypoints(images_names, processed_frames)
    colmap.replace_and_verificate_matches(match_table, images_names)
    num_sparse_points = colmap.run_mapper()
    print(num_sparse_points)

    colmap_sparse_to_simple_sfm_json_views(
        scene_colmap_sparse_path=os.path.join(capture_work_dir, 'colmap', 'sparse'),
        work_dir_path=os.path.join(capture_work_dir),
        relative_frames_path='frames',
    )

    views_data_path = Path(capture_work_dir, 'views_data.json')

    """Extracts and saves people masks from all views"""
    simple_sfm_dataset_utils.generate_ptf_masks(capture_work_dir, ptf_segm_model_weigths_path)
    """Takes colmap sparse pointcloud and saves sparse depth for each view"""
    simple_sfm_dataset_utils.generate_sparse_depth_from_colmap(capture_work_dir)

    views_data_oriented_path = Path(capture_work_dir, 'views_data_oriented.json')
    simple_sfm_dataset_utils.center_and_orient(views_data_path,
                                               views_data_oriented_path,
                                               orient_method='up')

    result_cameras = CameraMultiple.from_simple_sfm_json(views_data_oriented_path)
    plotly_plot_cameras_to_images(cameras=result_cameras, output_path=Path(capture_work_dir, 'cameras_plot'))
