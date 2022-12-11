import argparse
import logging
import sys
from pathlib import Path

import yaml

from simple_sfm.cameras import CameraMultiple
from simple_sfm.cameras import camera_samplers
from simple_sfm.utils.visualise import plotly_plot_cameras_to_images

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=None,
                        help='Path to json file of simple sfm dataset.')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to yaml file with sub scenes configurations')
    parser.add_argument('--generate-vis', type=bool, default=True,
                        help='Generates sampled cameras visualisation if True')

    opts = parser.parse_args()

    simple_sfm_json_path = Path(opts.input_path)
    dataset_dir_path = simple_sfm_json_path.parent
    dataset_config_name = (simple_sfm_json_path.name).split('.')[0]
    generate_vis = opts.generate_vis
    config_path = opts.config_path
    if config_path is None:
        # test cams automatically will be excluded from other sub_scenes.
        sub_scenes_config = {
            'test': {
                'num_cams': 1,
                'sample_method': 'most_distant',
            },
            'train': {
                'num_cams': 1,
                'sample_method': 'most_distant',
            }
        }
    else:
        with open(config_path, 'r') as stream:
            try:
                sub_scenes_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    cameras = CameraMultiple.from_simple_sfm_json(simple_sfm_json_path)
    scenes_cameras_list = {}
    if 'test' in sub_scenes_config:
        test_scene_config = sub_scenes_config.pop('test')
        sampler_func = getattr(camera_samplers, test_scene_config['sampler_type'])
        test_cameras, test_ids = sampler_func(cameras, test_scene_config['num_cams'])
        scenes_cameras_list[f'{dataset_config_name}_test'] = test_cameras
        cameras = cameras.get_cams_with_cams_index(list(set(cameras.cameras_ids[:, 0]) - set(test_ids[:, 0])))

    for key, item in sub_scenes_config.items():
        sampler_func = getattr(camera_samplers, item['sampler_type'])
        scene_cameras, _ = sampler_func(cameras, item['num_cams'])
        scenes_cameras_list[f'{dataset_config_name}_{key}'] = scene_cameras

    for key, scene_cameras in scenes_cameras_list.items():
        if generate_vis:
            scene_vis_path = Path(dataset_dir_path, 'cameras_plot', 'key')
            plotly_plot_cameras_to_images(scene_cameras, output_path=scene_vis_path)
        scene_cameras.to_simple_sfm_json(Path(dataset_dir_path, f'{key}.json'))

if __name__ == '__main__':
    main()
