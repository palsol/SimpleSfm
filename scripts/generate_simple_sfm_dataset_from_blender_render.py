import os
import argparse
import logging
import shutil
import sys
from pathlib import Path

from simple_sfm.cameras import CameraMultiple
from simple_sfm.dataset import simple_sfm_dataset_utils
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
    # TODO add mode in which render use folder with images

    # logger = logging.getLogger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=None,
                        help='Path to the scene input data. It can be as a folder with images as a video file.')
    parser.add_argument('--output-dir-path', type=str, default=None,
                        help='Path to output dir.')

    opts = parser.parse_args()

    input_data_path = opts.input_path
    output_dir_path = opts.output_dir_path
    simple_sfm_path = Path(os.path.abspath(__file__)).parent.parent

    # simple_sfm_path = Path(os.path.abspath(__file__)).parent.parent

    # modnet_weigths_path = Path(simple_sfm_path, 'weights/modnet_photographic_portrait_matting.torchscript')
    # ptf_segm_model_weigths_path = '/home/palsol/projects/SimpleSfm/notebooks/data/mma_multi_211122_l_cuda_0.torchscript.pt'

    output_dir_path = Path(output_dir_path)
    scene_name = input_data_path.split('/')[-1]
    scene_work_dir = Path(output_dir_path, scene_name)
    print(scene_work_dir)
    shutil.copytree(input_data_path, scene_work_dir)

    cameras = CameraMultiple.from_KRT_dataset(scene_work_dir)
    views_data_path = Path(scene_work_dir, 'views_data.json')
    cameras.to_simple_sfm_json(views_data_path)

    # """Extracts and saves people masks from all views"""
    # simple_sfm_dataset_utils.generate_modnet_masks(capture_work_dir, modnet_weigths_path)
    # """Takes colmap sparse pointcloud and saves sparse depth for each view"""
    # simple_sfm_dataset_utils.generate_sparse_depth_from_colmap(capture_work_dir)

    simple_sfm_dataset_utils.generate_masks_from_multi_label_segmentation(
        scene_work_dir,
        num_classes=5,
        target_class=2,
        output_name='object_of_interest_mask_path',
        input_segmentation_name='multi_label_segmentation',
    )

    simple_sfm_dataset_utils.generate_sparse_colmap(
        scene_work_dir,
        super_point_extractor_weights_path=Path(simple_sfm_path, 'weights/superpoint_v1.pth'),
        super_glue_weights_path=Path(simple_sfm_path, 'weights/superglue_outdoor.pth'),
    )

    simple_sfm_dataset_utils.generate_sparse_depth_from_colmap(scene_work_dir)

    views_data_oriented_path = Path(scene_work_dir, 'views_data_oriented.json')
    simple_sfm_dataset_utils.center_and_orient(views_data_path,
                                               views_data_oriented_path,
                                               orient_method=None)

    result_cameras = CameraMultiple.from_simple_sfm_json(views_data_oriented_path)
    plotly_plot_cameras_to_images(cameras=result_cameras, output_path=Path(scene_work_dir, 'cameras_plot'))


if __name__ == '__main__':
    main()
