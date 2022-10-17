__all__ = ['ColmapBdManager']

import os
import shutil
import sqlite3
import subprocess
import logging
import time
from typing import Dict, List, Tuple

import numpy as np

from simple_sfm.utils.colmap import read_write_colmap_data
from simple_sfm.matcher.matcher import Frame

logger = logging.getLogger(__name__)


def scale_camera_calibration(calibration_type: str,
                             params: List,
                             width: int,
                             height: int):
    if params is not None:
        if calibration_type == 'SIMPLE_RADIAL':
            params[0] = params[0] * width
            params[1] = int(params[1] * width)
            params[2] = int(params[2] * height)
        elif calibration_type == 'FULL_OPENCV' or calibration_type == 'OPENCV':
            print(width, height)
            params[0] = params[0] * width
            params[1] = params[1] * height
            params[2] = int(params[2] * width)
            params[3] = int(params[3] * height)
        else:
            logger.error('Unsupported camera calibration type!')

    return params


class ColmapBdManager(object):
    def __init__(self,
                 db_dir: str,
                 images_folder_path: str = None,
                 camera_type: str = None,
                 camera_params: List = None,
                 camera_size: Tuple[int, int] = None):
        """

        :param db_dir: database directory path
        :param images_folder_path: folder with images
        :param camera_type: camera type, if you want use your own camera intrinsic, name must be passed according :
                            https://colmap.github.io/cameras.html
        :param camera_params: camera params
        """

        self.db_dir = db_dir
        self.sparse_path = os.path.join(self.db_dir, 'sparse')
        self.dense_path = os.path.join(self.db_dir, 'dense')
        self.ply_path = os.path.join(self.dense_path, 'fused.ply')
        self.db_path = os.path.join(self.db_dir, 'database.db')
        self.images_folder_path = images_folder_path

        self.camera_type = camera_type
        self.camera_params = camera_params

        os.makedirs(self.sparse_path, exist_ok=True)
        os.makedirs(self.dense_path, exist_ok=True)
        self.width = camera_size[0]
        self.height = camera_size[1]
        self.camera_params = scale_camera_calibration(self.camera_type, self.camera_params, self.width, self.height)

        if ~os.path.isfile(self.db_path) and self.images_folder_path is not None:
            self.build_dummy_data_base()
        else:
            logger.error('For building new colmap database, the images path must be specify!')
            return

        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        logger.info(f'BD image width {self.width},  height {self.height}.')
        logger.info(f'BD camera_type {camera_type}.')
        logger.info(f'BD camera_params {self.camera_params}.')

    def __del__(self):
        self.close_bd()

    @staticmethod
    def image_ids_to_pair_id(image_id1, image_id2):
        if image_id1 > image_id2:
            return 2147483647 * image_id2 + image_id1
        else:
            return 2147483647 * image_id1 + image_id2

    @staticmethod
    def mapper(db_path, images_folder_path, sparse_path, camera_type=None, camera_params=None):
        logger.info('Starts mapper.')
        start = time.time()

        command = ['colmap', 'mapper',
                   '--Mapper.ba_refine_principal_point', '1',
                   '--Mapper.filter_max_reproj_error', '2',
                   '--Mapper.min_num_matches', '32',
                   '--Mapper.extract_colors', '1',
                   '--Mapper.max_num_models', '1',
                   '--database_path', db_path,
                   '--image_path', images_folder_path,
                   '--output_path', sparse_path]

        if camera_type is not None and camera_params is not None:
            command.extend(['--Mapper.ba_refine_focal_length', '1',
                            '--Mapper.ba_refine_extra_params', '1',
                            '--Mapper.ba_refine_principal_point', '1'])

        subprocess.run(command,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)

        end = time.time()
        logger.info(f'Finished'
                     f'Time of mapper {float(end - start)}.')

    @staticmethod
    def triangulation(db_path, images_folder_path, sparse_path, output_path):
        """
        Run colmap triangulation.
        :return:
        """

        logger.info('Starts point triangulator.')
        start = time.time()

        subprocess.run(['colmap', 'point_triangulator',
                        '--log_level', '1',
                        '--Mapper.init_min_tri_angle', '4',
                        '--Mapper.init_min_num_inliers', '25',
                        '--database_path', db_path,
                        '--image_path', images_folder_path,
                        '--input_path', sparse_path,
                        '--output_path', output_path],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)

        end = time.time()
        logger.info(f'Finished'
                     f'Time of triangulation {float(end - start)}.')

    @staticmethod
    def dense(images_folder_path, sparse_path, output_path, size):

        logger.info('Starts image_undistorter.')
        start = time.time()

        subprocess.run(['colmap',
                        'image_undistorter',
                        '--image_path', images_folder_path,
                        '--input_path', sparse_path,
                        '--output_path', output_path,
                        '--output_type', 'COLMAP',
                        '--min_scale', str(0.9),
                        '--max_image_size', str(size),
                        ],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)

        end = time.time()
        logger.info(f'Finished'
                     f'Time of image_undistorter {float(end - start)}.')

        logger.info('Starts patch_match_stereo.')
        start = time.time()

        subprocess.run([
            'colmap',
            'patch_match_stereo',
            '--workspace_path', output_path,
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.geom_consistency', 'true',
        ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True)

        end = time.time()
        logger.info(f'Finished'
                     f'Time of patch_match_stereo {float(end - start)}.')

    @staticmethod
    def generate_ply(output_path, dense_path):

        logger.info('Starts stereo_fusion.')
        start = time.time()

        subprocess.run(['colmap',
                        'stereo_fusion',
                        '--workspace_path', dense_path,
                        '--workspace_format', 'COLMAP',
                        '--input_type', 'geometric',
                        '--output_path', output_path,
                        ],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)

        end = time.time()
        logger.info(f'Finished'
                     f'Time of stereo_fusion {float(end - start)}.')

    def __create_images_file(self, poses):
        self.cursor.execute('SELECT * FROM images')
        with open(os.path.join(self.sparse_path, 'images.txt'), 'w+') as img_file:
            for row in self.cursor.fetchall():
                img_id = row[0]
                img_name = row[1]
                camera_id = row[2]
                img_timestamp = int(img_name.split('.')[0])
                pose = poses[img_timestamp]
                string = f"{img_id} "
                for q in pose.quat:
                    string += f"{q} "
                for t in pose.trans:
                    string += f"{t} "
                string += f"{camera_id} {img_name} \n\n"
                img_file.write(string)

    def __create_cameras_file(self, poses, w, h):

        if self.camera_type is not None and self.camera_params is not None:
            raise ValueError('Camera type nad params can not be set if you use cameras file generator.')

        with open(os.path.join(self.sparse_path, 'cameras.txt'), 'w+') as camera_file:
            focal_length_x, focal_length_y, principal_point_x, principal_point_y = list(poses.values())[0].intr[:4]
            string = f"1 PINHOLE {w} {h} " \
                     f"{focal_length_x * w} " \
                     f"{focal_length_y * h} " \
                     f"{principal_point_x * w} " \
                     f"{principal_point_y * h} "
            camera_file.write(string)

    def __create_points3d_file(self):
        with open(os.path.join(self.sparse_path, 'points3D.txt'), 'w+') as camera_file:
            pass

    def close_bd(self):
        self.cursor.close()
        self.connection.close()

    def build_dummy_data_base(self):
        """
        Build dummy colmap database, using feature_extractor command.
        :return:
        """
        # TODO  Implement creating database without calling feature_extractor command.

        command = ['colmap', 'feature_extractor',
                   '--log_level', '0',
                   '--ImageReader.single_camera', '1',
                   '--ImageReader.default_focal_length_factor', '0.85',
                   '--SiftExtraction.peak_threshold', '0.02',
                   '--SiftExtraction.octave_resolution', '3',
                   '--database_path', self.db_path,
                   '--image_path', self.images_folder_path]

        if self.camera_type is not None and self.camera_params is not None:
            command.extend(['--ImageReader.camera_model', self.camera_type,
                            '--ImageReader.camera_params', str(self.camera_params)[1:-1].replace(' ', '')])

        subprocess.run(command,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)

    def build_initial_folder_from_known_camera_parameters(self,
                                                          poses: Dict,
                                                          W: int,
                                                          H: int):
        """
        Build initial sparse folder from poses and W H.
        :param poses:
        :param W:
        :param H:
        :return:
        """

        self.__create_images_file(poses)
        self.__create_cameras_file(poses, W, H)
        self.__create_points3d_file()

    def replace_images_data(self, images_names: Dict[int, str]):
        """
        Replaces current image data with new. Indexes must be start from 1. Image names it's a image file names.
        :param images_names: Dict where the key is a index of image and the value is the name of image.
        :return:
        """

        self.cursor.execute('DELETE FROM images;')
        self.connection.commit()

        for item in images_names.items():
            self.cursor.execute('INSERT INTO images(image_id, name, camera_id, '
                                'prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) '
                                'VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);',
                                (item[0], item[1], 1,
                                 None, None, None, None, None, None, None))
        self.connection.commit()

    def replace_keypoints(self,
                          images_names: Dict[int, str],
                          frames: List[Frame]):
        """
        Replaces current images keypoints with new one.
        :param images_names: Dict where the key is a index of image and the value is the name of image.
        :param frames: Frames with filed keypoints_poses fields.
        :return:
        """

        self.cursor.execute('DELETE FROM keypoints;')
        self.cursor.execute('DELETE FROM descriptors;')
        self.connection.commit()

        for image_id_bd in images_names.keys():
            keypoints = frames[image_id_bd - 1].keypoints_poses.cpu().numpy()
            keypoints = np.concatenate([keypoints,
                                        np.ones((keypoints.shape[0], 1)),
                                        np.zeros((keypoints.shape[0], 1))], axis=1).astype(np.float32)

            keypoints_str = keypoints.tostring()
            self.cursor.execute('INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);',
                                (image_id_bd, keypoints.shape[0], keypoints.shape[1], keypoints_str))

        self.connection.commit()

    def replace_and_verificate_matches(self,
                                       match_table: Dict[Tuple[int, int], np.array],
                                       images_names: Dict[int, str]):
        """

        :param match_table: Dict where the key is a pair from the match_list,
                            and the value is the points that match between images in pair.
        :param images_names: Dict where the key is a index of image and the value is the name of image.
        :return:
        """

        logger.info('Starts matches importer.')
        start = time.time()
        self.cursor.execute('DELETE FROM matches;')
        self.connection.commit()

        match_table_txt = []
        match_table_insert_data = []

        for item in match_table.items():
            image_id1, image_id2 = item[0]
            matches = item[1]

            match_table_txt.append((images_names[image_id1], images_names[image_id2]))
            if image_id1 > image_id2:
                matches = matches[:, [1, 0]]

            image_pair_id = self.image_ids_to_pair_id(image_id1, image_id2)
            matches_str = matches.tostring()
            match_table_insert_data.append((int(image_pair_id), matches.shape[0], matches.shape[1], matches_str))

        self.cursor.executemany("INSERT INTO  matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                                match_table_insert_data)

        self.connection.commit()

        match_txt_path = os.path.join(self.db_dir, 'match.txt')
        np.savetxt(match_txt_path, match_table_txt, fmt="%s", delimiter=' ')

        subprocess.run(['colmap', 'matches_importer',
                        '--database_path', self.db_path,
                        '--match_list_path', match_txt_path,
                        '--match_type', 'pairs'],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL
                       )

        end = time.time()
        logger.info(f'Finished \n'
                     f'Elapsed time: {float(end - start)} \n')

    def run_mapper(self):
        self.mapper(self.db_path, self.images_folder_path, self.sparse_path, self.camera_type, self.camera_params)

        mapper_sparse_path = os.path.join(self.sparse_path, '0')
        file_names = os.listdir(mapper_sparse_path)
        _, _, points3D = read_write_colmap_data.read_model(mapper_sparse_path)
        num_sparse_points = len(points3D)

        for file_name in file_names:
            shutil.move(os.path.join(mapper_sparse_path, file_name), self.sparse_path)
        os.rmdir(mapper_sparse_path)

        return num_sparse_points

    def run_dense(self, size):
        self.dense(self.images_folder_path, self.sparse_path, self.dense_path, size)

    def run_generate_ply(self):
        self.generate_ply(self.ply_path, self.dense_path)
