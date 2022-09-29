__all__ = ['SyncedMultiviewVideoSceneProcesser',
           'OneVideoSceneProcesser']

import logging
import math
import os
from glob import glob

import cv2
import ffmpeg
from tqdm import tqdm

logger = logging.getLogger(__name__)


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if 'rotate' in meta_dict['streams'][0]['tags']:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def get_video_stats(video_path):
    vidcap = cv2.VideoCapture(video_path)
    rotateCode = check_rotation(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    return {
        'rotation': rotateCode,
        'num_frames': num_frames
    }


class SyncedMultiviewVideoSceneProcesser:
    def __init__(self,
                 data_path,
                 dataset_output_path,
                 video_format,
                 max_len=None,
                 skip=None,
                 center_crop=False,
                 scale_factor=None,
                 img_prefix='jpg'
                 ):

        self.data_path = data_path
        self.dataset_output_path = dataset_output_path
        self.dataset_frames_path = os.path.join(self.dataset_output_path, 'frames')
        self.max_len = max_len
        self.center_crop = center_crop
        self.scale_factor = scale_factor
        self.img_prefix = img_prefix

        os.makedirs(self.dataset_output_path, exist_ok=True)
        os.makedirs(self.dataset_frames_path, exist_ok=True)

        videos_paths_list = glob(self.data_path + '*.' + video_format)
        videos_stats = [get_video_stats(el) for el in videos_paths_list]
        videos_num_frames = [el['num_frames'] for el in videos_stats]
        self.scene_num_frames = min(videos_num_frames)
        self.videos_data = dict()

        for i, video_path in enumerate(videos_paths_list):
            video_name = video_path.split('/')[-1].split('.')[0]

            if videos_stats[i]['num_frames'] != self.scene_num_frames:
                logger.info(f"Video has different frame length {videos_stats[i]['num_frames']}"
                            f"must be {self.scene_num_frames}, "
                            f"video path: {videos_paths_list[i]}")

            self.videos_data[video_name] = {
                'video_path': video_path,
                'num_frames': videos_stats[i]['num_frames'],
                'rotation': videos_stats[i]['rotation'],
            }

        if skip is None:
            if self.max_len is None:
                self.skip = 1
            elif self.max_len < self.scene_num_frames:
                self.skip = math.ceil(self.scene_num_frames / self.max_len)
            else:
                self.skip = 1
        else:
            self.skip = skip

        logger.info(f'Every {self.skip} will be skipped')

    def run(self):
        times_ids = list(range(self.scene_num_frames))[::self.skip]

        times_frame_paths = dict()
        for time_id in times_ids:
            times_frame_paths[time_id] = os.path.join(self.dataset_frames_path, f'{time_id:09d}')
            os.makedirs(times_frame_paths[time_id], exist_ok=True)

        for video_name, video_data in tqdm(self.videos_data.items()):
            vidcap = cv2.VideoCapture(video_data['video_path'])
            success, image = vidcap.read()

            count = 0
            while success and count < self.scene_num_frames:
                if video_data['rotation'] is not None:
                    image = correct_rotation(image, video_data['rotation'])
                if count in times_ids:
                    frame_path = os.path.join(times_frame_paths[count], f'{video_name}_{count:09d}.' + self.img_prefix)
                    if self.center_crop:
                        height, width, channels = image.shape
                        if isinstance(self.center_crop, bool):
                            center_crop = min(height, width)
                        else:
                            center_crop = self.center_crop

                        image = image[height // 2 - center_crop // 2: height // 2 + center_crop // 2,
                                width // 2 - center_crop // 2: width // 2 + center_crop // 2]

                    if self.scale_factor is not None:
                        width = int(image.shape[1] * self.scale_factor)
                        height = int(image.shape[0] * self.scale_factor)
                        dim = (width, height)
                        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                    cv2.imwrite(frame_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                success, image = vidcap.read()
                count += 1


class OneVideoSceneProcesser:
    def __init__(self,
                 video_path,
                 dataset_output_path,
                 max_len=None,
                 skip=None,
                 center_crop=False,
                 scale_factor=None,
                 img_prefix='jpg',
                 filter_with_sharpness=False,
                 ):

        self.video_path = video_path
        self.dataset_output_path = dataset_output_path
        self.dataset_frames_path = os.path.join(self.dataset_output_path, 'frames')
        self.max_len = max_len
        self.center_crop = center_crop
        self.scale_factor = scale_factor
        self.img_prefix = img_prefix
        self.filter_with_sharpness = filter_with_sharpness

        os.makedirs(self.dataset_output_path, exist_ok=True)
        os.makedirs(self.dataset_frames_path, exist_ok=True)

        videos_stats = get_video_stats(self.video_path)
        self.scene_num_frames = videos_stats['num_frames']

        if self.max_len is not None:
            if self.max_len < self.scene_num_frames:
                self.skip = math.ceil(self.scene_num_frames / self.max_len)
        elif skip is not None:
            self.skip = skip
        else:
            self.skip = 1

        logger.info(f'Every {self.skip} will be skipped, filter with sharpness {self.filter_with_sharpness}')

    def run(self):

        vidcap = cv2.VideoCapture(self.video_path)
        rotateCode = check_rotation(self.video_path)

        success, image = vidcap.read()
        images_sharpness = []
        count = 0
        while success:
            if rotateCode is not None:
                image = correct_rotation(image, rotateCode)
            if count % self.skip == 0 or self.filter_with_sharpness:
                frame_path = os.path.join(self.dataset_frames_path, f'{count:05d}.' + self.img_prefix)
                if self.center_crop:
                    height, width, channels = image.shape
                    if isinstance(self.center_crop, bool):
                        center_crop = min(height, width)
                    else:
                        center_crop = self.center_crop

                    image = image[height // 2 - center_crop // 2: height // 2 + center_crop // 2,
                            width // 2 - center_crop // 2: width // 2 + center_crop // 2]

                if self.scale_factor is not None:
                    width = int(image.shape[1] * self.scale_factor)
                    height = int(image.shape[0] * self.scale_factor)
                    dim = (width, height)
                    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                image_sharpness = sharpness(image)
                images_sharpness.append([count, frame_path, image_sharpness])

                cv2.imwrite(frame_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            success, image = vidcap.read()
            count += 1

        # images_sharpness = sorted(images_sharpness, key=lambda x: x[1])
        #
        # for el in images_sharpness[:-self.max_len]:
        #     os.remove(el[0])

        shift = 0
        while shift < self.scene_num_frames:
            sub_seq = images_sharpness[shift:shift + self.skip]
            sub_seq = sorted(sub_seq, key=lambda x: x[2])
            for el in sub_seq[:-1]:
                os.remove(el[1])
            curr_pos = sub_seq[-1][0]

            for el in images_sharpness[shift + self.skip:curr_pos + self.skip]:
                os.remove(el[1])

            shift = curr_pos + self.skip
