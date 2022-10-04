__all__ = ['Matcher']

import itertools
import logging
import time
from typing import NamedTuple, Dict, Tuple, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from simple_sfm.models.superglue import SuperGlue
from simple_sfm.models.superpoint import SuperPoint
from simple_sfm.utils.video_streamer import VideoStreamer
from simple_sfm.utils.datasets import ImageFoldersDataset
from simple_sfm.scene_utils.simple_matcher import SimpleMatcher

logger = logging.getLogger(__name__)


class Frame(NamedTuple):
    descriptors: torch.Tensor
    keypoints_poses: torch.Tensor
    scores: torch.Tensor
    image: torch.Tensor


class Matcher(object):

    def __init__(self,
                 super_point_extractor_weights_path: str,
                 nms_radius: float,
                 keypoint_threshold: float,
                 matcher_type: str = 'simple',
                 super_glue_weights_path: str = None,
                 match_threshold: float = 0.4,
                 sinkhorn_iterations: int = 20,
                 super_glue_batch: int = 5):

        self.matcher_type = matcher_type

        self.super_point_extractor = SuperPoint(weights_path=super_point_extractor_weights_path,
                                                nms_radius=nms_radius,
                                                keypoint_threshold=keypoint_threshold)
        self.super_point_extractor.cuda()

        if self.matcher_type == 'simple':
            self.matcher = SimpleMatcher(match_threshold)
        elif self.matcher_type == 'super_glue':
            assert super_glue_weights_path is not None, 'Please set weights path for SuperGlue.'
            self.matcher = SuperGlue(weights_path=super_glue_weights_path,
                                     match_threshold=match_threshold,
                                     sinkhorn_iterations=sinkhorn_iterations)
            self.super_glue_batch = super_glue_batch
            self.matcher.cuda()

    def __super_glue_matcher(self,
                             frames: List[Frame],
                             match_list: List[Tuple[int, int]]) -> Dict[Tuple[int, int], np.array]:
        """

        :param frames: Frames with filed keypoints_poses, images, scores and descriptors fields.
        :param match_list: List with pairs which must be matched.
        :return: Dict where the key is a pair from the match_list,
                and the value is the points that match between images in pair.
        """

        match_table = {}

        for match_batch in [match_list[i:i + self.super_glue_batch] for i in
                            range(0, len(match_list), self.super_glue_batch)]:
            match_batch = match_batch - 1

            keys_1 = np.array(match_batch[:, 0])
            keys_2 = np.array(match_batch[:, 1])
            frames_1 = [frames[i] for i in keys_1]
            frames_2 = [frames[i] for i in keys_2]

            keypoints_max_1 = min([len(x.keypoints_poses) for x in frames_1])
            keypoints_max_2 = min([len(x.keypoints_poses) for x in frames_2])

            data = {
                'descriptors0': [x.descriptors[:, :keypoints_max_1] for x in frames_1],
                'descriptors1': [x.descriptors[:, :keypoints_max_2] for x in frames_2],
                'keypoints0': [x.keypoints_poses[:keypoints_max_1] for x in frames_1],
                'keypoints1': [x.keypoints_poses[:keypoints_max_2] for x in frames_2],
                'scores0': [x.scores[:keypoints_max_1] for x in frames_1],
                'scores1': [x.scores[:keypoints_max_2] for x in frames_2],
                'image0': [x.image for x in frames_1],
                'image1': [x.image for x in frames_2],
            }

            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])

            with torch.no_grad():
                res = self.matcher(data)

            for i, match_pair in enumerate(match_batch):
                matches = res['matches0'][i]
                valid = matches > -1
                matches = torch.stack([torch.nonzero(valid)[:, 0], matches[valid]], dim=1)

                matches = matches.cpu().numpy().astype(np.uint32)
                match_table[(match_pair[0] + 1, match_pair[1] + 1)] = matches

            del res
            torch.cuda.empty_cache()
        return match_table

    def __simple_matcher(self,
                         frames: List[Frame],
                         match_list: List[Tuple[int, int]]) -> Dict[Tuple[int, int], np.array]:
        """
        :param frames: Frames with filed descriptors field.
        :param match_list: List with pairs which must be matched.
        :return: Dict where the key is a pair from the match_list,
                and the value is the points that match between images in pair.
        """
        match_table = {}
        for image_id_1, image_id_2 in match_list:
            matches = self.matcher.match_two_way_cuda(frames[image_id_1 - 1].descriptors,
                                                      frames[image_id_2 - 1].descriptors).t()

            matches = matches[:, :2].cpu().numpy().astype(np.uint32)
            match_table[(image_id_1, image_id_2)] = matches
        return match_table

    def match_frames(self,
                     frames: List[Frame],
                     images_names: dict,
                     match_pairs_filter: callable = None) -> Tuple[List[Frame],
                                                                   Dict[Tuple[int, int], np.array],
                                                                   Dict[int, str]]:

        logger.info('Starts matching descriptors.')
        start = time.time()

        match_list = np.array(list(itertools.combinations(images_names.keys(), r=2)))
        match_list = np.array(list(filter(match_pairs_filter, match_list)))

        match_table = None

        if self.matcher_type == 'simple':
            match_table = self.__simple_matcher(frames, match_list)
        elif self.matcher_type == 'super_glue':
            match_table = self.__super_glue_matcher(frames, match_list)

        num_matched_points = 0
        for keys, value in match_table.items():
            num_matched_points += len(value)

        end = time.time()
        logger.info(f'Finished \n'
                    f'Elapsed time: {float(end - start)} \n'
                    f'Num matches: {len(match_list)} \n'
                    f'Num matched points: {num_matched_points} \n')

        torch.cuda.empty_cache()

        return frames, match_table, images_names

    def match_datasets(self,
                      datasets: Union[ImageFoldersDataset, List[ImageFoldersDataset]],
                      match_pairs_filter: callable = None) -> Tuple[List[Frame],
                                                                    Dict[Tuple[int, int], np.array],
                                                                    Dict[int, str],
                                                                    List[int]]:
        """
        Get ImageFoldersDataset dataset or list of ImageFoldersDatasets and processed all frames from it,
        image folders type must be set as 'rgb' and 'mask'(if masks are used).

        :param datasets: ImageFoldersDataset or list of ImageFoldersDataset
        :param match_pairs_filter: Condition for filtering pairs, if you want all pairs set None.
                                   For example, lambda x : np.abs(x[0] -x[1]) % 2 == 0,
                                   filter all match pairs which have even distance between each other.
        :return: processed_frames - List of Frame,
                 match_table - Dict where the key is a pair from the match_list,
                 and the value is the points that match between images in pair.
                 images_names - Dict where the key is a index of image and the value is the name of image.
                 dataset_split_bd_index - border index between frames from different datasets
        """

        if match_pairs_filter is None:
            match_pairs_filter = lambda x: True

        logger.info('Starts extracting descriptors.')
        start = time.time()
        image_bd_index = 1

        images_bd_ids = {}
        images_bd_index_names = {}
        processed_frames = []

        num_keypoints = 0

        if ~isinstance(datasets, List):
            datasets = [datasets]

        dataset_split_bd_index = []
        for dataset in datasets:
            image_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            for data in image_dataloader:

                images = data['rgb'].cuda()
                images_names = data['image_name']

                masks = None
                if 'mask' in data:
                    masks = data['mask'].cuda()

                with torch.no_grad():
                    data = {'image': images}
                    if masks is not None:
                        data['mask'] = masks > 0

                    result = self.super_point_extractor(data)

                keypoints_pos = result['keypoints']
                descriptors = result['descriptors']
                scores = result['scores']

                processed_frames.append(Frame(descriptors,
                                              keypoints_pos,
                                              scores,
                                              images[0]))

                images_bd_index_names[image_bd_index] = images_names

                image_bd_index += 1
                num_keypoints += len(keypoints_pos)

            dataset_split_bd_index.append(image_bd_index)

        end = time.time()

        logger.info(f'Finished \n '
                    f'Elapsed time: {float(end - start)} \n'
                    f'Num images: {len(processed_frames)} \n'
                    f'Num keypoints: {num_keypoints} \n')

        return *self.match_frames(processed_frames, images_bd_index_names, match_pairs_filter), dataset_split_bd_index

    def match_video_stream(self,
                           vs: VideoStreamer,
                           match_pairs_filter: callable = None,
                           vs_mask: VideoStreamer = None) -> Tuple[List[Frame],
                                                                   Dict[Tuple[int, int], np.array],
                                                                   Dict[int, str]]:
        """
        Get VideoStreamer and processed all frames from it,.
        :param vs: VideoStreamer
        :param match_pairs_filter: Condition for filtering pairs, if you want all pairs set None.
                                   For example, lambda x : np.abs(x[0] -x[1]) % 2 == 0,
                                   filter all match pairs which have even distance between each other.
        :param vs_mask: VideoStreamer: video streamer that provides mask for images
        :return: processed_frames - List of Frame,
                 match_table - Dict where the key is a pair from the match_list,
                 and the value is the points that match between images in pair.
                 images_names - Dict where the key is a index of image and the value is the name of image.
        """

        if match_pairs_filter is None:
            match_pairs_filter = lambda x: True

        logger.info('Starts extracting descriptors.')
        start = time.time()
        image_bd_index = 1

        images_bd_ids = {}
        images_bd_index_names = {}
        processed_frames = []

        num_keypoints = 0
        while True:
            image, status, img_path = vs.next_frame()
            if vs_mask is not None:
                image_mask, status_mask, img_mask_path = vs_mask.next_frame()
                status &= status_mask

            if status is False:
                break

            with torch.no_grad():
                image = torch.from_numpy(image)[None, None, :, :].cuda()
                data = {'image': image}
                if vs_mask:
                    image_mask = torch.from_numpy(image_mask > 0)[None, None, :, :].cuda()
                    data['mask'] = image_mask

                result = self.super_point_extractor(data)

            keypoints_pos = result['keypoints']
            descriptors = result['descriptors']
            scores = result['scores']

            processed_frames.append(Frame(descriptors,
                                          keypoints_pos,
                                          scores,
                                          image.squeeze(1)))

            image_name = img_path.split('/')[-1]
            images_bd_ids[image_name] = image_bd_index
            images_bd_index_names[image_bd_index] = image_name

            image_bd_index += 1
            num_keypoints += len(keypoints_pos)

        end = time.time()

        logger.info(f'Finished \n '
                    f'Elapsed time: {float(end - start)} \n'
                    f'Num images: {len(processed_frames)} \n'
                    f'Num keypoints: {num_keypoints} \n')

        return self.match_frames(processed_frames, images_bd_index_names, match_pairs_filter)