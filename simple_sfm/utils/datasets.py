__all__ = ['ImageFoldersDataset']

import logging
import os
from glob import glob
from typing import Dict

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ImageFoldersDataset(Dataset):
    def __init__(self,
                 folders_dict: dict,
                 transforms_dict: dict,
                 image_name_path_shift: int = 1,
                 ):
        """
        Dataset which allows to load corresponding images from different folders to same item.

        Args:
            folders_dict: folders with path to image folder for each type of images.
            transforms_dict: dict with transform with each type of images

        Returns:
            dict of different types of images for each images
        """
        super().__init__()
        if image_name_path_shift < 1:
            image_name_path_shift = 1

        self.transforms = transforms_dict

        images_paths_dict = dict()
        for key, path in folders_dict.items():
            images_paths_dict[key] = glob(os.path.join(path, '*.jpg'))

        self.images_meta_dict = dict()
        images_names = None
        for key, images_paths in images_paths_dict.items():
            images_names_curr = [('_'.join(el.split('/')[-image_name_path_shift:]).split('.')[0], el) for el in images_paths]
            sorted(images_names_curr, key=lambda x: x[0])

            if images_names is None:
                images_names = images_names_curr
                for i in range(len(images_names)):
                    self.images_meta_dict[images_names_curr[i][0]] = {}

            else:
                assert len(images_names_curr) == len(
                    images_names), f'There is different number of items in {key} image folder {len(images_names_curr)} - elements and others folders {len(images_names)} - elements'
                for i in range(len(images_names)):
                    assert images_names[i][0] == images_names_curr[i][
                        0], f'There is image name {images_names_curr[i]} in {key} folder that missing in other folders.'

            for i in range(len(images_names)):
                self.images_meta_dict[images_names_curr[i][0]][key] = images_names_curr[i][1]

    def __len__(self):
        return len(self.images_meta_dict)

    def __getitem__(self, idx):
        image_name = list(self.images_meta_dict.keys())[idx]
        data = self._read_frame(image_name)
        data['image_name'] = image_name

        return data

    def _read_frame(self,
                    image_name: str,
                    ) -> Dict[str, torch.Tensor]:

        image_meta = self.images_meta_dict[image_name]

        image_data = dict()
        for image_type, image_path in image_meta.items():

            try:
                with Image.open(image_path) as img:
                    image_data[image_type] = self.transforms[image_type](img)
            except OSError as e:
                logger.error(f'Possibly, image file is broken: {image_path}')
                raise e

        return image_data
