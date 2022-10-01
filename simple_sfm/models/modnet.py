__all__ = ['MODNETModel']

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from simple_sfm import utils
from tqdm import tqdm


class MODNETModel:
    def __init__(self,
                 modnet_checkpoint_path: str,
                 batch_size: int = 128,
                 device: str = 'cuda',
                 ref_size: int = 512,
                 ):
        self.batch_size = batch_size
        self.device = device
        self.ref_size = ref_size

        self.modnet = torch.jit.load(modnet_checkpoint_path)
        self.modnet.to(device)

        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def segment_tensor(self, image_tensor, ref_size=None):
        """

        Args:
            image_tensor: B x 3 x H x W (normalized between -1 and 1)
            ref_size: resolution for inference MODNET

        Returns:
            matte: B x 1 x H x W
        """
        if ref_size is None:
            ref_size = self.ref_size

        with torch.no_grad():
            old_im_h, old_im_w = image_tensor.shape[-2:]
            new_size = utils.get_image_size_near_ref_size([old_im_h, old_im_w], ref_size=ref_size, divisor=32)
            image_resized_tensor = F.interpolate(image_tensor, size=(new_size[0], new_size[1]), mode='area')
            matte = self.modnet(image_resized_tensor.cuda())
            matte = F.interpolate(matte, size=(old_im_h, old_im_w), mode='area')

        return matte

    def segment_batch(self, images_tensors):
        # unify image channels to 3
        _, _, h, w = images_tensors.shape

        # inference
        with torch.no_grad():
            matte_batch = []
            for start, end in tqdm(utils.chunker(len(images_tensors), self.batch_size)):
                matte = self.modnet(images_tensors[start:end] - 0.5)
                matte_batch.append(matte)

        return torch.cat(matte_batch, axis=0)

    def segment_image(self, im, ref_size=512):
        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = self.im_transform(im)
        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        old_im_h, old_im_w = im.shape[-2:]
        new_size = utils.get_image_size_near_ref_size(im.shape[-2:], ref_size=ref_size, divisor=32)
        im = F.interpolate(im, size=(new_size[0], new_size[1]), mode='area')

        # inference
        matte = self.modnet(im.cuda())

        matte = F.interpolate(matte, size=(old_im_h, old_im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()

        return matte
