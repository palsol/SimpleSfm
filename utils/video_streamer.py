__all__ = ['VideoStreamer']

import logging
import math
import os

from glob import glob
import cv2

logger = logging.getLogger(__name__)


class VideoStreamer(object):
    """ Class to help process image streams. Three types of possible inputs:"
      A directory of images (files in directory matching 'img_glob').
      A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, height, width, skip=None, max_len=100, img_glob='*.jpg'):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.i = 0
        self.max_len = None

        if skip is not None:
            self.skip = skip
            self.max_len = None
        elif max_len is not None:
            self.skip = None
            self.max_len = max_len
        else:
            self.skip = 1

        self.cap = cv2.VideoCapture(basedir)
        lastbit = basedir[-4:len(basedir)]
        if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
            raise IOError('Cannot open movie file')
        elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
            logger.info('==> Processing Video Input.')
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)

            if skip is None:
                if self.max_len is not None and self.max_len < num_frames:
                    self.skip = math.ceil(num_frames / self.max_len)
                else:
                    self.skip = 1

            self.listing = self.listing[::self.skip]
            self.camera = True
            self.video_file = True
            self.max_len = len(self.listing)
        else:
            logger.info('==> Processing Image Directory Input.')
            search = os.path.join(basedir, img_glob)
            self.listing = glob(search)
            self.listing.sort()
            num_frames = len(self.listing)

            if skip is None:
                if self.max_len is not self.max_len < num_frames:
                    self.skip = math.ceil(num_frames / self.max_len)
                else:
                    self.skip = 1
            self.listing = self.listing[::self.skip]
            self.max_len = len(self.listing)
            if self.max_len == 0:
                raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

    def get_resolution(self):
        image_file = self.listing[0]
        image = self.read_image(image_file, self.sizer)
        height, width = image.shape
        return width, height

    def read_image(self, impath, img_size):
        """ Read image as grayscale and resize to img_size.
        Inputs
          impath: Path to input image.
          img_size: (W, H) tuple specifying resize size.
        Returns
          grayim: float32 numpy array sized H x W with values in range [0, 1].
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        if img_size[0] is not None and img_size[1] is not None:
            grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
           image: Next H x W image.
           status: True or False depending whether image was loaded.
        """
        if self.i == self.max_len:
            return (None, False, None)
        else:
            image_file = self.listing[self.i]
            input_image = self.read_image(image_file, self.sizer)
        # Increment internal counter.
        self.i = self.i + 1
        input_image = input_image.astype('float32')
        return (input_image, True, image_file)
