import numpy as np
import cv2
import torch
import torch.nn.functional as F


def crop_image(image, bbox):
    """Crops area from image specified as bbox. Always returns image of bbox shape filling missing parts with zeros
    Args:
        image [numpy array (h, w, 3)]: input image
        bbox [tuple of size 4]: input bbox (left, top, right, bottom)
    Returns:
        image_cropped [numpy array (bbox_h, bbox_w, 3)]: resulting cropped image
    """
    h, w = image.shape[:2]
    left, top, right, bottom = (int(x) for x in bbox)

    assert (left < right) and (top < bottom)
    assert (left < w) and (top < h) and (right > 0) and (bottom > 0)

    # crop
    image = image[
            max(top, 0):min(bottom, h),
            max(left, 0):min(right, w),
            :
            ]

    # pad image
    left_pad = max(0 - left, 0)
    top_pad = max(0 - top, 0)
    right_pad = max(right - w, 0)
    bottom_pad = max(bottom - h, 0)

    if any((left_pad, top_pad, right_pad, bottom_pad)):
        image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)))

    return image


def crop_tensor(tensor, bbox):
    h, w = tensor.shape[-2:]
    left, top, right, bottom = (int(x) for x in bbox)

    assert (left < right) and (top < bottom)
    assert (left < w) and (top < h) and (right > 0) and (bottom > 0)

    tensor = tensor[:, :, max(top, 0):min(bottom, h), max(left, 0):min(right, w)]

    left_pad = max(0 - left, 0)
    top_pad = max(0 - top, 0)
    right_pad = max(right - w, 0)
    bottom_pad = max(bottom - h, 0)

    if any((left_pad, top_pad, right_pad, bottom_pad)):
        tensor = F.pad(tensor, (left_pad, right_pad, top_pad, bottom_pad), "constant", 0)

    return tensor


def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def resize_image_long_side(image, long_side_size):
    img_size = list(image.shape)[:2]
    scale = long_side_size / max(img_size)
    return cv2.resize(image, (int(img_size[1] * scale), int(img_size[0] * scale)), interpolation=cv2.INTER_AREA)


def scale_bbox(bbox, scale=1.0):
    bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    bbox_center_y, bbox_center_x = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2

    new_bbox_h, new_bbox_w = scale * bbox_h, scale * bbox_w

    l = int(bbox_center_x - new_bbox_w / 2)
    t = int(bbox_center_y - new_bbox_h / 2)
    r = l + int(new_bbox_w)
    b = t + int(new_bbox_h)

    return (l, t, r, b)


def central_crop(image, size):
    image_size = np.array(image.shape[:2])
    center = image_size / 2
    x = int(center[0] - min(size[0], image_size[0]) / 2)
    y = int(center[1] - min(size[1], image_size[1]) / 2)
    return image[x:x + size[0], y:y + size[1]]


def central_crop_tensor(image, size):
    image_size = np.array(image.shape[-2:])
    center = image_size / 2
    x = int(center[0] - min(size[0], image_size[0]) / 2)
    y = int(center[1] - min(size[1], image_size[1]) / 2)
    return image[:, :, x:x + size[0], y:y + size[1]]


def make_coordinate_grid(shape):
    h, w = shape
    X, Y = np.meshgrid(np.arange(h), np.arange(w))
    X = 2 * (X / 255) - 1
    Y = 2 * (Y / 255) - 1

    return np.stack([X, Y], axis=-1)


def make_overlay_mask(shape):
    identity_grid = make_coordinate_grid(shape)
    dist_to_edge_grid = ((np.abs(identity_grid) - np.array([[1, 1]], dtype=np.float32)) ** 2).min(axis=-1)
    mask = (1 - np.exp(-80 * dist_to_edge_grid)).reshape(shape)
    return mask


def overlay_one_rgba_image_above_another(
        background_image,
        foreground_image,
):
    img_type = background_image.dtype
    mask = foreground_image[:, :, -1:] / 255
    foreground_image = foreground_image[:, :, :3]
    background_image = background_image[:, :, :3]
    return overlay_one_image_above_another(background_image, foreground_image, mask).astype(img_type)


def overlay_one_image_above_another(
        background_image,
        foreground_image,
        overlay_mask,
        foreground_image_bbox=None
):
    canvas = np.copy(background_image)
    foreground_h, foreground_w = foreground_image.shape[:2]
    background_h, background_w = background_image.shape[:2]

    if foreground_image_bbox is not None:
        left_gap = 0 - min(foreground_image_bbox[0], 0)
        top_gap = 0 - min(foreground_image_bbox[1], 0)
        right_gap = max(foreground_image_bbox[2], background_w) - background_w
        bottom_gap = max(foreground_image_bbox[3], background_h) - background_h

        foreground_image_cropped = foreground_image[top_gap:foreground_w - bottom_gap:,
                                   left_gap:foreground_w - right_gap]
        overlay_mask_cropped = overlay_mask[top_gap:foreground_w - bottom_gap:, left_gap:foreground_w - right_gap]

        slice_0 = slice(max(foreground_image_bbox[1], 0), min(foreground_image_bbox[1] + foreground_h, background_h))
        slice_1 = slice(max(foreground_image_bbox[0], 0), min(foreground_image_bbox[0] + foreground_w, background_w))

        canvas[slice_0, slice_1] = canvas[slice_0, slice_1] * (
                1 - overlay_mask_cropped) + foreground_image_cropped * overlay_mask_cropped
    else:
        canvas = canvas * (1 - overlay_mask) + foreground_image * overlay_mask

    return canvas


def overlay_one_2dtensor_above_another(
        background_tensor,
        foreground_tensor,
        overlay_mask,
        foreground_tensor_bbox=None
):
    foreground_h, foreground_w = foreground_tensor.shape[-2:]
    background_h, background_w = background_tensor.shape[-2:]
    canvas = torch.clone(background_tensor)
    if foreground_tensor_bbox is not None:
        left_gap = 0 - min(foreground_tensor_bbox[0], 0)
        top_gap = 0 - min(foreground_tensor_bbox[1], 0)
        right_gap = max(foreground_tensor_bbox[2], background_w) - background_w
        bottom_gap = max(foreground_tensor_bbox[3], background_h) - background_h

        foreground_image_cropped = foreground_tensor[:, :,
                                   top_gap:foreground_w - bottom_gap:,
                                   left_gap:foreground_w - right_gap
                                   ]
        overlay_mask_cropped = overlay_mask[:, :,
                               top_gap:foreground_w - bottom_gap:,
                               left_gap:foreground_w - right_gap
                               ]
        slice_0 = slice(max(foreground_tensor_bbox[1], 0),
                        min(foreground_tensor_bbox[1] + foreground_h, background_h))
        slice_1 = slice(max(foreground_tensor_bbox[0], 0),
                        min(foreground_tensor_bbox[0] + foreground_w, background_w))

        canvas[:, :, slice_0, slice_1] = canvas[:, :, slice_0, slice_1] * (
                1 - overlay_mask_cropped) + foreground_image_cropped * overlay_mask_cropped
    else:
        canvas = canvas * (1 - overlay_mask) + foreground_tensor * overlay_mask

    return canvas


def get_image_size_near_ref_size(image_size, ref_size, divisor=None):
    im_h, im_w = image_size
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    if divisor is not None:
        im_rw = im_rw - im_rw % divisor
        im_rh = im_rh - im_rh % divisor

    return [im_rh, im_rw]
