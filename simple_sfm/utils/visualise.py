import multiprocessing
import os
import random
import shutil
import subprocess
import glob
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image
import torch

from simple_sfm.cameras import CameraMultiple

try:
    from matplotlib import cm

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def multiprocessing_save_image(params):
    ndarr, file_name = params
    image = Image.fromarray(ndarr)
    image.save(file_name, format='JPEG', subsampling=0, quality=80)


def write_loss(iterations, losses, train_writer):
    losses_tb = {}

    for optimizer_group in losses.keys():
        for loss_name, value in losses[optimizer_group].items():
            losses_tb['loss_' + optimizer_group + '_' + loss_name] = value

    for loss_name, value in losses_tb.items():
        train_writer.add_scalar(loss_name, losses_tb[loss_name], iterations)


def write_metrics(iterations, metrics, train_writer):
    metrics_tb = {}

    for metric_name, value in metrics.items():
        metrics_tb['metric_' + metric_name] = value

    for metric_name, value in metrics_tb.items():
        train_writer.add_scalar(metric_name, metrics_tb[metric_name], iterations)


def write_grad_norms(iterations, gradient_info, train_writer):
    for name, gradient_norm in gradient_info.items():
        train_writer.add_scalar(name.replace('.', '/'), gradient_norm, iterations)


def plot_square_grid(images, cols, rows, padding=5, first_pad=0):
    widths, heights = images[0].size

    if first_pad == 0:
        first_pad = padding

    paddings_horizont = [padding] * (cols)
    paddings_vertical = [padding] * (rows)
    paddings_horizont[-1] = 0
    paddings_vertical[-1] = 0

    paddings_horizont[0] = first_pad

    table_width = widths * cols + sum(paddings_horizont)
    table_height = heights * rows + sum(paddings_vertical)

    new_im = Image.new('RGBA', (table_width, table_height))

    x = 0
    for i in range(rows):
        new_im.paste(Image.fromarray(np.ones((paddings_vertical[i], table_width, 4), dtype=np.uint8) * 0),
                     (x, 0))
        x += heights + paddings_vertical[i]

    y = 0
    for i in range(cols):
        new_im.paste(Image.fromarray(np.ones((table_height, paddings_horizont[i], 4), dtype=np.uint8) * 0),
                     (0, y))
        y += widths + paddings_horizont[i]

    y = 0
    for i in range(rows):
        x = 0
        for j in range(cols):
            new_im.paste(images[j + i * cols], (x, y))
            x += (widths + paddings_horizont[j])
        y += (heights + paddings_vertical[i])

    return new_im


def image_folder_to_video(
        output_path: str,
        folder_path: str,
        image_name_format: str = '%06d.jpg',
        remove_image_folder: bool = False
):
    """
    Convert folder with images to video, folder must contain only images which will be used in generating video.
    Args:
        output_path: output video path
        folder_path: folder with images
        image_name_format: name format for ffmpeg
        remove_image_folder: remove image folder after video generating, or not.

    """
    num_images = len(glob.glob(os.path.join(folder_path, '*.' + image_name_format.split('.')[-1])))
    command = (
            f'ffmpeg -hide_banner -loglevel warning -y -r 30 -i {folder_path}/{image_name_format} '
            + f'-vframes {num_images} '
            + '-vcodec libx264 -crf 18 '
            + '-pix_fmt yuv420p '
            + output_path
    ).split()
    subprocess.call(command)
    if remove_image_folder:
        shutil.rmtree(folder_path)


class ImageWriter:
    def __init__(self,
                 output_path,
                 n_threads=16,
                 ):
        self.output_path = output_path
        self.write_pool = multiprocessing.Pool(n_threads)
        os.makedirs(self.output_path, exist_ok=True)

    def __del__(self):
        self.write_pool.close()

    @staticmethod
    def write_image(inputs):
        folder, name, numpy_image = inputs
        numpy_image = np.clip(numpy_image, 0., 255.).astype(np.uint8)
        if numpy_image.shape[1] % 2 != 0:
            # ffmpeg cannot process such frames
            numpy_image = numpy_image[:, :-1]
        filename = f'{name}.jpg'
        image = Image.fromarray(numpy_image)
        image.save(os.path.join(folder, filename))
        del image

    def save_images(self, images: np.array, names: List[str] = None):
        """
        Save array images to folder.
        Args:
            images: N x H x W x C
            names:  list of length N with names, if None, saver uses frame idx in array.

        Returns:

        """
        num_images = images.shape[0]
        if names is None:
            names = [f'{i:06d}' for i in range(num_images)]

        # for name, image in zip(names, images):
        #     self.write_image((self.output_path, name, image))
        self.write_pool.map(self.write_image,
                            zip([self.output_path] * num_images,
                                names,
                                images)
                            )


class VideoWriter:
    def __init__(self,
                 output_path,
                 n_threads=16,
                 tmp_root='/tmp',
                 ):
        self.output_path = output_path
        self.write_pool = multiprocessing.Pool(n_threads)
        self.n_items = 0
        self.random_id = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        self.tmp_dir = os.path.join(tmp_root, self.random_id)

        self.image_writer = ImageWriter(output_path=self.tmp_dir, n_threads=n_threads)

    def process_batch(self, images: torch.Tensor):
        images = images.data.permute(0, 2, 3, 1).to('cpu').numpy()
        batch_size = images.shape[0]
        names = [f'{(self.n_items + i):06d}' for i in range(batch_size)]
        self.image_writer.save_images(images, names)
        self.n_items += batch_size

    def finalize(self):
        command = (
                f'ffmpeg -hide_banner -loglevel warning -y -r 30 -i {self.tmp_dir}/%06d.jpg '
                + f'-vframes {self.n_items} '
                + '-vcodec libx264 -crf 18 '
                + '-pix_fmt yuv420p '
                + self.output_path
        ).split()
        subprocess.call(command)
        shutil.rmtree(self.tmp_dir)


def grayscale_to_cmap(array: np.ndarray,
                      cmap: str = 'viridis'):
    if MATPLOTLIB_AVAILABLE:
        cmap = cm.get_cmap(cmap)
        return cmap(array)[..., :3]
    else:
        return np.stack([array] * 3, axis=-1)


class VideoWriterPool:
    def __init__(self,
                 output_path,
                 n_threads=16,
                 tmp_root='/tmp',
                 ):
        self.output_path = output_path
        self.write_pool = multiprocessing.Pool(n_threads)
        self.n_items = 0
        self.random_id = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        self.tmp_dir = os.path.join(tmp_root, self.random_id)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def __del__(self):
        self.write_pool.close()

    @staticmethod
    def write_image(inputs):
        folder, frame_number, numpy_image = inputs
        numpy_image = np.clip(numpy_image, 0., 255.).astype(np.uint8)
        if numpy_image.shape[1] % 2 != 0:
            # ffmpeg cannot process such frames
            numpy_image = numpy_image[:, :-1]
        filename = f'{frame_number:06d}.jpg'
        Image.fromarray(numpy_image).save(os.path.join(folder, filename))

    @staticmethod
    def write_pil_image(inputs):
        folder, frame_number, pil_image = inputs
        filename = f'{frame_number:06d}.jpg'
        pil_image.save(os.path.join(folder, filename), quality=100)

    def process_batch(self, images: torch.Tensor):
        images = images.data.permute(0, 2, 3, 1).to('cpu').numpy()
        batch_size = images.shape[0]
        self.write_pool.map(self.write_image,
                            zip([self.tmp_dir] * len(images),
                                self.n_items + np.arange(batch_size),
                                images)
                            )
        self.n_items += batch_size

    def process_pil_list(self, images: List):
        batch_size = len(images)
        self.write_pool.map(self.write_pil_image,
                            zip([self.tmp_dir] * len(images),
                                self.n_items + np.arange(batch_size),
                                images)
                            )
        self.n_items += batch_size

    def finalize(self):
        command = (
                f'ffmpeg -hide_banner -loglevel warning -y -r 30 -i {self.tmp_dir}/%06d.jpg '
                + f'-vframes {self.n_items} '
                + '-vcodec libx264 -crf 18 '
                + '-pix_fmt yuv420p '
                + self.output_path
        ).split()
        subprocess.call(command)
        shutil.rmtree(self.tmp_dir)


def ploty_plot_extrinsics(
        fig,
        extrinsics: torch.Tensor,
        cams_ids: List = None,
        axis_scale: float = 0.01,
        line_width: int = 4,
        marker_size: float = 1,
        marker_color: str = 'blue',
        opacity: float = 1
):
    """
    Method for ploting camera extrinsict on 3d plotly figure.
    :param fig:
    :param extrinsics: tensor of shape num_cams x 4 x 4
    :param cams_ids: list of cameras ids or names
    :param axis_scale: plot axis scale
    :param line_width: camera axis plot line width
    :param marker_size: size of camera marker
    :param marker_color: marker color
    :param opacity: plot opacity
    :return:
    """
    assert PLOTLY_AVAILABLE, 'There is no plotly lib!'

    translation = extrinsics[..., :3, -1:]
    rotation = extrinsics[..., :3, :3]

    cameras_world_positions = -rotation.transpose(-1, -2) @ translation
    cameras_world_positions = cameras_world_positions[:, :, 0]
    cameras_R = rotation

    if cams_ids is None:
        cams_ids = list(range(cameras_R.shape[0]))

    fig = fig.add_trace(go.Scatter3d(
        x=cameras_world_positions[:, 0],
        y=cameras_world_positions[:, 1],
        z=cameras_world_positions[:, 2],
        mode="markers+text",
        marker={"size": marker_size, 'color': marker_color},
        text=[str(el) for el in cams_ids],
        textposition="middle left",
        opacity=opacity
    ))

    for i in range(cameras_R.shape[0]):
        R, T = cameras_R[i], cameras_world_positions[i]

        fig.add_trace(go.Scatter3d(
            x=[T[0], T[0] + R[0, 0] * axis_scale],
            y=[T[1], T[1] + R[0, 1] * axis_scale],
            z=[T[2], T[2] + R[0, 2] * axis_scale],
            line={"color": 'red', 'width': line_width},
            mode='lines',
            showlegend=False,
            opacity=opacity
        ))
        fig.add_trace(go.Scatter3d(
            x=[T[0], T[0] + R[1, 0] * axis_scale],
            y=[T[1], T[1] + R[1, 1] * axis_scale],
            z=[T[2], T[2] + R[1, 2] * axis_scale],
            line={"color": 'green', 'width': line_width},
            mode='lines',
            showlegend=False,
            opacity=opacity
        ))
        fig.add_trace(go.Scatter3d(
            x=[T[0], T[0] + R[2, 0] * axis_scale],
            y=[T[1], T[1] + R[2, 1] * axis_scale],
            z=[T[2], T[2] + R[2, 2] * axis_scale],
            line={"color": 'blue', 'width': line_width},
            mode='lines',
            showlegend=False,
            opacity=opacity
        ))

    return fig


def plotly_plot_cameras(
        cameras: CameraMultiple
):
    assert PLOTLY_AVAILABLE, 'There is no plotly lib!'
    extrinsics = cameras.extrinsics.cpu()
    fig = go.Figure()
    translation = extrinsics[..., :3, -1:]
    rotation = extrinsics[..., :3, :3]

    cameras_world_positions = -rotation.transpose(-1, -2) @ translation
    cameras_world_positions = cameras_world_positions[:, :, 0]
    axis_min, _ = cameras_world_positions.min(axis=0)
    axis_max, _ = cameras_world_positions.max(axis=0)
    axis_diff = axis_max - axis_min
    scale = max(axis_diff).item()
    axis_center = (axis_min + axis_max) / 2
    print(scale)
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[axis_center[0] - scale / 2, axis_center[0] + scale / 2], ),
            yaxis=dict(nticks=4, range=[axis_center[1] - scale / 2, axis_center[1] + scale / 2], ),
            zaxis=dict(nticks=4, range=[axis_center[2] - scale / 2, axis_center[2] + scale / 2], ),
        ),
    )

    fig.update_layout(scene_aspectmode='cube')

    fig.update_layout(showlegend=False)
    fig.update_layout(
        autosize=False,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    fig = ploty_plot_extrinsics(fig,
                                extrinsics,
                                cams_ids=cameras.cameras_ids[:, 0],
                                opacity=0.9,
                                axis_scale=0.1 * scale,
                                marker_size=10,
                                marker_color='orange'
                                )
    return fig


def plotly_plot_cameras_to_images(
        cameras: CameraMultiple,
        output_path: str,
        resolution=None,
        plotly_scale=1
):
    if resolution is None:
        resolution = [800, 600]

    assert PLOTLY_AVAILABLE, 'There is no plotly lib!'
    os.makedirs(output_path, exist_ok=True)
    fig = plotly_plot_cameras(cameras)

    ## plot xyz
    fig.write_image(os.path.join(output_path, "cameras_plot_xyz.jpg"),
                    width=resolution[0] / plotly_scale,
                    height=resolution[0] / plotly_scale,
                    scale=plotly_scale
                    )
    ## plot xz
    camera = dict(
        eye=dict(x=0., y=2.0, z=0.)
    )
    fig.update_layout(scene_camera=camera, title='xz')
    fig.write_image(os.path.join(output_path, "cameras_plot_xz.jpg"),
                    width=resolution[0] / plotly_scale,
                    height=resolution[0] / plotly_scale,
                    scale=plotly_scale
                    )
    ## plot xy
    camera = dict(
        eye=dict(x=0., y=0, z=2.0)
    )
    fig.update_layout(scene_camera=camera, title='xy')
    fig.write_image(os.path.join(output_path, "cameras_plot_xy.jpg"),
                    width=resolution[0] / plotly_scale,
                    height=resolution[0] / plotly_scale,
                    scale=plotly_scale
                    )
    ## plot yz
    camera = dict(
        eye=dict(x=2.0, y=0, z=0)
    )
    fig.update_layout(scene_camera=camera, title='yz')
    fig.write_image(os.path.join(output_path, "cameras_plot_y.jpg"),
                    width=resolution[0] / plotly_scale,
                    height=resolution[0] / plotly_scale,
                    scale=plotly_scale
                    )
