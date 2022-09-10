import logging
import struct

import numpy as np

logger = logging.getLogger(__name__)

def load_colmap_binary_cloud(filename: str) -> np.ndarray:
    """
    See
    https://github.com/colmap/colmap/blob/d84ccf885547c6e91c58b2e5dab8a6a34e636be3/scripts/python/read_write_model.py#L128
    """
    list_points = []

    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    with open(filename, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)

            list_points.append(xyz)
    return np.stack(list_points, axis=0)


def load_colmap_bin_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def save_pointcloud_to_ply(filename, xyz_points, rgb_points=None):
    """
    Creates a .pkl file of the point clouds generated

    Args:
        filename:
        xyz_points:
        rgb_points: from 0 to 255

    """

    assert xyz_points.shape[1] == 3, 'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape, \
        'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename, 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    xyz_points = xyz_points.astype(float)
    rgb_points = rgb_points.astype(np.uint8)
    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",
                                        xyz_points[i, 0],
                                        xyz_points[i, 1],
                                        xyz_points[i, 2],
                                        rgb_points[i, 0].tostring(),
                                        rgb_points[i, 1].tostring(),
                                        rgb_points[i, 2].tostring())))
    fid.close()
