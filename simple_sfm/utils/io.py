from typing import List, Union, Tuple, NamedTuple

import logging
import struct

import numpy as np

logger = logging.getLogger(__name__)

VIDEO_FILE_EXTENSIONS = (
    '.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep',
    '.aepx',
    '.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx',
    '.avb',
    '.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin',
    '.bix',
    '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine',
    '.cip',
    '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav',
    '.dce',
    '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d',
    '.dmss',
    '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx',
    '.dxr',
    '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp',
    '.fcproject',
    '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264',
    '.hdmov',
    '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr',
    '.ivs',
    '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v',
    '.m21',
    '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2',
    '.mjp',
    '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie',
    '.mp21',
    '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex',
    '.mpl',
    '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb',
    '.mvc',
    '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc',
    '.ogm',
    '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist',
    '.plproj',
    '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr',
    '.pxv',
    '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd',
    '.rmd',
    '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk',
    '.sbt',
    '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi',
    '.smi',
    '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf',
    '.swi',
    '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp',
    '.ttxt',
    '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg', '.vem', '.vep', '.vf',
    '.vft',
    '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7',
    '.vpj',
    '.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot',
    '.wp3',
    '.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv',
    '.zeg',
    '.zm1', '.zm2', '.zm3', '.zmv')


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


def is_video_file(filename):
    if filename.lower().endswith((VIDEO_FILE_EXTENSIONS)):
        return True
    return False


def load_krt_data(path: str,
                  ) -> dict:
    """
    Load KRT file containing intrinsic and extrinsic parameters.

    Args:
        path (str): Path to KRT file.

    Returns:
        Dictionary: dict with cameras parameters in normal coordinate system.

    """
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrinsic = np.array([[float(x) for x in f.readline().split()] for i in range(3)])
            dist = np.array([float(x) for x in f.readline().split()])
            extrinsic = np.array([[float(x) for x in f.readline().split()] for i in range(3)])


            f.readline()

            cameras[name[:-1]] = {
                'intrinsic': intrinsic.astype(np.float32),
                'dist': dist.astype(np.float32),
                'extrinsic': extrinsic.astype(np.float32),
            }

    return cameras
