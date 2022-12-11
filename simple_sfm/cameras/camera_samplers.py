import torch

from simple_sfm.utils.geometry import find_most_distant_point, find_most_distant_point_a_to_b


def most_distant(cameras, k):
    """
    Samples k most distant from each other cameras.
    :param cameras: CameraMultiple
    :param k: num cameras
    :return: CameraMultiple, cameras_ids
    """

    assert len(cameras) > k, 'Num cameras must be larger than k'
    cam_pos = cameras.world_position

    first_point_id = find_most_distant_point(cam_pos)
    result_id = first_point_id[None]

    for i in range(k - 1):
        next_point_id = find_most_distant_point_a_to_b(cam_pos, cam_pos[result_id])
        result_id = torch.cat([result_id, next_point_id[None]])

    cameras_ids = cameras.cameras_ids[result_id]
    print(cameras_ids)
    return cameras.get_cams_with_cams_index(cameras_ids), cameras_ids
