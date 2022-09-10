
def read_re10k_views(views_file_path, scale=1.0):
  with open(views_file_path) as f:
    text = f.read()

  rows = text.split('\n')[1:-1]
  intrinsics = {}
  extrinsics = {}

  for cam_idx, row in enumerate(rows):
    row_name = row.split(' ')[0]
    (focal_length_x, focal_length_y, principal_point_x, principal_point_y) = np.array(row.split(' ')[1:5]).astype(np.float32)

    intrinsics[cam_idx] = np.array([[focal_length_x, 0, principal_point_x],
                                      [0, focal_length_y, principal_point_y],
                                      [0, 0, 1]])
    extrs = np.array(row.split(' ')[7:19]).astype(np.float32).reshape((3, 4), order='C')
    extrs[:, -1] = extrs[:, -1] / scale

    extrinsics[cam_idx] = extrs

  return intrinsics, extrinsics