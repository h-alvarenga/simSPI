import numpy as np
from simSPI import geometric_micelle
from scipy.spatial.transform import Rotation

def test_project_rotated_ellipsoid():
  '''
  test that projecting then summing is the same as the analytical formula.
  errors arise from discretiziation on 2D array mesh, so set atol to test close enough
  '''

  scale = 5
  a = 3 * scale
  b = 2 * scale
  c = 1 * scale
  step_x = step_y = 1
  max_axis = np.max([a, b])
  x_mesh, y_mesh = np.meshgrid(np.arange(-max_axis, max_axis, step=step_x), np.arange(-max_axis, max_axis, step=step_y))
  axis = np.array([0, 0, 1])
  deg = 45
  angle = np.deg2rad(deg)
  r = Rotation.from_rotvec(angle * axis).as_matrix()

  proj_ellipsoid = geometric_micelle.project_rotated_ellipsoid(x_mesh, y_mesh, a, b, c, r)
  volume = proj_ellipsoid.sum()

  analytic_vol_ellipsoid = 4 / 3 * np.pi * a * b * c
  rel_error = np.abs(analytic_vol_ellipsoid - volume) / analytic_vol_ellipsoid

  assert np.isclose(rel_error, 0, atol=1e-3)
