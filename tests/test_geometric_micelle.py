import math
import numpy as np
import torch
from torch import tensor
from scipy.spatial.transform import Rotation

from simSPI.geometric_micelle import project_rotated_ellipsoid, projected_rotated_circle, project_rotated_cylinder, two_phase_micelle


def test_rotation_conventions():
  angle_y = 45
  angle_z = 90
  angle_x = 10
  convention = 'XZY'
  Ry = Rotation.from_euler(convention,[0,0,angle_y],degrees=True).as_matrix()
  Rz = Rotation.from_euler(convention,[0,angle_z,0],degrees=True).as_matrix()
  Rx = Rotation.from_euler(convention,[angle_x,0,0],degrees=True).as_matrix()
  Rxzy = Rotation.from_euler(convention,[angle_x,angle_z,angle_y],degrees=True).as_matrix()
  assert np.allclose(Rxzy,Rx@Rz@Ry)

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
  max_axis = max(a, b)
  y_mesh, x_mesh = torch.meshgrid(torch.arange(-max_axis, max_axis, step=step_x), torch.arange(-max_axis, max_axis, step=step_y))
  axis = tensor([0, 0, 1]).float()
  deg = 45
  angle = np.deg2rad(deg)
  rotations = torch.from_numpy(Rotation.from_rotvec(angle * axis).as_matrix())

  proj_ellipsoid = project_rotated_ellipsoid(x_mesh, y_mesh, a, b, c, rotations)
  volume = proj_ellipsoid.sum()

  analytic_vol_ellipsoid = 4 / 3 * math.pi * a * b * c
  rel_error = torch.abs(analytic_vol_ellipsoid - volume) / analytic_vol_ellipsoid

  assert np.isclose(rel_error, 0, atol=1e-3)

def test_projected_rotated_circle():
  a, b = 1, 1
  step_x, step_y = 0.01, 0.01
  y_mesh, x_mesh = torch.meshgrid(torch.arange(-a,a,step=step_x).float(),torch.arange(-b,b,step=step_y).float())
  x, y = x_mesh, y_mesh

  ## test identity
  radius_circle=1
  circle_bool = projected_rotated_circle(x, y, radius_circle, rotation=torch.eye(3))
  normalized_area_circle = circle_bool.float().mean()
  area_square = 2*2
  assert np.isclose(normalized_area_circle, math.pi*radius_circle**2/area_square,atol=1e-3) # pi r^2 / 4 with r=1, box is area 4

  # rotate in z plane
  random_angle = np.random.uniform(low=0,high=360)
  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[random_angle,0,0],degrees=True).as_matrix())
  circle_random = projected_rotated_circle(x, y, radius_circle, rotation=rotation)
  circle_identity = projected_rotated_circle(x, y, radius_circle, rotation=torch.eye(3))
  assert np.allclose(circle_random,circle_identity)

  # test case of 45 deg which should shrink area to known ratio
  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[0,0,45],degrees=True).as_matrix())
  ellipse = projected_rotated_circle(x, y, radius_circle, rotation=rotation)
  area_circle = circle_identity.sum()
  area_ellipse = ellipse.sum()
  axis_contraction = math.sqrt(2)
  assert np.isclose(area_circle / area_ellipse, axis_contraction, atol=1e-3) # pi r^2 / (pi a b), with a = r/sqrt(2), b=r

  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[0,45,0],degrees=True).as_matrix())
  ellipse = projected_rotated_circle(x, y, radius_circle=1, rotation=rotation)
  area_ellipse = ellipse.sum()
  axis_contraction = math.sqrt(2)
  assert np.isclose(area_circle / area_ellipse, axis_contraction, atol=1e-3) # pi r^2 / (pi a b), with a = r/sqrt(2), b=r

  ## test direction
  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[0,0,45],degrees=True).as_matrix())
  ellipse = projected_rotated_circle(x, y, radius_circle=1, rotation=rotation)
  assert ellipse.sum(0).max() < ellipse.sum(1).max()

  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[0,45,0],degrees=True).as_matrix())
  ellipse = projected_rotated_circle(x, y, radius_circle=1, rotation=rotation)
  assert ellipse.sum(0).max() > ellipse.sum(1).max()

  ## test 90 deg
  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[0,90,0],degrees=True).as_matrix())
  ellipse = projected_rotated_circle(x, y, radius_circle=1, rotation=rotation)
  assert np.isclose(ellipse.sum(),0)

  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[0,0,90],degrees=True).as_matrix())
  ellipse = projected_rotated_circle(x, y, radius_circle=1, rotation=rotation)
  assert np.isclose(ellipse.sum(),0)

  rotation = torch.from_numpy(Rotation.from_euler('ZYX',[0,90,90],degrees=True).as_matrix())
  ellipse = projected_rotated_circle(x, y, radius_circle=1, rotation=rotation)
  assert np.isclose(ellipse.sum(),0)


def test_project_rotated_cylinder():
  n = 256
  arr_1d = torch.arange(-n // 2, n // 2, 1).float()
  y, x = torch.meshgrid(arr_1d, arr_1d)

  # same direction, h ratio
  rotation = torch.from_numpy(Rotation.from_euler('XYZ', [45, 45, 0], degrees=True).as_matrix())
  h_short = n // 4
  h_long = n // 2
  proj_cylinder_short = project_rotated_cylinder(x, y, radius_circle=32, h=h_short, rotation=rotation)
  proj_cylinder_long = project_rotated_cylinder(x, y, radius_circle=32, h=h_long, rotation=rotation)
  assert np.isclose(proj_cylinder_long.sum() / proj_cylinder_short.sum(), h_long / h_short, atol=1e-1)

  small = 1e-3
  overlap = torch.logical_and(proj_cylinder_short > small, proj_cylinder_short > small)
  assert np.allclose(overlap, proj_cylinder_short > small)

  # same direction radius_circle ratio
  rotation = torch.from_numpy(Rotation.from_euler('XYZ', [45, 45, 0], degrees=True).as_matrix())
  rad_fat = n // 8
  rad_thin = rad_fat // 2
  h = n // 2
  proj_cylinder_thin = project_rotated_cylinder(x, y, radius_circle=rad_thin, h=h, rotation=rotation)
  proj_cylinder_fat = project_rotated_cylinder(x, y, radius_circle=rad_fat, h=h, rotation=rotation)
  assert np.isclose(proj_cylinder_fat.sum() / proj_cylinder_thin.sum(), (rad_fat / rad_thin) ** 2, atol=1e-1)

  overlap = np.logical_and(proj_cylinder_thin > small, proj_cylinder_fat > small)
  assert np.allclose(overlap, proj_cylinder_thin > small)

  # identity, circle of area h*pi*r^2
  rotation_z_random = torch.from_numpy(Rotation.from_euler('XYZ', [0, 0, 45], degrees=True).as_matrix())
  proj_cylinder_z_random = project_rotated_cylinder(x, y, radius_circle=32, h=1, rotation=rotation_z_random)
  proj_cylinder_id = project_rotated_cylinder(x, y, radius_circle=32, h=1, rotation=torch.eye(3))
  assert np.allclose(proj_cylinder_z_random, proj_cylinder_id)

  # rectangle
  for p90, m90, axis in [[[0, 90, 0], [0, -90, 0], 1],
                         [[90, 0, 0], [-90, 0, 0], 0],
                         ]:
    rotation = torch.from_numpy(Rotation.from_euler('XYZ', p90, degrees=True).as_matrix())
    h = 128
    radius_circle = 32
    proj_cylinder_side_p90 = project_rotated_cylinder(x, y, radius_circle=radius_circle, h=h, rotation=rotation)
    rotation = torch.from_numpy(Rotation.from_euler('XYZ', m90, degrees=True).as_matrix())
    proj_cylinder_side_m90 = project_rotated_cylinder(x, y, radius_circle=radius_circle, h=h, rotation=rotation)
    assert np.allclose(proj_cylinder_side_p90, proj_cylinder_side_m90)
    assert np.isclose(proj_cylinder_side_p90.sum(axis).max(), h)
    diameter = 2 * radius_circle
    area_rectangle = h * diameter
    assert np.isclose((proj_cylinder_side_p90 > small).sum(), area_rectangle, atol=h)

  for p, m, convention in [
    [[0, 90, 45], [0, 90, -45], 'XZY'],
    [[0, 90, 45], [0, 90, -45], 'YZX'],
  ]:
    rotation_p = torch.from_numpy(Rotation.from_euler(convention, p, degrees=True).as_matrix())
    rotation_m = torch.from_numpy(Rotation.from_euler(convention, m, degrees=True).as_matrix())

    h = 128
    radius_circle = 32
    proj_cylinder_p = project_rotated_cylinder(x, y, radius_circle=radius_circle, h=h, rotation=rotation_p)
    proj_cylinder_m = project_rotated_cylinder(x, y, radius_circle=radius_circle, h=h, rotation=rotation_m)
    assert np.allclose(proj_cylinder_p, proj_cylinder_m)

def test_two_phase_micelle():
  scale = 32
  a = 8 * scale
  b = 8 * scale
  c = 2 * scale
  radius_circle = 3 * scale
  step_x = step_y = 1
  max_axis = max(a, b)
  y_mesh, x_mesh = torch.meshgrid(torch.arange(-max_axis, max_axis, step=step_x), torch.arange(-max_axis, max_axis, step=step_y))

  # same volume
  np.random.seed(1)
  rotations = torch.from_numpy(Rotation.random(num=2).as_matrix())

  micelle_1 = two_phase_micelle(x_mesh, y_mesh,
                                a=a,
                                b=b,
                                c=c,
                                rotation=rotations[0],
                                radius_circle=radius_circle,
                                inner_shell_ratio=0.9,
                                shell_density_ratio=100)

  micelle_2 = two_phase_micelle(x_mesh, y_mesh,
                                a=a,
                                b=b,
                                c=c,
                                rotation=rotations[1],
                                radius_circle=radius_circle,
                                inner_shell_ratio=0.9,
                                shell_density_ratio=100)

  assert np.isclose(0, (micelle_1 - micelle_2).sum() / (micelle_2 + micelle_2).sum(), atol=1e-4)

  # volume scales with cube of units
  micelle_vol = []
  scales = [16, 32]
  for idx, scale in enumerate(scales):
    a = 8 * scale
    b = 8 * scale
    c = 2 * scale
    radius_circle = 3 * scale
    step_x = step_y = 1
    max_axis = max(a, b)
    y_mesh, x_mesh = torch.meshgrid(torch.arange(-max_axis, max_axis, step=step_x),
                                 torch.arange(-max_axis, max_axis, step=step_y))

    micelle = two_phase_micelle(x_mesh, y_mesh,
                                a=a,
                                b=b,
                                c=c,
                                rotation=rotations[idx],
                                radius_circle=radius_circle,
                                inner_shell_ratio=0.9,
                                shell_density_ratio=100)

    micelle_vol.append(micelle.sum())

  volume_factor = (scales[0] / scales[1]) ** 3
  assert np.isclose(micelle_vol[0] / micelle_vol[1], volume_factor, atol=1e-3)
