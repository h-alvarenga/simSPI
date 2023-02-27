import numpy as np
from compSPI import transforms
import torch
from torch import tensor

def project_rotated_ellipsoid(x,y,a,b,c,rotation):
  '''

  :param x: numpy ndarray, shape (n_points, n_points). 2d array meshgrid
  :param y: numpy ndarray, shape (n_points, n_points). 2d array meshgrid
  :param a: float, scaling of x
  :param b: float, scaling of y
  :param c: float, scaling of z
  :param rotation: numpy ndarray, shape (3,3). may need to use rotation.T for input, see comments
  :return: numpy ndarray, shape (n_points, n_points)

  Comments
  --------
  From ellipsoid equation (x'/a)^2 + (y'/b)^2  + (z'/c)^2 < 1 , with rotated points R[x y z]^T = [x' y' z'], i.e. [x y z]^T = R^T[x' y' z']^T
  # piece_1 cancels out
  # note that the z0 and z1 terms are very similar, except for a sign, so we can precompute the pieces
  piece_1 = (-Rxx*Rxz*b**2*c**2*x - Rxy*Rxz*b**2*c**2*y - Ryx*Ryz*a**2*c**2*x - Ryy*Ryz*a**2*c**2*y - Rzx*Rzz*a**2*b**2*x - Rzy*Rzz*a**2*b**2*y)
  z0 = (piece_1 - piece_2) / piece_3
  z1 = (piece_1 + piece_2) / piece_3
  z = np.abs(z0 - z1) = 2*piece_2/piece_3
  '''
  Rxx,Rxy,Rxz = rotation[0]
  Ryx,Ryy,Ryz = rotation[1]
  Rzx,Rzy,Rzz = rotation[2]
  piece_2 = a*b*c*np.sqrt(-Rxx**2*Ryz**2*c**2*x**2 - Rxx**2*Rzz**2*b**2*x**2 - 2*Rxx*Rxy*Ryz**2*c**2*x*y - 2*Rxx*Rxy*Rzz**2*b**2*x*y + 2*Rxx*Rxz*Ryx*Ryz*c**2*x**2 + 2*Rxx*Rxz*Ryy*Ryz*c**2*x*y + 2*Rxx*Rxz*Rzx*Rzz*b**2*x**2 + 2*Rxx*Rxz*Rzy*Rzz*b**2*x*y - Rxy**2*Ryz**2*c**2*y**2 - Rxy**2*Rzz**2*b**2*y**2 + 2*Rxy*Rxz*Ryx*Ryz*c**2*x*y + 2*Rxy*Rxz*Ryy*Ryz*c**2*y**2 + 2*Rxy*Rxz*Rzx*Rzz*b**2*x*y + 2*Rxy*Rxz*Rzy*Rzz*b**2*y**2 - Rxz**2*Ryx**2*c**2*x**2 - 2*Rxz**2*Ryx*Ryy*c**2*x*y - Rxz**2*Ryy**2*c**2*y**2 - Rxz**2*Rzx**2*b**2*x**2 - 2*Rxz**2*Rzx*Rzy*b**2*x*y - Rxz**2*Rzy**2*b**2*y**2 + Rxz**2*b**2*c**2 - Ryx**2*Rzz**2*a**2*x**2 - 2*Ryx*Ryy*Rzz**2*a**2*x*y + 2*Ryx*Ryz*Rzx*Rzz*a**2*x**2 + 2*Ryx*Ryz*Rzy*Rzz*a**2*x*y - Ryy**2*Rzz**2*a**2*y**2 + 2*Ryy*Ryz*Rzx*Rzz*a**2*x*y + 2*Ryy*Ryz*Rzy*Rzz*a**2*y**2 - Ryz**2*Rzx**2*a**2*x**2 - 2*Ryz**2*Rzx*Rzy*a**2*x*y - Ryz**2*Rzy**2*a**2*y**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
  piece_3 = (Rxz**2*b**2*c**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
  z_nans = 2*piece_2/piece_3
  proj_ellipsoid = np.nan_to_num(z_nans, 0)
  return proj_ellipsoid

def projected_rotated_circle(x, y, radius_circle, rotation):
  '''
  https://math.stackexchange.com/a/2962856
  TODO: fails around 'ZYX',[45,90,0], dotted lines or single point at origin for for 'ZYX',[_,90,0] and extent longer than radius_circle for 'ZYX',[45,90,0]
    proposed fix: separate case when ellipse is fully on side: it's just a line of length circle diameter somewhere on xy plane.
                  perhaps make line similar to in project_rotated_cylinder
                  or handle case directly in project_rotated_cylinder (will be rectangle), and throw error if encoutered here
  '''
  normal_axis = rotation[:, -1]  # rotation dotted with zaxis (projection axis)
  nx, ny, nz = normal_axis
  ellipse = (nx * nx + nz * nz) * x * x + 2 * nx * ny * x * y + (nz * nz + ny * ny) * y * y - nz * nz * radius_circle * radius_circle < 0
  return ellipse

def project_rotated_cylinder(x, y, radius_circle, h, rotation, n_crop=None):
  '''
  Comments
  --------
  h in units of pixels

  TODO:
    h=0 case still shows thin line
    'ZYX' [45,90,0] fails
    'XZY',[0,1,91] problems in two phase / stripes
    make blurring to anti-alias
  '''
  assert x.shape == y.shape
  n = x.shape[0]
  Rxz, Ryz, Rzz = rotation[:, -1]

  if np.isclose(rotation[-1, -1], 1, atol=1e-4):
    case = 'about z-axis'
    circle = projected_rotated_circle(x, y, radius_circle, rotation=np.eye(3))
    fill_factor = h
    proj_cylinder = fill_factor * circle
    print(case)
    return proj_cylinder

  elif np.isclose(np.abs(Rxz), 1):  # 90 deg, line along y-axis
    case = '90 deg, line along x-axis'
    line = torch.zeros(n, n)
    line[n // 2, :] = 1
    n_border_x = int(np.round(n / 2 - h / 2))
    line[:, :n_border_x] = line[:, -n_border_x:] = 0


  elif np.isclose(np.abs(Ryz), 1):  #
    case = '90 deg, line along y-axis'
    line = torch.zeros(n, n)
    line[:, n // 2] = 1
    n_border_y = int(np.round(n / 2 - h / 2))
    line[:n_border_y, :] = line[-n_border_y:, :] = 0

  elif np.isclose(Ryz, 0) and not np.isclose(Rzz, 1):
    case = '90 deg, line along x-axis with z-tilt'
    line = torch.zeros(n, n)
    line[n // 2, :] = 1
    n_border_x = int(np.round(n / 2 - h / 2 * np.abs(Rxz)))
    line[:, :n_border_x] = line[:, -n_border_x:] = 0

  elif np.isclose(Rxz, 0) and not np.isclose(Rzz, 1):
    case = '90 deg, line along y-axis with z-tilt'
    line = torch.zeros(n, n)
    line[:, n // 2] = 1
    n_border_y = int(np.round(n / 2 - h / 2 * np.abs(Ryz)))
    line[:n_border_y, :] = line[-n_border_y:, :] = 0

  else:
    case = 'else'
    line_test = Rxz * y - Ryz * x  # intercept zero since cylinder centered

    line_width_factor = 2 # sqrt 2 ?
    line_clipped = line_width_factor - np.clip(np.abs(line_test), a_min=0, a_max=line_width_factor)


    n_border_x = int(np.round(n / 2 - h / 2 * np.abs(Rxz)))
    n_border_y = int(np.round(n / 2 - h / 2 * np.abs(Ryz)))

    # fails if n_border 0 or n//2
    line_clipped[:n_border_y, :] = line_clipped[-n_border_y:, :] = line_clipped[:, :n_border_x] = line_clipped[:,
                                                                                                  -n_border_x:] = 0
    line = line_clipped

  line_f = transforms.primal_to_fourier_2D(line)

  ellipse = projected_rotated_circle(x, y, radius_circle, rotation)
  ellipse_f = transforms.primal_to_fourier_2D(ellipse)
  product = ellipse_f * line_f
  if n_crop is not None:
    idx_start, idx_end = n//2-n_crop//2, n//2+n_crop//2
    product = product[idx_start:idx_end,idx_start:idx_end]
  convolve = transforms.fourier_to_primal_2D(product)
  proj_cylinder = convolve.real.numpy()
  print(case)
  return proj_cylinder

def two_phase_micelle(x_mesh, y_mesh, a, b, c, rotation, radius_circle, inner_shell_ratio, shell_density_ratio):
  '''

  :param x_mesh:
  :param y_mesh:
  :param a:
  :param b:
  :param c:
  :param rotation:
  :param radius_circle:
  :param inner_shell_ratio:
  :param shell_density_ratio:
  :return:

  Comments:
  --------
  Note that the rotation input to project_rotated_ellipsoid and project_rotated_cylinder are transpose to each other
  '''
  proj_ellipsoid_outer = project_rotated_ellipsoid(x_mesh, y_mesh, a, b, c, rotation.T)
  proj_ellipsoid_inner = project_rotated_ellipsoid(x_mesh, y_mesh,
                                                                     a * inner_shell_ratio,
                                                                     b * inner_shell_ratio,
                                                                     c * inner_shell_ratio,
                                                                     rotation.T)
  shell = proj_ellipsoid_outer - proj_ellipsoid_inner

  h_outer = c * 2
  volume_outer = np.pi * radius_circle ** 2 * h_outer
  proj_cylinder_outer = project_rotated_cylinder(tensor(x_mesh),
                                                                   tensor(y_mesh),
                                                                   radius_circle=radius_circle,
                                                                   h=h_outer,
                                                                   rotation=rotation)
  proj_cylinder_outer = volume_outer * proj_cylinder_outer / proj_cylinder_outer.sum()

  h_inner = c * inner_shell_ratio * 2
  volume_inner = np.pi * radius_circle ** 2 * h_inner
  proj_cylinder_inner = project_rotated_cylinder(tensor(x_mesh),
                                                                   tensor(y_mesh),
                                                                   radius_circle=radius_circle,
                                                                   h=h_inner,
                                                                   rotation=rotation)
  proj_cylinder_inner = volume_inner * proj_cylinder_inner / proj_cylinder_inner.sum()

  proj_cylinder_shell = proj_cylinder_outer - proj_cylinder_inner
  micelle = shell_density_ratio * shell + proj_ellipsoid_inner - (
            shell_density_ratio * proj_cylinder_shell + proj_cylinder_inner)
  return micelle