import numpy as np

def project_rotated_ellipsoid(x,y,a,b,c,r):
  '''

  :param x:
  :param y:
  :param a:
  :param b:
  :param c:
  :param r:
  :return:

  Comments
  --------
  # piece_1 cancels out
  # note that the z0 and z1 terms are very similar, except for a sign, so we can precompute the pieces
  piece_1 = (-Rxx*Rxz*b**2*c**2*x - Rxy*Rxz*b**2*c**2*y - Ryx*Ryz*a**2*c**2*x - Ryy*Ryz*a**2*c**2*y - Rzx*Rzz*a**2*b**2*x - Rzy*Rzz*a**2*b**2*y)
  z0 = (piece_1 - piece_2) / piece_3
  z1 = (piece_1 + piece_2) / piece_3
  z = np.abs(z0 - z1) = 2*piece_2/piece_3
  '''
  Rxx,Rxy,Rxz = r[0]
  Ryx,Ryy,Ryz = r[1]
  Rzx,Rzy,Rzz = r[2]
  piece_2 = a*b*c*np.sqrt(-Rxx**2*Ryz**2*c**2*x**2 - Rxx**2*Rzz**2*b**2*x**2 - 2*Rxx*Rxy*Ryz**2*c**2*x*y - 2*Rxx*Rxy*Rzz**2*b**2*x*y + 2*Rxx*Rxz*Ryx*Ryz*c**2*x**2 + 2*Rxx*Rxz*Ryy*Ryz*c**2*x*y + 2*Rxx*Rxz*Rzx*Rzz*b**2*x**2 + 2*Rxx*Rxz*Rzy*Rzz*b**2*x*y - Rxy**2*Ryz**2*c**2*y**2 - Rxy**2*Rzz**2*b**2*y**2 + 2*Rxy*Rxz*Ryx*Ryz*c**2*x*y + 2*Rxy*Rxz*Ryy*Ryz*c**2*y**2 + 2*Rxy*Rxz*Rzx*Rzz*b**2*x*y + 2*Rxy*Rxz*Rzy*Rzz*b**2*y**2 - Rxz**2*Ryx**2*c**2*x**2 - 2*Rxz**2*Ryx*Ryy*c**2*x*y - Rxz**2*Ryy**2*c**2*y**2 - Rxz**2*Rzx**2*b**2*x**2 - 2*Rxz**2*Rzx*Rzy*b**2*x*y - Rxz**2*Rzy**2*b**2*y**2 + Rxz**2*b**2*c**2 - Ryx**2*Rzz**2*a**2*x**2 - 2*Ryx*Ryy*Rzz**2*a**2*x*y + 2*Ryx*Ryz*Rzx*Rzz*a**2*x**2 + 2*Ryx*Ryz*Rzy*Rzz*a**2*x*y - Ryy**2*Rzz**2*a**2*y**2 + 2*Ryy*Ryz*Rzx*Rzz*a**2*x*y + 2*Ryy*Ryz*Rzy*Rzz*a**2*y**2 - Ryz**2*Rzx**2*a**2*x**2 - 2*Ryz**2*Rzx*Rzy*a**2*x*y - Ryz**2*Rzy**2*a**2*y**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
  piece_3 = (Rxz**2*b**2*c**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
  z_nans = 2*piece_2/piece_3
  proj_ellipsoid = np.nan_to_num(z_nans, 0)
  return proj_ellipsoid