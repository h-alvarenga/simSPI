"""Module to add in the projection of a micelle."""
import torch
from simSPI.geometric_micelle import two_phase_micelle
from simSPI.linear_simulator.shift_utils import Shift
from compSPI.transforms import fourier_to_primal_2D, primal_to_fourier_2D

class Micelle(torch.nn.Module):
  """Class to corrupt the projection with noise.

  Written by Geoffrey Woollard

  TODO: think about about how translation of micelle plays with translation of protein and final projection to the measured image.

  Parameters
  ----------
  config: class
      contains parameters of the micelle

  """

  def __init__(self, config):
    super(Micelle, self).__init__()
    self.micelle_scale_factor = config.micelle_scale_factor
    self.micelle_axis_a = config.micelle_axis_a
    self.micelle_axis_b = config.micelle_axis_b
    self.micelle_axis_c = config.micelle_axis_c
    self.micelle_radius_cavity = config.micelle_radius_cavity
    self.micelle_inner_shell_ratio = config.micelle_inner_shell_ratio
    self.micelle_shell_density_ratio = config.micelle_shell_density_ratio
    self.micelle_box_size = config.micelle_box_size
    self.micelle_translation_z = config.micelle_translation_z
    self.micelle_shift = Shift(config)


  def get_micelle(self, rot_params, micelle_params):
    assert isinstance(micelle_params,dict)
    step_x = step_y = 1
    box_size = self.micelle_box_size
    assert max(self.micelle_axis_a,self.micelle_axis_b,self.micelle_axis_c) <= box_size // 2, f'use larger box_size={box_size} for micelle params'
    y_mesh, x_mesh = torch.meshgrid(torch.arange(-box_size//2, box_size//2, step=step_x),
                                 torch.arange(-box_size//2, box_size//2, step=step_y))

    rotations = rot_params["rotmat"]
    # np.random.seed(1)
    batch_size = len(rot_params["rotmat"])
    # rotations = torch.from_numpy(Rotation.random(num=batch_size).as_matrix())

    micelles = torch.empty(batch_size, 1, box_size, box_size)
    for idx in range(batch_size): # TODO: vectorize
      micelles[idx, 0, :, :] = two_phase_micelle(x_mesh, y_mesh,
                                a=self.micelle_axis_a,
                                b=self.micelle_axis_b,
                                c=self.micelle_axis_c,
                                rotation=rotations[idx],
                                radius_circle=self.micelle_radius_cavity,
                                inner_shell_ratio=self.micelle_inner_shell_ratio,
                                shell_density_ratio=self.micelle_shell_density_ratio)
    return micelles

  def shift_micelle_given_rotation_translation(self, micelle_f, rot_params):
    '''

    TODO: incorporate micelle_translation_x, micelle_translation_y.
    :param micelle_f:
    :param rot_params:
    :return:
    '''

    rotations = rot_params["rotmat"]
    translations = rotations[:,[0,1],-1]*self.micelle_translation_z

    shift_params = {'shift_x': translations[:,0],
                    'shift_y': translations[:,1]}
    micelle_shifted_f = self.micelle_shift(micelle_f, shift_params)
    return micelle_shifted_f

  def batch_scale_norm(self, arr_4d):
    '''
    Normalize over dimensions such that on batch dimension is all positive and sums to 1
    TODO: use unsqueeze for reshaping?

    :param arr_4d: shape (batch,1,nx,ny)
    :return: arr_4d_sum1: shape (batch,1,nx,ny)
    '''

    min_batch = arr_4d.reshape(len(arr_4d), 1, -1).min(dim=-1).values
    arr_4d_posiivereals = arr_4d - min_batch[..., None, None]

    arr_4d_sum = arr_4d_posiivereals.reshape(len(arr_4d), 1, -1).sum(dim=-1)
    arr_4d_sum1 = arr_4d_posiivereals / arr_4d_sum[..., None, None]

    return arr_4d_sum1


  def forward(self, proj, rot_params, micelle_params):
    """Add micelle to projections.

    Currently, only supports ellipsoid micelles with a cylindrical cavity.

    TODO: stay in Fourier space

    Parameters
    ----------
    proj: torch.Tensor
        input projection of shape (batch_size,1,side_len,side_len)

    Returns
    -------
    out: torch.Tensor
        noisy projection of shape (batch_size,1,side_len,side_len)
    """
    if micelle_params is not None:
      micelle = self.get_micelle(rot_params, micelle_params)
      micelle_f = primal_to_fourier_2D(micelle)
      micelle_shifted_f = self.shift_micelle_given_rotation_translation(micelle_f,rot_params)
      micelle_shifted = fourier_to_primal_2D(micelle_shifted_f).real
      micelle_shifted_sum1 = self.batch_scale_norm(micelle_shifted)

      proj = proj + self.micelle_scale_factor*micelle_shifted_sum1

    return proj
