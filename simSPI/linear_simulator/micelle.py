"""Module to add in the projection of a micelle."""
import torch
from torch import tensor

from simSPI.geometric_micelle import two_phase_micelle
from simSPI.linear_simulator.shift_utils import Shift
from compSPI.transforms import primal_to_fourier_2D

class Micelle(torch.nn.Module):
  """Class to generate micelle in Fourier domain.

  Generates micelle in Fourier domain, which has been rotated.
  Generalized to account for micelle not centred:
    generates centred using simSPI.geometric_micelle
    and then shifts this numerically with FFTs by rotated displacement to new origin


  Written by Geoffrey Woollard

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
    self.micelle_translation = tensor([config.micelle_translation_x, config.micelle_translation_y, config.micelle_translation_z])
    self.micelle_shift = Shift(config)


  def get_micelle(self, rot_params, micelle_params):
    assert isinstance(micelle_params,dict)
    step_x = step_y = 1
    box_size = self.micelle_box_size
    assert max(self.micelle_axis_a,self.micelle_axis_b,self.micelle_axis_c) <= box_size // 2, f'use larger box_size={box_size} for micelle params'
    y_mesh, x_mesh = torch.meshgrid(torch.arange(-box_size//2, box_size//2, step=step_x),
                                 torch.arange(-box_size//2, box_size//2, step=step_y))

    rotations = rot_params["rotmat"]
    batch_size = len(rot_params["rotmat"])

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

    :param micelle_f:
    :param rot_params:
    :return:

    Notes
    -----
    Large shifts can change (shifting and flipping) scale because of a numerical instability (torch.exp) in Shift.phase_shift for large translation.
      Consider promoting to double
    '''

    rotations = rot_params["rotmat"]
    translations = rotations@self.micelle_translation

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
    arr_4d_posivereals = arr_4d - min_batch[..., None, None]
    print(min_batch)

    arr_4d_sum = arr_4d_posivereals.reshape(len(arr_4d), 1, -1).sum(dim=-1)
    arr_4d_sum1 = arr_4d_posivereals / arr_4d_sum[..., None, None]
    print(arr_4d_sum)

    return arr_4d_sum1


  def forward(self, rot_params, micelle_params):
    """Generate micelle.

    Currently, only supports ellipsoid micelles with a cylindrical cavity.

    Parameters
    ----------
    rot_params: dict
    micelle_params: AttrDict

    Returns
    -------
    out: torch.Tensor
        noisy projection of shape (batch_size,1,side_len,side_len)

    """
    if micelle_params is not None:
      micelle = self.get_micelle(rot_params, micelle_params)
      micelle_sum1 = self.batch_scale_norm(micelle)
      micelle_scaled = self.micelle_scale_factor*micelle_sum1
      micelle_scaled_f = primal_to_fourier_2D(micelle_scaled)
      micelle_scaled_shifted_f = self.shift_micelle_given_rotation_translation(micelle_scaled_f,rot_params)

    return micelle_scaled_shifted_f
