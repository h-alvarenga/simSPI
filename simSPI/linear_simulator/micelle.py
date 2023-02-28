"""Module to add in the projection of a micelle."""
import torch
from scipy.spatial.transform import Rotation
import numpy as np
from simSPI.geometric_micelle import two_phase_micelle


class Micelle(torch.nn.Module):
  """Class to corrupt the projection with noise.

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

  def forward(self, proj, rot_params, micelle_params):
    """Add noise to projections.

    Currently, only supports ellipsoid micelles with a cylindrical cavity.

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
      micelle = micelle / micelle.max()
      proj = proj + self.micelle_scale_factor*micelle

    return proj
