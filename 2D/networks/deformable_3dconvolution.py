"""Deformable 3D convolution
Adapted from https://github.com/oeway/pytorch-deform-conv/
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def flatten(a):
    return a.contiguous().view(a.nelement())


def repeat(a, repeats, axis=0):
    assert len(a.size()) == 1
    return flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def batch_map_coordinates(input, coords, order=1):
    """Interpolate (trilinear) input values given coordinates
    Notations:
        l - left
        r - right
        t - top
        b - bottom
        a - anterior (front)
        p - posterior (back)
            
    ltp------rtp
    |\        |\
    | lta------rta
    | |       | |
    lbp------rbp|
     \|        \|
      lba------rba
    
    """

    batch_size = input.size(0)
    input_depth = input.size(1)
    input_height = input.size(2)
    input_width = input.size(3)

    n_coords = coords.size(1)
    
    coords = torch.cat((
        torch.clamp(coords.narrow(2, 0, 1), 0, input_depth - 1),
        torch.clamp(coords.narrow(2, 1, 1), 0, input_height - 1), 
        torch.clamp(coords.narrow(2, 2, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)
    
    coords_lta = coords.floor().long()
    coords_rbp = coords.ceil().long()
    
    coords_ltp = torch.stack([coords_lta[..., 0], coords_lta[..., 1], coords_rbp[..., 2]], 2)
    coords_rtp = torch.stack([coords_rbp[..., 0], coords_lta[..., 1], coords_rbp[..., 2]], 2)
    coords_rta = torch.stack([coords_rbp[..., 0], coords_lta[..., 1], coords_lta[..., 2]], 2)
    coords_lba = torch.stack([coords_lta[..., 0], coords_rbp[..., 1], coords_lta[..., 2]], 2)
    coords_lbp = torch.stack([coords_lta[..., 0], coords_rbp[..., 1], coords_rbp[..., 2]], 2)
    coords_rba = torch.stack([coords_rbp[..., 0], coords_rbp[..., 1], coords_lta[..., 2]], 2)
    
    idx = repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, flatten(coords[..., 0]), flatten(coords[..., 1]), flatten(coords[..., 2])
        ], 1)
        inds = indices[:, 0]*input.size(1)*input.size(2)*input.size(3) \
        + indices[:, 1]*input.size(2)*input.size(3) + indices[:, 2]*input.size(3) + indices[:, 3]
        
        vals = flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals
    
    vals_lta = _get_vals_by_coords(input, coords_lta.detach())
    vals_rbp = _get_vals_by_coords(input, coords_rbp.detach())
    vals_ltp = _get_vals_by_coords(input, coords_ltp.detach())
    vals_rtp = _get_vals_by_coords(input, coords_rtp.detach())
    vals_rta = _get_vals_by_coords(input, coords_rta.detach())
    vals_lba = _get_vals_by_coords(input, coords_lba.detach())
    vals_lbp = _get_vals_by_coords(input, coords_lbp.detach())
    vals_rba = _get_vals_by_coords(input, coords_rba.detach())

    # trilinear interpolation
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    coords_offset_lta = coords - coords_lta.type(coords.data.type())
    coords_offset_rbp = coords - coords_rbp.type(coords.data.type())

    vals_ta = coords_offset_lta[..., 0]*(vals_rta - vals_lta) + vals_lta
    vals_ba = coords_offset_lta[..., 0]*(vals_rba - vals_lba) + vals_lba
    
    vals_tp = coords_offset_rbp[..., 0]*(vals_rtp - vals_ltp) + vals_ltp
    vals_bp = coords_offset_rbp[..., 0]*(vals_rbp - vals_lbp) + vals_lbp

    # interpolate top
    vals_t = coords_offset_lta[..., 2]* (vals_tp - vals_ta) + vals_ta
    
    # interpolate bottom
    vals_b = coords_offset_rbp[..., 2]* (vals_bp - vals_ba) + vals_ba
    
    mapped_vals = coords_offset_lta[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


def generate_grid(batch_size, input_depth, input_height, input_width, dtype, cuda):
    """Generate grid for coordinates of the input
    Parameters
    ---------
    batch_size : int
    input_depth : int
    input_height : int
    input_width : int
    dtype : torch.dtype
    cuda : boolean
    
    Returns
    -------
    torch.Tensor. shape = (b, d*h*w, 3)
    """
    grid = np.meshgrid(
        range(input_depth), range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 3)

    grid = np.expand_dims(grid, 0)
    grid = np.tile(grid, [batch_size, 1, 1])
    
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)


def batch_map_offsets(input, offsets, grid=None, order=1):
    """(Batch) map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (b, s, s, s)
    offsets: torch.Tensor. shape = (b, s, s, s, 3)
    
    Returns
    -------
    torch.Tensor. shape = (b, s, s, s)
    """
    batch_size = input.size(0)
    input_depth = input.size(1)
    input_height = input.size(2)
    input_width = input.size(3)

    offsets = offsets.view(batch_size, -1, 3)
    
    if grid is None:
        grid = generate_grid(batch_size, 
                                input_depth, 
                                input_height, 
                                input_width, 
                                offsets.data.type(), 
                                offsets.data.is_cuda)

    coords = offsets + grid

    mapped_vals = batch_map_coordinates(input, coords)
    return mapped_vals


class ConvOffset3D(nn.Conv3d):
    """Convolutional layer responsible for learning the 3D offsets and output the
    deformed feature map using trilinear interpolation. This layer does not perform
    convolution on the deformed map.
    """
    def __init__(self, in_channels, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = in_channels
        self._grid_param = None
        super(ConvOffset3D, self).__init__(self.filters, self.filters * 3, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        x_shape = x.size()
        offsets = super(ConvOffset3D, self).forward(x)

        # offsets: (b*c, d, h, w, 3)
        offsets = self._to_bc_d_h_w_3(offsets, x_shape)

        # x: (b*c, d, h, w)
        x = self._to_bc_d_h_w(x, x_shape)

        # X_offset: (b*c, d, h, w)
        x_offset = batch_map_offsets(x, offsets, grid=self._get_grid(self,x))

        # x_offset: (b, d, h, w, c)
        x_offset = self._to_b_c_d_h_w(x_offset, x_shape)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_depth, input_height, input_width = x.size(0), x.size(1), x.size(2), x.size(3)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_depth, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_depth, input_height, input_width, dtype, cuda)
        self._grid = generate_grid(batch_size, input_depth, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3) * weights.size(4)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_d_h_w_3(x, x_shape):
        """(b, c, d, h, w) -> (b*c, d, h, w, 3)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]), 3)
        return x

    @staticmethod
    def _to_bc_d_h_w(x, x_shape):
        """(b, c, d, h, w) -> (b*c, d, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))
        return x

    @staticmethod
    def _to_b_c_d_h_w(x, x_shape):
        """(b*c, d, h, w) -> (b, c, d, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))
        return x

    
def deform_conv3d(in_c, out_c, kernel_size, **kwargs):
    """Deformable convolution layer: convolution + offset"""
    return nn.Sequential(
        ConvOffset3D(out_c),
        nn.Conv3d(in_c, out_c, kernel_size, **kwargs)
        
    )

    