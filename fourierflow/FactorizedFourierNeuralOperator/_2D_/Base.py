#------------------------------------------------------------------------------
#
#           Ocean subgrid parameterization using machine learning
#
#                             Graduation work
#
#------------------------------------------------------------------------------
# @ Victor Mangeleer
#
# -----------------
#       Notes
# -----------------
# The authors of this remastered code are :
#
#                    Zongyi Li and Daniel Zhengyu Huang
#
# This code is based on the papers :
#
#                      Factorized Fourier Neural Operators
#                       (https://arxiv.org/abs/2111.13802)
#
#        Fourier Neural Operator for Parametric Partial Differential Equations
#                       (https://arxiv.org/abs/2010.08895)
#
# and comes from:
#
#                   https://github.com/alasdairtran/fourierflow
#
#                 https://github.com/neuraloperator/neuraloperator
#
# -----------------
#     Librairies
# -----------------
#
# --------- Standard ---------
import torch

import numpy               as np
import torch.nn            as nn
import torch.nn.functional as F

from einops import rearrange

# --------- Own ---------
from ...feedforward import FeedForward
from ...linear      import WNLinear

# -----------------------------------------------------
#                    SPECTRAL CONVOLUTION
# -----------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self,  in_dim,
                       out_dim,
                       modes_x,
                       modes_y,
                 mode = "full",
               n_ff_layers = 2,
                    factor = 4,
                 dropout = 0.1,
         ff_weight_norm = True,
              use_fork = False,
           layer_norm  = False,
         fourier_weight = None,
            forecast_ff = None,
            backcast_ff = None):
        super().__init__()
        """
        Documentation
        -------------
        - in_dim         : input dimension
        - out_dim        : output dimension
        - modes_x        : modes to keep along the x-direction
        - modes_y        : modes to keep along the y-direction
        - fourier_weight : set of weights shared across layers
        - mode           :
                            - "full"       : FFT + WEIGHTED MODE SELECTION + IFFT + FEEDFORWARD
                            - "low-pass"   : FFT + LOW-PASS FILTERING      + IFFT + FEEDFORWARD
                            - "no-fourier" : FeedForward

        --- FEEDFORWARD ---
        - n_ff_layers    : number of layers
        - factor         : size amplification factor
        - dropout        : probability of not using a neuron
        - ff_weight_norm : normalization of the weights
        - backcast_ff    : neural network used
        - layer_norm     : add a layer normalization layer after the activation functions

        --- FORECASTING ---
        - use_fork    : add an additional head after IFFT
        - forecast_ff : neural network used
        """
        # Security
        assert mode in ['no-fourier', 'full', 'low-pass'], \
            "(FFNO2D - Spectral convolution) mode should be one of use-fourier or no-fourier"

        # Initialization
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.modes_x        = modes_x
        self.modes_y        = modes_y
        self.mode           = mode
        self.use_fork       = use_fork
        self.fourier_weight = fourier_weight

        # Shared set of weights
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])

            # Initialization using Xavier Normal technique
            for n_modes in [modes_x, modes_y]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param  = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        # Additionnal network at the head of the FFNO block
        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # Head of the FFNO to sum x and y after IFFT
        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        """
        Documentation
        -------------
        - Forward pass of the whole FFNO block (Fourier + FeedForward)
        - x.shape == [batch_size, grid_size, grid_size, in_dim]
        """
        # Going through Fourier Domain
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        # Concatenation of x and y after IFFT
        b = self.backcast_ff(x)

        # Additionnal separated head after coming back from Fourier Space
        f = self.forecast_ff(x) if self.use_fork else None

        return b, f

    def forward_fourier(self, x):
        """
        Documentation
        -------------
        - Forward pass in the Fourier Domain (first part of FNO block)
        - x.shape == [batch_size, grid_size, grid_size, in_dim]
        """
        # Shaping (1) - x.shape == [batch_size, in_dim, grid_size, grid_size]
        x = rearrange(x, 'b m n i -> b i m n')

        # Retreiving dimensions
        B, I, M, N = x.shape

        # # # --------- Dimesion Y --------- # # #
        x_fty = torch.fft.rfft(x, dim = -1, norm = 'ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        # CASE 1 - Einstein Sumation (Weighted sum of modes)
        if self.mode == 'full':
            out_ft[:, :, :, :self.modes_y] = torch.einsum("bixy,ioy->boxy",
                x_fty[:, :, :, :self.modes_y], torch.view_as_complex(self.fourier_weight[1]))

        # CASE 2 - Low-Pass filtering, i.e. keeping modes < k_max
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.modes_y] = x_fty[:, :, :, :self.modes_y]

        # Coming back to original space
        xy = torch.fft.irfft(out_ft, n = N, dim = -1, norm = 'ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # --------- Dimesion X --------- # # #
        x_ftx = torch.fft.rfft(x, dim = -2, norm = 'ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        # CASE 1 - Einstein Sumation (Weighted sum of modes)
        if self.mode == 'full':
            out_ft[:, :, :self.modes_x, :] = torch.einsum("bixy,iox->boxy",
                x_ftx[:, :, :self.modes_x, :],
                torch.view_as_complex(self.fourier_weight[0]))

        # CASE 2 - Low-Pass filtering, i.e. keeping modes < k_max
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.modes_x, :] = x_ftx[:, :, :self.modes_x, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy
        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

# -----------------------------------------------------
#            Factorized Fourier Neural Operator
# -----------------------------------------------------
class FFNO(nn.Module):
    def __init__(self, input_dim,
                      output_dim,
                         modes_x,
                         modes_y,
                           width,
                    n_layers = 4,
                      factor = 4,
                 n_ff_layers = 2,
             share_weight = True,
           ff_weight_norm = True,
             layer_norm  = False):
        super().__init__()
        """
        Documentation
        -------------
        - in_dim         : input dimension
        - out_dim        : output dimension
        - modes_x        : modes to keep along the x-direction
        - modes_y        : modes to keep along the y-direction
        - width          : space size to which lift the input
        - n_layers       : number of convolution inside FFNO
        - fourier_weight : set of weights shared across layers
        - share_weight   : determine wether or not weights are shared across layers

        --- FEEDFORWARD ---
        - n_ff_layers    : number of layers
        - factor         : size amplification factor
        - ff_weight_norm : normalization of the weights
        - layer_norm     : add a layer normalization layer after the activation functions
        """
        # Initialization
        self.padding   = 8
        self.modes_x   = modes_x
        self.modes_y   = modes_y
        self.width     = width
        self.input_dim = input_dim
        self.n_layers  = n_layers

        # Lifting layer
        self.in_proj = WNLinear(input_dim + 2, self.width, wnorm = ff_weight_norm)

        # Shared set of weights
        self.fourier_weight = None

        if share_weight:
            self.fourier_weight = nn.ParameterList([])

            # Initialization using Xavier Normal technique
            for n_modes in [modes_x, modes_y]:
                weight = torch.FloatTensor(width, width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        # Stores spectral layers of FFNO
        self.spectral_layers = nn.ModuleList([])

        # Initialization of the layers
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim         = width,
                                                       out_dim        = width,
                                                       modes_x        = modes_x,
                                                       modes_y        = modes_y,
                                                       forecast_ff    = None,
                                                       backcast_ff    = None,
                                                       fourier_weight = self.fourier_weight,
                                                       factor         = factor,
                                                       ff_weight_norm = ff_weight_norm,
                                                       n_ff_layers    = n_ff_layers,
                                                       layer_norm     = layer_norm,
                                                       use_fork       = False,
                                                       dropout        = 0.1,
                                                       mode           = 'full'))

        # Projection layer
        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm = ff_weight_norm),
            WNLinear(128, output_dim, wnorm = ff_weight_norm))

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim = -1)               # [B, X, Y, 4]
        x = self.in_proj(x)                              # [B, X, Y, H]
        x = x.permute(0, 3, 1, 2)                        # [B, H, X, Y]
        x = F.pad(x, [0, self.padding, 0, self.padding]) # [B, H, X, Y]
        x = x.permute(0, 2, 3, 1)                        # [B, X, Y, H]
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, _ = layer(x)
            x = x + b
        b = b[..., :-self.padding, :-self.padding, :]
        output = self.out(b)
        return output

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# -----------------------------------------------------
#                        Testing
# -----------------------------------------------------
if __name__ == '__main__':

    # Input vector
    x = torch.zeros([64, 3, 64, 64])

    # ------- Spectral Convolution -------
    layer = SpectralConv2d(in_dim = x.shape[1], out_dim = 1, modes_x = 12, modes_y = 12)
