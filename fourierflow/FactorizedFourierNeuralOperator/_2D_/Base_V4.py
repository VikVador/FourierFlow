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
from ...parallel_feedforward import FeedForward
from ...parallel_linear      import WNLinear

# -----------------------------------------------------
#                  Spectral Convolution
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
            ff_weight_norm  = True,
                  use_fork  = False,
               layer_norm   = False,
           fourier_weight_x = None,
           fourier_weight_y = None,
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
        - fourier_weight : set of weights shared across layers (x and y directions)
        - mode           :
                            - "full"       : FFT + WEIGHTED MODE SELECTION + IFFT + FEEDFORWARD
                            - "filtering"  : FFT + FILTERING + IFFT + FEEDFORWARD
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
        assert mode in ['no-fourier', 'full', 'filtering'], \
            "(FFNO2D - Spectral convolution) mode should be one of use-fourier or no-fourier"

        assert modes_x[1] - modes_x[0] > 0, \
            "(FFNO2D - Spectral convolution) modes in x-direction should be m1 < m2"

        assert modes_y[1] - modes_y[0] > 0, \
            "(FFNO2D - Spectral convolution) modes in x-direction should be m1 < m2"

        assert in_dim - out_dim == 0, \
            "(FFNO2D - Spectral convolution) input and output dimension must be equal"

        # Initialization
        self.in_dim           = in_dim
        self.out_dim          = out_dim
        self.modes_x          = modes_x
        self.modes_y          = modes_y
        self.mode             = mode
        self.use_fork         = use_fork
        self.fourier_weight_x = fourier_weight_x
        self.fourier_weight_y = fourier_weight_y

        # Shared set of weights
        if self.fourier_weight_x is None:

            # Initialization using Xavier Normal technique
            for i, n_modes in enumerate([modes_x[1] - modes_x[0], modes_y[1] - modes_y[0]]):

                # Performing normalization
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param  = nn.Parameter(weight)
                nn.init.xavier_normal_(param)

                # Saving weights
                if i == 0:
                    self.fourier_weight_x = param
                else:
                    self.fourier_weight_y = param

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
        - Forward pass of the whole FFNO block
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
        - Forward pass in the Fourier domain
        """
        # Re-shaping (1)
        x = rearrange(x, 'b m n i -> b i m n')

        # Retreiving dimensions
        B, I, M, N = x.shape

        # ----------------------------------------
        #                Dimension Y
        # ----------------------------------------
        # Computing Fourier Transform (y-direction)
        f_transform = torch.fft.rfft(x, dim = -1, norm = 'ortho')

        # Stores the results after playing with modes
        out_ft = f_transform.new_zeros(B, I, M, N // 2 + 1)

        # ------------- CASE 1 - Weightening -------------
        if self.mode == 'full':
            out_ft[:, :, :, self.modes_y[0]:self.modes_y[1]] = torch.einsum("b i x y, i o y -> b o x y",
                f_transform[:, :, :, self.modes_y[0]:self.modes_y[1]], torch.view_as_complex(self.fourier_weight_y))

        # ------------- CASE 2 - Low Pass Filtering -------------
        elif self.mode == 'filtering':
            out_ft[:, :, :, self.modes_y[0]:self.modes_y[1]] = f_transform[:, :, :, self.modes_y[0]:self.modes_y[1]]

        # Coming back to original space
        xy = torch.fft.irfft(out_ft, n = N, dim = -1, norm = 'ortho')

        # ----------------------------------------
        #                Dimension X
        # ----------------------------------------
         # Computing Fourier Transform (x-direction)
        f_transform = torch.fft.rfft(x, dim = -2, norm = 'ortho')

        # Stores the results after playing with modes
        out_ft = f_transform.new_zeros(B, I, M // 2 + 1, N)

        # ------------- CASE 1 - Weightening -------------
        if self.mode == 'full':
            out_ft[:, :, self.modes_x[0]:self.modes_x[1], :] = torch.einsum("bixy,iox->boxy",
                f_transform[:, :, self.modes_x[0]:self.modes_x[1], :], torch.view_as_complex(self.fourier_weight_x))

        # ------------- CASE 2 - Low Pass Filtering -------------
        elif self.mode == 'filtering':
            out_ft[:, :, self.modes_x[0]:self.modes_x[1], :] = f_transform[:, :, self.modes_x[0]:self.modes_x[1], :]

        # Coming back to original space
        xx = torch.fft.irfft(out_ft, n = M, dim = -2, norm='ortho')

        # ----------------------------------------
        #                Combining
        # ----------------------------------------
        x = xx + xy
        x = rearrange(x, 'b i m n -> b m n i')

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
        Slightly modified FFNO operator now working as a pass-band filter and it can be parralelized !

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

        # Security
        assert len(modes_x) == 2 & len(modes_y) == 2, \
            "(FFNO2D V3 - Module) modes should be a 2D tuple containing the min and max modes defining the pass-band filter"

        # ------ Lifting layer ------
        self.in_proj = WNLinear(input_dim + 2, self.width, wnorm = ff_weight_norm)

        # --------- Weights ---------
        self.fourier_weight_x = None
        self.fourier_weight_y = None

        if share_weight:

            # Initialization using Xavier Normal technique
            for i, n_modes in enumerate([modes_x[1] - modes_x[0], modes_y[1] - modes_y[0]]):

                # Performing normalization
                weight = torch.FloatTensor(width, width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)

                # Saving weights
                if i == 0:
                    self.fourier_weight_x = param
                else:
                    self.fourier_weight_y = param


        # --------- Spectral Convolutions ---------
        for i in range(n_layers):

            # Creation of the layer
            spec_layer = SpectralConv2d(in_dim           = width,
                                        out_dim          = width,
                                        modes_x          = modes_x,
                                        modes_y          = modes_y,
                                        forecast_ff      = None,
                                        backcast_ff      = None,
                                        fourier_weight_x = self.fourier_weight_x,
                                        fourier_weight_y = self.fourier_weight_y,
                                        factor           = factor,
                                        ff_weight_norm   = ff_weight_norm,
                                        n_ff_layers      = n_ff_layers,
                                        layer_norm       = layer_norm,
                                        use_fork         = False,
                                        dropout          = 0.1,
                                        mode             = 'full')

            # Saving spectral layer
            setattr(self, f'ffno_spectral_layer_{i}', spec_layer)

        # --------- Projection Layer ---------
        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm = ff_weight_norm),
            WNLinear(128, output_dim, wnorm = ff_weight_norm))

    def forward(self, x):
        """
        Documentation
        -------------
        Forwarding throughout the whole FFNO block

        """
        # Creation of grid containing x and y coordinates
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim = -1)               # [B, X, Y, 4]

        # Lifting the input
        x = self.in_proj(x)                              # [B, X, Y, H]
        x = x.permute(0, 3, 1, 2)                        # [B, H, X, Y]
        x = F.pad(x, [0, self.padding, 0, self.padding]) # [B, H, X, Y]
        x = x.permute(0, 2, 3, 1)                        # [B, X, Y, H]

        # Performing Spectral Convolution
        for i in range(self.n_layers):

            # Retreiving the layer
            layer = getattr(self, f'ffno_spectral_layer_{i}')

            # Convolution
            b, _ = layer(x)

            # Skip connection
            x = x + b

        # Removing useless dimensions
        b = b[..., :-self.padding, :-self.padding, :]

        # Projection
        output = self.out(b)

        return output

    def get_grid(self, shape, device):
        """
        Documentation
        -------------
        Creation of a grid containing coordinates of cells (used by Fourier Transform)
        """
        # Retreiving dimensions
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]

        # X-direction
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype = torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])

        # Y-direction
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype = torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

        return torch.cat((gridx, gridy), dim = -1).to(device)