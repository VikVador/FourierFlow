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
import torch.nn as nn

from einops        import rearrange

# --------- Own ---------
from ..feedforward import FeedForward
from ..linear      import WNLinear

# -----------------------------------------------------
#                      Functions
# -----------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(
                    in_dim, out_dim, n_modes, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        x_ft = torch.fft.rfft2(x, s=(M, N), norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ft.new_zeros(B, I, N, M // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :self.n_modes] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :self.n_modes, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))

            out_ft[:, :, -self.n_modes:, :self.n_modes] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -self.n_modes:, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            raise

        x = torch.fft.irfft2(out_ft, s=(M, N), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class FNOPlus2DBlock(nn.Module):
    def __init__(self, modes, width, input_dim=12, dropout=0.0, in_dropout=0.0,
                 n_layers=4, share_weight: bool = False,
                 share_fork=False, factor=2,
                 ff_weight_norm=False, n_ff_layers=2,
                 gain=1, layer_norm=False, use_fork=False, mode='full'):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork

        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = FeedForward(
                    width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = FeedForward(
                width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(width, width, modes, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                       out_dim=width,
                                                       n_modes=modes,
                                                       forecast_ff=self.forecast_ff,
                                                       backcast_ff=self.backcast_ff,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       use_fork=use_fork,
                                                       dropout=dropout,
                                                       mode=mode))

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, 1, wnorm=ff_weight_norm))

    def forward(self, x, **kwargs):
        # x.shape == [n_batches, *dim_sizes, input_size]
        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        forecast_list = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, f = layer(x)

            if self.use_fork:
                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            x = x + b

        if not self.use_fork:
            forecast = self.out(b)

        return {
            'forecast': forecast,
            'forecast_list': forecast_list,
        }
