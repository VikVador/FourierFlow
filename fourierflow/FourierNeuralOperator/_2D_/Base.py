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

# -----------------------------------------------------
#                      Functions
# -----------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels,
                      out_channels,
                            modes1,
                            modes2):
        super(SpectralConv2d, self).__init__()
        """
        Documentation
        -------------
        - in_dim    : input dimension
        - out_dim   : output dimension
        - modes_x   : modes to keep along the x-direction
        - modes_y   : modes to keep along the y-direction
        """
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1       # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2       = modes2
        self.scale        = (1 / (in_channels * out_channels))
        self.weights1     = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cfloat))
        self.weights2     = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :,  :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :,  :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x

class FNO(nn.Module):
    def __init__(self, input_channels, output_channels, modes1, modes2, width, n_layers = 4):
        super().__init__()
        """
        Documentation
        -------------
        The network is composed of:
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u). W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input        : The solution of the coefficient function and locations (a(x, y), x, y)
        input shape  : (batchsize, x = s, y = s, c = 3)
        output       : The solution
        output shape : (batchsize, x = s, y = s, c = 1)
        """
        self.modes1   = modes1
        self.modes2   = modes2
        self.width    = width
        self.padding  = 18                                         # Pad the domain if input is non-periodic
        self.fc0      = nn.Linear(input_channels + 2, self.width)  # input channel is 3: (a(x, y), x, y)
        self.n_layers = n_layers
        self.convs    = nn.ModuleList([])
        self.ws       = nn.ModuleList([])

        # Initiazation of the layers
        for _ in range(n_layers):
            conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            w    = nn.Conv2d(self.width, self.width, 1)
            self.convs.append(conv)
            self.ws.append(w)

        # Projection to output space
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_channels)

    def forward(self, x):

        # Initialization
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim = -1)

        # Projection to bigger latent space
        x = self.fc0(x)

        # Adjusting dimensions (1)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # Fourier Neural Blocs
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # Adjusting dimensions (2)
        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

         # Projection to output space
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
