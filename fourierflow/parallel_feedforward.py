# -----------------
#     Librairies
# -----------------
#
# --------- Standard ---------
import torch.nn as nn

# --------- Custom ---------
from .parallel_linear import WNLinear

# ----------------------------------------------------------------------------------------------------------
#
#                                               Architecture
#
# ----------------------------------------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    Documentation
    -------------
    This is a custom multiple linear normalized layer architecture which
    has been slightly revisited to work with nn.DataParallel
    """
    def __init__(self, dim,
                    factor,
            ff_weight_norm,
                  n_layers,
                layer_norm,
                   dropout):
        super().__init__()

        # Saving the total number of layers
        self.nb_layers = n_layers

        # Initialization of the layers
        for i in range(n_layers):

            # Retreiving dimensions
            in_dim  = dim if i == 0            else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor

            # Creation of the layer
            l = nn.Sequential(
                    WNLinear(in_dim, out_dim, wnorm = ff_weight_norm),
                    nn.Dropout(dropout),
                    nn.ReLU(inplace = True) if i < n_layers - 1 else nn.Identity(),
                    nn.LayerNorm(out_dim) if layer_norm and i == n_layers - 1 else nn.Identity(),
            )

            # Adding the layer to the module
            setattr(self, f'layer_{i}', l)

    def forward(self, x):

        for l in range(self.nb_layers):

            # Retreiving the current layer
            layer = getattr(self, f'layer_{l}')

            # Forward pass
            x = layer(x)

        return x
