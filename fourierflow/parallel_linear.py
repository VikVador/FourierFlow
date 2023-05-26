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
import copy
import math
import logging
import torch.nn as nn

from torch.nn.utils             import weight_norm
from torch.nn.utils.weight_norm import WeightNorm

# Used for debugging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------
#
#                                               Architectures
#
# ----------------------------------------------------------------------------------------------------------
class GehringLinear(nn.Linear):
    """
    Documentation
    -------------
    A linear layer with Gehring initialization and weight normalization.
    """
    def __init__(self, in_features,
                      out_features,
                       dropout = 0,
                       bias = True,
                weight_norm = True):

        self.dropout     = dropout
        self.weight_norm = weight_norm

        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """
        Documentation
        -------------
        1 - One problem with initialization from the uniform distribution is that
            the distribution of the outputs has a variance that grows with the
            number of inputs. It turns out that we can normalize the variance of
            each neuron’s output to 1 by scaling its weight vector by the square
            root of its fan-in (i.e. its number of inputs). Dropout further
            increases the variance of each input, so we need to scale down std.

                            See A.3. in Gehring et al (2017):

                            https://arxiv.org/pdf/1705.03122

        2 - Weight normalization is a reparameterization that decouples the
            magnitude of a weight tensor from its direction.

                              See Salimans and Kingma (2016):

                              https://arxiv.org/abs/1602.07868.
        """
        # Compute standard deviation of objective distribution for the weights
        std = math.sqrt((1 - self.dropout) / self.in_features)

        # Normalization
        self.weight.data.normal_(mean = 0, std = std)

        if self.bias is not None:
            self.bias.data.fill_(0)

        if self.weight_norm:
            nn.utils.weight_norm(self)


class WNLinear(nn.Linear):
    """
    Documentation
    -------------
    Normalized weights linear layer !
    """
    def __init__(self, in_features: int,
                      out_features: int,
                      bias: bool = True,
                          device = None,
                           dtype = None,
                           wnorm = False):

        super().__init__(in_features = in_features,
                        out_features = out_features,
                                bias = bias,
                              device = device,
                               dtype = dtype)

        # Normalization of the weights
        if wnorm:
            weight_norm(self)

        # Fixing the deep copy problem
        self._fix_weight_norm_deepcopy()

    def _fix_weight_norm_deepcopy(self):
        """
        Documentation
        -------------
        Fix bug where deepcopy doesn't work with weightnorm. Taken from:

          https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348

        """

        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):

            # Save and delete all weightnorm weights on self
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)

            # Remove this deepcopy method, restoring the object's original one if necessary
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__

            # Actually do the copy
            result = copy.deepcopy(self)

            # Restore weights and method on self
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result

        # bind __deepcopy__ to the weightnorm'd layer
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)
