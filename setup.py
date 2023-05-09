from setuptools import find_packages, setup

VERSION          = '1.0.0'
DESCRIPTION      = 'Fourier Neural Operators'
LONG_DESCRIPTION = 'Fourier Neural Operators - Contains the Fourier Neural Operator and the Factorized Fourier Neural Operator'

setup(
      name             = "fourierflow",
      version          = VERSION,
      url              = 'https://github.com/alasdairtran/fourierflow',
      author           = 'Alasdair Tran',
      author_email     = 'alasdair.tran@anu.edu.au',
      description      = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      packages         = ["FourierFlow",
                          "fourierflow",
                          "fourierflow.FactorizedFourierNeuralOperator._2D_",
                          "fourierflow.FactorizedFourierNeuralOperator._3D_",
                          "fourierflow.FourierNeuralOperator",
                          "fourierflow.FourierNeuralOperator._2D_",
                          "fourierflow.FourierNeuralOperator._3D_"],
)
