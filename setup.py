from setuptools import find_packages, setup

VERSION          = '1.1.0'
DESCRIPTION      = 'Fourier Flow'
LONG_DESCRIPTION = 'Fourier Flow - Contains the Fourier Neural Operator and the Factorized Fourier Neural Operator'

setup(
      name             = "fourierflow",
      version          = VERSION,
      url              = 'https://github.com/VikVador/FourierFlow/tree/main',
      author           = 'Victor Mangeleer',
      author_email     = 'vmangeleer@student.uliege.be',
      description      = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      packages         = ["fourierflow",
                          "fourierflow.FactorizedFourierNeuralOperator._2D_",
                          "fourierflow.FactorizedFourierNeuralOperator._3D_",
                          "fourierflow.FourierNeuralOperator._2D_",
                          "fourierflow.FourierNeuralOperator._3D_"],
)
