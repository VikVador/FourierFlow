<img src="assets/header_fourierflow.gif" />
<hr>
<p align="center">
<b style="font-size:1.5vw;">Fourier Flow</b>
</p>
<hr>

In this repository, you will find the code for the:
- Fourier Neural Operators (**FNO**) : Base (V1/V2), Time Dependant & Cloud point

- Factorized Fourier Neural Operators (**FFNO**) : Base, Time Dependant  (V1/V2), & Cloud point

The reason behind the creation of this repository is to have something simpler and more convenient to install than what is available online. It is important to notice that I do not own the credit of most of the code !

<hr>
<p align="center">
<b style="font-size:1.5vw;">Fourier Neural Operator</b>
</p>
<hr>

```python
import fourierflow.FourierNeuralOperator._2D_.Base as FNO_2D

# Loading a spectral convolutional layer
layer = FNO_2D.SpectralConv2d(in_channels = 3,
			out_channels = 1,
			modes1 = 12,
			modes2 = 12)

# Loading FNO
model = FNO_2D.FNO(input_channels = 3,
			output_channels = 1,
			modes1 = 12,
			modes2 = 12,
			width = 32,
			n_layers = 4)
```
<img src="assets/fno.png" />
<hr>
<p align="center">
<b style="font-size:1.5vw;">Factorized Fourier Neural Operator</b>
</p>
<hr>

```python
import fourierflow.FactorizedFourierNeuralOperator._2D_.Base as FFNO_2D

# Loading a spectral convolutional layer
layer = FFNO_2D.SpectralConv2d(in_dim = 3,
			out_dim = 1,
			modes_x = 12,
			modes_y = 12,
			mode = "full",
			n_ff_layers = 2,
			factor = 4,
			dropout = 0.1,
			ff_weight_norm = True,
			use_fork = False,
			layer_norm  = False,
			fourier_weight = None,
			forecast_ff = None,
			backcast_ff = None)

# Loading FNO
model = FFNO_2D.FFNO(input_dim = 3,
			output_dim = 1,
			modes_x = 12,
			modes_y = 12,
			width = 32,
			n_layers = 4,
			factor = 4,
			n_ff_layers = 2,
			share_weight = True,
			ff_weight_norm = True,
			layer_norm  = False)
```

<img src="assets/ffno.png" />
<hr>
<p  style="font-size:20px; font-weight:bold;" align="center">
Installation
</p>
<hr>

```
pip install git+https://github.com/VikVador/FourierFlow
```

<hr>
<p  style="font-size:1.5vw; font-weight:bold;" align="center">
Original Papers & GitHubs
</p>
<hr>
The authors of this remastered code are Zongyi Li and Daniel Zhengyu Huang ! This code is based on the papers :

                          Factorized Fourier Neural Operators
                          (https://arxiv.org/abs/2111.13802)

                  Fourier Neural Operator for Parametric Partial Differential Equations
                              (https://arxiv.org/abs/2010.08895)

 and comes from:

                    https://github.com/alasdairtran/fourierflow

                  https://github.com/neuraloperator/neuraloperator
