
<p align='center'>
  <a href='https://github.com/alec-tschantz/pymdp'>
    <img src='.github/logo.png' />
  </a> 
</p>

An implementation of predictive coding in Python, utilising `pytorch` for GPU acceleration. We compare the use of the free energy as a loss function vs mean squared error. In addition we also test these predictive coding networks against a normal backpropagation network.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Running

# Free Energy Loss Function

```bash
python -m scripts.supervised
```

# Mean Squared Error Loss Function

```bash
python -m scripts.supervised_mse
```

# Backpropagation

```bash
python -m scripts.supervised_bp
```

## Requirements
- `numpy`
- `torch`
- `torchvision` 

## Authors
- [Alexander Tschantz](https://github.com/alec-tschantz) 
- [Beren Millidge](https://github.com/BerenMillidge)

## Extended by
- [Jose Urruticoechea](https://github.com/jurruti) 
- [Panos_Syrgkanis](https://github.com/psyrgkan)
- [Serdar Sungun](https://github.com/theSStranger)
