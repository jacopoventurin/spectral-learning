# spectral-learning
PyThorch implementation of some tools related to spectral learning as described in https://journals.aps.org/pre/pdf/10.1103/PhysRevE.104.054312.

The repository also contain some tools for the application of spectral pruning as described in https://www.nature.com/articles/s41598-022-14805-7

### Installation
Install the package from source by calling `pip install .` 
from the repository's root directory.

### Example usage
`
import torch
import torch.nn as nn
from Spectral import Spectral, Pruning, update_train

#define model
spectral_model = nn.Sequential(Spectral(10,200), nn.Tanh(), Spectral(200,1))

#some training...

#pruning
Pruning(spectral_model, 70, input_fixed=True)

#allow training on all parameters
update_train(spectral_model, base=True, diag_start=True, diag_end=True)
`
