import torch
import torch.nn as nn
from Spectral import Spectral


def update_train(net: nn.Module, base: bool = True, diag_start: bool = True, diag_end: bool = True):
    """
      Function for change trainable parameters in spectral network.
      If set to true the elements of each spectral layer in net are trainable.
      """

    try:
        for module in net.modules():
            if isinstance(module, Spectral):
                module.diag_start.requires_grad_(diag_start)
                module.diag_end.requires_grad_(diag_end)
                module.base.requires_grad_(base)

                module.start_grad = diag_start
                module.end_grad = diag_end
                module.base_grad = base
        print("Modified model: ")

    except:
        print(f"Check for your model: seems {net}.module() don't exist!")
        print("Your model:")

    print(net)





