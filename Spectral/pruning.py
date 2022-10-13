import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Spectral import Spectral, update_train


class Pruning:
    """
    Class for the application of pruning procedure. In this case only eigenvalues in diag_start are pruned,
    while eigenvalues in diag_end are used as control to match the right number of nodes.
    This class work correctly only if all the layers in the network are Spectral.

    Parameters:
    -----------
    model:
        nn.Module based model. Only Spectral layer should be use in the feedforward neural network
        in order to obtain the correct pruned model.
    perc:
        given percentile for pruning.
    input_fixed:
        if set to True the dimension of the input layer is fixed. The dimension of the output layer
        is always fixed

    Example:
    --------
        import torch.nn as nn
        from Spectral import Spectral, Pruning

        model = nn.Sequential(Spectral(20,20), Spectral(20,30), Spectral(30,1))
        Pruning(model, 50)


    """

    def __init__(self,
                 model: nn.Module,
                 perc: float,
                 input_fixed: bool = True,
                 ):

        self.model = model
        self.perc = perc
        self.input_fixed = input_fixed

        # New indexes in each diag_start
        new_index_start = self.return_index()

        j = 0  # Iterator associated to the j-th layer in self.model
        self.deep = len(new_index_start) - 1

        for module in self.model.modules():
            if isinstance(module, Spectral):
                self.prune_model(module, new_index_start, j)
                j += 1

        self.show_result()

    def return_shape(self):
        """
        Method that return shape of Spectral diag_start in the original model except for the last.
        This should be the input of `np.split`
        """
        shape = []
        length = 0
        for module in self.model.modules():
            if isinstance(module, Spectral):
                length += len(module.diag_start)
                shape.append(length)
        shape = shape[:-1]
        return shape

    def return_index(self):
        """
        Method that return only indices of eigenvalue in diag_start that shouldn't be pruned,
        reshaped according to [[indexes layer1],[indexes layer2]...]
        """
        # Collects all the diag_start's value
        diag = []

        if self.input_fixed:
            i = 0
            hidden = []
            for module in self.model.modules():
                if isinstance(module, Spectral):
                    if i == 0:
                        diag.extend(module.diag_start.detach().numpy())
                    else:
                        hidden.extend(module.diag_start.detach().numpy())
                    i += 1

            # Put elements under threshold equal 0
            hidden = np.array(hidden, dtype=object)
            abs_diag = np.abs(hidden)
            threshold = np.percentile(abs_diag, self.perc)
            hidden[abs_diag < threshold] = 0.0

            # Include input_diag
            diag.extend(hidden)
            diag = np.array(diag, dtype=object)

        else:
            for module in self.model.modules():
                if isinstance(module, Spectral):
                    diag.extend(module.diag_start.detach().numpy())

            # Put elements under threshold equal 0
            diag = np.array(diag, dtype=object)
            abs_diag = np.abs(diag)
            threshold = np.percentile(abs_diag, self.perc)
            diag[abs_diag < threshold] = 0.0

        # Reshape indexes
        original_shape = self.return_shape()
        diag = np.split(diag, original_shape, axis=0)
        # Maintain indexes different from 0 only
        index = []
        for i in range(np.shape(diag)[0]):
            index.append(diag[i].nonzero()[0].tolist())

        return index

    def find_top_end(self, module, dim):
        """
        Method to find index of dim higher value in diag_end of a given module. It returns new index
        and prune diag end
        """
        diag_end = module.diag_end.detach()
        abs_diag_end = torch.abs(diag_end)
        _, index = torch.topk(abs_diag_end, dim, dim=1)
        index = torch.reshape(index,(dim,))
        diag_end = diag_end[0, index]

        module.diag_end = nn.Parameter(diag_end)

        return torch.Tensor(index)

    def prune_diag_end(self, module, new_index):
        diag_end = module.diag_end.detach()
        diag_end = diag_end[0, new_index]
        module.diag_end = nn.Parameter(diag_end)


    def prune_diag_start(self, module, new_index):
        """
        Method to prune diag_start using new indexes

        """
        diag_start = module.diag_start.detach()
        diag_start = diag_start[new_index]
        module.diag_start = nn.Parameter(diag_start)

    def prune_base(self, module, new_index_start, new_index_end=None):
        """
        Method to prune base and eventual bias using new indexes.
        If new_index_end = None only input is pruned

        """
        base = module.base

        if new_index_end is not None:
            base = base[:, new_index_end]
            if module.bias is not None:
                bias = module.bias.detach()
                bias = bias[new_index_end]
                module.bias = nn.Parameter(bias)

            base = base[new_index_start, :]
            module.base = nn.Parameter(base)
        else:
            base = base[new_index_start, :]
            module.base = nn.Parameter(base)



    def prune_model(self, module: Spectral, new_index_start: list, j: int):
        """Method that prune a module deep j, according to new_index"""
        if j < self.deep:
            self.prune_diag_start(module, new_index_start[j])
            self.prune_diag_end(module, new_index_start[j+1])
            self.prune_base(module, new_index_start[j], new_index_start[j+1])
            module.in_dim = len(new_index_start[j])
            module.out_dim = len(new_index_start[j+1])

        else:
            self.prune_diag_start(module, new_index_start[j])
            start_dim = len(new_index_start[j])
            end_dim = len(module.diag_end)
            new_index_end = self.find_top_end(module, end_dim)
            self.prune_base(module, new_index_start[j], new_index_end)
            module.in_dim = len(new_index_start[j])



    def show_result(self):
        print("New pruned model:")
        print(self.model)
