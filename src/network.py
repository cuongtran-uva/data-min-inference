import torch, copy
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class MultiClassMLP(nn.Module):
    """
    Implement a MLP with  single hidden layer. The choice of activation
    function can be passed as argument when init the network
    """

    def __init__(self, options={'num_feats': 20, 'act_func':'Relu', 'n_out':3}):
        super(MultiClassMLP, self).__init__()
        self.input_layer = nn.Linear(options['num_feats'], 10)
        C = options['n_out']
        if options['act_func'] =='Relu':
          self.act_func = nn.ReLU()
        elif options['act_func'] =='Tanh':
          self.act_func = nn.Tanh()
        else:
          self.act_func = nn.Sigmoid()
        self.o_layer = nn.Linear(10, C)

    def forward(self, x):
        output = self.act_func(self.input_layer(x))
        output = self.o_layer(output)
        return output

class MultiClassCLF(object):
    def __init__(self, train_loader):
      self.train_loader = train_loader
    def fit(self, options):
        """
        train a neural network model
        """
        torch.manual_seed(0)
        model = MultiClassMLP(options)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'])
        for _ in range(options['epochs']):
            for inputs, targets in self.train_loader:
                targets = targets
                model.zero_grad()
                optimizer.zero_grad()
                outputs = model(inputs)
                clf_loss = loss_func(outputs, targets)
                clf_loss.backward()
                optimizer.step()
        self.model = model


class BayesianLR(nn.Module):
    """
    Implement a logistic regression classifiers
    """

    def __init__(self, options={'d': 20, 'o':10}):
        super(BayesianLR, self).__init__()
        self.input_layer = nn.Linear(options['d'], options['o'])
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.input_layer(x)
        return self.tanh(output)





