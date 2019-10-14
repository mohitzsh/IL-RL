import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.initalize()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def initalize(self):
        for parameters in self.parameters():
            nn.init.normal_(parameters,0,1e-2)

#class MultiHeadMLP(nn.Module):
#
#    def __init__(self, input_size, output_size, nheads, hidden_size=128):
#        super(MultiHeadMLP, self).__init__()
#        self.nheads = nheads 
#        self._head = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU(),
#            nn.Linear(hidden_size, output_size)
#        )
#
#        self.heads = nn.ModuleList([self._head for _ in range(self.nheads)])
#
#        self.initalize()
#
#    def forward(self, x, k=None):
#        if k is not None:
#            """
#            A head is specified. Only pass x through that specific head.
#            """
#            _head = self.heads[k]
#            x = _head(x)
#        else:
#            """
#            No head is specified, pass the x through every head.
#            """
#            _x = []
#            for _head in self.heads:
#                _x.append(_head(x).unsqueeze(0))
#            x = torch.cat(_x, dim=0) 
#            del _x
#        return x
#
#    def initalize(self):
#        for parameters in self.parameters():
#            nn.init.normal_(parameters,0,1e-2)