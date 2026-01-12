# models.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, num_classes, layers=3, hidden=64):
        """
        input_dim: διάσταση διανυσμάτων (d)
        num_classes: αριθμός partitions (m)
        layers: συνολικός αριθμός γραμμικών layers (π.χ. 3)
        hidden: αριθμός νευρώνων σε κάθε κρυφό layer
        """
        super().__init__()
        modules=[]
        # (layers - 1) κρυφά layers με ReLU
        for _ in range(max(layers-1, 0)):
            modules.append(nn.Linear(d_in, hidden))
            modules.append(nn.ReLU())
            d_in=hidden
        # τελικό layer: πάει σε num_classes (m)
        modules.append(nn.Linear(d_in, num_classes))
        self.net = nn.Sequential(*modules)
    def forward(self, x):
         # Forward pass through the network
        return self.net(x)  
