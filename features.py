import torch.nn as nn

prod_features = [
    nn.Linear(270, 128),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(128, 2)
]

n_neurons = 128

exp_features = [
    nn.Linear(270, n_neurons),
    nn.Dropout(p=0.2),
    nn.ReLU(),
    nn.Linear(n_neurons, n_neurons),
    nn.Dropout(p=0.2),
    nn.ReLU(),
    nn.Linear(n_neurons, 2)
]