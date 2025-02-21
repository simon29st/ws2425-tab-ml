import torch
from torch import nn

import pandas as pd


# TODO: add #groups argument and split the features to their respective nets like so: https://discuss.pytorch.org/t/grouping-some-features-before-going-to-fully-connected-layers/162444/5
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, total_layers, nodes_per_hidden_layer):
        super().__init__()

        # input layer
        layers = list()
        layers.append(nn.Linear(input_size, nodes_per_hidden_layer))
        layers.append(nn.ReLU())

        # hidden layers
        for _ in range(1, total_layers-1):
            layers.append(nn.Linear(nodes_per_hidden_layer, nodes_per_hidden_layer))
            layers.append(nn.ReLU())

        # output layer, without activation (loss function includes activation)
        layers.append(nn.Linear(nodes_per_hidden_layer, output_size))
        
        self.net = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.net(x)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, class_pos: str):
        self.X = torch.tensor(X.values, dtype=torch.float)
        self.y = torch.zeros(len(y.index))
        self.y[y.reset_index(drop=True) == class_pos] = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GroupStructure():
    def __init__(self, all_features: set, excluded: set, *included):
        self.all_features = set()

        self.excluded = excluded
        self.all_features.update(self.excluded)

        for g_k in included:
            if isinstance(g_k, tuple) and len(g_k) == 2 and isinstance(g_k[0], set) and isinstance(g_k[1], int) and g_k[1] in {0, 1}:
                self.all_features.update(g_k[0])
            else:
                raise Exception('invalid group', g_k)
        self.included = included

        if all_features != self.all_features:
            raise Exception('feature mismatch', all_features, 'vs', self.all_features)
    
    def get_number_of_included_groups(self):
        return len(self.included)
    
    def mutate(slf):
        pass  # TODO

    @classmethod
    def crossover(cls):
        pass  # TODO