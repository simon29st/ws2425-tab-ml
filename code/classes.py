import torch
from torch import nn

import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self, group_structure, output_size, total_layers, nodes_per_hidden_layer):
        super().__init__()

        self.group_structure = group_structure

        # split input into multiple networks (depending on group structure), cf. https://discuss.pytorch.org/t/implement-selected-sparse-connected-neural-network/45517/2
        self.networks = nn.ModuleList()

        # one network per group in group_structure
        for (features, _) in self.group_structure.get_included_groups():
            modules = list()

            # input layer
            modules.append(nn.Linear(len(features), nodes_per_hidden_layer))
            modules.append(nn.ReLU())

            # hidden layers
            for _ in range(1, total_layers-1):
                modules.append(nn.Linear(nodes_per_hidden_layer, nodes_per_hidden_layer))
                modules.append(nn.ReLU())
            
            network_for_group = nn.Sequential(*modules)
            self.networks.append(network_for_group)

        # output layer, without activation (loss function already includes activation)
        self.layer_out = nn.Linear(nodes_per_hidden_layer * self.group_structure.get_number_of_included_groups(), output_size)
            

    def forward(self, *xs):  # xs (list) as this receives one "x tensor" per group in group_structure
        output_networks = list()
        for i, x in enumerate(xs):
            output_networks.append(self.networks[i](x))
        return self.layer_out(torch.cat(output_networks, 1))
    

    def get_networks(self):
        return self.networks
    

    def get_group_structure(self):
        return self.group_structure


# cf .https://stackoverflow.com/a/70330290
class WeightClipper:
    def __init__(self, w_min: int, w_max: int):
        self.w_min = w_min
        self.w_max = w_max
    

    def __call__(self, module: nn.Module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(self.w_min, self.w_max)
            module.weight.data = w


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, class_pos: str, group_structure):
        self.X = torch.tensor(X.values, dtype=torch.float)

        self.y = torch.zeros(len(y.index))
        self.y[y.reset_index(drop=True) == class_pos] = 1
        
        self.feature_groups = group_structure.get_included_groups_features()


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, idx: int):
        return tuple(self.X[idx, group] for group in self.feature_groups), self.y[idx]


class GroupStructure:
    def __init__(self, all_features: set, excluded: set, *included):
        self.all_features = set()

        self.excluded = excluded
        self.all_features.update(self.excluded)

        for g_k in included:
            if isinstance(g_k, tuple) and len(g_k) == 2 and isinstance(g_k[0], tuple) and isinstance(g_k[1], int) and g_k[1] in {0, 1}:
                self.all_features.update(g_k[0])
            else:
                raise Exception('invalid group', g_k)
        self.included = included

        if all_features != self.all_features:
            raise Exception('feature mismatch', all_features, 'vs', self.all_features)
        

    def __str__(self):
        return f'({self.excluded}, {self.included})'
    

    def get_number_of_included_groups(self):
        return len(self.included)
    

    def get_included_groups(self):
        return self.included
    

    def get_included_groups_features(self) -> tuple:  # only get feature sets of the groups
        return tuple(group[0] for group in self.included)
    

    def get_included_features(self) -> set:
        return set(feature for group in self.included for feature in group[0])
    

    def get_unconstrained_groups(self):  # groups without monotonicity constraint
        return tuple(group for group in self.included if group[1] == 0)
    

    def get_unconstrained_features(self):  # features of groups without monotonicity constraint
        return set(feature for group in self.get_unconstrained_groups() for feature in group[0])
    

    def get_all_features(self) -> set:
        return self.all_features
    

    def gga_mutate(slf):
        pass  # TODO


    @classmethod
    def gga_crossover(cls):
        pass  # TODO