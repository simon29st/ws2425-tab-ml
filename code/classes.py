import torch
from torch import nn

import numpy as np
import pandas as pd

from copy import deepcopy


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
        # TODO: incorporate monotonicity effect
        return tuple(self.X[idx, group] for group in self.feature_groups), self.y[idx]


class GroupStructure:
    def __init__(self, all_features: set, excluded: set, *included):
        self.all_features = set()

        self.excluded = excluded
        self.all_features.update(self.excluded)

        for g_k in included:
            if isinstance(g_k, list) and len(g_k) == 2 and isinstance(g_k[0], list) and isinstance(g_k[1], int) and g_k[1] in {-1, 0, 1}:
                if any(feature in self.all_features for feature in g_k[0]):
                    raise Exception(f'a feature in group {g_k} has already been used in another group in this group structure')
                else:
                    self.all_features.update(g_k[0])
            else:
                raise Exception('invalid group', g_k)
        self.included = list(included)

        if all_features != self.all_features:
            raise Exception('feature mismatch', all_features, 'vs', self.all_features)
        

    def __str__(self):
        return f'({self.excluded}, {self.included})'
    

    def __len__(self):
        return 1 + len(self.included)
    

    def get_number_of_included_groups(self):
        return len(self.included)
    

    def get_included_groups(self):
        return self.included
    

    def get_included_groups_features(self) -> list:  # only get feature sets of the groups
        return [group[0] for group in self.included]
    

    def get_included_features(self) -> set:
        return set(feature for group in self.included for feature in group[0])
    

    def get_unconstrained_groups(self):  # groups without monotonicity constraint
        return [group for group in self.included if group[1] == 0]
    

    def get_unconstrained_features(self):  # features of groups without monotonicity constraint
        return set(feature for group in self.get_unconstrained_groups() for feature in group[0])
    

    def get_all_features(self) -> set:
        return self.all_features
    

    def gga_mutate(self):
        copy_excluded = self.excluded.copy()
        copy_included = self.included[:]

        for feature_excl in copy_excluded:
            if Prob.should_do(Prob.p_gga_mutate_feature):
                self.excluded.remove(feature_excl)
                index_group_new = np.random.randint(low=0, high=len(copy_included), size=1).item()
                self.included[index_group_new][0].append(feature_excl)
        
        for i, group in enumerate(copy_included):
            for feature_incl in group[0]:
                if Prob.should_do(Prob.p_gga_mutate_feature):
                    self.included[i][0].remove(feature_incl)
                    index_group_new = np.random.randint(low=0, high=1 + len(copy_included), size=1).item()
                    if index_group_new == 0:
                        self.excluded.add(feature_incl)
                    else:
                        self.included[index_group_new - 1][0].append(feature_incl)
        
        for i, group in enumerate(copy_included):
            if Prob.should_do(Prob.p_gga_mutate_monotonicity):
                self.included[i][1] = np.random.randint(low=-1, high=2, size=1).item()
    

    def get_crossing_section(self, bounds: list) -> list:
        crossing_section = list()

        lower, upper = bounds
        if lower == 0:
            crossing_section.append(self.excluded)
            lower += 1
        
        crossing_section += self.included[lower-1:upper]
        return crossing_section
    

    def insert_crossing_section(self, crossing_section):
        features_to_insert = set()
        for group in crossing_section:
            if isinstance(group, set):  # exclusion group in crossing section
                features_to_insert.update(group)
            else:  # "regular" group
                for feature in group[0]:
                    features_to_insert.add(feature)
        
        # remove features to be inserted form current groups
        for feature in features_to_insert:
            self.excluded.discard(feature)
        for group in self.included:
            group[0] = [feature for feature in group[0] if feature not in features_to_insert]
        
        # remove empty "included" groups (excluded would remain as a ~placeholder if empty)
        included = deepcopy(self.included)
        for i in range(len(self.included)-1, -1, -1):  # go from back to front so indices to remove are still valid after consecutive removals
            if len(self.included[i][0]) == 0:
                del included[i]
        self.included = included
        
        # insert crossing section
        for group in crossing_section:
            if isinstance(group, set):  # exclusion group in crossing section
                self.excluded.update(group)
            else:  # "regular" group
                self.included.append(group)


    @classmethod
    def gga_crossover(cls, parent_1, parent_2):
        bounds_1 = sorted(np.random.randint(low=0, high=len(parent_1), size=2))
        bounds_2 = sorted(np.random.randint(low=0, high=len(parent_2), size=2))

        crossing_section_1 = parent_1.get_crossing_section(bounds_1)
        crossing_section_2 = parent_2.get_crossing_section(bounds_2)

        child_1 = deepcopy(parent_1)
        child_2 = deepcopy(parent_2)

        child_1.insert_crossing_section(crossing_section_2)
        child_2.insert_crossing_section(crossing_section_1)


        return child_1, child_2


class Prob:
    p_sample_hps = 0.5

    p_ea_crossover_overall = 0.7
    p_ea_crossover_param = 0.5
    p_ea_mutate_overall = 0.3
    p_ea_mutate_param = 0.2

    p_gga_crossover = 0.7
    p_gga_mutate_overall = 0.3
    p_gga_mutate_feature = 0.2
    p_gga_mutate_monotonicity = 0.2


    @staticmethod
    def r_trunc_geom(p: float, samples: int, val_min: int = 3, val_max: int = 10):
        a = val_min - 1
        b = val_max

        draws_unif = np.random.uniform(low=0, high=1, size=samples)
        draws_trunc_geom = np.ceil(  # np.ceil, as support of trunc geom in (a, b], cf. https://en.wikipedia.org/wiki/Truncated_distribution
            np.log(np.pow(1 - p, a) - draws_unif * (np.pow(1 - p, a) - np.pow(1 - p, b))) / np.log(1 - p),
        ).astype(int)
        
        return draws_trunc_geom
    

    @staticmethod
    def should_do(p: float):
        return 1  # TODO: remove
        return np.random.uniform() <= p