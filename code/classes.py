import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import spearmanr

from copy import deepcopy
from collections import defaultdict


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

        # TODO: incorporate monotonicity effect
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
    

    @staticmethod
    def detector_features(data, categorical_indicator) -> set:
        p = data.shape[1] - 1  # not a probability, total # of features
        num_inclduded_features = Prob.r_trunc_geom(Prob.p_sample_features_selected, samples=1, val_min=1, val_max=p)
        info_gain = mutual_info_classif(
            X=data.loc[:, data.columns != 'class'],
            y=data.loc[:, 'class'],
            discrete_features=categorical_indicator[:-1]  # TODO: check if a Dataset can be passed (instead of the pandas dataframe) + if so save categorical_indicator in the dataset
        )
        p_info_gain = info_gain / np.sum(info_gain)

        feats_selected = np.random.choice(
            a = data.shape[1]-1,
            size=num_inclduded_features,
            replace=False,
            p=p_info_gain
        )

        return set(feats_selected.tolist())


    @staticmethod
    def detector_interactions(data, features_included: set) -> list:  # returns groups of included features without monotonicity attribute
        p = len(features_included)  # not a probability, # of features included
        num_interactions = Prob.r_trunc_geom(Prob.p_sample_interactions, samples=1, val_min=1, val_max=p * (p - 1) / 2).item()
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        feature_names = data.loc[:, data.columns != 'class'].columns
        poly.feature_names_in_ = feature_names

        X_interaction_terms = poly.fit_transform(  # X_interaction_terms doesn't include 'class', hence don't name 'data_*' but 'X_*'
            X=data.iloc[:, data.columns != 'class'],
            y=data.loc[:, 'class']
        )
        X_interaction_terms_columns = poly.get_feature_names_out().tolist()

        data_train, data_test = train_test_split(  # as pandas data frame to preserve original indices (numpy ndarray discards them)
            data,
            train_size=0.8,
            shuffle=True,
            stratify=data.loc[:, 'class']
        )
        idx_train = data_train.index
        idx_test = data_test.index
        # data as numpy
        X_interaction_terms_train = X_interaction_terms[idx_train]
        X_interaction_terms_test = X_interaction_terms[idx_test]

        features_included_list = list(features_included)
        score_interaction = dict()
        # iterate over feature combinations
        # interaction_terms + feature_names include ALL features + interaction effects, i.e. also those that are excluded
        for i in features_included:
            for j in features_included:
                feature_name_i = feature_names[i]
                feature_name_j = feature_names[j]

                try:
                    idx_interaction = X_interaction_terms_columns.index(f'{feature_name_i} {feature_name_j}')
                except:
                    # combination (i, j) not in interactions
                    # either (j, i) will be in interactions (if i, j are in features_included)
                    # or neither (i, j) nor (j, i) will be in interactions (if any of them are not in features_included)
                    # in any case, continue with next iteration
                    continue
                
                log_mod = LogisticRegression(penalty=None)
                log_mod = log_mod.fit(X=X_interaction_terms_train[:, features_included_list + [idx_interaction]], y=data.loc[idx_train, 'class'])
                score_interaction[(i, j)] = log_mod.score(X=X_interaction_terms_test[:, features_included_list + [idx_interaction]], y=data.loc[idx_test, 'class'])

        score_interactions_sorted = sorted(score_interaction.items(), key=lambda elem: elem[1], reverse=True)[:num_interactions]  # descending
        included_interactions = [elem[0] for elem in score_interactions_sorted]

        groups_included = [set(interaction) for interaction in included_interactions]
        len_last_groups_included = -1
        while len_last_groups_included != len(groups_included):
            len_last_groups_included = len(groups_included)
            i = 0
            while i < len(groups_included):
                j = i + 1
                while j < len(groups_included):
                    if groups_included[i].isdisjoint(groups_included[j]):
                        j += 1
                        continue
                    else:
                        groups_included[i].update(groups_included[j])
                        del groups_included[j]
                i += 1
        
        # in case some included features are not yet in the interaction groups, add them to individual groups
        for feature in features_included:
            feature_placed = False
            for group in groups_included:
                if feature in group:
                    feature_placed = True
                    break
            if not feature_placed:
                groups_included.append([feature])

        return [list(group) for group in groups_included]
        

    @staticmethod
    def detector_monotonicity(data, included_groups_without_monotonicity: list) -> list:
        groups_included = list()
        for group in included_groups_without_monotonicity:
            group_scores = list()
            group_signs = list()

            for feature in group:
                rhos = list()
                for _ in range(10):
                    data_train = resample(
                        data.iloc[:, :],
                        replace=True,
                        n_samples=round(0.9 * len(data.index)),
                        stratify=data.loc[:, 'class']
                    )
                    idx_data_train = data_train.index
                    data_test = data.loc[~data.index.isin(idx_data_train), :]

                    dec_tree = DecisionTreeClassifier(max_depth=30, min_samples_split=20)
                    dec_tree = dec_tree.fit(X=data_train.iloc[:, [feature]], y=data_train.loc[:, 'class'])  # expects DataFrame for X
                    y_pred = dec_tree.predict(X=data_test.iloc[:, [feature]])  # expects DataFrame for X

                    rhos.append(spearmanr(a=data_test.iloc[:, feature], b=y_pred).statistic)

                rho_mean = np.mean(rhos)
                rho_sign = np.sign(rho_mean)

                score = (np.abs(rho_mean) - 0) / (1 - 0) * (0.8 - 0.2) + 0.2  # scale to [0.2, 0.8], cf. https://stats.stackexchange.com/a/281164

                group_scores.append(score)
                group_signs.append(rho_sign)
            
            groups_included.append([
                group,
                round(np.random.binomial(n=1, p=np.mean(group_scores), size=1).item()),
                group_signs
            ])

        return groups_included
    

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
    p_sample_features_selected = 0.5  # original paper uses relative # of features used across 10 decision trees, for our NN implementation just use 0.5
    p_sample_interactions = 0.5  # original paper uses relative # of pairwise interactions used across 10 decision trees, for our NN implementation just use 0.5

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
        draws_trunc_geom = np.ceil(  # round with np.ceil, as support of trunc geom in (a, b], cf. https://en.wikipedia.org/wiki/Truncated_distribution
            np.log(np.pow(1 - p, a) - draws_unif * (np.pow(1 - p, a) - np.pow(1 - p, b))) / np.log(1 - p),
        ).astype(int)
        
        return draws_trunc_geom
    

    @staticmethod
    def should_do(p: float):
        return 1  # TODO: remove
        return np.random.uniform() <= p
    

    @staticmethod
    def fit_decision_tree(data):
        pass