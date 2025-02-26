import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from scipy.stats import spearmanr

from nds import ndomsort

from math import comb
from copy import deepcopy
from datetime import datetime, timedelta
import os
from pathlib import Path
import json


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

        # incorporate monotonicity for individual features (swap sign if -1)
        for group in self.feature_groups:
            for feature in group:
                self.X[:, feature] *= group_structure.get_feature_signs()[feature]


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, idx: int):
        return tuple(self.X[idx, group] for group in self.feature_groups), self.y[idx]


class GroupStructure:
    def __init__(self, all_features: list, feature_signs: list, excluded: list, *included):
        self.all_features = list()

        self.excluded = excluded

        if len(all_features) != len(feature_signs):
            raise Exception(f'all_features must be same length as feature_signs: {all_features} vs {feature_signs}')
        elif any(feat_sign not in {-1, 1} for feat_sign in feature_signs):
            raise Exception(f'feature monotonicity must be encoded in {-1, 1}: {feature_signs}')
        self.all_features += self.excluded
        self.feature_signs = feature_signs  # used for swapping feature sign in Dataset

        for g_k in included:
            if isinstance(g_k, list) and len(g_k) == 2 and isinstance(g_k[0], list) and isinstance(g_k[1], int) and g_k[1] in {0, 1}:
                if any(feature in self.all_features for feature in g_k[0]):
                    raise Exception(f'a feature in group {g_k} has already been used in another group in this group structure')
                elif any(not isinstance(feature, int) for feature in g_k[0]):
                    raise Exception(f'features must be ints: {g_k[0]}')
                else:
                    self.all_features += g_k[0]
            else:
                raise Exception('invalid group', g_k)
        self.included = list(included)

        if set(all_features) != set(self.all_features):
            raise Exception(f'feature mismatch: {all_features} vs {self.all_features}')
        elif len(all_features) != len(self.all_features):
            raise Exception(f'some feature is missing / too much: {all_features} vs {excluded} + {included}')
        

    def __str__(self):
        return f'({self.excluded}, {self.included})'
    

    def __len__(self):
        return 1 + len(self.included)
    

    # used for saving to json
    def to_dict(self):
        return {
            'all_features': self.all_features,
            'feature_signs': self.feature_signs,
            'excluded': self.excluded,
            'included': self.included
        }
    

    # used for loading from json
    @classmethod
    def from_dict(cls, inp):
        return cls(
            inp['all_features'],
            inp['feature_signs'],
            inp['excluded'],
            *inp['included']
        )
    

    def get_number_of_included_groups(self):
        return len(self.included)
    

    def get_included_groups(self):
        return self.included
    

    def get_included_groups_features(self) -> list:  # only get feature lists of all groups
        return [group[0] for group in self.included]
    

    def get_included_features(self) -> list:
        return list(feature for group in self.included for feature in group[0])
    

    def get_unconstrained_groups(self):  # groups without monotonicity constraint
        return [group for group in self.included if group[1] == 0]
    

    def get_unconstrained_features(self) -> list:  # features of groups without monotonicity constraint
        return list(feature for group in self.get_unconstrained_groups() for feature in group[0])
    

    def get_all_features(self) -> list:
        return self.all_features
    
    
    def get_feature_signs(self) -> list:
        return self.feature_signs
    

    @staticmethod
    def detector_features(data, categorical_indicator: list, class_column: str) -> list:
        p = data.shape[1] - 1  # not a probability, total # of features
        num_inclduded_features = Prob.r_trunc_geom(Prob.p_sample_features_selected, samples=1, val_min=1, val_max=p)
        info_gain = mutual_info_classif(
            X=data.loc[:, data.columns != class_column],
            y=data.loc[:, class_column],
            discrete_features=categorical_indicator[:-1]  # TODO: check if a Dataset can be passed (instead of the pandas dataframe) + if so save categorical_indicator in the dataset
        )
        p_info_gain = info_gain / np.sum(info_gain)

        try:
            feats_selected = np.random.choice(
                a=data.shape[1]-1,
                size=num_inclduded_features,
                replace=False,
                p=p_info_gain
            )
        except:
            # sometimes num_inclduded_features (size) can be > # of non-zero values in p_info_gain (p)
            # then simply return only the features with non-zero values in p_info_gain
            feats_selected = np.nonzero(p_info_gain)[0]

        return list(feats_selected.tolist())


    @staticmethod
    def detector_interactions(data, features_included: set, class_column: str) -> list:  # returns groups of included features without monotonicity attribute
        p = len(features_included)  # not a probability, # of features included
        num_interactions = Prob.r_trunc_geom(Prob.p_sample_interactions, samples=1, val_min=1, val_max=p * (p - 1) / 2).item()
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        feature_names = data.loc[:, data.columns != class_column].columns
        poly.feature_names_in_ = feature_names

        X_interaction_terms = poly.fit_transform(  # X_interaction_terms doesn't include class_column, hence don't name 'data_*' but 'X_*'
            X=data.iloc[:, data.columns != class_column],
            y=data.loc[:, class_column]
        )
        X_interaction_terms_columns = poly.get_feature_names_out().tolist()

        data_train, data_test = train_test_split(  # as pandas data frame to preserve original indices (numpy ndarray discards them)
            data,
            train_size=0.8,
            shuffle=True,
            stratify=data.loc[:, class_column]
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
                log_mod = log_mod.fit(X=X_interaction_terms_train[:, features_included_list + [idx_interaction]], y=data.loc[idx_train, class_column])
                score_interaction[(i, j)] = log_mod.score(X=X_interaction_terms_test[:, features_included_list + [idx_interaction]], y=data.loc[idx_test, class_column])

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
    def detector_monotonicity(data, included_groups_without_monotonicity: list, class_column: str) -> tuple:
        feature_scores = list()
        feature_signs = list()
        for feature in range(len(data.columns) - 1):  # only iterate over feature columns (data includes class_column column)
            rhos = list()
            for _ in range(10):
                data_train = resample(
                    data.iloc[:, :],
                    replace=True,
                    n_samples=round(0.9 * len(data.index)),
                    stratify=data.loc[:, class_column]
                )
                idx_data_train = data_train.index
                data_test = data.loc[~data.index.isin(idx_data_train), :]

                dec_tree = DecisionTreeClassifier(max_depth=30, min_samples_split=20)
                dec_tree = dec_tree.fit(X=data_train.iloc[:, [feature]], y=data_train.loc[:, class_column])  # expects DataFrame for X
                y_pred = dec_tree.predict(X=data_test.iloc[:, [feature]])  # expects DataFrame for X

                if data_test.iloc[:, feature].nunique() == 1 or np.unique(y_pred).shape[0] == 1:  # spearmanr only defined if no input is constant
                    rhos.append(0)
                else:
                    rhos.append(spearmanr(a=data_test.iloc[:, feature], b=y_pred).statistic)

            rho_mean = np.mean(rhos)
            rho_sign = round(np.sign(rho_mean).item())
            if rho_sign == 0:
                rho_sign = 1

            score = (np.abs(rho_mean) - 0) / (1 - 0) * (0.8 - 0.2) + 0.2  # scale to [0.2, 0.8], cf. https://stats.stackexchange.com/a/281164

            feature_scores.append(score)
            feature_signs.append(rho_sign)
        
        groups_included = list()
        for group in included_groups_without_monotonicity:
            group_scores = list()

            for feature in group:
                group_scores.append(feature_scores[feature])
            
            groups_included.append([
                group,
                round(np.random.binomial(n=1, p=np.mean(group_scores), size=1).item()),
            ])

        return groups_included, feature_signs
    

    def gga_mutate(self):
        copy_excluded = self.excluded.copy()

        for feature_excl in copy_excluded:
            if Prob.should_do(Prob.p_gga_mutate_feature):
                self.excluded.remove(feature_excl)
                if len(self.included) == 0:  # in case some previous mutation yielded a featureless learner (i.e. empty self.included) -> append new included group and add feature there
                    self.included.append([
                        [feature_excl],
                        self.get_feature_signs()[feature_excl]
                    ])
                else:  # regular case, there is at least 1 included group, i.e. at least 1 feature used in the learner
                    index_group_new = np.random.randint(low=0, high=len(self.included), size=1).item()
                    self.included[index_group_new][0].append(feature_excl)
                    
        copy_included = deepcopy(self.included)
        
        for i, group in enumerate(copy_included):
            for feature_incl in group[0]:
                if Prob.should_do(Prob.p_gga_mutate_feature):
                    self.included[i][0].remove(feature_incl)
                    index_group_new = np.random.randint(low=0, high=1 + len(copy_included), size=1).item()
                    if index_group_new == 0:
                        self.excluded.append(feature_incl)
                    else:
                        self.included[index_group_new - 1][0].append(feature_incl)
        
        for i, group in enumerate(copy_included):
            if Prob.should_do(Prob.p_gga_mutate_monotonicity):
                self.included[i][1] = np.random.randint(low=-1, high=2, size=1).item()


    def get_crossing_section(self, bounds: list) -> list:
        crossing_section = list()
        with_exclusion_group = False

        lower, upper = bounds
        if lower == 0:
            crossing_section.append(self.excluded)
            with_exclusion_group = True
            lower += 1
        
        crossing_section += self.included[lower-1:upper]
        return crossing_section, with_exclusion_group
    

    def insert_crossing_section(self, crossing_section, with_exclusion_group):
        features_to_insert = set()
        for i, group in enumerate(crossing_section):  
            if i == 0 and with_exclusion_group:  # exclusion group in crossing section, first group in crossing section is then always the exclusion group
                features_to_insert.update(set(group))
            else:  # remainder are always groups of included features
                for feature in group[0]:
                    features_to_insert.add(feature)
        
        # remove features to be inserted form current groups
        for feature in features_to_insert:
            try:
                self.excluded.remove(feature)
            except:
                pass  # expected behaviour, feature to be inserted was not in self.excluded, then it must be in self.included
        for group in self.included:
            group[0] = [feature for feature in group[0] if feature not in features_to_insert]
        
        # remove empty "included" groups (excluded would remain as a ~placeholder if empty)
        included = deepcopy(self.included)
        for i in range(len(self.included)-1, -1, -1):  # go from back to front so indices to remove are still valid after consecutive removals
            if len(self.included[i][0]) == 0:
                del included[i]
        self.included = included
        
        # insert crossing section
        for i, group in enumerate(crossing_section):  
            if i == 0 and with_exclusion_group:  # exclusion group in crossing section
                self.excluded += group  # no need to check for duplicates, as those have already been removed above
            else:
                self.included.append(group)


    @classmethod
    def gga_crossover(cls, parent_1, parent_2):
        bounds_1 = sorted(np.random.randint(low=0, high=len(parent_1), size=2))
        bounds_2 = sorted(np.random.randint(low=0, high=len(parent_2), size=2))

        crossing_section_1, cs_1_with_exclusion_group = parent_1.get_crossing_section(bounds_1)
        crossing_section_2, cs_2_with_exclusion_group = parent_2.get_crossing_section(bounds_2)

        child_1 = deepcopy(parent_1)
        child_2 = deepcopy(parent_2)

        child_1.insert_crossing_section(crossing_section_2, cs_2_with_exclusion_group)
        child_2.insert_crossing_section(crossing_section_1, cs_1_with_exclusion_group)

        return child_1, child_2


class Prob:
    p_sample_hps = 0.5
    p_sample_features_selected = 0.5  # original paper uses relative # of features used across 10 decision trees, no straightforward to retrieve this from sklearn, for our NN implementation just use 0.5
    p_sample_interactions = 0.5  # original paper uses relative # of pairwise interactions used across 10 decision trees, no straightforward to retrieve this from sklearn, for our NN implementation just use 0.5

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


class EAGGA:
    def __init__(self, oml_dataset, class_positive, hps: dict[str, tuple | int | float], batch_size: int, patience: int, secs_per_fold: int, secs_total: int, file_path: str = None):
        self.data, _, self.categorical_indicator, _ = oml_dataset.get_data()
        self.class_column = oml_dataset.default_target_attribute
        self.class_positive = class_positive

        if self.data.isnull().values.any():  # from manual checks: dataset 'jm1' has NaN values in 5 rows -> simply remove
            print(f'Drop {self.data.loc[self.data.isnull().any(axis=1), :].shape[0]} NaN values of dataset {oml_dataset.name}')
            self.data = self.data.dropna(axis=0, how='any')
            self.data = self.data.reset_index(drop=True)
        
        self.hps = hps

        # outer split, holdout testing
        self.data_train_val, self.data_test = train_test_split(
            self.data,
            train_size=self.hps['holdout_train_size'],
            shuffle=True,
            stratify=self.data.loc[:, self.class_column]
        )
        # reset indices as StratifiedKFold assumes consecutive index
        self.data_train_val = self.data_train_val.reset_index(drop=True)
        self.data_test = self.data_test.reset_index(drop=True)

        # inner split, k-fold cross validation
        self.cv = StratifiedKFold(
            n_splits=self.hps['cv_k'],
            shuffle=True
        )

        self.batch_size = batch_size
        self.patience = patience

        self.secs_per_fold = secs_per_fold
        self.secs_total = secs_total
        self.file_path = file_path  # if not None -> autosave each generation, else do nothing

        self.monotonicity_clipper = WeightClipper(0, None)  # enforce monotonicity by clipping weights to [0, infty) after each epoch (in def train)

        self.offspring = self.init_population()
        self.population = list()


    def init_population(self):
        print('Starting init population')
        population_layers = Prob.r_trunc_geom(Prob.p_sample_hps, self.hps['mu'], self.hps['total_layers'][0], self.hps['total_layers'][1])
        population_nodes = Prob.r_trunc_geom(Prob.p_sample_hps, self.hps['mu'], self.hps['nodes_per_hidden_layer'][0], self.hps['nodes_per_hidden_layer'][1])

        all_features = list(i for i in range(len(self.data_train_val.columns) - 1))
        population_features_included = [GroupStructure.detector_features(self.data_train_val, self.categorical_indicator, self.class_column) for _ in range(self.hps['mu'])]
        population_features_excluded = [list(set(all_features) - set(features_included)) for features_included in population_features_included]
        population_interactions = [GroupStructure.detector_interactions(self.data_train_val, features_included, self.class_column) for features_included in population_features_included]
        population_monotonicity_constraints = [GroupStructure.detector_monotonicity(self.data_train_val, groups_without_monotonicity, self.class_column) for groups_without_monotonicity in population_interactions]
        print('Finished init population')

        return [{
            'total_layers': population_layers[i].item(),
            'nodes_per_hidden_layer': population_nodes[i].item(),
            'group_structure': GroupStructure(
                all_features,
                population_monotonicity_constraints[i][1],
                population_features_excluded[i],
                *population_monotonicity_constraints[i][0]
            )
        } for i in range(self.hps['mu'])]


    def start_eagga(self):
        time_start = datetime.now()
        print(f'Start EAGGA at {time_start.isoformat()}')

        gen = 0
        while(datetime.now() < time_start + timedelta(seconds=self.secs_total)):
            print(f'Generation {gen+1}, evaluate {len(self.offspring)} individuals')

            for individual in self.offspring:
                individual['metrics'] = self.run_cv(individual)
                self.population.append(individual)

                if datetime.now() >= time_start + timedelta(seconds=self.secs_total):
                    break  # don't train any more individuals if time ran out
            
            ranks_nds = ndomsort.non_domin_sort(
                [(
                    np.mean(individual['metrics']['performance']['auc']),
                    individual['metrics']['performance']['nf'],
                    individual['metrics']['performance']['ni'],
                    individual['metrics']['performance']['nnm'],
                ) for individual in self.population],
                get_objectives=lambda performance_tup: (1 - performance_tup[0], *[performance_tup[i] for i in range(1, len(performance_tup))]),  # compute pareto fronts w.r.t. reference (worst) point (0, 1, 1, 1)
                only_front_indices=True
            )
            for i in range(len(self.population)):
                self.population[i]['rank_nds'] = ranks_nds[i]
            self.population = sorted(self.population, key=lambda individual: individual['rank_nds'])[:self.hps['mu']]  # ascending order, lower ranks are better, choose best mu individuals
            
            if datetime.now() >= time_start + timedelta(seconds=self.secs_total):
                self.offspring = list()  # re-set offspring so in case of json export the same individuals won't be saved as part of offspring (without metrics) and population (with metrics)
                self.autosave(gen)
                break  # don't generate offspring if time ran out anyway
            
            self.offspring = self.generate_offspring()
            self.autosave(gen)

            gen += 1
        print(f'Finished EAGGA at {datetime.now().isoformat()}')


    # runs k-fold cv training with stopping early
    def run_cv(self, individual) -> dict:
        total_layers = individual['total_layers']
        nodes_per_hidden_layer = individual['nodes_per_hidden_layer']
        group_structure = individual['group_structure']
        print(f'Running {self.hps['cv_k']}-fold CV for individual: {total_layers} total layers, {nodes_per_hidden_layer} nodes per hidden layer, gs: {group_structure}')

        metrics = {
            'performance': list(),
            'epochs': list()
        }

        for fold, (indices_train, indices_val) in enumerate(self.cv.split(X=self.data_train_val, y=self.data_train_val.loc[:, self.class_column])):
            data_train_stop_early = self.data_train_val.loc[indices_train, :]
            data_train, data_stop_early = train_test_split(
                data_train_stop_early,
                train_size=0.8,
                shuffle=True,
                stratify=data_train_stop_early.loc[:, self.class_column]
            )
            dataset_train = Dataset(
                X=data_train.loc[:, data_train.columns != self.class_column],
                y=data_train.loc[:, self.class_column],
                class_pos=self.class_positive,
                group_structure=group_structure
            )
            dataset_stop_early = Dataset(
                X=data_stop_early.loc[:, data_stop_early.columns != self.class_column],
                y=data_stop_early.loc[:, self.class_column],
                class_pos=self.class_positive,
                group_structure=group_structure
            )

            data_val = self.data_train_val.loc[indices_val, :]
            dataset_val = Dataset(
                X=data_val.loc[:, data_val.columns != self.class_column],
                y=data_val.loc[:, self.class_column],
                class_pos=self.class_positive,
                group_structure=group_structure
            )

            model = NeuralNetwork(
                group_structure=group_structure,
                output_size=1,  # we only use binary datasets
                total_layers=total_layers,
                nodes_per_hidden_layer=nodes_per_hidden_layer
            )

            optimizer = torch.optim.AdamW(model.parameters())
            loss_fn = nn.BCEWithLogitsLoss()

            model, stop_epoch = self.train(optimizer, loss_fn, model, dataset_train, dataset_stop_early)

            metrics['performance'].append(self.eval(loss_fn, model, dataset_val))
            metrics['epochs'].append(stop_epoch)
            print(f'Fold {fold + 1}/{self.cv.get_n_splits()} | trained for {stop_epoch} epochs | metrics: {metrics['performance'][-1]}')  # TODO: remove

        return {
            'performance': {
                # NF, NI, NNM constant for each fold (always same group structure)
                'nf': metrics['performance'][0]['nf'],
                'ni': metrics['performance'][0]['ni'],
                'nnm': metrics['performance'][0]['nnm'],
                'auc': [performance['auc'] for performance in metrics['performance']],
                'loss' : [performance['loss'] for performance in metrics['performance']]
            },
            'epochs': metrics['epochs']
        }


    # does one training fold
    # cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    def train(self, optimizer, loss_fn, model: NeuralNetwork, dataset_train: Dataset, dataset_stop_early: Dataset) -> tuple:
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        epoch = 0
        loss_history_early_stopping = list()
        time_start = datetime.now()
        while(datetime.now() < time_start + timedelta(seconds=self.secs_per_fold)):
            model.train()  # training mode, put here as we switch to eval mode at the end of each epoch in def stop_early
            running_epoch_loss = 0

            for batch_input, batch_target in loader_train:  # divide data in mini batches
                optimizer.zero_grad()  # set gradients to 0
                batch_output = model(*batch_input).flatten()  # expand batch_input as it is a list of tuples (Dataset getter splits according to group structure)

                batch_loss = loss_fn(batch_output, batch_target)
                running_epoch_loss += batch_loss.detach().item()
                batch_loss.backward()  # compute gradients

                optimizer.step()  # update weights

            running_epoch_loss /= len(loader_train)

            if self.monotonicity_clipper is not None:  # used for monotonicity constraint
                for i, (_, monotonicity_constraint) in enumerate(model.get_group_structure().get_included_groups()):
                    if monotonicity_constraint == 1:
                        model.networks[i].apply(self.monotonicity_clipper)  # applies weight clipper recursively to network + its children, cf. https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply
            
            should_stop_early, stop_epoch = self.stop_early(loss_history_early_stopping, loss_fn, model, dataset_stop_early)
            if should_stop_early:
                return model, stop_epoch
            epoch += 1
        
        return model, epoch
    

    def stop_early(self, loss_history: list, loss_fn, model: NeuralNetwork, dataset_stop_early: Dataset) -> tuple:
        # stopping criterion: mean of 'patience' previous losses over [t-patience, t] < current loss at t+1
        # if True -> go back to min loss within [t-patience, t]
        loss = self.eval(loss_fn, model, dataset_stop_early, only_loss=True)['loss']
        loss_history.append(loss)

        if len(loss_history) > self.patience and np.mean(loss_history[-self.patience-1:-1]) < loss:
            mask = np.ones_like(loss_history)
            mask[:-self.patience-1] = np.inf
            stop_epoch = np.argmin(mask * loss_history) + 1  # +1 to make it 1-based (as opposed to 0-based from np.argmin indexing)
            print(f'Stop early: {np.mean(loss_history[-self.patience-1:-1])} < {loss}, epoch stop: {stop_epoch}')
            return True, stop_epoch.item()
        return False, -1
    

    # cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    def eval(self, loss_fn, model: NeuralNetwork, dataset_eval: Dataset, only_loss: bool = False) -> dict:
        res = dict()

        loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=self.batch_size)
        
        model.eval()  # eval mode
        running_loss = 0

        batch_predictions = list()
        batch_targets = list()
        for batch_input, batch_target in loader_eval:
            with torch.no_grad():
                batch_output = model(*batch_input).flatten()
            
            batch_loss = loss_fn(batch_output, batch_target)
            batch_prediction = torch.sigmoid(batch_output)

            batch_predictions.append(batch_prediction)
            batch_targets.append(batch_target)

            running_loss += batch_loss.detach().item()
        res['loss'] = running_loss / len(loader_eval)

        if only_loss:
            return res

        # compute AUC
        predictions = torch.cat(batch_predictions)
        targets = torch.cat(batch_targets)
        res['auc'] = roc_auc_score(targets, predictions)
        
        # compute interpretability metrics
        #   NF via group structure: relative # of included features
        #   NI via group structure:
        #       for each group with > 1 features: sum over range from 1 to (n - 1 = # of features in group - 1) -> # handshakes in party -> same as {n choose 2}
        #       then divide by # of all possible interactions {all_features choose 2}
        #   NNM via group structure: (# of features in groups without monotonicity constraint) / (total # of features)
        gs = model.get_group_structure()
        num_features_all = len(gs.get_all_features())
        num_features_included = len(gs.get_included_features())
        num_features_unconstrained = len(gs.get_unconstrained_features())

        res['nf'] = num_features_included / num_features_all
        combs = 0
        for (features, _) in gs.get_included_groups():
            num_features_in_group = len(features)
            if num_features_in_group > 1:
                combs += comb(num_features_in_group, 2)  # combs += sum(range(1, num_features_in_group))
        res['ni'] = combs / comb(num_features_all, 2)
        res['nnm'] = num_features_unconstrained / num_features_all
        
        return res


    def generate_offspring(self):
        offspring = list()

        while len(offspring) <= self.hps['lambda']:
            id_parent_1 = self.binary_tournament()
            id_parent_2 = self.binary_tournament()
            parent_1 = self.population[id_parent_1]
            parent_2 = self.population[id_parent_2]

            child_1 = deepcopy(parent_1)
            child_2 = deepcopy(parent_2)
            del child_1['metrics']
            del child_2['metrics']

            # EA on hyperparams
            if Prob.should_do(Prob.p_ea_crossover_overall):  # uniform crossover
                if Prob.should_do(Prob.p_ea_crossover_param):
                    child_1['total_layers'] = parent_2['total_layers']
                    child_2['total_layers'] = parent_1['total_layers']
                if Prob.should_do(Prob.p_ea_crossover_param):
                    child_1['nodes_per_hidden_layer'] = parent_2['nodes_per_hidden_layer']
                    child_2['nodes_per_hidden_layer'] = parent_1['nodes_per_hidden_layer']
            for child in [child_1, child_2]:  # Gaussian mutation
                if Prob.should_do(Prob.p_ea_mutate_overall):
                    if Prob.should_do(Prob.p_ea_mutate_param):
                        child['total_layers'] = EAGGA.ea_mutate_gaussian(child['total_layers'], self.hps['total_layers'][0], self.hps['total_layers'][1])
                        child['total_layers'] = round(child['total_layers'])
                    if Prob.should_do(Prob.p_ea_mutate_param):
                        child['nodes_per_hidden_layer'] = EAGGA.ea_mutate_gaussian(child['nodes_per_hidden_layer'], self.hps['nodes_per_hidden_layer'][0], self.hps['nodes_per_hidden_layer'][1])
                        child['nodes_per_hidden_layer'] = round(child['nodes_per_hidden_layer'])

            # GGA on group structure
            if Prob.should_do(Prob.p_gga_crossover):  # crossover
                gs_1 = child_1['group_structure']
                gs_2 = child_2['group_structure']
                gs_1, gs_2 = GroupStructure.gga_crossover(gs_1, gs_2)
                child_1['group_structure'] = gs_1
                child_2['group_structure'] = gs_2
            for child in [child_1, child_2]:  # mutate
                if Prob.should_do(Prob.p_gga_mutate_overall):
                    child['group_structure'].gga_mutate()

            offspring.append(child_1)
            offspring.append(child_2)

        return offspring[:self.hps['lambda']]  # in case offsping is longer than lambda (while-loop always adds 2 individuals), discard offspring after lambda
    

    def binary_tournament(self):
        '''
        steps
        (1) sample (without replacement) two random ids from population
        (2) non-dominated sorting -> rank pareto fronts
        (3)
        (a) if the sampled ids are in differently ranked pareto fronts -> return id with lower pareto front rank
        (b) else (sampled ids are from same pareto front)
        (i)     compute crowding distance
        (ii)    return id with larger crowding distance
        '''
        # (1)
        ids = np.random.choice(a=len(self.population), size=2, replace=False)#.tolist()
        # (2) ranks_nds taken from EA loop
        # (3)
        if self.population[ids[0]]['rank_nds'] != self.population[ids[1]]['rank_nds']:  # (a)
            return sorted(ids, key=lambda id: self.population[id]['rank_nds'])[0]  # ascending order, lower ranks are better
        else:  # (b)
            # skip (3, b, i+ii) for now, TODO: if time -> implement crowding distance, computation cf. https://de.mathworks.com/matlabcentral/fileexchange/65809-on-the-calculation-of-crowding-distance
            return ids[0]
        

    def autosave(self, gen):
        if self.file_path is not None:
            file_path = EAGGA.create_file_path(self.file_path)
            Path(file_path).mkdir(parents=True, exist_ok=True)

            print(f'Autosaving generation {gen + 1} to {file_path}')

            self.save_population(gen)
    

    def save_population(self, gen):
        population = deepcopy(self.population)
        offspring = deepcopy(self.offspring)
        for individual in population:
            individual['group_structure'] = individual['group_structure'].to_dict()
        for individual in offspring:
            individual['group_structure'] = individual['group_structure'].to_dict()

        file_content = {
            'population': population,
            'offspring': offspring
        }
        with open(EAGGA.create_file_path(os.path.join(self.file_path, f'gen-{gen}.json')), 'w') as f:
            json.dump(file_content, f)
    

    def load_population(self, gen):
        with open(EAGGA.create_file_path(os.path.join(self.file_path, f'gen-{gen}.json')), 'r') as f:
            file_content = json.load(f)
        
        self.population = file_content['population']
        self.offspring = file_content['offspring']
        for individual in self.population:
            individual['group_structure'] = GroupStructure.from_dict(individual['group_structure'])
        for individual in self.offspring:
            individual['group_structure'] = GroupStructure.from_dict(individual['group_structure'])


    @staticmethod
    def create_file_path(file_path):
        return file_path.replace('\\', '/')
    

    @staticmethod
    def ea_mutate_gaussian(val, lower, upper):
        val = (val - lower) / (upper - lower)
        val = np.random.normal(loc=val, scale=0.1, size=1).item()
        val = (val * (upper - lower)) + lower
        return min(max(val, lower), upper)