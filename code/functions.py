import numpy as np

import torch
from torch import nn

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from classes import NeuralNetwork, Dataset, GroupStructure, Prob

from math import comb
from copy import deepcopy
from datetime import datetime, timedelta

from nds import ndomsort


def run_eagga_cv(mu, lambd, cv_k, data_train_val, categorical_indicator, epochs: int, batch_size: int, patience: int, weight_clipper=None, secs_per_fold: int = 5 * 60 * 60, secs_total: int = 30 * 60 * 60):
    # inner split
    cv_inner = StratifiedKFold(
        n_splits=cv_k,
        shuffle=False  # TODO: set to True
    )

    hp_bounds = {
        'total_layers': (3, 10),
        'nodes_per_hidden_layer': (3, 20)
    }

    population_layers = Prob.r_trunc_geom(Prob.p_sample_hps, mu, hp_bounds['total_layers'][0], hp_bounds['total_layers'][1])
    population_nodes = Prob.r_trunc_geom(Prob.p_sample_hps, mu, hp_bounds['nodes_per_hidden_layer'][0], hp_bounds['nodes_per_hidden_layer'][1])

    all_features = list(i for i in range(len(data_train_val.columns) - 1))
    population_features_included = [GroupStructure.detector_features(data_train_val, categorical_indicator) for _ in range(mu)]
    population_features_excluded = [list(set(all_features) - set(features_included)) for features_included in population_features_included]
    population_interactions = [GroupStructure.detector_interactions(data_train_val, features_included) for features_included in population_features_included]
    population_monotonicity_constraints = [GroupStructure.detector_monotonicity(data_train_val, groups_without_monotonicity) for groups_without_monotonicity in population_interactions]

    population = list()

    # set offspring = initial population, makes looping easier in subsequent iterations, this way we can use a single for loop for both initial population + subsequent offspring (as we only evaluate offspring in each round)
    offspring = [{
        'total_layers': population_layers[i].item(),
        'nodes_per_hidden_layer': population_nodes[i].item(),
        'group_structure': GroupStructure(
            all_features,
            population_monotonicity_constraints[i][1],
            population_features_excluded[i],
            *population_monotonicity_constraints[i][0]
        )
    } for i in range(mu)]

    print('initial population')
    [print(f'total layers {individual['total_layers']}, nodes_per_hidden_layer {individual['nodes_per_hidden_layer']}, gs: {individual['group_structure']}') for individual in offspring]

    
    time_start = datetime.now()
    print(f'start EA at {time_start.isoformat()}')

    # evolutionary algorithm
    i_evolution = 0
    while(datetime.now() < time_start + timedelta(seconds=secs_total)):
        print(f'Evolution {i_evolution+1}, evaluate {len(offspring)} individuals')
        for i, individual in enumerate(offspring):
            total_layers = individual['total_layers']
            nodes_per_hidden_layer = individual['nodes_per_hidden_layer']
            gs = individual['group_structure']

            print(f'running HPO for individual {i+1}/{len(offspring)}: {total_layers} total_layers, {nodes_per_hidden_layer} nodes per hidden layer')
            metrics = run_cv(cv_inner, data_train_val, total_layers, nodes_per_hidden_layer, gs, epochs, batch_size, patience, weight_clipper, secs_per_fold)

            offspring[i]['metrics'] = metrics

            if datetime.now() >= time_start + timedelta(seconds=secs_total):
                break
        
        population += offspring
        ranks_nds = ndomsort.non_domin_sort(
            [individual['metrics']['performance']['mean'] for individual in population],
            get_objectives=lambda elem: (1 - elem[0], *[elem[i] for i in range(1, len(elem))]),  # compute pareto fronts w.r.t. reference (worst) point (0, 1, 1, 1)
            only_front_indices=True
        )
        for i in range(len(population)):
            population[i]['rank_nds'] = ranks_nds[i]
        population = sorted(population, key=lambda individual: individual['rank_nds'])[:mu]  # ascending order, lower ranks are better, then choose best mu individuals
        print(f'population: {population}')
        
        offspring = generate_offspring(lambd, population, ranks_nds, hp_bounds)
        for ind in offspring:
            print(ind, ind['group_structure'])

        i_evolution += 1
    
    print(f'finished EA at {datetime.now().isoformat()}')


def run_cv(cv, data_train_val, total_layers: int, nodes_per_hidden_layer: int, group_structure: GroupStructure, epochs: int, batch_size: int, patience: int, weight_clipper=None, secs: int = 30 * 60 * 60):
    metrics = {
        'performance': list(),
        'epochs': list()
    }

    for fold, (indices_train, indices_val) in enumerate(cv.split(X=data_train_val, y=data_train_val.loc[:, 'class'])):
        print(f'fold {fold + 1}/{cv.get_n_splits()}', end=' | ')  # TODO: remove

        data_train_stop_early = data_train_val.loc[indices_train, :]
        data_train, data_stop_early = train_test_split(
            data_train_stop_early,
            train_size=0.8,
            shuffle=True,
            stratify=data_train_stop_early.loc[:, 'class']
        )
        dataset_train = Dataset(
            X=data_train.loc[:, data_train.columns != 'class'],
            y=data_train.loc[:, 'class'],
            class_pos='tested_positive',
            group_structure=group_structure
        )
        dataset_stop_early = Dataset(
            X=data_stop_early.loc[:, data_stop_early.columns != 'class'],
            y=data_stop_early.loc[:, 'class'],
            class_pos='tested_positive',
            group_structure=group_structure
        )

        data_val = data_train_val.loc[indices_val, :]
        dataset_val = Dataset(
            X=data_val.loc[:, data_val.columns != 'class'],
            y=data_val.loc[:, 'class'],
            class_pos='tested_positive',
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

        model, optimal_epoch = train(optimizer, loss_fn, model, epochs, dataset_train, dataset_stop_early, batch_size, patience, weight_clipper, secs)

        metrics['performance'].append(eval(model, dataset_val, batch_size))
        metrics['epochs'].append(optimal_epoch)
        print(metrics['performance'][-1], metrics['epochs'][-1])  # TODO: remove debug metrics output

    # TODO: no need to record NF, NI, NNM over folds and compute mean + var for them, as they stay the same in each fold
    return {
        'performance': {
            'mean': np.mean(metrics['performance'], axis=0),
            'var': np.var(metrics['performance'], axis=0),
            'folds': metrics['performance']
        },
        'epochs': {
            'mean': np.mean(metrics['epochs'], axis=0),
            'folds': metrics['epochs']
        }
    }


# cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train(optimizer, loss_fn, model: NeuralNetwork, epochs: int, dataset_train: Dataset, dataset_stop_early: Dataset, batch_size: int, patience: int, weight_clipper=None, secs: int = 30 * 60 * 60) -> tuple:
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)  # TODO: set shuffle to True

    epoch = 0
    early_stop_loss_history = list()
    time_start = datetime.now()
    while(datetime.now() < time_start + timedelta(seconds=secs)):
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

        if weight_clipper is not None:  # used for monotonicity constraint
            for i, (_, monotonicity_constraint) in enumerate(model.get_group_structure().get_included_groups()):
                if monotonicity_constraint == 1:
                    model.networks[i].apply(weight_clipper)  # applies weight clipper recursively to network + its children, cf. https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply
        
        should_stop_early, optimal_epoch = stop_early(early_stop_loss_history, model, loss_fn, dataset_stop_early, batch_size, patience)
        if should_stop_early:
            return model, optimal_epoch
        epoch += 1
    
    return model, epoch


def stop_early(loss_history: list, model: NeuralNetwork, loss_fn, dataset_stop_early: Dataset, batch_size: int, patience: int = 10) -> tuple:
    # cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    loader_stop_early = torch.utils.data.DataLoader(dataset_stop_early, batch_size=batch_size)  # no need to shuffle in eval
    
    model.eval()  # eval mode
    running_loss = 0

    for batch_input, batch_target in loader_stop_early:
        with torch.no_grad():
            batch_output = model(*batch_input).flatten()
            
        batch_loss = loss_fn(batch_output, batch_target)
        running_loss += batch_loss.detach().item()
    running_loss /= len(loader_stop_early)

    # stopping criterion: mean of >patience< previous losses < current loss
    # if True -> go back to min loss within [t-patience, t]

    loss_history.append(running_loss)
    #print(f'stop_early loss_history: {loss_history}')

    if len(loss_history) <= patience:
        #print('stop_early len(loss_history) <= patience')
        return False, -1
    elif np.mean(loss_history[-patience-1:-1]) < running_loss:
        #print(f'stop_early loss_history[-patience-1:-1]: {loss_history[-patience-1:-1]}')
        mask = np.ones_like(loss_history)
        mask[:-patience-1] = np.inf
        optimal_epoch = np.argmin(mask * loss_history) + 1  # +1 to make it 1-based (as opposed to 0-based from np.argmin indexing)
        print(f'stop early: {np.mean(loss_history[-patience-1:-1])} < {running_loss}, optimal epoch {optimal_epoch}', end=' | ')
        return True, optimal_epoch
    
    #print(f'stop_early loss_history[-patience-1:-1]: {loss_history[-patience-1:-1]}')
    return False, -1


def eval(model: NeuralNetwork, dataset_eval: Dataset, batch_size: int) -> tuple:
    loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size)
    
    model.eval()  # eval mode

    batch_predictions = list()
    batch_targets = list()
    for batch_input, batch_target in loader_eval:
        with torch.no_grad():
            batch_output = model(*batch_input).flatten()
        batch_prediction = torch.sigmoid(batch_output)

        batch_predictions.append(batch_prediction)
        batch_targets.append(batch_target)
    predictions = torch.cat(batch_predictions)
    targets = torch.cat(batch_targets)

    auc = roc_auc_score(targets, predictions)
    
    gs = model.get_group_structure()
    num_features_all = len(gs.get_all_features())
    num_features_included = len(gs.get_included_features())
    num_features_unconstrained = len(gs.get_unconstrained_features())

    nf = num_features_included / num_features_all  # via gs: relative # of included features
    ni = 0  # via group structure:
    #   for each group with > 1 features: sum over range from 1 to (# of features in group - 1) -> # handshakes in party -> n choose 2
    #   then divide by # of all possible interactions (all_features choose 2)
    for (features, _) in gs.get_included_groups():
        num_features_in_group = len(features)
        if num_features_in_group > 1:
            #ni += sum(range(1, num_features_in_group))
            ni += comb(num_features_in_group, 2)
    ni /= comb(num_features_all, 2)
    nnm = num_features_unconstrained / num_features_all  # via gs: (# of features in groups without monotonicity constraint) / (total # of features)
    
    return auc.item(), nf, ni, nnm


def binary_tournament(population, ranks_nds):
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
    ids = np.random.choice(a=len(population), size=2, replace=False)#.tolist()
    # (2) ranks_nds taken from EA loop
    # (3)
    if ranks_nds[ids[0]] != ranks_nds[ids[1]]:  # (a)
        return sorted(ids, key=lambda id: ranks_nds[id])[0]  # ascending order, lower ranks are better
    else:  # (b)
        # skip (3, b, i+ii) for now, TODO: if time -> implement crowding distance, computation cf. https://de.mathworks.com/matlabcentral/fileexchange/65809-on-the-calculation-of-crowding-distance
        return ids[0]


def generate_offspring(lambd, population, ranks_nds, hp_bounds):
    offspring = list()

    while len(offspring) <= lambd:
        id_parent_1 = binary_tournament(population, ranks_nds)
        id_parent_2 = binary_tournament(population, ranks_nds)
        parent_1 = population[id_parent_1]
        parent_2 = population[id_parent_2]

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
                    child['total_layers'] = ea_mutate_gaussian(child['total_layers'], hp_bounds['total_layers'][0], hp_bounds['total_layers'][1])
                    child['total_layers'] = round(child['total_layers'])
                if Prob.should_do(Prob.p_ea_mutate_param):
                    child['nodes_per_hidden_layer'] = ea_mutate_gaussian(child['nodes_per_hidden_layer'], hp_bounds['nodes_per_hidden_layer'][0], hp_bounds['nodes_per_hidden_layer'][1])
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

    return offspring[:lambd]  # in case offsping is longer than lambda (while always adds 2 offspring)


def ea_mutate_gaussian(val, lower, upper):
    val = (val - lower) / (upper - lower)
    val = np.random.normal(loc=val, scale=0.1, size=1).item()
    val = (val * (upper - lower)) + lower
    return min(max(val, lower), upper)