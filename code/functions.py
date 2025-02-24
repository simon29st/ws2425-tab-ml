import numpy as np

import torch
from sklearn.metrics import roc_auc_score

from classes import NeuralNetwork, Dataset, GroupStructure, Prob

from math import comb
from copy import deepcopy

from nds import ndomsort


def run_eagga_cv(mu, la, cv_inner, data_train_test, epochs: int, batch_size: int, weight_clipper=None):
    hp_bounds = {
        'total_layers': (3, 10),
        'nodes_per_hidden_layer': (3, 20)
    }

    population_layers = Prob.r_trunc_geom(Prob.p_sample_hps, mu, hp_bounds['total_layers'][0], hp_bounds['total_layers'][1])
    population_nodes = Prob.r_trunc_geom(Prob.p_sample_hps, mu, hp_bounds['nodes_per_hidden_layer'][0], hp_bounds['nodes_per_hidden_layer'][1])
    population = list()

    # call initial population offspring, makes looping easier in subsequent iterations (as we only evaluate offspring in each round and usually mu != lambda)
    offspring = [{
        'total_layers': population_layers[i].item(),
        'nodes_per_hidden_layer': population_nodes[i].item(),
        'group_structure': GroupStructure(  # TODO: init group structure with detectors
            {0, 1, 2, 3, 4, 5, 6, 7},
            {0, 1},
            [[2, 5], 1],
            [[4], 0],
            [[7, 3, 6], 1]
        )
    } for i in range(mu)]

    # evolutionary algorithm
    i_evolution = 0  # TODO: remove, used to stop after 3 evolutions
    while(True):  # TODO: define stopping criterion + remove break from end of loop
        print(f'Evolution {i_evolution+1}, evaluate {len(offspring)} individuals')
        for i, individual in enumerate(offspring):
            total_layers = individual['total_layers']
            nodes_per_hidden_layer = individual['nodes_per_hidden_layer']
            gs = individual['group_structure']

            print(f'running HPO for individual {i+1}/{len(offspring)}: {total_layers} total_layers, {nodes_per_hidden_layer} nodes per hidden layer')
            metrics = run_cv(cv_inner, data_train_test, total_layers, nodes_per_hidden_layer, gs, epochs, batch_size, weight_clipper)

            offspring[i]['metrics'] = metrics
        
        population += offspring
        ranks_nds = ndomsort.non_domin_sort(
            [individual['metrics']['mean'] for individual in population],
            get_objectives=lambda elem: (1 - elem[0], *[elem[i] for i in range(1, len(elem))]),  # compute pareto fronts w.r.t. reference (worst) point (0, 1, 1, 1)
            only_front_indices=True
        )
        for i in range(len(population)):
            population[i]['rank_nds'] = ranks_nds[i]
        population = sorted(population, key=lambda individual: individual['rank_nds'])[:mu]  # ascending order, lower ranks are better, then choose best mu individuals
        print(f'population: {population}')
        
        offspring = generate_offspring(la, population, ranks_nds, hp_bounds)
        for ind in offspring:
            print(ind, ind['group_structure'])

        # TODO: remove, used to stop after 3 evolutions
        if i_evolution > 1:
            break
        i_evolution += 1


def run_cv(cv, data_train_test, total_layers: int, nodes_per_hidden_layer: int, group_structure: GroupStructure, epochs: int, batch_size: int, weight_clipper=None):
    metrics = list()

    for i, (indices_train, indices_test) in enumerate(cv.split(X=data_train_test, y=data_train_test.loc[:, 'class'])):
        print(f'fold {i + 1}/{cv.get_n_splits()}')

        data_train = data_train_test.loc[indices_train, :]
        dataset_train = Dataset(
            X=data_train.loc[:, data_train.columns != 'class'],
            y=data_train.loc[:, 'class'],
            class_pos='tested_positive',
            group_structure=group_structure
        )

        data_test = data_train_test.loc[indices_test, :]
        dataset_test = Dataset(
            X=data_test.loc[:, data_test.columns != 'class'],
            y=data_test.loc[:, 'class'],
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
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train(optimizer, loss_fn, model, epochs, dataset_train, dataset_test, batch_size, weight_clipper)#, verbose=True)

        metrics.append(eval(model, dataset_test, batch_size))

        # TODO: remove debug metrics output
        print(metrics[-1])

    # TODO: no need to record NF, NI, NNM over folds and compute mean + var for them, as they stay the same in each fold
    return {
        'mean': np.mean(metrics, axis=0),
        'var': np.var(metrics, axis=0),
        'folds': metrics
    }


# cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train(optimizer, loss_fn, model: NeuralNetwork, epochs: int, dataset_train: Dataset, dataset_test: Dataset, batch_size: int, weight_clipper=None, verbose=False):
    if verbose:
        eval_loss = train_eval(loss_fn, model, dataset_test, batch_size)
        print(f'epoch 0/{epochs}, eval loss {eval_loss}')

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)  # TODO: set shuffle to True

    for epoch in range(epochs):
        model.train()  # training mode, put here as we eval at the end of each epoch
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

        if verbose:
            print(f'epoch {epoch + 1}/{epochs}, train loss {running_epoch_loss}')
            eval_loss = train_eval(loss_fn, model, dataset_test, batch_size)
            print(f'epoch {epoch + 1}/{epochs}, eval loss {eval_loss}')


# evaluate for training, i.e. compute loss on test set, actual evaluation (AUC, NF, NI, NNM) will be done in separate function 'eval'
# cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_eval(loss_fn, model: NeuralNetwork, dataset_test: Dataset, batch_size: int, verbose=False) -> float:
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)  # no need to shuffle in test
    
    model.eval()  # eval mode
    running_loss = 0

    for batch_input, batch_target in loader_test:
        with torch.no_grad():
            batch_output = model(*batch_input).flatten()
            
        batch_loss = loss_fn(batch_output, batch_target)
        running_loss += batch_loss.detach().item()
    running_loss /= len(loader_test)

    if verbose:
        print(f'eval loss {running_loss}')
    return running_loss


def eval(model: NeuralNetwork, dataset_test: Dataset, batch_size: int) -> tuple:
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    
    model.eval()

    batch_predictions = list()
    batch_targets = list()
    for batch_input, batch_target in loader_test:
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
    ni = 0  # via gs:
    # for each group with > 1 features: sum over range from 1 to (# of features in group - 1) -> # handshakes in party -> n choose 2
    # then divide by # of all possible interactions (all_features choose 2)
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


def generate_offspring(la, population, ranks_nds, hp_bounds):
    offspring = list()

    while len(offspring) <= la:
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

    return offspring[:la]  # in case offsping is longer than la (while always adds 2 offspring)


def ea_mutate_gaussian(val, lower, upper):
    val = (val - lower) / (upper - lower)
    val = np.random.normal(loc=val, scale=0.1, size=1).item()
    val = (val * (upper - lower)) + lower
    return min(max(val, lower), upper)