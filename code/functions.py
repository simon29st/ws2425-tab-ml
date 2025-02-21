import torch
from sklearn.metrics import roc_auc_score

from classes import NeuralNetwork, Dataset, GroupStructure

from math import comb


def run_HPO_CV(cv_inner, data_train_test, epochs, batch_size, weight_clipper=None):
    # TODO: generate initial HPs for network
    hp_configs = {
        'total_layers': [3],
        'nodes_per_hidden_layer': [3]
    }
    # TODO: generate initial group structures
    gs = GroupStructure(
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1},
        ((2, 5), 1),
        ((4,), 0),
        ((7, 3, 6), 1)
    )

    while(True):  # TODO: define stopping criterion + remove break from end of loop
        total_layers = hp_configs['total_layers'][-1]
        nodes_per_hidden_layer = hp_configs['nodes_per_hidden_layer'][-1]

        print(f'running HPO: {total_layers} total_layers, {nodes_per_hidden_layer} nodes per hidden layer')

        run_CV(cv_inner, data_train_test, total_layers, nodes_per_hidden_layer, gs, epochs, batch_size, weight_clipper)

        # TODO: evaluate performance + add to auc, NI, NF, NNM

        # TODO: EA on hp_total_layers + hp_nodes
        # TODO: GGA on group structure

        # until stopping criterion is met
        break


def run_CV(cv, data_train_test, total_layers, nodes_per_hidden_layer, group_structure, epochs, batch_size, weight_clipper=None):
    for i, (indices_train, indices_test) in enumerate(cv.split(X=data_train_test, y=data_train_test.loc[:, 'class'])):
        print(f'fold {i + 1} / {cv.get_n_splits()}')

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

        train(optimizer, loss_fn, model, epochs, dataset_train, dataset_test, batch_size, weight_clipper, verbose=True)

        metrics = eval(model, dataset_test, batch_size)

        # TODO: remove debug output, this is to check network for non-neg weights (for monotonicity constraint)
        print(metrics)
        print(model)
        for i, network in enumerate(model.networks):
            print(f'Network {i+1}')
            for j, layer in enumerate(network):
                if hasattr(layer, 'weight'):
                    print(f'Layer {j+1}')
                    print(layer.weight, layer.weight.shape)


# cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train(optimizer, loss_fn, model, epochs, dataset_train, dataset_test, batch_size, weight_clipper=None, verbose=False):
    if verbose:
        eval_loss = train_eval(loss_fn, model, dataset_test, batch_size)
        print(f'epoch 0 / {epochs}, eval loss {eval_loss}')

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
                    model.networks[i].apply(weight_clipper)  # will apply weight clipper recursively to network + its children, cf. https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply

        if verbose:
            print(f'epoch {epoch + 1} / {epochs}, train loss {running_epoch_loss}')
            eval_loss = train_eval(loss_fn, model, dataset_test, batch_size)
            print(f'epoch {epoch + 1} / {epochs}, eval loss {eval_loss}')


# evaluate for training, i.e. compute loss on test set, actual evaluation (AUC, NF, NI, NNM) will be done in separate function 'eval'
# cf. https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_eval(loss_fn, model, dataset_test, batch_size, verbose=False):
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


def eval(model: NeuralNetwork, dataset_test: Dataset, batch_size: int):
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


def eval_nn_eagga(task_train, task_test):
    pass