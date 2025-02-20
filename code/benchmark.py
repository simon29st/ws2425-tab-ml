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

        # output layer
        layers.append(nn.Linear(nodes_per_hidden_layer, output_size))
        #if total_layers == 1:
        #    layers.append(nn.Sigmoid())
        #else:
        #    layers.append(nn.Softmax())
        
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


def eval_nn_eagga(task_train, task_test):
    pass


if __name__ == '__main__':
    eval_nn_eagga()
    nn = NeuralNetwork(10, 1, 5, 10)
    print(nn)