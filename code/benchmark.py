from torch import nn


# TODO: add #groups argument and split the features to their respective nets like so: https://discuss.pytorch.org/t/grouping-some-features-before-going-to-fully-connected-layers/162444/5
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, total_layers, nodes_per_hidden_layer):
        super().__init__()
        
        self.net = nn.Sequential()

        # input layer
        self.net.append(nn.Linear(input_size, nodes_per_hidden_layer))
        self.net.append(nn.ReLU())

        # hidden layers
        for _ in range(1, total_layers-1):
            self.net.append(nn.Linear(nodes_per_hidden_layer, nodes_per_hidden_layer))
            self.net.append(nn.ReLU())

        # output layer
        self.net.append(nn.Linear(nodes_per_hidden_layer, output_size))
        if total_layers == 1:
            self.net.append(nn.Sigmoid())
        else:
            self.net.append(nn.Softmax())
    

    def forward(self, x):
        return self.net(x)


def eval_nn_eagga(task_train, task_test):
    pass


if __name__ == '__main__':
    eval_nn_eagga()
    nn = NeuralNetwork(10, 1, 5, 10)
    print(nn)