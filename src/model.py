import torch
import torch.nn as nn


def define_net_regression(trial, n_input_params, n_output_params):
    ''' chose number of hidden layers, number of hidden neurons and the activation function
    the input and output layer is assumed as linear layer with linear transformation
    two input features (torque and rotation), one output feature (efficiency)'''

    layers = []

    hidden_layers = trial.suggest_int(
        "hidden_layers", 2, 2)  # number of hidden layers
    print("number of hidden layers", hidden_layers)
    x_in = n_input_params

    # create hidden layers:
    for i in range(hidden_layers):
        # number of neurons in this layer
        hidden_neurons = trial.suggest_int(
            'hidden_neurons_' + str(i+1), (n_input_params*2-i)*2, 30)
        print("neurons in hidden layer [", i+1, "]", hidden_neurons)
        layers.append(nn.Linear(x_in, hidden_neurons))  # append this layer
        layers.append(nn.BatchNorm1d(hidden_neurons))
        act_func_name = trial.suggest_categorical(
            'act_func_' + str(i+1), ["ReLU"])  # , "Sigmoid", "Tanh"
        act_func = getattr(nn, act_func_name)()
        print("act_func[", i+1, "]", act_func)
        layers.append(act_func)
        x_in = hidden_neurons  # number of inputs for next layer = number of neurons of this layer

    layers.append(nn.Linear(x_in, n_output_params))  # output layer
    net = nn.Sequential(*layers)

    # Initialization: Xavier for weights, zeros for bias
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    return net