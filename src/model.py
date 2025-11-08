import torch
import torch.nn as nn
import yaml
import os

# Load config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

network_config = config.get("network", {})

def define_net_regression(trial, n_input_params, n_output_params):
    """
    Create a neural network with parameters read from config.yaml.
    Input/output layers are linear; hidden layers, neurons, activation functions, and initializations are configurable.
    """

    layers = []
    x_in = n_input_params

    # Get number of hidden layers from trial within config bounds
    hl_low = network_config.get("hidden_layers", {}).get("low", 2)
    hl_high = network_config.get("hidden_layers", {}).get("high", 2)
    hidden_layers = trial.suggest_int("hidden_layers", hl_low, hl_high)
    print("Number of hidden layers:", hidden_layers)

    for i in range(hidden_layers):
        # Hidden neurons
        hn_low = network_config.get("hidden_neurons", {}).get("low", (n_input_params*2-i)*2)
        hn_high = network_config.get("hidden_neurons", {}).get("high", 30)
        hidden_neurons = trial.suggest_int(f"hidden_neurons_{i+1}", hn_low, hn_high)
        print(f"Neurons in hidden layer [{i+1}]:", hidden_neurons)
        layers.append(nn.Linear(x_in, hidden_neurons))
        layers.append(nn.BatchNorm1d(hidden_neurons))

        # Activation function
        act_choices = network_config.get("activation_functions", {}).get("choices", ["ReLU"])
        act_func_name = trial.suggest_categorical(f"act_func_{i+1}", act_choices)
        act_func = getattr(nn, act_func_name)()
        print(f"Activation function [{i+1}]:", act_func)
        layers.append(act_func)

        x_in = hidden_neurons

    # Output layer
    layers.append(nn.Linear(x_in, n_output_params))
    net = nn.Sequential(*layers)

    # Initialize weights and biases based on config
    weight_init = network_config.get("weight_initialization", "xavier_uniform")
    bias_init = network_config.get("bias_initialization", "zeros")

    for m in net:
        if isinstance(m, nn.Linear):
            if weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            # Add more schemes if needed
            if bias_init == "zeros":
                nn.init.zeros_(m.bias)
            elif bias_init == "ones":
                nn.init.ones_(m.bias)

    return net