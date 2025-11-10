import torch
import torch.nn as nn
import yaml
import os
import optuna
# Assuming nn.BatchNorm1d is correct for your data shape.
# If your input data is 2D (batch_size, features), BatchNorm1d is typically correct.

# Load config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: config.yaml not found at {config_path}")
    config = {}

network_config = config.get("network", {})

def define_net_regression(params_or_trial, n_input_params, n_output_params):
    """
    Create a neural network architecture.
    
    The first argument can be:
    1. An Optuna Trial object (during hyperparameter search).
    2. A dictionary of fixed best parameters (during final model creation).
    """

    layers = []
    x_in = n_input_params
    
    # Determine the input type
    is_trial = isinstance(params_or_trial, optuna.trial.Trial)
    source = params_or_trial # Either the Trial object or the parameters dict

    # --- 1. Get Number of Hidden Layers ---
    hl_low = network_config.get("hidden_layers", {}).get("low", 2)
    hl_high = network_config.get("hidden_layers", {}).get("high", 2)
    
    if is_trial:
        hidden_layers = source.suggest_int("hidden_layers", hl_low, hl_high)
    else:
        # Check if the key exists before accessing
        if "hidden_layers" not in source:
             raise ValueError("Required parameter 'hidden_layers' missing from best_params dictionary.")
        hidden_layers = source["hidden_layers"]
        
    print(f"Model Definition - Hidden Layers: {hidden_layers}")

    # --- 2. Build Hidden Layers ---
    for i in range(hidden_layers):
        
        # Hidden neurons
        hn_low = network_config.get("hidden_neurons", {}).get("low", (n_input_params * 2 - i) * 2)
        hn_high = network_config.get("hidden_neurons", {}).get("high", 30)
        hn_key = f"hidden_neurons_{i+1}"
        
        # Activation function
        act_choices = network_config.get("activation_functions", {}).get("choices", ["ReLU"])
        act_key = f"act_func_{i+1}"
        
        if is_trial:
            hidden_neurons = source.suggest_int(hn_key, hn_low, hn_high)
            act_func_name = source.suggest_categorical(act_key, act_choices)
        else:
            # Check keys
            if hn_key not in source or act_key not in source:
                 raise ValueError(f"Required parameter '{hn_key}' or '{act_key}' missing from best_params dictionary.")
            hidden_neurons = source[hn_key]
            act_func_name = source[act_key]

        print(f"Layer [{i+1}] - Neurons: {hidden_neurons}, Activation: {act_func_name}")
        
        # Layer construction
        layers.append(nn.Linear(x_in, hidden_neurons))
        layers.append(nn.BatchNorm1d(hidden_neurons))
        
        act_func = getattr(nn, act_func_name)()
        layers.append(act_func)

        x_in = hidden_neurons

    # --- 3. Output Layer ---
    layers.append(nn.Linear(x_in, n_output_params))
    net = nn.Sequential(*layers)
    

    # --- 4. Initialize Weights and Biases ---
    weight_init = network_config.get("weight_initialization", "xavier_uniform")
    bias_init = network_config.get("bias_initialization", "zeros")

    for m in net:
        if isinstance(m, nn.Linear):
            # Weight initialization
            if weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif weight_init == "kaiming_uniform":
                # Kaiming is often preferred for ReLU activations
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # Add more schemes if needed

            # Bias initialization
            if m.bias is not None:
                if bias_init == "zeros":
                    nn.init.zeros_(m.bias)
                elif bias_init == "ones":
                    nn.init.ones_(m.bias)
                # Add more schemes if needed

    return net