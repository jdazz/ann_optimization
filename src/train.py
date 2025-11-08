import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
import numpy as np
import pickle, os
from src.model import define_net_regression
from .model_test import unseen_test
import yaml

config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get configurations parameters
kfold_splits = config.get("cross_validation", {}).get("kfold", 5)
generate_plots = config.get("generate_plot", True)
r2_target = config.get("targets", {}).get("r2_target", 0.99)
nmae_target = config.get("targets", {}).get("nmae_target", 0.08)
mre_threshold = config.get("targets", {}).get("mre_threshold", 25)

loss_fn_name = config.get("loss_function", {}).get("type", "MSELoss")
if hasattr(F, loss_fn_name.lower()):  # handle 'MSELoss'
    loss_function = getattr(F, loss_fn_name.lower())
else:
    loss_function = F.mse_loss  # fallback

# Hyperparameter search space
hyperparams_config = config.get("hyperparameter_search_space", {})
learning_rate_config = hyperparams_config.get("learning_rate", {})
batch_size_config = hyperparams_config.get("batch_size", {})
epochs_config = hyperparams_config.get("epochs", {})
optimizer_config = hyperparams_config.get("optimizer_name", {})


def data_crossvalidation(n_input_params, n_output_params, train, validate, batch_size):
    # split Subset back into Inputs and Outputs
    input_train = np.array([row[:n_input_params] for row in train], dtype='float32')
    output_train = np.array([row[n_input_params:n_input_params+n_output_params] for row in train], dtype='float32')
    input_validate = np.array([row[:n_input_params] for row in validate], dtype='float32')
    output_validate = np.array([row[n_input_params:n_input_params+n_output_params] for row in validate], dtype='float32')

    # Convert to Tensors
    input_train = torch.from_numpy(input_train).to(Device)
    output_train = torch.from_numpy(output_train).to(Device)
    input_validate = torch.from_numpy(input_validate).to(Device)
    output_validate = torch.from_numpy(output_validate).to(Device)

    train_data_loader = DataLoader(TensorDataset(input_train, output_train), batch_size, shuffle=True, drop_last=True)
    validation_data_loader = DataLoader(TensorDataset(input_validate, output_validate), batch_size, shuffle=False)

    return train_data_loader, validation_data_loader, output_validate


# === Crossvalidation ===
def crossvalidation(trial, n_input_params, n_output_params, data_train):
    print("Creating net")
    model = define_net_regression(trial, n_input_params, n_output_params).to(Device)

    learning_rate = trial.suggest_loguniform(
        "learning_rate", learning_rate_config.get("low", 0.0001), learning_rate_config.get("high", 0.01)
    )
    batch_size = trial.suggest_int(
        "batch_size", batch_size_config.get("low", 50), batch_size_config.get("high", 150)
    )
    epochs = trial.suggest_int(
        "epochs", epochs_config.get("low", 200), epochs_config.get("high", 200)
    )
    optimizer_name = trial.suggest_categorical("opt", optimizer_config.get("choices", ["Adam"]))
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        kfold = KFold(kfold_splits)
        for train_idx, validate_idx in kfold.split(data_train):
            train_subset = data_train[train_idx]
            validate_subset = data_train[validate_idx]

            train_loader, val_loader, output_validate = data_crossvalidation(
                n_input_params, n_output_params, train_subset, validate_subset, batch_size
            )

            # Training step
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_function(pred, y)
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    pred_validate = model(x)
                    accuracy = loss_function(pred_validate, y)

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Save intermediate model
    os.makedirs("models", exist_ok=True)
    torch.save(model, os.path.join("models", f"trial_{trial.number}.pt"))

    return accuracy


# === Optimization ===
def optimization(dataset, data_train):
    sampler_type = config.get("sampler", {}).get("type", "TPESampler")
    study_direction = config.get("study", {}).get("direction", "minimize")

    sampler = getattr(optuna.samplers, sampler_type)()
    study = optuna.create_study(sampler=sampler, direction=study_direction)
    study.optimize(
        lambda trial: crossvalidation(trial, dataset.n_input_params, dataset.n_output_params, data_train),
        n_trials=hyperparams_config.get("n_samples", 25)
    )

    # Load best model
    best_model_path = os.path.join("models", f"trial_{study.best_trial.number}.pickle")
    best_model_path = os.path.join("models", f"trial_{study.best_trial.number}.pt")
    best_model = torch.load(best_model_path)

    # Cleanup other trial models
    for file in os.listdir("models"):
        if file.startswith("trial_") and not file.endswith(f"{study.best_trial.number}.pickle"):
            os.remove(os.path.join("models", file))

    return best_model


# === Find best model ===
def find_best_model(dataset, train_subset, test_set):
    print("Running optimization...")
    best_model = optimization(dataset, train_subset)

    print("Saving best model...")
    best_model_path = os.path.join("models", "ANN_best_model.pt")
    torch.save(best_model, best_model_path)

    print("Testing best model on unseen data...")
    error_prob, nmae, r2 = unseen_test(test_set, best_model_path)

    print(f"Final Best NMAE: {nmae}")
    print(f"Final Best R2: {r2}")
    print(f"Final Best P(error<={mre_threshold}%): {error_prob}")