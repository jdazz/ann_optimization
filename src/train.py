import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
import numpy as np
import os
from src.model import define_net_regression
import yaml
import copy
from .model_test import test

config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get parameters from config
kfold_splits = config.get("cross_validation", {}).get("kfold", 5)
loss_fn_name = config.get("loss_function", {}).get("type", "MSELoss")
if hasattr(F, loss_fn_name.lower()):
    loss_function = getattr(F, loss_fn_name.lower())
else:
    loss_function = F.mse_loss

hyperparams_config = config.get("hyperparameter_search_space", {})
learning_rate_config = hyperparams_config.get("learning_rate", {})
batch_size_config = hyperparams_config.get("batch_size", {})
epochs_config = hyperparams_config.get("epochs", {})
optimizer_config = hyperparams_config.get("optimizer_name", {})




def data_crossvalidation(n_input_params, n_output_params, train, validate, batch_size):
    """Helper to create DataLoaders (Unchanged)"""
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


def crossvalidation(trial, n_input_params, n_output_params, dataset):
    """
    REWRITTEN: Optuna objective function.
    Performs K-Fold CV and returns the average validation loss.
    """
    print(f"\nTrial {trial.number}: ", end="")
    
    # --- 1. Define Model and Hyperparameters ---
    # Model template is defined *once* per trial using suggested params
    try:
        model_template = define_net_regression(trial, n_input_params, n_output_params).to(Device)
    except Exception as e:
        print(f"Model definition failed: {e}")
        raise optuna.exceptions.TrialPruned()

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
    
    data_train = dataset.full_data
    
    # KFold object is created *once* per trial
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
    fold_losses = []
    global_step = 0

    # --- 2. K-Fold Loop (OUTER loop) ---
    
    for fold_idx, (train_idx, validate_idx) in enumerate(kfold.split(data_train)):
        
        # --- 2a. Reset model and optimizer for each fold ---
        fold_model = copy.deepcopy(model_template).to(Device)
        optimizer = getattr(torch.optim, optimizer_name)(fold_model.parameters(), lr=learning_rate)

        train_subset = data_train[train_idx]
        validate_subset = data_train[validate_idx]

        train_loader, val_loader, _ = data_crossvalidation(
            n_input_params, n_output_params, train_subset, validate_subset, batch_size
        )
        
        last_epoch_val_loss = 0.0

        # --- 3. Epoch Loop (INNER loop) ---
        for epoch in range(epochs):
            fold_model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = fold_model(x)
                loss = loss_function(pred, y)
                loss.backward()
                optimizer.step()

            # Validation step
            fold_model.eval()
            with torch.no_grad():
                val_loss_sum = 0
                for x, y in val_loader:
                    pred_validate = fold_model(x)
                    val_loss_sum += loss_function(pred_validate, y).item()
                # Average validation loss for this epoch
                last_epoch_val_loss = val_loss_sum / len(val_loader)

            # Report to Optuna for pruning
            trial.report(last_epoch_val_loss, global_step)
            global_step += 1
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Store the final validation loss for this fold
        fold_losses.append(last_epoch_val_loss)

    # --- 4. Final Metric ---
    # The trial's value is the average of the final validation loss from all folds
    avg_loss = np.mean(fold_losses)
    print(f"Avg Loss: {avg_loss:.6f}")

    # --- 5. Return Metric ---
    # NO MODEL SAVING HERE
    return avg_loss


def optimization(dataset):
    """
    REWRITTEN: Runs HPO and returns the BEST HYPERPARAMETERS.
    """
    sampler_type = config.get("sampler", {}).get("type", "TPESampler")
    study_direction = config.get("study", {}).get("direction", "minimize")

    sampler = getattr(optuna.samplers, sampler_type)()
    study = optuna.create_study(sampler=sampler, direction=study_direction)
    
    print("Starting Optuna Hyperparameter Optimization...")
    study.optimize(
        lambda trial: crossvalidation(trial, dataset.n_input_params, dataset.n_output_params, dataset),
        n_trials=hyperparams_config.get("n_samples", 25)
    )

    print("\nOptimization Finished.")
    print(f"  Best trial number: {study.best_trial.number}")
    print(f"  Best value (avg loss): {study.best_value:.6f}")
    print(f"  Best parameters: {study.best_params}")

    # --- NO MODEL LOADING OR CLEANUP ---
    
    # Return the dictionary of best parameters
    return study.best_params, study.best_value


def train_final_model(model, data_train, best_params, n_input_params, n_output_params):
    """
    NEW: Helper function to train the final model on the full dataset.
    """
    print(f"Retraining final model on full dataset ({len(data_train)} samples)...")
    
    # 1. Create full dataset loader
    input_train = np.array([row[:n_input_params] for row in data_train], dtype='float32')
    output_train = np.array([row[n_input_params:n_input_params+n_output_params] for row in data_train], dtype='float32')
    
    input_train_tensor = torch.from_numpy(input_train).to(Device)
    output_train_tensor = torch.from_numpy(output_train).to(Device)
    
    full_dataset = TensorDataset(input_train_tensor, output_train_tensor)
    
    # 2. Get training settings from best_params
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    optimizer_name = best_params['opt']
    epochs = best_params['epochs']
    
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        
        # Optional: print training progress
        # if (epoch + 1) % 20 == 0:
        #     print(f"  Final training epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_train):.6f}")

    print("Final model retraining complete.")
    return model


def find_best_model(dataset):
    """
    REWRITTEN: Main function to orchestrate the entire pipeline.
    
    1. Runs HPO to find best params.
    2. Retrains a new model on the full dataset with those params.
    3. Saves and returns the final model.
    """
    # --- 1. Run Hyperparameter Optimization ---
    best_params, best_cv_loss = optimization(dataset)

    # --- 2. Create and Retrain Final Model ---
    print("\n--- Creating Final Model ---")
    # Create the final model using the best_params dictionary
    final_model = define_net_regression(
        best_params, 
        dataset.n_input_params, 
        dataset.n_output_params
    ).to(Device)

    # Train this model on the *entire* dataset.full_data
    final_model = train_final_model(
        final_model,
        dataset.full_data,
        best_params,
        dataset.n_input_params,
        dataset.n_output_params
    )
    
    # --- 3. Save Final Model ---
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "ANN_best_model.pt")
    
    # Save the model's STATE (weights), not the whole object
    torch.save(final_model.state_dict(), best_model_path)
    print(f"Final best model's weights saved to: {best_model_path}")

    # --- 4. Test Final Model (Optional but recommended) ---
    print("\n--- Testing Final Model on Unseen Data ---")
    # 'unseen_test' must also be adapted to use 'best_params' to build the model

    
    return final_model, best_params

