# src/train.py - Corrected and Unified

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
import numpy as np
import os
# from src.model import define_net_regression # Assuming this is correctly imported
import yaml
import copy
import queue
import threading
from streamlit.runtime.state import SessionStateProxy # For type hinting

# Assuming define_net_regression is defined or imported correctly.
# We'll use a placeholder import for the final script structure.
from src.model import define_net_regression


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEW GLOBAL TRACKING ---
BEST_MODEL_STATE = None
BEST_LOSS = float('inf')
TRIALS_DONE = 0
# ---------------------------

def send_update(update_queue: queue.Queue, key, value):
    """Helper to send a structured update to the Streamlit main thread via the queue."""
    update_queue.put({'key': key, 'value': value})

# (stop_callback, data_crossvalidation, get_loss_function remain unchanged)

def data_crossvalidation(n_input_params, n_output_params, train, validate, batch_size):
    # ... (function body remains the same) ...
    input_train = np.array([row[:n_input_params] for row in train], dtype='float32')
    output_train = np.array([row[n_input_params:n_input_params+n_output_params] for row in train], dtype='float32')
    input_validate = np.array([row[:n_input_params] for row in validate], dtype='float32')
    output_validate = np.array([row[n_input_params:n_input_params+n_output_params] for row in validate], dtype='float32')
 
    input_train = torch.from_numpy(input_train).to(Device)
    output_train = torch.from_numpy(output_train).to(Device)
    input_validate = torch.from_numpy(input_validate).to(Device)
    output_validate = torch.from_numpy(output_validate).to(Device)

    train_data_loader = DataLoader(TensorDataset(input_train, output_train), batch_size, shuffle=True, drop_last=True)
    validation_data_loader = DataLoader(TensorDataset(input_validate, output_validate), batch_size, shuffle=False)

    return train_data_loader, validation_data_loader, output_validate


def get_loss_function(config):
    loss_fn_name = config.get("loss_function", {}).get("type", "MSELoss")
    
    # 1. Convert to the standard functional form (snake_case)
    # E.g., 'MSELoss' -> 'mse_loss' (or just 'mse_loss' if it's in config)
    snake_case_name = loss_fn_name.lower().replace("loss", "_loss")
    
    if hasattr(F, snake_case_name):
        return getattr(F, snake_case_name)
    elif hasattr(F, loss_fn_name.lower()):
        return getattr(F, loss_fn_name.lower()) # For things like 'relu'
    else:
        # Check for the loss function name directly if the above failed
        print(f"Warning: Loss function '{loss_fn_name}' not found. Defaulting to F.mse_loss.")
        return F.mse_loss


# --- CRITICAL FIX 1: Simplify and unify the crossvalidation signature ---
def crossvalidation(trial, model_builder, dataset, config, update_queue: queue.Queue, stop_event: threading.Event):
    """
    Optuna objective function.
    Performs K-Fold CV, saves the best model found so far, and returns the average validation loss.
    
    Arguments are ordered to match the objective lambda.
    """
    global BEST_MODEL_STATE, BEST_LOSS, TRIALS_DONE
    TRIALS_DONE += 1
    
    print(f"\nTrial {trial.number}: ", end="")
    
    # Get parameters from dataset object (simplifies signature)
    n_input_params = dataset.n_input_params
    n_output_params = dataset.n_output_params

    # Parse parameters from config
    kfold_splits = config.get("cross_validation", {}).get("kfold", 5)
    loss_function = get_loss_function(config)
    
    # ... (Hyperparameter suggestions) ...
    hyperparams_config = config.get("hyperparameter_search_space", {})
    learning_rate_config = hyperparams_config.get("learning_rate", {})
    batch_size_config = hyperparams_config.get("batch_size", {})
    epochs_config = hyperparams_config.get("epochs", {})
    optimizer_config = config.get("optimizer_name", {}) # Use config directly if not in search space

    # Define Model and Hyperparameters using Optuna trial suggestions
    try:
        # Use the passed model_builder function
        model_template = model_builder(trial, n_input_params, n_output_params).to(Device)
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
    # Correctly suggesting optimizer name using config defaults
    optimizer_name = trial.suggest_categorical("opt", optimizer_config.get("choices", ["Adam"]))
    
    data_train = dataset.full_data
    
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
    fold_losses = []
    global_step = 0

    # K-Fold Loop
    for fold_idx, (train_idx, validate_idx) in enumerate(kfold.split(data_train)):
        
        fold_model = copy.deepcopy(model_template).to(Device)
        optimizer = getattr(torch.optim, optimizer_name)(fold_model.parameters(), lr=learning_rate)

        train_subset = data_train[train_idx]
        validate_subset = data_train[validate_idx]

        train_loader, val_loader, _ = data_crossvalidation(
            n_input_params, n_output_params, train_subset, validate_subset, batch_size
        )
        
        last_epoch_val_loss = 0.0

        for epoch in range(epochs):
            # --- CRITICAL FIX 2: Replace UNSAFE st_state access with queue ---
            if stop_event.is_set():
                # Use thread-safe queue:
                send_update(update_queue, 'log_messages', f"Stop signal received during Trial {trial.number}, Fold {fold_idx}. Pruning trial.")
                # Gracefully stop this trial
                raise optuna.exceptions.TrialPruned()
            
            # ... (Rest of training loop remains the same) ...
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
                last_epoch_val_loss = val_loss_sum / len(val_loader)

            # Report to Optuna for pruning
            trial.report(last_epoch_val_loss, global_step)
            global_step += 1
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        

        fold_losses.append(last_epoch_val_loss)

    # compute average of the final validation loss from all folds
    avg_loss = np.mean(fold_losses)
    print(f"Avg Loss: {avg_loss:.6f}")
    if trial.number == 0:
        # If it's the very first trial, the current loss is the best so far.
        current_best_loss = float('inf') 
    else:
        # For Trial 1 and beyond, Optuna will have completed trials and best_value is safe.
        # We need to access the best_value safely, assuming the study object is available
        # and has at least one completed trial (Trial 0).
        try:
            current_best_loss = trial.study.best_value
        except ValueError:
            # Fallback for extreme cases (shouldn't happen after Trial 0 fix)
            current_best_loss = float('inf')

    # --- NEW: Check and save the best intermediate model (uses send_update correctly) ---
    if avg_loss < current_best_loss:
        # Update BEST_LOSS logic must now use the queue to tell the UI
        # and rely on Optuna's internal study.best_value/best_trial
        
        send_update(update_queue, 'best_loss_so_far', avg_loss)
        send_update(update_queue, 'best_params_so_far', trial.params)
        send_update(update_queue, 'log_messages', f"🎉 New best loss: {avg_loss:.6f}. Saving intermediate model...")
        
        # Save the model state dict (this now only needs to be local/saved to disk)
        fold_model_state = fold_model.state_dict() 
        save_path = os.path.join("models", "ANN_best_intermediate_model.pt")
        os.makedirs("models", exist_ok=True)
        torch.save(fold_model_state, save_path)
        print(f" (New Best Intermediate Model saved)")

    return avg_loss

def optimization(dataset, config, update_queue: queue.Queue, st_state: SessionStateProxy, stop_event: threading.Event):
    """
    Orchestrates the HPO using Optuna with real-time callbacks.
    """
    BEST_LOSS = float('inf')
    BEST_MODEL_STATE = None
    TRIALS_DONE = 0
    # Also reset the session state mirror for the UI:
    send_update(update_queue, 'best_loss_so_far', float('inf'))
    send_update(update_queue, 'best_params_so_far', {})
    
    # Define the callback function INSIDE optimization
    def optuna_callback(study: optuna.Study, trial: optuna.Trial):
        
        # 1. Update trial progress: USE QUEUE
        send_update(update_queue, 'current_trial_number', trial.number)
        
        # 2. Check for stop signal (between trials)
        if stop_event.is_set():
            send_update(update_queue, 'log_messages', "Stop signal received. Stopping HPO study.")
            study.stop()
            return

        # 3. Log trial completion: USE QUEUE
        if trial.state == optuna.trial.TrialState.COMPLETE:
            send_update(update_queue, 'log_messages', f"Trial {trial.number} finished. Loss: {trial.value:.6f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            send_update(update_queue, 'log_messages', f"Trial {trial.number} pruned.")

        pass

    # --- End of Callback Definition ---

    # --- CRITICAL FIX 3: Correct objective lambda to pass ALL 6 arguments ---
    # The arguments must match the crossvalidation definition:
    # (trial, model_builder, dataset, config, update_queue, stop_event)
    objective = lambda trial: crossvalidation(
        trial, 
        define_net_regression, # 2. Model Builder
        dataset,               # 3. Dataset
        config,                # 4. Config
        update_queue,          # 5. Queue
        stop_event             # 6. Stop Event
        # NOTE: st_state is intentionally omitted here because crossvalidation no longer needs it.
    )
    
    study = optuna.create_study(direction="minimize")
    
    # Use config for n_trials (avoids the st_state crash)
    n_trials_safe = config.get('hyperparameter_search_space', {}).get('n_samples', 50) 
    
    study.optimize(
        objective, 
        n_trials = n_trials_safe, 
        callbacks=[optuna_callback]
    )
    
    if stop_event.is_set():
        send_update(update_queue, 'log_messages', "Optimization was stopped.")
        if study.best_trial:
            return study.best_trial.params
        else:
            return None

    send_update(update_queue, 'log_messages', "Optimization finished.")
    return study.best_params

def train_final_model(model, data_train, best_params, n_input_params, n_output_params, config):
    # ... (function body remains the same) ...
    print(f"Retraining final model on full dataset ({len(data_train)} samples)...")
    
    input_train = np.array([row[:n_input_params] for row in data_train], dtype='float32')
    output_train = np.array([row[n_input_params:n_input_params+n_output_params] for row in data_train], dtype='float32')
    
    input_train_tensor = torch.from_numpy(input_train).to(Device)
    output_train_tensor = torch.from_numpy(output_train).to(Device)
    
    full_dataset = TensorDataset(input_train_tensor, output_train_tensor)
    
    # Get training settings from best_params
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    optimizer_name = best_params['opt']
    epochs = best_params['epochs']
    
    # Get loss function from the config
    loss_function = get_loss_function(config)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Training Loop
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
        
    print("Final model retraining complete.")
    return model


def find_best_model(dataset):
    """
    NOTE: This function appears to be unused by your core pipeline. 
    It is kept here but would be executed by the main thread.
    """
    # Load config internally
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Run Hyperparameter Optimization
    # This call is likely incorrect for a main thread execution and should be removed 
    # if it's not being used for the pipeline.
    # The actual pipeline execution happens in run_training_pipeline.
    
    # ... (This logic should be handled by run_training_pipeline) ...

    # MODIFIED RETURN SIGNATURE (based on your intent)
    # return final_model, best_params, study 
    pass