# src/train.py - Corrected and Unified

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
import numpy as np
import pandas as pd
import os
# from src.model import define_net_regression # Assuming this is correctly imported
import yaml
import copy
import queue
import threading

# Assuming define_net_regression is defined or imported correctly.
# We'll use a placeholder import for the final script structure.
from src.dataset import Dataset
from src.model import define_net_regression
from utils.save_onnx import export_to_onnx
from src.model_test import test as test_model
from utils.save_scaler import save_scaler_to_json
from utils.run_manager import append_run_log, update_best_metrics, zip_run_dir, write_value_file


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEW GLOBAL TRACKING ---
BEST_MODEL_STATE = None
BEST_LOSS = float('inf')
TRIALS_DONE = 0
BEST_CV_LOSS = float('inf')   # best (lowest) cross-validation loss so far
BEST_R2 = -float('inf')       # best test R² so far
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
def crossvalidation(trial, model_builder, dataset, config, update_queue: queue.Queue, stop_event: threading.Event, dataset_test, run_dir):
    """
    Optuna objective function.
    Performs K-Fold cross validation, saves the best model found so far, and returns
    the average validation loss for the trial.

    - The objective (what Optuna minimizes) is the average validation loss.
    - When a new best trial is found (by this loss), we:
        * Retrain the model on the full training data using train_final_model.
        * Save the intermediate best model (PT and optional ONNX).
        * Evaluate that model on the provided dataset_test via model_test.test,
          and send test metrics to the UI via update_queue.

    Arguments order matches the objective lambda from optimization().
    """
    global BEST_MODEL_STATE, BEST_LOSS, TRIALS_DONE, BEST_CV_LOSS, BEST_R2
    TRIALS_DONE += 1
    # Local logger that writes to both queue and summary.txt on disk.
    # This keeps summary.txt updated even if the Streamlit page reloads and
    # misses queue processing events.
    def log_and_record(message: str):
        append_run_log(run_dir, str(message))
        update_queue.put({"key": "log_messages", "value": message, "logged": True})

    print(f"\nTrial {trial.number}: ", end="")
    log_and_record(f"Trial {trial.number} started.")

    # 1. Basic setup
    # -------------------------------------------------------------------------
    n_input_params = dataset.n_input_params
    n_output_params = dataset.n_output_params

    cv_cfg = config.get("cross_validation", {})
    kfold_splits = cv_cfg.get("kfold", 5)
    loss_function = get_loss_function(config)

    hyperparams_config = config.get("hyperparameter_search_space", {})
    learning_rate_config = hyperparams_config.get("learning_rate", {})
    batch_size_config = hyperparams_config.get("batch_size", {})
    epochs_config = hyperparams_config.get("epochs", {})
    optimizer_config = hyperparams_config.get("optimizer_name", {})

    # -------------------------------------------------------------------------
    # 2. Define model + hyperparameters with Optuna
    # -------------------------------------------------------------------------
    try:
        # model_builder(trial, n_input, n_output) should return a torch.nn.Module
        model_template = model_builder(trial, n_input_params, n_output_params).to(Device)
    except Exception as e:
        print(f"Model definition failed: {e}")
        raise optuna.exceptions.TrialPruned()

    learning_rate = trial.suggest_loguniform(
        "learning_rate",
        learning_rate_config.get("low", 0.0001),
        learning_rate_config.get("high", 0.01),
    )
    batch_size = trial.suggest_int(
        "batch_size",
        batch_size_config.get("low", 50),
        batch_size_config.get("high", 150),
    )
    epochs = trial.suggest_int(
        "epochs",
        epochs_config.get("low", 200),
        epochs_config.get("high", 200),
    )
    optimizer_name = trial.suggest_categorical(
        "optimizer_name",
        optimizer_config.get("choices", ["Adam"]),
    )

    # `dataset.full_data` is [inputs | outputs] for training (train split only if used)
    data_train = dataset.full_data

    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
    fold_losses = []
    global_step = 0

    # -------------------------------------------------------------------------
    # 3. K-Fold loop
    # -------------------------------------------------------------------------
    for fold_idx, (train_idx, validate_idx) in enumerate(kfold.split(data_train)):
        fold_model = copy.deepcopy(model_template).to(Device)
        optimizer = getattr(torch.optim, optimizer_name)(fold_model.parameters(), lr=learning_rate)

        train_subset = data_train[train_idx]
        validate_subset = data_train[validate_idx]

        train_loader, val_loader, _ = data_crossvalidation(
            n_input_params,
            n_output_params,
            train_subset,
            validate_subset,
            batch_size,
        )

        last_epoch_val_loss = 0.0

        for epoch in range(epochs):
            # Cooperative stop between epochs
            if stop_event.is_set():
                send_update(
                    update_queue,
                    "log_messages",
                    f"Stop signal received during Trial {trial.number}, Fold {fold_idx}. Pruning trial.",
                )
                raise optuna.exceptions.TrialPruned()

            # -----------------------------
            # Training loop
            # -----------------------------
            fold_model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = fold_model(x)
                loss = loss_function(pred, y)
                loss.backward()
                optimizer.step()

            # -----------------------------
            # Validation loop
            # -----------------------------
            fold_model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                for x, y in val_loader:
                    pred_validate = fold_model(x)
                    val_loss_sum += loss_function(pred_validate, y).item()
                last_epoch_val_loss = val_loss_sum / max(1, len(val_loader))

            # Report to Optuna for pruning
            trial.report(last_epoch_val_loss, global_step)
            global_step += 1
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        fold_losses.append(last_epoch_val_loss)

    # -------------------------------------------------------------------------
    # 4. Aggregate loss and compare against current best
    # -------------------------------------------------------------------------
    avg_loss = np.mean(fold_losses)
    print(f"Avg Loss: {avg_loss:.6f}")

    # -------------------------------------------------------------------------
    # 5. CV-based candidate filter using global BEST_CV_LOSS (selection by CV only)
    # -------------------------------------------------------------------------
    if avg_loss < BEST_CV_LOSS:
        BEST_CV_LOSS = avg_loss

        # Notify UI about new best CV loss and params
        # send_update(update_queue, "best_loss_so_far", avg_loss)
        # send_update(update_queue, "best_params_so_far", trial.params)
        log_and_record(
            f"New best CV loss: {avg_loss:.6f}. Retraining candidate best model on full training data..."
        )

        # ------------------------------------------------------------------
        # A) RETRAIN MODEL ON FULL TRAINING DATA USING train_final_model
        # ------------------------------------------------------------------
        data_train_full = dataset.full_data      # shape: [N_train, n_input + n_output]
        n_samples = data_train_full.shape[0]

        if n_samples < 2:
            # Pathological small dataset: skip retrain to avoid BN issues
            log_and_record(
                "Training set has fewer than 2 samples. Skipping full-data retrain for this best model."
            )
            best_model_full = model_template  # fall back to template / last fold
        else:
            # Build a fresh model with the same hyperparameters
            best_model_full = model_builder(trial, n_input_params, n_output_params).to(Device)

            # Use your centralized training routine
            best_model_full = train_final_model(
                model=best_model_full,
                data_train=data_train_full,
                best_params=trial.params,
                n_input_params=n_input_params,
                n_output_params=n_output_params,
                config=config,
            )

            log_and_record(
                "Full-data training for new best model completed (via train_final_model)."
            )

        # ------------------------------------------------------------------
        # B) Evaluate candidate on TEST data (if available)
        # ------------------------------------------------------------------
        test_metrics = None
        r2 = -float("inf")
        if dataset_test is not None:
            try:
                import pandas as pd
                from torch.utils.data import TensorDataset, DataLoader

                # --- 1) Prepare test_data, n_input, n_output (like in test()) ---
                if hasattr(dataset_test, "full_data") and hasattr(dataset_test, "n_input_params"):
                    # Dataset-like object
                    test_data = dataset_test.full_data
                    n_input = dataset_test.n_input_params
                    n_output = dataset_test.n_output_params
                elif isinstance(dataset_test, pd.DataFrame):
                    # DataFrame: assume same column layout as training dataset
                    test_data = dataset_test.to_numpy(dtype="float32")
                    n_input = dataset.n_input_params
                    n_output = dataset.n_output_params
                else:
                    raise ValueError(
                        "dataset_test must be a Dataset-like object with 'full_data' "
                        "or a pandas DataFrame."
                    )

                # Match test() logic
                input_data = test_data[:, :n_input].astype("float32")
                output_data = test_data[:, n_input : n_input + n_output].astype("float32")

                # Align test data scaling with training pipeline so live metrics match final metrics
                scaler = getattr(dataset, "scaler", None)
                if getattr(dataset, "should_standardize", False) and scaler is not None:
                    try:
                        input_data = scaler.transform(input_data)
                        send_update(
                            update_queue,
                            "log_messages",
                            "Applied fitted scaler to test data for live metrics.",
                        )
                    except Exception as e:
                        send_update(
                            update_queue,
                            "log_messages",
                            f"⚠️ Failed to scale test data for live metrics: {e}",
                        )

                inputs_tensor = torch.from_numpy(input_data).to(Device)
                outputs_tensor = torch.from_numpy(output_data).to(Device)

                # Use testing settings from config (like model_test.test does via YAML)
                targets_cfg = config.get("targets", {})
                MRE_THRESHOLD = targets_cfg.get("mre_threshold", 25)
                TEST_BATCH_SIZE = config.get("testing", {}).get("batch_size", 1)

                test_loader = DataLoader(
                    TensorDataset(inputs_tensor, outputs_tensor),
                    batch_size=TEST_BATCH_SIZE,
                    shuffle=False,
                )

                # --- 2) Evaluation Loop (same as in test()) ---
                best_model_full.eval()
                mre_list, y_pred_list, y_true_list = [], [], []

                with torch.no_grad():
                    for x, y in test_loader:
                        y_pred = best_model_full(x)
                        y_true = y

                        y_pred_np = y_pred.cpu().numpy().flatten()
                        y_true_np = y_true.cpu().numpy().flatten()

                        rel_error = np.abs((y_pred_np - y_true_np) / (y_true_np + 1e-6))

                        mre_list.extend(rel_error * 100)
                        y_pred_list.extend(y_pred_np)
                        y_true_list.extend(y_true_np)

                y_true_array = np.array(y_true_list)
                y_pred_array = np.array(y_pred_list)
                y_mean = np.mean(y_true_array)

                # R-squared
                SS_res = np.sum((y_true_array - y_pred_array) ** 2)
                SS_tot = np.sum((y_true_array - y_mean) ** 2)
                r2 = 1 - (SS_res / SS_tot) if SS_tot != 0 else 0.0

                # Normalized Mean Absolute Error (NMAE)
                if y_mean != 0:
                    nmae = np.sum(np.abs(y_pred_array - y_true_array)) / (
                        len(y_true_array) * y_mean
                    )
                else:
                    nmae = 0.0

                # Percentage of samples within MRE_THRESHOLD
                if len(y_true_list) > 0:
                    test_accuracy = (
                        np.sum(np.array(mre_list) <= MRE_THRESHOLD)
                        / len(y_true_list)
                        * 100
                    )
                else:
                    test_accuracy = 0.0

                test_metrics = {
                    "NMAE": float(nmae),
                    "R2": float(r2),
                    "Accuracy": float(test_accuracy),
                    "y_pred": y_pred_array,
                    "y_true": y_true_array,
                }

            except Exception as e:
                send_update(
                    update_queue,
                    "log_messages",
                    f"Failed to compute test metrics for candidate best model: {e}",
                )

        # ------------------------------------------------------------------
        # C) Commit as best model only if test R² improves
        # ------------------------------------------------------------------
        if test_metrics is not None and r2 > BEST_R2:
            BEST_R2 = r2  # keep tracking for display purposes

        # Notify UI about new best CV loss and params (selection is CV-driven)
        send_update(update_queue, "best_loss_so_far", avg_loss)
        send_update(update_queue, "best_params_so_far", trial.params)

        # Save model and export ONNX
        os.makedirs(run_dir, exist_ok=True)
        save_path = os.path.join(run_dir, "best_model.pt")
        torch.save(best_model_full.state_dict(), save_path)
        send_update(update_queue, "best_model_path", save_path)
        append_run_log(run_dir, f"[best_model_saved] {save_path}")
        print(" (New Best Intermediate Model saved from full-data training)")

        intermediate_onnx_path = None
        try:
            intermediate_onnx_path = export_to_onnx(
                best_model_full,
                dataset,
                os.path.join(run_dir, "best_model"),
            )
            send_update(update_queue, "best_onnx_path", intermediate_onnx_path)
            print(" (New Best Intermediate Model saved in PT and ONNX)")
        except Exception as e:
            log_and_record(f"Failed to export intermediate ONNX model: {e}")
            print(
                " (New Best Intermediate Model saved in PT only. ONNX export failed: "
                f"{e})"
            )

        # Send metrics/logs to UI for accepted best model
        if test_metrics is not None:
            send_update(update_queue, "live_best_test_metrics", test_metrics)
            send_update(update_queue, "best_intermediate_r2", r2)
            send_update(update_queue, "best_intermediate_nmae", test_metrics["NMAE"])
            send_update(update_queue, "best_intermediate_accuracy", test_metrics["Accuracy"])
            log_and_record("New best model found.")
            # Persist metrics directly from the worker to survive UI reloads
            update_best_metrics(
                run_dir,
                {
                    "NMAE": test_metrics.get("NMAE"),
                    "R2": test_metrics.get("R2"),
                    "Accuracy": test_metrics.get("Accuracy"),
                    "best_cv_loss": avg_loss,
                },
            )
            # Persist scaler parameters if standardization was used
            try:
                if getattr(dataset, "should_standardize", False) and getattr(dataset, "scaler", None):
                    scaler_json = save_scaler_to_json(dataset.scaler, getattr(dataset, "input_vars", []))
                    if scaler_json:
                        scaler_path = os.path.join(run_dir, "scaler_params.json")
                        with open(scaler_path, "w") as f:
                            f.write(scaler_json)
                        append_run_log(run_dir, f"[scaler_params] {scaler_path}")
                        send_update(update_queue, "scaler_params_path", scaler_path)
            except Exception as e:
                log_and_record(f"Failed to save scaler parameters: {e}")
        else:
            log_and_record(
                f"New best model chosen by CV loss ({avg_loss:.6f}). Test metrics unavailable."
            )
            update_best_metrics(run_dir, {"best_cv_loss": avg_loss})

        # Persist predictions of the best model to CSV in the current run folder
        if test_metrics is not None:
            try:
                y_pred_arr = np.array(test_metrics.get("y_pred", []))
                y_true_arr = np.array(test_metrics.get("y_true", []))
                if y_pred_arr.size > 0 and y_true_arr.size > 0:
                    y_pred_arr = y_pred_arr.reshape(len(y_pred_arr), -1)
                    y_true_arr = y_true_arr.reshape(len(y_true_arr), -1)
                    min_len = min(len(y_pred_arr), len(y_true_arr))
                    y_pred_arr = y_pred_arr[:min_len]
                    y_true_arr = y_true_arr[:min_len]

                    data = {}
                    for idx in range(y_true_arr.shape[1]):
                        data[f"y_true_{idx+1}"] = y_true_arr[:, idx]
                    for idx in range(y_pred_arr.shape[1]):
                        data[f"y_pred_{idx+1}"] = y_pred_arr[:, idx]

                    preds_df = pd.DataFrame(data)
                    preds_path = os.path.join(run_dir, "best_predictions.csv")
                    preds_df.to_csv(preds_path, index=False)
                    append_run_log(run_dir, f"[best_predictions_csv] {preds_path}")
                    send_update(update_queue, "best_predictions_path", preds_path)
            except Exception as e:
                log_and_record(f"Failed to save best predictions CSV: {e}")
        else:
            # Still persist the best CV loss even if we lack test metrics
            update_best_metrics(run_dir, {"best_cv_loss": avg_loss})

        # Zip the current run folder so it is downloadable even after reloads
        zip_path = zip_run_dir(run_dir)
        if zip_path:
            send_update(update_queue, "intermediate_zip_path", zip_path)
            log_and_record(
                f"Run artifacts zipped for download: {os.path.basename(zip_path)}"
            )

        # --- Alternative selection by test R² (kept for reference) ---
        # if test_metrics is not None and r2 > BEST_R2:
        #     BEST_R2 = r2
        #     BEST_CV_LOSS = avg_loss
        #     ... (save model and send updates)
        # else:
        #     ... (log that R² did not improve)

    # This is the value Optuna will minimize
    return avg_loss

def optimization(
    dataset,
    dataset_test,
    config,
    update_queue: queue.Queue,
    stop_event: threading.Event,
    is_resume: bool,
    optuna_study=None,
    run_dir=None,
):
    """
    Orchestrates the HPO using Optuna with real-time callbacks, supporting resume capability.

    Args:
        dataset: Training data structure.
        config: Configuration dictionary for HPO settings.
        update_queue: Queue for sending real-time updates to the main thread (UI).
        stop_event: Threading event to signal manual stop.
        is_resume (bool): Flag indicating whether to resume an existing study.
        optuna_study: Optional pre-existing Optuna Study object (for resume).
    """
    
    # 1. Initialize & Reset UI Metrics
    # (These initial resets are only truly necessary for a new run, but setting them 
    # here ensures the UI starts with a clean slate unless resuming.)
    if not is_resume:
        global BEST_CV_LOSS, BEST_R2
        BEST_CV_LOSS = float("inf")
        BEST_R2 = -float("inf")
        send_update(update_queue, 'best_loss_so_far', float('inf'))
        send_update(update_queue, 'best_params_so_far', {})
        send_update(update_queue, 'best_onnx_path', None)
        send_update(update_queue, 'current_trial_number', 0)
        # --- NEW: Reset all live test metrics shown during training ---
        send_update(update_queue, 'live_best_test_metrics', None)

        # Optional: Reset individual metric keys if you store them separately
        send_update(update_queue, 'best_intermediate_r2', None)
        send_update(update_queue, 'best_intermediate_nmae', None)
        send_update(update_queue, 'best_intermediate_accuracy', None)

        # --- Reset final results from previous runs ---
        send_update(update_queue, 'test_results', None)
        send_update(update_queue, 'final_model_path', None)
        send_update(update_queue, 'final_onnx_path', None)


    
    n_total_trials = config.get('hyperparameter_search_space', {}).get('n_samples', 50)
    
    # --- 2. Load or Create Optuna Study ---
    
    if is_resume and optuna_study is not None:
        # RESUME SCENARIO
        study = optuna_study
        n_completed_trials = len(study.trials)
        n_trials_to_run = max(0, n_total_trials - n_completed_trials)
        
        send_update(update_queue, 'log_messages', 
                    f"Resuming study. {n_completed_trials} trials complete. Running {n_trials_to_run} more.")
        
        # Sync UI state with the loaded study's progress
        send_update(update_queue, 'current_trial_number', n_completed_trials)
        
        if study.best_trial:
             send_update(update_queue, 'best_loss_so_far', study.best_trial.value)
             send_update(update_queue, 'best_params_so_far', study.best_trial.params)
             
    else:
        # NEW STUDY SCENARIO (or resume=False)
        study = optuna.create_study(direction="minimize")
        n_trials_to_run = n_total_trials
        send_update(update_queue, 'log_messages', f"Starting new Optuna study for {n_total_trials} trials.")

    # Check if there's any work to do
    if n_trials_to_run <= 0:
         send_update(update_queue, 'log_messages', "Optimization finished. Trials to run is zero.")
         return study.best_params if study.best_trial else None, study

    # --- 3. Define the Callback Function ---
    def optuna_callback(study: optuna.Study, trial: optuna.Trial):
        
        # 1. Update trial progress: USE QUEUE (use len(study.trials) for accurate count)
        current_trials = len(study.trials)
        send_update(update_queue, 'current_trial_number', current_trials)
        write_value_file(run_dir, "current_trial_number.txt", current_trials)
        
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
        # Note: Best loss update is handled within crossvalidation

    # --- 4. Define Objective and Run Optimization ---
    
    objective = lambda trial: crossvalidation(
        trial, 
        define_net_regression, 
        dataset,               
        config,                
        update_queue,          
        stop_event,
        dataset_test,
        run_dir,
    )

    # Persist initial trial count (handles resume or fresh start)
    initial_trials = len(study.trials)
    send_update(update_queue, 'current_trial_number', initial_trials)
    write_value_file(run_dir, "current_trial_number.txt", initial_trials)

    try:
        study.optimize(
            objective, 
            n_trials = n_trials_to_run, # IMPORTANT: Only run the REMAINING trials
            callbacks=[optuna_callback]
        )
    except Exception as e:
        send_update(update_queue, 'log_messages', f"Optimization Error: {e}")
        return None, study

    # --- 5. Final Cleanup and State Saving ---
    
    # NEW: Also send it to the UI thread via the update queue
    send_update(update_queue, 'optuna_study', study)
    
    if stop_event.is_set():
        send_update(update_queue, 'log_messages', "Optimization was manually stopped.")
        return study.best_params if study.best_trial else None, study

    send_update(update_queue, 'log_messages', "Optimization finished.")
    return study.best_params, study

    

def train_final_model(model, data_train, best_params, n_input_params, n_output_params, config, stop_event: threading.Event | None = None):
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
    optimizer_name = best_params['optimizer_name']
    epochs = best_params['epochs']
    
    # Get loss function from the config
    loss_function = get_loss_function(config)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        if stop_event is not None and stop_event.is_set():
            print("Stop signal received during final training. Aborting.")
            break
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
