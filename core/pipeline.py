# FILE: core/pipeline.py (MODIFIED)

import os
import io
import contextlib
import tempfile 
import shutil
import streamlit as st

# Import your custom source files
from src.dataset import Dataset
from src.train import find_best_model # Assume this now returns study object
from src.model_test import test
from src.model import define_net_regression
from utils.config_utils import load_config # <-- Added to get full config for HPO trials

def run_training_pipeline(train_file, test_file):
    """
    Executes the full training and testing pipeline.
    
    Handles temp file creation, data loading, model training,
    model testing, and log capturing.
    
    Returns:
        (dict): A dictionary of results if successful, else None.
        (str): The captured stdout log.
        (optuna.study.Study or None): The Optuna study object. <-- NEW RETURN
        (Exception): An exception object if one occurred, else None.
    """
    log_stream = io.StringIO()
    temp_dir = None
    results = None
    # Initialize new return values
    study = None 
    best_model_path = os.path.join("models", "ANN_best_model.pt")
    best_intermediate_model_path = os.path.join("models", "ANN_best_intermediate_model.pt")


    with contextlib.redirect_stdout(log_stream):
        try:
            print("--- Streamlit App: Starting Training ---")

            # --- NEW: Delete old model checkpoints to enforce consistency ---
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Removed old checkpoint: {best_model_path}")
            
            inter_model_path = os.path.join("models", "ANN_best_intermediate_model.pt")
            if os.path.exists(inter_model_path):
                os.remove(inter_model_path)
                print(f"Removed old checkpoint: {inter_model_path}")
            # --- END NEW BLOCK ---
            
            temp_dir = tempfile.mkdtemp()
            
            # --- Handle Uploaded Files (Unchanged) ---
            train_path = os.path.join(temp_dir, train_file.name)
            test_path = os.path.join(temp_dir, test_file.name)
            
            with open(train_path, "wb") as f:
                f.write(train_file.getbuffer())
            with open(test_path, "wb") as f:
                f.write(test_file.getbuffer())

            # --- Load Data (Unchanged) ---
            print(f"Loading training data from: {train_path}")
            dataset_train = Dataset(train_path)
            
            print(f"Loading testing data from: {test_path}")
            dataset_test = Dataset(test_path)

            # --- Find Best Model (UPDATED CAPTURE) ---
            print("Starting find_best_model()...")
            # CAPTURE THE NEW RETURN VALUE: study
            best_model, best_param, study = find_best_model(dataset_train) 
            print("find_best_model() complete.")
            
            # --- Intermediate Model Performance (NEW BLOCK) ---
            # Test the BEST INTERMEDIATE model found during HPO 
            if study and os.path.exists(best_intermediate_model_path):
                print("Testing best intermediate model...")
                # We use the best_param found by the study for the test metrics
                # Note: This assumes test() can load a model from best_intermediate_model_path
                inter_accuracy, inter_nmae, inter_r2, _, _, _ = test(
                    dataset_test, best_intermediate_model_path, best_param
                )
                print("Intermediate model testing complete.")
            else:
                inter_accuracy, inter_nmae, inter_r2 = None, None, None


            # --- Test Final Model (Unchanged Logic, uses best_model_path) ---
            print("Starting test()...")
            test_accuracy, nmae, r2, mre_list, y_pred, y_true = test(
                dataset_test, best_model_path, best_param
            )
            print("test() complete.")
            
            # --- Compile Results (UPDATED) ---
            model_structure = define_net_regression(
                best_param, 
                dataset_train.n_input_params, 
                dataset_train.n_output_params
            )

            results = {
                # Final Model Results
                "test_accuracy": test_accuracy,
                "nmae": nmae,
                "r2": r2,
                "best_param": best_param,
                "model_structure": str(model_structure),
                "y_true": y_true,
                "y_pred": y_pred,
                "mre_list": mre_list,
                
                # Intermediate Model Results
                "inter_accuracy": inter_accuracy, 
                "inter_nmae": inter_nmae,
                "inter_r2": inter_r2,
                "inter_model_path": best_intermediate_model_path if study else None
            }
            
            print("--- Streamlit App: Training Complete ---")

        except Exception as e:
            print(f"\n--- AN ERROR OCCURRED ---")
            print(str(e))
            log_output = log_stream.getvalue()
            # Return study object even if an error occurred
            return None, log_output, study, e
            
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")

    log_output = log_stream.getvalue()
    
    # UPDATED RETURN SIGNATURE
    return results, log_output, study, None