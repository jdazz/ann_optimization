# In src/train.py or core/pipeline.py

import os
import torch
import threading
import traceback
import queue 
from streamlit.runtime.state import SessionStateProxy
# Note: Remove import of SessionStateProxy as we no longer use it for writes
from src.train import optimization, train_final_model
from src.model import define_net_regression

from utils.save_onnx import export_to_onnx

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function Signature Change: Remove st_state, use update_queue as the first parameter
def run_training_pipeline(dataset_train, config, update_queue: queue.Queue, st_state: SessionStateProxy, stop_event: threading.Event):

    def send_update(key, value):
        update_queue.put({'key': key, 'value': value})

    try:
        # 1. Run Hyperparameter Optimization (HPO)
        send_update('log_messages', "Starting HPO optimization...")
        
        # --- 1. CAPTURE BOTH best_params AND THE OPTUNA STUDY OBJECT ---
        best_params, study = optimization(dataset_train, config, update_queue, st_state, stop_event)

        if not best_params:
            send_update('log_messages', "Optimization stopped or failed to find best parameters.")
            return
            
        # --- 2. SEND THE STUDY OBJECT TO STREAMLIT STATE ---
        send_update('optuna_study', study)
        send_update('log_messages', "Optimization complete. Optuna study object saved for visualization.")
        
        # 2. Define the Final Model Architecture
        send_update('log_messages', "\n--- Creating Final Model Architecture ---")
        final_model = define_net_regression(
            best_params, 
            dataset_train.n_input_params, 
            dataset_train.n_output_params,
        ).to(Device)
        
        # 3. Train the Final Model on the Full Dataset
        send_update('log_messages', "Starting final training on full dataset...")
        final_model = train_final_model(
            final_model,
            dataset_train.full_data,
            best_params,
            dataset_train.n_input_params,
            dataset_train.n_output_params,
            config 
        )

        # 4. Save the Final Model
        os.makedirs("models", exist_ok=True)
        final_model_path = os.path.join("models", "ANN_final_model.pt")
        torch.save(final_model.state_dict(), final_model_path)

        # 6. Export Model to ONNX Format
        final_onnx_path = export_to_onnx(final_model, dataset_train, os.path.join("models", "ANN_final_model"))
        
        # 5. Send Final Status Updates to Streamlit
        send_update('final_model_path', final_model_path)
        send_update('final_onnx_path', final_onnx_path)
        send_update('best_params_so_far', best_params)
        send_update('log_messages', f"Final best model saved to: {final_model_path}")
        send_update('is_running', False) # Signal completion to the UI


    except Exception as e:
        # Use traceback to get better error detail
        full_traceback = traceback.format_exc()
        send_update('log_messages', f"❌ An error occurred in the training pipeline: {e}\n{full_traceback}")
        send_update('is_running', False)
        
    finally:
        send_update('log_messages', "--- Training thread finished ---")
        send_update('stop_event', None)