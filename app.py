# app.py - REVISED FOR PERSISTING INTERMEDIATE RESULTS AFTER MANUAL STOP

import streamlit as st
import pandas as pd
import threading
import time
import os
import yaml
import json
import io
import zipfile
import numpy as np
from collections import deque
import queue
from sklearn.preprocessing import StandardScaler
import optuna.visualization as ov

# Import your project modules
from src.dataset import Dataset
from core.pipeline import run_training_pipeline
from src.model_test import test
from src.plot import make_plot, make_plotly_figure

# --- Import Sidebar and Config Utilities ---
from ui.sidebar import render_sidebar
from utils.config_utils import load_config, save_config

CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG = load_config(CONFIG_PATH)



def save_scaler_to_json(scaler, input_vars):
    """
    Extracts mean and scale from a fitted StandardScaler and formats it as JSON.
    """
    if scaler is None or not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        return None 

    scaler_data = {}
    
    if isinstance(scaler.mean_, np.ndarray):
        n_features = len(scaler.mean_)
    else:
        return None 

    loop_len = min(n_features, len(input_vars))

    if loop_len == 0:
        return json.dumps({}, indent=4)

    for i in range(loop_len):
        var_name = input_vars[i]
        scaler_data[var_name] = {
            "mean": scaler.mean_[i].item(),
            "std": scaler.scale_[i].item()
        }

    return json.dumps(scaler_data, indent=4)


# --- 1. Session State Initialization ---

def initialize_session_state():
    """Initializes all required session state variables."""

    if "config" not in st.session_state:
        st.session_state.config = load_config(CONFIG_PATH)

    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    
    # THREAD PERSISTENCE KEYS
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = None
    if "training_thread" not in st.session_state:
        st.session_state.training_thread = None
    # END THREAD PERSISTENCE KEYS

    if "log_messages" not in st.session_state:
        st.session_state.log_messages = deque(maxlen=200)
    if "update_queue" not in st.session_state:
        st.session_state.update_queue = queue.Queue()

    if "current_trial_number" not in st.session_state:
        st.session_state.current_trial_number = 0
    if "total_trials" not in st.session_state:
        st.session_state.total_trials = st.session_state.config.get('hyperparameter_search_space', {}).get('n_samples', 50)
    if "best_loss_so_far" not in st.session_state:
        st.session_state.best_loss_so_far = float("inf")
    if "best_params_so_far" not in st.session_state:
        st.session_state.best_params_so_far = {}

    if "current_ui_config" not in st.session_state:
        st.session_state.current_ui_config = st.session_state.config

    if "best_model_path" not in st.session_state:
        st.session_state.best_model_path = "models/ANN_best_intermediate_model.pt"
    if "best_onnx_path" not in st.session_state:
        st.session_state.best_onnx_path = None
    
    if "final_model_path" not in st.session_state:
        st.session_state.final_model_path = None

    if "final_onnx_path" not in st.session_state:
        st.session_state.final_onnx_path = None
    if "test_results" not in st.session_state:
        st.session_state.test_results = None

    # Scaler Storage
    if "fitted_scaler" not in st.session_state:
        st.session_state.fitted_scaler = None
    if "dataset_input_vars" not in st.session_state:
        st.session_state.dataset_input_vars = []
    
    # Optuna Study Storage
    if "optuna_study" not in st.session_state:
        st.session_state.optuna_study = None
    
    # New state to track if training was manually stopped
    if "was_stopped_manually" not in st.session_state:
        st.session_state.was_stopped_manually = False


# --- 2. Queue Processor ---

def process_queue_updates():
    q = st.session_state.update_queue
    updates_processed = 0

    while not q.empty():
        try:
            update = q.get_nowait()
            key = update['key']
            value = update['value']

            if key == 'log_messages':
                st.session_state.log_messages.append(value)
            elif key == 'is_running':
                st.session_state.is_running = value
            # The thread object and stop event are handled in the main loop and initialization
            # We don't need to overwrite them based on queue updates unless signaling a full cleanup
            else:
                st.session_state[key] = value

            q.task_done()
            updates_processed += 1

        except queue.Empty:
            break
        except Exception as e:
            st.error(f"Error processing queue update: {e}")

    return updates_processed > 0

# ----------------------------------------------------------------------------------
## 3. Main App Execution
# ----------------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("ANN Optimization Dashboard")

initialize_session_state()

# --- CRITICAL THREAD RE-ATTACHMENT LOGIC ---
if st.session_state.training_thread and st.session_state.training_thread.is_alive():
    # If a thread exists and is still running after a page reload/rerun, 
    # re-assert the running state and inform the user.
    st.session_state.is_running = True
   
    
# Check if the thread object is present but dead (stopped during a rerun)
elif st.session_state.training_thread and not st.session_state.training_thread.is_alive():
    # Clean up the dead thread object if it's no longer alive
    st.session_state.training_thread = None
    st.session_state.stop_event = None
    # Do NOT reset is_running to False here if we want to show final state after completion.
    # The 'is_running' flag is better managed at the end of the script in Section 7.

# --- 4. Sidebar Integration ---
render_sidebar(DEFAULT_CONFIG, CONFIG_PATH)

uploaded_train_file = st.session_state.get('uploaded_train_file')
uploaded_test_file = st.session_state.get('uploaded_test_file')

# --- 5. Main Panel: Control and Live Status ---

col1, col2 = st.columns([1, 1])

should_standardize_config = st.session_state.config.get("cross_validation", {}).get("standardize_features", False)
if should_standardize_config:
    st.warning("⚠️ **Model Scaling Active**")
    st.markdown(
        "Your input features were scaled. To ensure correct predictions with the downloaded model, use the **`scaler_params.json`** values to transform new input data (`x`) before feeding it to the model."
    )
    st.code("x_scaled = (x - mean) / std")

with col1:
    st.header("Training Control")

    start_button_disabled = st.session_state.is_running or not uploaded_train_file
    if st.button("Start Training & Testing", disabled=start_button_disabled, type="primary"):
        if uploaded_train_file:

            latest_ui_config = st.session_state.current_ui_config
            try:
                save_config(latest_ui_config, CONFIG_PATH)
                st.session_state.config = load_config(CONFIG_PATH)

                st.session_state.total_trials = st.session_state.config.get(
                    'hyperparameter_search_space', {}
                ).get('n_samples', 50)

                st.success(f"Configuration saved successfully! Starting pipeline with {st.session_state.total_trials} trials.")
                st.session_state.log_messages.append("Configuration updated and saved to config.yaml.")
            except Exception as e:
                st.error(f"Failed to save or load configuration: {e}")
                st.rerun()

            os.makedirs("temp_data", exist_ok=True)
            model_dir = os.path.dirname(st.session_state.best_model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)

            train_path = f"temp_data/train_data.{uploaded_train_file.name.split('.')[-1]}"
            test_path = f"temp_data/test_data.{uploaded_test_file.name.split('.')[-1]}" if uploaded_test_file else None

            with open(train_path, "wb") as f: f.write(uploaded_train_file.getvalue())
            if uploaded_test_file:
                with open(test_path, "wb") as f: f.write(uploaded_test_file.getvalue())

            # --- RESET ALL STATE FOR NEW RUN ---
            st.session_state.log_messages.clear()
            st.session_state.log_messages.append("Initializing...")
            st.session_state.best_loss_so_far = float("inf")
            st.session_state.current_trial_number = 0
            st.session_state.final_model_path = None
            st.session_state.final_onnx_path = None
            st.session_state.best_onnx_path = None
            st.session_state.test_results = None
            st.session_state.fitted_scaler = None
            st.session_state.optuna_study = None
            st.session_state.dataset_input_vars = []
            st.session_state.was_stopped_manually = False # Reset manual stop flag
            # ------------------------------------

            # --- SCALING LOGIC ---
            dataset_train = Dataset(
                source=train_path,
                config=st.session_state.config,
                update_queue=st.session_state.update_queue)
            
            st.session_state.dataset_input_vars = dataset_train.input_vars

            should_standardize = st.session_state.config.get("cross_validation", {}).get("standardize_features", False)

            if should_standardize:
                st.session_state.log_messages.append("Standardization enabled. Fitting and transforming training data.")
                scaler = StandardScaler()
                dataset_train.apply_scaler(scaler=scaler, is_fitting=True)
                st.session_state.fitted_scaler = dataset_train.scaler
            # ---------------------

            update_queue = st.session_state.update_queue
            
            # --- CRITICAL THREAD START (SAVING THREAD AND EVENT) ---
            st.session_state.stop_event = threading.Event()
            thread = threading.Thread(
                target=run_training_pipeline,
                args=(
                    dataset_train,
                    st.session_state.config,
                    update_queue,
                    st.session_state,
                    st.session_state.stop_event
                ),
                daemon=True
            )
            st.session_state.training_thread = thread # Save the thread object
            st.session_state.is_running = True
            thread.start()
            st.rerun()

    stop_button_disabled = not st.session_state.is_running
    if st.button("STOP Training", disabled=stop_button_disabled):
        if st.session_state.stop_event:
            st.session_state.log_messages.append("--- STOP signal sent! Finishing current step... ---")
            st.session_state.stop_event.set()
            st.session_state.was_stopped_manually = True # Set flag when stop button is pressed
        st.rerun()
    
    # if st.button("Reset Application State", type="secondary", help="Clear all results and return to initial state."):
    #     reset_app_state()
    #     # This final rerun triggers the app to reflect the cleared state
    #     st.rerun()


    # --- MODIFIED: Show results if running OR if best model exists (to show intermediate results) ---
    best_model_exists = os.path.exists(st.session_state.best_model_path)

    if st.session_state.is_running or best_model_exists:

        st.subheader("Live HPO Status")

        if st.session_state.is_running:
            st.info("Training is in progress...")
        elif st.session_state.was_stopped_manually:
            st.warning("Training stopped manually! Displaying best results achieved so far.") # New message
        elif st.session_state.final_model_path:
            st.success("Training finished!")
        else:
            # This case should only happen if the best model was saved but the final stage failed.
            st.warning("Training finished or stopped.") 

        zero_indexed_trial = st.session_state.current_trial_number
        total_trials = st.session_state.total_trials
        display_trial_number = min(zero_indexed_trial + 1, total_trials)

        if not st.session_state.is_running and zero_indexed_trial >= total_trials:
            progress_percent = 1.0
            display_trial_number = total_trials
        else:
            progress_percent = (display_trial_number / total_trials)

        st.progress(progress_percent, text=f"Trial {display_trial_number} / {total_trials}")

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Best Loss (So Far)", f"{st.session_state.best_loss_so_far:.6f}")

        # --- DOWNLOAD BEST-SO-FAR MODEL ---
        # The logic will only proceed if the file exists AND we have recorded a valid best loss, 
        # ensuring at least one successful trial has finished.
        has_best_loss = st.session_state.best_loss_so_far != float("inf")

        if best_model_exists and has_best_loss:
            try:
                # Attempt to read the file first to confirm it's available
                with open(st.session_state.best_model_path, "rb") as f:
                    model_data = f.read()
                
                # Check if the file is non-empty before proceeding with display logic
                if len(model_data) == 0:
                    st.info("Best-so-far model will be available after the first successful trial completes saving.")
                else:
                    # File read successfully, proceed with download options
                    fitted_scaler = st.session_state.get('fitted_scaler')
                    best_onnx_path = st.session_state.get('best_onnx_path')
                    intermediate_onnx_exists = best_onnx_path and os.path.exists(best_onnx_path)
                    # Check for the PT file existence is now redundant as we successfully read it
                    
                    with st.expander("Download Intermediate Model"):

                        if fitted_scaler and should_standardize_config:
                            scaler_json_data = save_scaler_to_json(fitted_scaler, st.session_state.dataset_input_vars)
                            
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                                # Use model_data which was already read
                                zip_file.writestr('ANN_best_intermediate_model.pt', model_data) 
                                zip_file.writestr('scaler_params.json', scaler_json_data)
                                
                                if intermediate_onnx_exists:
                                    with open(best_onnx_path, "rb") as f_onnx:
                                        zip_file.writestr('ANN_best_intermediate_model.onnx', f_onnx.read())

                            st.download_button(
                                label="Download Best-So-Far Deployment (.zip)",
                                data=zip_buffer.getvalue(),
                                file_name="ANN_intermediate_model.zip",
                                mime="application/zip",
                                type="primary"
                            )
                        else:
                            dl_col_int1, dl_col_int2 = st.columns(2) 
                            
                            # PT download is always available here since model_data is valid
                            with dl_col_int1:
                                st.download_button(
                                    label="Download PT",
                                    data=model_data,
                                    file_name="best_intermediate.pt",
                                    mime="application/octet-stream",
                                    key="dl_best_pt"
                                )
                            
                            if intermediate_onnx_exists:
                                with dl_col_int2:
                                    with open(best_onnx_path, "rb") as f_onnx:
                                        st.download_button(
                                            label="Download ONNX",
                                            data=f_onnx.read(),
                                            file_name="best_intermediate.onnx",
                                            mime="application/octet-stream",
                                            type="secondary",
                                            key="dl_best_onnx"
                                        )
                            else:
                                with dl_col_int2:
                                    st.info("ONNX N/A")
                                        
            except Exception as e:
                st.error(f"Error preparing model download: {e}")
                st.info("Best-so-far model will be available after the first successful trial.")
        else:
            # This executes if best_model_exists is False OR has_best_loss is False (i.e., inf)
            st.info("Best-so-far model will be available after the first successful trial.")

        st.subheader("Best Hyperparameters")
        st.json(st.session_state.best_params_so_far, expanded=False)

with col2:
    if st.session_state.is_running or st.session_state.final_model_path or best_model_exists: # Check for best_model_exists here too
        st.header("Live Logs")
        log_container = st.empty()
        log_text = "\n".join(list(st.session_state.log_messages)[::-1])
        log_container.text_area("Logs", value=log_text, height=400, disabled=True)
# ----------------------------------------------------------------------------------
## 6. Final Results Section
# ----------------------------------------------------------------------------------

st.divider()
st.header("Final Results")

# --- MODIFIED: The Final Results section requires the FINAL model path ---
if st.session_state.final_model_path and uploaded_test_file:
    # ... (Rest of Section 6 logic remains the same, as it only runs after full HPO completion or final export)
    
    if not st.session_state.test_results:
        if not st.session_state.is_running:
            st.session_state.log_messages.append("Running evaluation on test set...")
            try:
                test_path = f"temp_data/test_data.{uploaded_test_file.name.split('.')[-1]}"

                dataset_test = Dataset(
                            source=test_path,
                            config=st.session_state.config,
                            update_queue=st.session_state.update_queue
                        )
                fitted_scaler = st.session_state.fitted_scaler

                if fitted_scaler is not None:
                    dataset_test.apply_scaler(scaler=fitted_scaler, is_fitting=False)

                test_metrics = test(
                    dataset_test,
                    st.session_state.final_model_path,
                    st.session_state.best_params_so_far
                )
                st.session_state.test_results = test_metrics
                st.rerun()

            except Exception as e:
                st.error(f"Failed to run test: {e}")

    if st.session_state.test_results:
        st.subheader("Test Metrics")
        results = st.session_state.test_results
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Test NMAE", f"{results['NMAE']:.4f}")
        r_col2.metric("Test R² Score", f"{results['R2']:.4f}")
        r_col3.metric("Test Accuracy", f"{results['Accuracy']:.2f}%")

        # --- DOWNLOAD FINAL MODEL ---
        try:
            with open(st.session_state.final_model_path, "rb") as f:
                final_model_data = f.read()

            fitted_scaler = st.session_state.get('fitted_scaler')
            final_onnx_path = st.session_state.get('final_onnx_path')
            onnx_exists = final_onnx_path and os.path.exists(final_onnx_path)

            if fitted_scaler and should_standardize_config:
                 # Case 1: Standardization IS used -> Offer ZIP deployment package
                scaler_json_data = save_scaler_to_json(fitted_scaler, st.session_state.dataset_input_vars)
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr('final_model.pt', final_model_data)
                    zip_file.writestr('scaler_params.json', scaler_json_data)
                    
                    if onnx_exists:
                        with open(final_onnx_path, "rb") as f_onnx:
                            zip_file.writestr('final_model.onnx', f_onnx.read())

                st.download_button(
                    label="Download final model (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="ANN_final_deployment_package.zip",
                    mime="application/zip",
                    type="primary"
                )
            else:
                # Case 2: No Standardization -> Offer PT and ONNX separately
                dl_col1, dl_col2 = st.columns(2)
                
                with dl_col1:
                    st.download_button(
                        label="Download PyTorch (.pt)",
                        data=final_model_data,
                        file_name="final_model.pt",
                        mime="application/octet-stream",
                        type="primary"
                    )
                
                if onnx_exists:
                    with dl_col2:
                         with open(final_onnx_path, "rb") as f_onnx:
                            st.download_button(
                                label="Download ONNX (.onnx)",
                                data=f_onnx.read(),
                                file_name="final_model.onnx",
                                mime="application/octet-stream",
                                type="secondary"
                            )
                else:
                    with dl_col2:
                         st.info("ONNX file not available.")

        except Exception as e:
            st.error(f"Error preparing final model for download: {e}")

        
        config = st.session_state.current_ui_config 
        display_config = config.get("display", {})
        st.markdown("---")
        
        # --- 1. Predictions vs. Actual Plot (Parity Plot) ---
        if display_config.get("show_prediction_plot", True): 
            st.subheader("Final Model Prediction Analysis (Parity Plot)")
            fig_parity = make_plotly_figure(results['y_pred'], results['y_true'])
            st.plotly_chart(fig_parity, use_container_width=True)
            st.markdown("---")

        # --- 2. Optuna Optimization History Plot (INCLUDING PARAMETER IMPORTANCE) ---
        if display_config.get("show_optuna_plots", True):
            if 'optuna_study' in st.session_state and st.session_state.optuna_study is not None:
                st.subheader("Hyperparameter Optimization Analysis")
                
                # Plot 1: Optimization History
                try:
                    st.markdown("#### Trial History")
                    fig_history = ov.plot_optimization_history(st.session_state.optuna_study)
                    st.plotly_chart(fig_history, use_container_width=True) # 
                except Exception as e:
                    st.warning(f"Could not generate Optimization History plot: {e}")

                # Plot 2: Parameter Importance
                try:
                    st.markdown("#### Parameter Importance")
                    fig_importance = ov.plot_param_importances(st.session_state.optuna_study)
                    st.plotly_chart(fig_importance, use_container_width=True) # 
                except Exception as e:
                    st.warning(f"Could not generate Parameter Importance plot: {e}")
                    
                st.markdown("---")
            else:
                st.info("Optuna study results not found in session state.")

# --- NEW: Condition to handle manual stop ---
elif st.session_state.was_stopped_manually and best_model_exists:
    st.info("Training was manually stopped. View the best intermediate results above.")

elif st.session_state.is_running:
    st.info("Waiting for training to complete to display final results...")
else:
    st.warning("Upload a test file and run training to see final results.")


# ----------------------------------------------------------------------------------
## 7. Auto-Rerun and Thread Completion Handler
# ----------------------------------------------------------------------------------

is_thread_alive = st.session_state.training_thread and st.session_state.training_thread.is_alive()
queue_is_empty = st.session_state.update_queue.empty()
rerun_needed = False

# 1. Process Queue
if is_thread_alive or not queue_is_empty:
    if process_queue_updates():
        rerun_needed = True

# 2. Handle Thread Death
if st.session_state.is_running and not is_thread_alive:
    if queue_is_empty:
        # If the thread died and the queue is empty, finalize state
        st.session_state.is_running = False
        st.session_state.log_messages.append("--- Training Thread finished. Final state reached. ---")
        st.session_state.was_stopped_manually = False # Reset flag if training finished naturally
        rerun_needed = True
    else:
        # If the thread died but queue has updates, process them and rerun
        rerun_needed = True

# 3. Auto-Rerun for live monitoring
if is_thread_alive:
    time.sleep(0.5)
    st.rerun()

# 4. Final Rerun (if state was just updated)
if rerun_needed and not is_thread_alive:
    st.rerun()