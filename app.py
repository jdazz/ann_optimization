# app.py
import streamlit as st
import pandas as pd
import threading
import time
import os
import yaml
import json  # NEW: For saving scaler parameters
import io    # NEW: For in-memory file handling
import zipfile # NEW: For creating ZIP archives
import numpy as np
from collections import deque
import queue
from sklearn.preprocessing import StandardScaler

# Import your project modules
from src.dataset import Dataset
from core.pipeline import run_training_pipeline
from src.model_test import test
from src.plot import make_plot

# --- Import Sidebar and Config Utilities ---
from ui.sidebar import render_sidebar
from utils.config_utils import load_config, save_config

CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG = load_config(CONFIG_PATH)


# --- NEW UTILITY FUNCTION: Create JSON of Scaler Parameters ---
def save_scaler_to_json(scaler, input_vars):
    """
    Extracts mean and scale from a fitted StandardScaler and formats it as JSON.
    Handles cases where input_vars might have more entries than scaler features.
    """
    if scaler is None or not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        # This should ideally not happen if called correctly, but good for safety.
        return None 

    scaler_data = {}
    
    # Check if mean_ is a numpy array and get its length
    if isinstance(scaler.mean_, np.ndarray):
        n_features = len(scaler.mean_)
    else:
        # Fallback if scaler.mean_ is somehow not an array (e.g., None or empty list)
        return None 

    # Use min length to avoid index errors if config input_vars doesn't match data columns
    loop_len = min(n_features, len(input_vars))

    if loop_len == 0:
        # Critical safety check: If no features were processed, return None or an empty JSON
        return json.dumps({}, indent=4)


    for i in range(loop_len):
        var_name = input_vars[i]
        scaler_data[var_name] = {
            "mean": scaler.mean_[i].item(),  # .item() converts numpy float to python float
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
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = None
    if "training_thread" not in st.session_state:
        st.session_state.training_thread = None

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
    if "final_model_path" not in st.session_state:
        st.session_state.final_model_path = None

    if "final_onnx_path" not in st.session_state:
        st.session_state.final_onnx_path = None
    if "test_results" not in st.session_state:
        st.session_state.test_results = None

    # Scaler Storage
    if "fitted_scaler" not in st.session_state:
        st.session_state.fitted_scaler = None
    # Storage for input variable names (needed for JSON export)
    if "dataset_input_vars" not in st.session_state:
        st.session_state.dataset_input_vars = []


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
            elif key == 'stop_event':
                st.session_state.stop_event = None
            elif key == 'training_thread':
                st.session_state.training_thread = None
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
st.title("🧠 ANN Optimization Dashboard")

if "best_model_path" in st.session_state and st.session_state.best_model_path == "models/ANN_best_intermediate_models.pt":
    del st.session_state["best_model_path"]

initialize_session_state()

# --- 4. Sidebar Integration ---
render_sidebar(DEFAULT_CONFIG, CONFIG_PATH)

uploaded_train_file = st.session_state.get('uploaded_train_file')
uploaded_test_file = st.session_state.get('uploaded_test_file')

# --- 5. Main Panel: Control and Live Status ---

col1, col2 = st.columns([1, 1])

# --- NEW: Display Scaling Warning if Standardization is Active ---
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
    if st.button("🚀 Start Training & Testing", disabled=start_button_disabled, type="primary"):
        if uploaded_train_file:

            latest_ui_config = st.session_state.current_ui_config
            try:
                save_config(latest_ui_config, CONFIG_PATH)
                st.session_state.config = load_config(CONFIG_PATH)

                st.session_state.total_trials = st.session_state.config.get(
                    'hyperparameter_search_space', {}
                ).get('n_samples', 50)

                st.success(f"✅ Configuration saved successfully! Starting pipeline with {st.session_state.total_trials} trials.")
                st.session_state.log_messages.append("Configuration updated and saved to config.yaml.")
            except Exception as e:
                st.error(f"🚨 Failed to save or load configuration: {e}")
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

            st.session_state.log_messages.clear()
            st.session_state.log_messages.append("Initializing...")
            st.session_state.best_loss_so_far = float("inf")
            st.session_state.current_trial_number = 0
            st.session_state.final_model_path = None
            st.session_state.test_results = None
            st.session_state.fitted_scaler = None
            # CRITICAL RESET: Ensure input vars are clear before new assignment
            st.session_state.dataset_input_vars = []

            # --- SCALING LOGIC ---
            # 1. Create Dataset (initially unscaled)
            dataset_train = Dataset(
                source=train_path,
                config=st.session_state.config,
                update_queue=st.session_state.update_queue)
            
            # CRITICAL FIX: Capture input variables NOW, right after the dataset loads the data
            st.session_state.dataset_input_vars = dataset_train.input_vars

            should_standardize = st.session_state.config.get("cross_validation", {}).get("standardize_features", False)

            if should_standardize:
                st.session_state.log_messages.append("Standardization enabled. Fitting and transforming training data.")
                scaler = StandardScaler()
                dataset_train.apply_scaler(scaler=scaler, is_fitting=True)
                st.session_state.fitted_scaler = dataset_train.scaler
            
            # The 'dataset_train' object now contains the correctly scaled data.
            # ---------------------

            update_queue = st.session_state.update_queue

            st.session_state.stop_event = threading.Event()
            st.session_state.training_thread = threading.Thread(
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
            st.session_state.is_running = True
            st.session_state.training_thread.start()
            st.rerun()

    stop_button_disabled = not st.session_state.is_running
    if st.button("🛑 STOP Training", disabled=stop_button_disabled):
        if st.session_state.stop_event:
            st.session_state.log_messages.append("--- STOP signal sent! Finishing current step... ---")
            st.session_state.stop_event.set()
        st.rerun()


    if st.session_state.is_running or st.session_state.final_model_path:

        st.subheader("Live HPO Status")

        if st.session_state.is_running:
            st.info("Training is in progress...")
        elif st.session_state.final_model_path:
            st.success("Training finished!")
        else:
            st.warning("Training has not started or was stopped.")

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

        # --- DOWNLOAD BEST-SO-FAR MODEL (MODIFIED) ---
        best_model_exists = os.path.exists(st.session_state.best_model_path)

        if best_model_exists:
            try:
                with open(st.session_state.best_model_path, "rb") as f:
                    model_data = f.read()

                fitted_scaler = st.session_state.get('fitted_scaler')

                if fitted_scaler and should_standardize_config:
                    # Case 1: Standardization IS used -> Offer ZIP file
                    scaler_json_data = save_scaler_to_json(fitted_scaler, st.session_state.dataset_input_vars)
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                        zip_file.writestr('ANN_best_intermediate_model.pt', model_data)
                        zip_file.writestr('scaler_params.json', scaler_json_data)
                        # README.txt REMOVED AS REQUESTED

                    m_col2.download_button(
                        label="📥 Download Best-So-Far Model (.zip)",
                        data=zip_buffer.getvalue(),
                        file_name="ANN_intermediate_model.zip",
                        mime="application/zip"
                    )
                else:
                    # Case 2: No Standardization -> Offer PT file
                    m_col2.download_button(
                        label="📥 Download Best-So-Far Model (.pt)",
                        data=model_data,
                        file_name="best_intermediate.pt",
                        mime="application/octet-stream"
                    )
            except Exception as e:
                m_col2.error(f"Error preparing model download: {e}")
                m_col2.info("Best-so-far model will be available after the first successful trial.")
        else:
            m_col2.info("Best-so-far model will be available after the first successful trial.")

        st.subheader("Best Hyperparameters (So Far)")
        st.json(st.session_state.best_params_so_far, expanded=False)

with col2:
    if st.session_state.is_running or st.session_state.final_model_path:
        st.header("Live Logs")
        log_container = st.empty()
        log_text = "\n".join(list(st.session_state.log_messages)[::-1])
        log_container.text_area("Logs", value=log_text, height=400, disabled=True)


# ----------------------------------------------------------------------------------
## 6. Final Results Section
# ----------------------------------------------------------------------------------

st.divider()
st.header("Final Results")

if st.session_state.final_model_path and uploaded_test_file:

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

        # --- DOWNLOAD FINAL MODEL (MODIFIED) ---
        try:
            with open(st.session_state.final_model_path, "rb") as f:
                final_model_data = f.read()

            fitted_scaler = st.session_state.get('fitted_scaler')

            if fitted_scaler and should_standardize_config:
                 # Case 1: Standardization IS used -> Offer ZIP file
                scaler_json_data = save_scaler_to_json(fitted_scaler, st.session_state.dataset_input_vars)
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr('final_model.pt', final_model_data)
                    zip_file.writestr('scaler_params.json', scaler_json_data)
                    # README.txt REMOVED AS REQUESTED

                st.download_button(
                    label="🏆 Download FINAL Trained Model (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="ANN_final_model.zip",
                    mime="application/zip",
                    type="primary"
                )
            else:
                # Case 2: No Standardization -> Offer PT file
                st.download_button(
                    label="🏆 Download FINAL Trained Model (.pt)",
                    data=final_model_data,
                    file_name="final_model.pt",
                    mime="application/octet-stream",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error preparing final model for download: {e}")

        st.subheader("Performance Plot")
        fig = make_plot(results['y_pred'], results['y_true'])
        st.pyplot(fig)

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

if is_thread_alive or not queue_is_empty:
    if process_queue_updates():
        rerun_needed = True

if st.session_state.is_running and not is_thread_alive:
    if queue_is_empty:
        st.session_state.is_running = False
        st.session_state.log_messages.append("--- Training Thread finished. Final state reached. ---")
        rerun_needed = True
    else:
        rerun_needed = True

if rerun_needed:
    st.rerun()

if is_thread_alive:
    time.sleep(0.5)
    st.rerun()