# app.py 
import streamlit as st
import pandas as pd
import threading
import time
import os
import yaml
from collections import deque
import queue

# Import your project modules
# (Make sure src is in PYTHONPATH)
from src.dataset import Dataset
from core.pipeline import run_training_pipeline 
from src.model_test import test 
from src.plot import make_plot 

# --- Import Sidebar and Config Utilities ---
# Assumed path, adjust if necessary
from ui.sidebar import render_sidebar 
from utils.config_utils import load_config, save_config 

CONFIG_PATH = "config.yaml"
# Load default config once to pass to sidebar reset function
DEFAULT_CONFIG = load_config(CONFIG_PATH)


# --- 1. Session State Initialization ---

def initialize_session_state():
    """Initializes all required session state variables."""
    
    # Config must be loaded first
    if "config" not in st.session_state:
        st.session_state.config = load_config(CONFIG_PATH)
        
    # State flags
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = None
    if "training_thread" not in st.session_state:
        st.session_state.training_thread = None

    # Thread Communication
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = deque(maxlen=200) 
    if "update_queue" not in st.session_state:
        st.session_state.update_queue = queue.Queue()
        
    # Live HPO Data
    if "current_trial_number" not in st.session_state:
        st.session_state.current_trial_number = 0
    # Use config for total trials
    if "total_trials" not in st.session_state:
        st.session_state.total_trials = st.session_state.config.get('hyperparameter_search_space', {}).get('n_samples', 50) 
    if "best_loss_so_far" not in st.session_state:
        st.session_state.best_loss_so_far = float("inf")
    if "best_params_so_far" not in st.session_state:
        st.session_state.best_params_so_far = {}
        
    # UI State: Holds configuration from sidebar widgets
    if "current_ui_config" not in st.session_state:
        st.session_state.current_ui_config = st.session_state.config

    # Paths and Results
    if "best_model_path" not in st.session_state:
        st.session_state.best_model_path = "models/ANN_best_intermediate_models.pt"
    if "final_model_path" not in st.session_state:
        st.session_state.final_model_path = None
    if "test_results" not in st.session_state:
        st.session_state.test_results = None


# --- 2. Queue Processor (No changes needed) ---

def process_queue_updates():
    """
    Reads the update_queue and applies changes to st.session_state.
    Returns True if any update was processed, signaling a rerun might be needed.
    """
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

initialize_session_state()

# --- 4. Sidebar Integration ---
# This calls render_sidebar, which updates st.session_state.uploaded_..._file 
# and st.session_state.current_ui_config.
# We pass the default config so the reset button works correctly.
render_sidebar(DEFAULT_CONFIG, CONFIG_PATH)

# Fetch uploaded files from session state
uploaded_train_file = st.session_state.get('uploaded_train_file')
uploaded_test_file = st.session_state.get('uploaded_test_file')

# --- 5. Main Panel: Control and Live Status ---

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Training Control")
    
    # --- START/STOP Buttons ---
    
    start_button_disabled = st.session_state.is_running or not uploaded_train_file
    if st.button("🚀 Start Training & Testing", disabled=start_button_disabled, type="primary"):
        if uploaded_train_file:
            
            # --- CRITICAL FIX 1: Save UI Config to file and state ---
            latest_ui_config = st.session_state.current_ui_config
            try:
                save_config(latest_ui_config, CONFIG_PATH)
                st.session_state.config = latest_ui_config # Update the main config used by the thread
                
                # Update total trials from the saved config
                st.session_state.total_trials = latest_ui_config.get('hyperparameter_search_space', {}).get('n_samples', 50)
                
                # --- ADDED: Visible Success Message ---
                st.success("✅ Configuration saved successfully! Starting training pipeline...")
                st.session_state.log_messages.append("Configuration updated and saved to config.yaml.")
            except Exception as e:
                st.error(f"🚨 Failed to save configuration: {e}")
                st.rerun() # Rerun to display error and stop
                
            # Setup data paths and save uploaded files
            os.makedirs("temp_data", exist_ok=True)
            train_path = f"temp_data/train_data.{uploaded_train_file.name.split('.')[-1]}"
            test_path = f"temp_data/test_data.{uploaded_test_file.name.split('.')[-1]}" if uploaded_test_file else None
            
            with open(train_path, "wb") as f: f.write(uploaded_train_file.getvalue())
            if uploaded_test_file:
                with open(test_path, "wb") as f: f.write(uploaded_test_file.getvalue())
            
            # Reset state for a new run
            st.session_state.log_messages.clear()
            st.session_state.log_messages.append("Initializing...")
            st.session_state.best_loss_so_far = float("inf")
            st.session_state.current_trial_number = 0 
            st.session_state.final_model_path = None
            st.session_state.test_results = None
            
            # Create Dataset 
            dataset_train = Dataset(train_path) 
            
            update_queue = st.session_state.update_queue 

            # Create stop event and thread
            st.session_state.stop_event = threading.Event()
            st.session_state.training_thread = threading.Thread(
                target=run_training_pipeline,
                args=(
                    dataset_train, 
                    st.session_state.config, # Use the freshly saved and updated config
                    update_queue,                 
                    st.session_state,             
                    st.session_state.stop_event   
                ),
                daemon=True
            )
            st.session_state.is_running = True
            st.session_state.training_thread.start()
            st.rerun() # Immediately rerun to trigger the main loop logic
            
    stop_button_disabled = not st.session_state.is_running
    if st.button("🛑 STOP Training", disabled=stop_button_disabled):
        if st.session_state.stop_event:
            st.session_state.log_messages.append("--- STOP signal sent! Finishing current step... ---")
            st.session_state.stop_event.set()
        st.rerun() 
    
    
    # --- Live Status Display ---
    st.subheader("Live HPO Status")
    
    if st.session_state.is_running:
        st.info("Training is in progress...")
    elif st.session_state.final_model_path:
        st.success("Training finished!")
    else:
        st.warning("Training has not started or was stopped.")

    # --- Progress Bar Logic ---
    zero_indexed_trial = st.session_state.current_trial_number
    total_trials = st.session_state.total_trials
    
    display_trial_number = min(zero_indexed_trial + 1, total_trials)
    
    if not st.session_state.is_running and zero_indexed_trial >= total_trials:
        progress_percent = 1.0
        display_trial_number = total_trials
    else:
        progress_percent = (display_trial_number / total_trials)
    
    st.progress(progress_percent, text=f"Trial {display_trial_number} / {total_trials}")
    
    # Metrics
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Best Loss (So Far)", f"{st.session_state.best_loss_so_far:.6f}")
    
    # Download Best-So-Far Model 
    best_model_exists = os.path.exists(st.session_state.best_model_path)

    if best_model_exists:
        
        # Read the file content into a bytes object outside the 'with open' block
        # so the file descriptor is closed before the function returns.
        try:
            with open(st.session_state.best_model_path, "rb") as f:
                model_data = f.read()
                
            m_col2.download_button(
                label="📥 Download Best-So-Far Model",
                data=model_data, # Pass the actual binary data
                file_name="best_intermediate.pt",
                mime="application/octet-stream",
                disabled=not best_model_exists
            )
        except Exception as e:
            m_col2.error(f"Error reading model file: {e}")
            m_col2.info("Best-so-far model will be available after the first successful trial.")
    else:
        m_col2.info("Best-so-far model will be available after the first successful trial.")


    st.subheader("Best Hyperparameters (So Far)")
    st.json(st.session_state.best_params_so_far, expanded=False)

with col2:
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
        # Only run test if thread is not running
        if not st.session_state.is_running: 
            st.session_state.log_messages.append("Running evaluation on test set...")
            try:
                test_path = f"temp_data/test_data.{uploaded_test_file.name.split('.')[-1]}"
                dataset_test = Dataset(test_path)
                
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
        # Ensure test_results is a dict (based on previous fix)
        results = st.session_state.test_results
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Test NMAE", f"{results['NMAE']:.4f}")
        r_col2.metric("Test R² Score", f"{results['R2']:.4f}")
        r_col3.metric("Test Accuracy", f"{results['Accuracy']:.2f}%")
        
        with open(st.session_state.final_model_path, "rb") as f:
            st.download_button(
                label="🏆 Download FINAL Trained Model",
                data=f,
                file_name="final_model.pt",
                mime="application/octet-stream",
                type="primary"
            )
            
        st.subheader("Performance Plot")
        # Ensure make_plot returns a Figure object (based on previous fix)
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

# 1. Process updates if thread is alive OR if queue has pending items
if is_thread_alive or not queue_is_empty:
    if process_queue_updates():
        rerun_needed = True

# 2. Handle thread completion (The thread just died, and we might have cleared the queue)
if st.session_state.is_running and not is_thread_alive:
    # Check the queue one last time after the last processing block
    if queue_is_empty:
        # The thread is dead and the queue is clear. Final state.
        st.session_state.is_running = False
        st.session_state.log_messages.append("--- Training Thread finished. Final state reached. ---")
        rerun_needed = True
    else:
        # The thread is dead, but the queue still has items. Force another rerun.
        rerun_needed = True


if rerun_needed:
    st.rerun()

if is_thread_alive:
    time.sleep(0.5)  # Slow down to avoid excessive reruns
    st.rerun()

