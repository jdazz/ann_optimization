# utils/state_manager.py

import streamlit as st
import threading
import os

def reset_resumable_flag():
    """Helper to reset the flag when a new run starts or a run completes."""
    if 'is_resumable' in st.session_state:
        st.session_state.is_resumable = False

def initialize_run_state():
    """Initializes/resets all run-specific session state variables for a NEW training job."""
    # NOTE: This function is called only when the user clicks 'Start New Study'
    st.session_state.log_messages.clear()
    st.session_state.log_messages.append("Initializing New Run...")
    st.session_state.best_loss_so_far = float("inf")
    st.session_state.current_trial_number = 0
    st.session_state.final_model_path = None
    st.session_state.final_onnx_path = None
    st.session_state.best_onnx_path = None
    st.session_state.test_results = None
    st.session_state.fitted_scaler = None
    st.session_state.was_stopped_manually = False 
    st.session_state.test_dataset_df = None
    
    # CRITICAL: This wipes the study for a NEW run
    st.session_state.optuna_study = None 
    # Must be set to False for a new run start
    st.session_state.is_resumable = False

def handle_thread_reattachment():
    """
    Handles thread re-attachment and detects resumable state.
    This runs on every script execution.
    """
    thread = st.session_state.get('training_thread')
    is_thread_alive = thread and isinstance(thread, threading.Thread) and thread.is_alive()
    was_stopped = st.session_state.get('was_stopped_manually', False)
    study_exists = st.session_state.get('optuna_study') is not None

    # --- 1. Thread is Active (Rerun, not Reload) ---
    if is_thread_alive:
        st.session_state.is_running = True
        st.session_state.is_resumable = True # A running job is always resumable if interrupted
        st.session_state.was_stopped_manually = False # Clear manual stop flag
        return

    # --- 2. Thread is Dead (Finished, Stopped, or Reloaded) ---
    st.session_state.is_running = False

    # Check for Study Completion
    current_trial = st.session_state.get('current_trial_number', 0)
    total_trials = st.session_state.get('total_trials', 50)
    
    is_completed = (current_trial >= total_trials)
    
    # Cleanup thread references if it's dead
    if not is_thread_alive:
        st.session_state.training_thread = None
        st.session_state.stop_event = None

    # --- 3. Determine Resumable State (Prioritizing Completion) ---
    if study_exists and is_completed:
        # The study is finished, regardless of how we got here.
        st.session_state.is_resumable = False
        st.session_state.log_messages.append("Optimization finished and locked.")
        return

    if study_exists and not is_completed:
        # A study object exists, and work remains (this is the state after a crash/reload).
        # We enforce is_resumable = True to allow recovery.
        st.session_state.is_resumable = True
        if not was_stopped:
            # Only log recovery if it wasn't a manual stop (which sends its own message)
            if not st.session_state.get('is_resumable_logged', False):
                 st.session_state.log_messages.append(f"⚠️ Recovered persistent study state. Ready to resume from Trial {current_trial}.")
                 st.session_state.is_resumable_logged = True
        return
        
    # If no study exists, it's not resumable (fresh start)
    st.session_state.is_resumable = False
    st.session_state.is_resumable_logged = False
    
