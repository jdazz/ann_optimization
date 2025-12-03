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
    This runs on every script execution. If a background training thread is
    already alive, we reuse its stop_event and update_queue and mark the UI as
    running so progress/stop controls stay active after a reload.
    """
    ss = st.session_state

    thread = ss.get("training_thread")
    is_thread_alive = thread and isinstance(thread, threading.Thread) and thread.is_alive()

    # If no tracked thread, try to find a worker by name and recover its stop_event.
    if not is_thread_alive:
        for t in threading.enumerate():
            if t.name.startswith("training_worker_") and t.is_alive():
                thread = t
                is_thread_alive = True
                ss.training_thread = t
                if hasattr(t, "stop_event"):
                    ss.stop_event = t.stop_event
                break

    if is_thread_alive:
        # Thread exists → keep existing queue/stop_event references.
        ss.is_running = True
        ss.is_resumable = True  # can be stopped/resumed after reload
        ss.was_stopped_manually = False
        return

    # Thread is not alive → clean references but do not recreate queue here
    ss.is_running = False
    ss.training_thread = None
    ss.stop_event = None
    # If a run was previously marked IN_PROGRESS but no worker exists, clear run_status
    if ss.get("run_status") == "IN_PROGRESS":
        ss.run_status = None
    # Ensure resume flag is cleared when no worker is present
    ss.is_resumable = False

    was_stopped = ss.get("was_stopped_manually", False)
    study_exists = ss.get("optuna_study") is not None
    current_trial = ss.get("current_trial_number", 0)
    total_trials = ss.get("total_trials", 50)
    is_completed = current_trial >= total_trials

    if study_exists and is_completed:
        ss.is_resumable = False
        ss.log_messages.append("Optimization finished and locked.")
        return

    if study_exists and not is_completed:
        ss.is_resumable = True
        if not was_stopped and not ss.get("is_resumable_logged", False):
            ss.log_messages.append(
                f"⚠️ Recovered persistent study state. Ready to resume from Trial {current_trial}."
            )
            ss.is_resumable_logged = True
        return

    ss.is_resumable = False
    ss.is_resumable_logged = False
