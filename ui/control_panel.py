# ui/control_panel.py

import streamlit as st
import threading
import time
import os
import io
import zipfile
from sklearn.preprocessing import StandardScaler
from utils.config_utils import save_config, load_config
from utils.state_manager import initialize_run_state, reset_resumable_flag
from utils.data_utils import initialize_training_dataset
from utils.save_scaler import save_scaler_to_json
from core.pipeline import run_training_pipeline 

CONFIG_PATH = "config.yaml"

def handle_run_pipeline(is_resume: bool):
    """
    General handler for starting a new run or resuming an incomplete one.
    """
    # 1. Configuration Setup
    latest_ui_config = st.session_state.current_ui_config
    try:
        save_config(latest_ui_config, CONFIG_PATH)
        st.session_state.config = load_config(CONFIG_PATH)
        st.session_state.total_trials = st.session_state.config.get(
            'hyperparameter_search_space', {}
        ).get('n_samples', 50)

        # --- CRITICAL: CLEAR STOP FLAG ON RESTART/RESUME ---
        st.session_state.was_stopped_manually = False
        st.session_state.is_resumable = True
        # ---------------------------------------------------

        if is_resume:
            st.success(f"Resuming pipeline from Trial {st.session_state.current_trial_number}...")
        else:
            initialize_run_state()
            st.session_state.optuna_study = None # Ensure new run starts clean
            st.success(f"Starting NEW pipeline with {st.session_state.total_trials} trials.")
        
    except Exception as e:
        st.error(f"Failed to save or load configuration: {e}")
        return 

    # 2. File and Directory Setup (Standard)
    os.makedirs("temp_data", exist_ok=True)
    model_dir = os.path.dirname(st.session_state.best_model_path)
    if model_dir: os.makedirs(model_dir, exist_ok=True)
    
    persistent_train_file = st.session_state.get('persistent_train_file_obj')
    persistent_test_file = st.session_state.get('persistent_test_file_obj')
    
    train_path = None
    test_path = None
    
    if persistent_train_file:
        file_ext = persistent_train_file.name.split('.')[-1]
        train_path = f"temp_data/train_data.{file_ext}"
        
        # Save files if they don't exist or if starting fresh
        if not is_resume or not os.path.exists(train_path):
            with open(train_path, "wb") as f: f.write(persistent_train_file.getvalue())
        
        if persistent_test_file:
            test_ext = persistent_test_file.name.split('.')[-1]
            test_path = f"temp_data/test_data.{test_ext}"
            if not is_resume or not os.path.exists(test_path):
                with open(test_path, "wb") as f: f.write(persistent_test_file.getvalue())
    else:
        st.error("No data found. Please upload a training file.")
        return 
        
    # 3. Initialize Dataset
    dataset_train, dataset_test = initialize_training_dataset(train_path, test_path)
    st.session_state.dataset_input_vars = dataset_train.input_vars

    # 4. Standardization
    should_standardize = st.session_state.config.get("cross_validation", {}).get("standardize_features", False)
    if should_standardize:
        if is_resume and st.session_state.fitted_scaler:
            dataset_train.apply_scaler(scaler=st.session_state.fitted_scaler, is_fitting=False)
        else:
            scaler = StandardScaler()
            dataset_train.apply_scaler(scaler=scaler, is_fitting=True)
            st.session_state.fitted_scaler = dataset_train.scaler

    # 5. Start Thread
    st.session_state.stop_event = threading.Event()
    thread = threading.Thread(
        target=run_training_pipeline,
        args=(
            dataset_train,
            dataset_test,
            st.session_state.config,
            st.session_state.update_queue,
            st.session_state,
            st.session_state.stop_event,
            is_resume
        ),
        daemon=True
    )
    st.session_state.training_thread = thread
    st.session_state.is_running = True
    thread.start()
    st.rerun()

def handle_stop_training():
    """Logic executed when the Stop button is pressed."""
    if st.session_state.stop_event:
        st.session_state.log_messages.append("--- STOP signal sent! ---")
        st.session_state.stop_event.set()
        
        # Mark as manually stopped
        st.session_state.was_stopped_manually = True
        
        # Force UI to think it's stopped immediately (enables Resume button on next rerun)
        st.session_state.is_running = False
        st.session_state.is_resumable = True
    time.sleep(0.5)
    st.rerun()

def render_control_panel():
    """Renders the main control panel with 3 distinct buttons."""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Training Control")

        is_running = st.session_state.is_running
        is_resumable = st.session_state.get('is_resumable', False)
        is_data_ready = st.session_state.get('is_data_ready_for_new_run', False)

        # 1. Start Training (New Study)
        if st.button(
            "Start Training (New Study)", 
            type="primary", 
            disabled=is_running or not is_data_ready,
            help="Starts a completely new optimization study."
        ):
            handle_run_pipeline(is_resume=False)

        # 2. Stop Training
        if st.button(
            "Stop Training", 
            type="secondary", 
            disabled=not is_running
        ):
            handle_stop_training()


        # 3. Resume Training
        # Enabled only if NOT running AND Resumable
        if st.button(
            "Resume Training", 
            disabled=is_running or not is_resumable,
            help="Resumes the previous study from where it left off."
        ):
             handle_run_pipeline(is_resume=True)

        # ... (Rest of status rendering) ...
        best_model_exists = os.path.exists(st.session_state.best_model_path)
        if is_running or best_model_exists or is_resumable:
            render_live_status(col1, best_model_exists)

    with col2:
        if is_running or st.session_state.final_model_path or best_model_exists or is_resumable:
            st.header("Live Logs")
            log_container = st.empty()
            log_text = "\n".join(list(st.session_state.log_messages)[::-1])
            log_container.text_area("Logs", value=log_text, height=400, disabled=True)

def render_live_status(col, best_model_exists):
    with col:
        ss = st.session_state

        st.subheader("Live HPO Status")

        # ---------------------------------------------------------------------
        # High-level status
        # ---------------------------------------------------------------------
        if ss.get("is_running", False):
            st.info("Training is in progress...")
        elif ss.get("is_resumable", False):
            st.warning("Training Paused. Resume or Start New.")
        elif ss.get("final_model_path"):
            st.success("Training finished!")
        else:
            st.warning("Ready to start.")

        # ---------------------------------------------------------------------
        # Trial progress bar
        # ---------------------------------------------------------------------
        total = ss.get("total_trials", 0) or 0
        current = ss.get("current_trial_number", 0) or 0

        if total > 0:
            progress = min(current / total, 1.0)
        else:
            progress = 0.0

        st.progress(progress, text=f"Trials: {current} / {total}")

        # ---------------------------------------------------------------------
        # Best CV loss + live best test metrics
        # ---------------------------------------------------------------------
        best_loss = ss.get("best_loss_so_far", float("inf"))
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            if best_loss == float("inf"):
                st.metric("Best CV Loss (So Far)", "—")
            else:
                st.metric("Best CV Loss (So Far)", f"{best_loss:.6f}")

        # Live test metrics for the best model so far (if available)
        live_test = ss.get("live_best_test_metrics")
        if isinstance(live_test, dict) and live_test:
            nmae = live_test.get("NMAE")
            r2 = live_test.get("R2")
            acc = live_test.get("Accuracy")

            with m_col2:
                if nmae is not None:
                    st.metric("Best Test NMAE", f"{nmae:.4f}")
                else:
                    st.metric("Best Test NMAE", "—")

            with m_col3:
                if r2 is not None:
                    st.metric("Best Test R²", f"{r2:.4f}")
                else:
                    st.metric("Best Test R²", "—")

            # # Optional: add a second row just for Accuracy
            # if acc is not None:
            #     a_col1, a_col2, a_col3 = st.columns(3)
            #     with a_col1:
            #         st.metric("Best Test Accuracy", f"{acc:.2f}%")
        else:
            # No test metrics yet – keep layout stable
            with m_col2:
                st.metric("Best Test NMAE", "—")
            with m_col3:
                st.metric("Best Test R²", "—")

        # ---------------------------------------------------------------------
        # Intermediate model download
        # ---------------------------------------------------------------------
        render_intermediate_download(best_model_exists)

        # ---------------------------------------------------------------------
        # Best hyperparameters
        # ---------------------------------------------------------------------
        st.subheader("Best Hyperparameters")
        best_params = ss.get("best_params_so_far", {})
        st.json(best_params, expanded=False)

def render_intermediate_download(best_model_exists):
    """Renders the intermediate best model download expander."""
    has_best_loss = st.session_state.best_loss_so_far != float("inf")
    
    if best_model_exists and has_best_loss:
        try:
            with open(st.session_state.best_model_path, "rb") as f:
                model_data = f.read()
            
            if len(model_data) > 0:
                best_onnx_path = st.session_state.get("best_onnx_path")
                with st.expander("Download Intermediate Model"):
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            label="Download best .pt model",
                            data=model_data,
                            file_name="best_intermediate.pt",
                            mime="application/octet-stream"
                        )
                    with dl_col2:
                        if best_onnx_path and os.path.exists(best_onnx_path):
                            with open(best_onnx_path, "rb") as f_onnx:
                                st.download_button(
                                    label="Download best .onnx model",
                                    data=f_onnx.read(),
                                    file_name="best_intermediate.onnx",
                                    mime="application/octet-stream"
                                )
                        else:
                            st.info("ONNX for best model not available yet.")
        except Exception:
            pass
