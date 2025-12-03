# ui/control_panel.py

import streamlit as st
import threading
import queue
import time
import os
import io
import zipfile
from sklearn.preprocessing import StandardScaler
from utils.run_manager import read_value_file
from utils.config_utils import save_config, load_config
from utils.state_manager import initialize_run_state, reset_resumable_flag
from utils.data_utils import initialize_training_dataset
from utils.save_scaler import save_scaler_to_json
from utils.run_manager import derive_dataset_name, make_config_hash, find_existing_run, finalize_run_folder
from core.pipeline import run_training_pipeline 

CONFIG_PATH = "config.yaml"

def handle_run_pipeline(is_resume: bool):
    """
    General handler for starting a new run or resuming an incomplete one.

    - Saves current UI config to CONFIG_PATH and reloads it into st.session_state.config.
    - Prepares temp_data files for train/test.
    - Initializes Dataset objects (train + test).
    - Applies standardization if enabled.
    - Starts the training pipeline in a background thread using run_training_pipeline.
    """

    # Guard against starting a new thread while one is already running
    active_thread = st.session_state.get("training_thread")
    if active_thread and isinstance(active_thread, threading.Thread) and active_thread.is_alive():
        st.warning("A training run is already in progress. Please stop it before starting a new one.")
        return
    if st.session_state.get("run_status") == "IN_PROGRESS":
        st.warning("Found an on-disk run marked IN_PROGRESS. Please finish/clean it before starting a new one.")
        return

    # 1. Configuration Setup
    latest_ui_config = st.session_state.current_ui_config
    try:
        # Persist UI config to disk, then reload as the "authoritative" config
        save_config(latest_ui_config, CONFIG_PATH)
        st.session_state.config = load_config(CONFIG_PATH)

        st.session_state.total_trials = (
            st.session_state.config
            .get("hyperparameter_search_space", {})
            .get("n_samples", 50)
        )

        # --- CRITICAL: CLEAR STOP FLAG ON RESTART/RESUME ---
        st.session_state.was_stopped_manually = False
        st.session_state.is_resumable = True
        # ---------------------------------------------------

        if is_resume:
            st.success(
                f"Resuming pipeline from Trial {st.session_state.current_trial_number}..."
            )
        else:
            # Fresh run: reset run-related state (best loss, metrics, etc.)
            initialize_run_state()
            st.session_state.optuna_study = None  # Ensure new run starts clean
            st.success(
                f"Starting NEW pipeline with {st.session_state.total_trials} trials."
            )

    except Exception as e:
        st.error(f"Failed to save or load configuration: {e}")
        return

    # 2. File and Directory Setup
    os.makedirs("temp_data", exist_ok=True)
    model_dir = os.path.dirname(st.session_state.best_model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    persistent_train_file = st.session_state.get("persistent_train_file_obj")
    persistent_test_file = st.session_state.get("persistent_test_file_obj")

    train_path = None
    test_path = None

    if persistent_train_file:
        file_ext = persistent_train_file.name.split(".")[-1]
        train_path = f"temp_data/train_data.{file_ext}"

        # Save files if they don't exist or if starting fresh
        if not is_resume or not os.path.exists(train_path):
            with open(train_path, "wb") as f:
                f.write(persistent_train_file.getvalue())

        if persistent_test_file:
            test_ext = persistent_test_file.name.split(".")[-1]
            test_path = f"temp_data/test_data.{test_ext}"
            if not is_resume or not os.path.exists(test_path):
                with open(test_path, "wb") as f:
                    f.write(persistent_test_file.getvalue())
    else:
        st.error("No data found. Please upload a training file.")
        return

    # 3. Initialize Dataset objects (train + test)
    dataset_train, dataset_test = initialize_training_dataset(train_path, test_path)
    st.session_state.dataset_input_vars = dataset_train.input_vars
    st.session_state.dataset_name = derive_dataset_name(dataset_train)
    st.session_state.config_hash = make_config_hash(st.session_state.config)

    # 4. Standardization
    should_standardize = (
        st.session_state.config.get("cross_validation", {}).get("standardize_features", False)
    )
    if should_standardize:
        if is_resume and st.session_state.get("fitted_scaler") is not None:
            # Apply previously-fitted scaler to the training dataset
            dataset_train.apply_scaler(
                scaler=st.session_state.fitted_scaler,
                is_fitting=False,
            )
        else:
            scaler = StandardScaler()
            dataset_train.apply_scaler(scaler=scaler, is_fitting=True)
            st.session_state.fitted_scaler = dataset_train.scaler

    # Ensure the update_queue exists (reuse if already present)
    if "update_queue" not in st.session_state or st.session_state.update_queue is None:
        st.session_state.update_queue = queue.Queue()

    # 5. Start Background Thread (NO st.session_state passed into worker)
    stop_event = threading.Event()
    st.session_state.stop_event = stop_event

    thread = threading.Thread(
        target=run_training_pipeline,
        args=(
            dataset_train,                 # training Dataset
            dataset_test,                  # test Dataset or DataFrame-like
            st.session_state.config,       # config dict
            st.session_state.update_queue, # Queue for UI updates
            stop_event,                    # stop signal
            is_resume,                     # resume flag
            st.session_state.get("optuna_study"),  # existing study (for resume)
        ),
        name=f"training_worker_{st.session_state.dataset_name}__{st.session_state.config_hash}",
        daemon=True,
    )
    # Attach stop_event to thread so we can recover it on reattachment
    thread.stop_event = stop_event

    st.session_state.training_thread = thread
    st.session_state.is_running = True

    thread.start()
    st.rerun()

def handle_stop_training():
    """Logic executed when the Stop button is pressed."""
    ss = st.session_state

    # Try to recover missing stop_event/training_thread (after reload) so we can signal the worker
    if not ss.get("stop_event"):
        for t in threading.enumerate():
            if t.name.startswith("training_worker_") and t.is_alive():
                ss.training_thread = t
                if hasattr(t, "stop_event"):
                    ss.stop_event = t.stop_event
                break

    if ss.get("stop_event"):
        ss.log_messages.append("--- STOP signal sent! ---")
        ss.stop_event.set()
        time.sleep(0.5)
        t = ss.get("training_thread")
        if t and hasattr(t, "join"):
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

    # Rename current run folder to _ABORTED if it exists
    run_dir = ss.get("current_run_dir")
    if run_dir:
        try:
            finalize_run_folder(run_dir, "ABORTED")
            ss.current_run_dir = os.path.join(
                os.path.dirname(run_dir),
                os.path.basename(run_dir).rsplit("__", 1)[0] + "__ABORTED",
            )
        except Exception:
            pass

    # Clear references and flags so Start is re-enabled
    ss.training_thread = None
    ss.stop_event = None
    ss.was_stopped_manually = True
    ss.is_running = False
    ss.is_resumable = False
    ss.run_status = "ABORTED"

    time.sleep(0.2)
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
        # Show logs whenever we have a run directory, even after reloads.
        if (
            is_running
            or st.session_state.final_model_path
            or best_model_exists
            or is_resumable
            or st.session_state.get("current_run_dir")
        ):
            st.header("Live Logs")
            log_container = st.empty()

            summary_text = ""
            run_dir = st.session_state.get("current_run_dir")
            if run_dir:
                summary_path = os.path.join(run_dir, "summary.txt")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r") as f:
                            lines = f.readlines()
                        # newest first
                        summary_text = "".join(reversed(lines))
                    except Exception:
                        summary_text = ""

            if not summary_text:
                summary_text = "\n".join(list(st.session_state.log_messages)[::-1])

            log_container.text_area("Logs", value=summary_text, height=400, disabled=True)

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
        # Helper to normalize metric values
        # ---------------------------------------------------------------------
        def _normalize_metric(val):
            """Convert file/session values into a clean numeric or None."""
            if val is None:
                return None
            if isinstance(val, str):
                s = val.strip()
                if s == "" or s.lower() in ("none", "nan"):
                    return None
                try:
                    return float(s)
                except Exception:
                    return None
            # Already numeric
            if isinstance(val, (int, float)):
                return float(val)
            return None

        # ---------------------------------------------------------------------
        # Best CV loss + live best test metrics
        # ---------------------------------------------------------------------
        run_dir = ss.get("current_run_dir")

        best_loss = read_value_file(run_dir, "best_cv_loss.txt")
        best_r2 = read_value_file(run_dir, "best_intermediate_r2.txt")
        best_nmae = read_value_file(run_dir, "best_intermediate_nmae.txt")

        if best_loss is None:
            best_loss = ss.get("best_loss_so_far")
        if best_r2 is None:
            best_r2 = ss.get("best_intermediate_r2")
        if best_nmae is None:
            best_nmae = ss.get("best_intermediate_nmae")

        # Normalize values
        best_loss = _normalize_metric(best_loss)
        best_r2 = _normalize_metric(best_r2)
        best_nmae = _normalize_metric(best_nmae)

        live_test = ss.get("live_best_test_metrics")
        if not isinstance(live_test, dict):
            live_test = {}

        # 3 metric columns: CV Loss, R², NMAE
        m_col1, m_col2, m_col3 = st.columns(3)

        # Before first trial completes, we want to show "--"
        first_trial_not_done = (current == 0)

        # ----------------- Best CV Loss -----------------
        with m_col1:
            if first_trial_not_done or best_loss is None or best_loss == float("inf"):
                st.metric("Best CV Loss (So Far)", "--")
            else:
                try:
                    st.metric("Best CV Loss (So Far)", f"{float(best_loss):.6f}")
                except Exception:
                    st.metric("Best CV Loss (So Far)", "--")

        # ----------------- Best R² -----------------
        with m_col2:
            # Fallback to live_test["R2"] if no intermediate best yet
            val = best_r2 if best_r2 is not None else _normalize_metric(live_test.get("R2"))

            if first_trial_not_done or val is None:
                st.metric("Best R²", "--")
            else:
                st.metric("Best R²", f"{val:.4f}")

        # ----------------- Best NMAE -----------------
        with m_col3:
            # Fallback to live_test["NMAE"] if no intermediate best yet
            val = best_nmae if best_nmae is not None else _normalize_metric(live_test.get("NMAE"))

            if first_trial_not_done or val is None:
                st.metric("Best NMAE", "--")
            else:
                st.metric("Best NMAE", f"{val:.4f}")

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
