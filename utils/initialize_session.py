import streamlit as st
from collections import deque
import queue

from utils.config_utils import load_config

CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG = load_config(CONFIG_PATH)

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

    
    # Store the split ratio (e.g., 0.8 for train, 0.2 for test)
    if "test_split_ratio" not in st.session_state:
        st.session_state.test_split_ratio = 0.2

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

    # --- NEW: Storage for the Test DataFrame (when split occurs) ---
    if "test_dataset_df" not in st.session_state:
        st.session_state.test_dataset_df = None