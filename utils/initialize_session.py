import streamlit as st
from collections import deque
import queue
import copy

from utils.config_utils import load_config

CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG = load_config(CONFIG_PATH)

def initialize_session_state():
    """
    Initializes all required session state variables, ensuring run metrics 
    (like best_loss_so_far and current_trial_number) are preserved across reloads.
    """

    # --- 1. CONFIGURATION (Must be loaded first) ---
    # Load configuration from disk path only once, or if it doesn't exist.
    if "config" not in st.session_state:
        # Assuming load_config is accessible in this file's environment
        st.session_state.config = load_config(CONFIG_PATH)
        st.session_state.current_ui_config = copy.deepcopy(st.session_state.config)
    
    # current_ui_config should be a separate copy updated by the UI
    if "current_ui_config" not in st.session_state:
        st.session_state.current_ui_config = copy.deepcopy(st.session_state.config)


    # --- 2. RUN CONTROL & THREAD PERSISTENCE (Crucial for Re-attachment) ---
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    
    # Thread objects and events MUST be preserved to re-attach
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = None
    if "training_thread" not in st.session_state:
        st.session_state.training_thread = None
    if "update_queue" not in st.session_state:
        st.session_state.update_queue = queue.Queue()


    # --- 3. RUN METRICS (Preserved across reloads to show progress) ---
    # Total trials must be defined once, preferably using the configuration.
    if "total_trials" not in st.session_state:
        st.session_state.total_trials = st.session_state.config.get(
            'hyperparameter_search_space', {}
        ).get('n_samples', 50)

    if "current_trial_number" not in st.session_state:
        st.session_state.current_trial_number = 0
    if "best_loss_so_far" not in st.session_state:
        st.session_state.best_loss_so_far = float("inf")
    if "best_params_so_far" not in st.session_state:
        st.session_state.best_params_so_far = {}
    
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = deque(maxlen=200)

    # State flag to indicate manual interruption (used for final log messages)
    if "was_stopped_manually" not in st.session_state:
        st.session_state.was_stopped_manually = False
    
    if "is_resumable" not in st.session_state:
        st.session_state.is_resumable = False


    # --- 4. MODEL / DATA STORAGE & PATHS (Preserved for post-run analysis/download) ---
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

    # Scaler/Dataset info
    if "fitted_scaler" not in st.session_state:
        st.session_state.fitted_scaler = None
    if "dataset_input_vars" not in st.session_state:
        st.session_state.dataset_input_vars = []
        
    # Data Splitting Info
    if "test_split_ratio" not in st.session_state:
        st.session_state.test_split_ratio = 0.2
    if "test_dataset_df" not in st.session_state:
        st.session_state.test_dataset_df = None # Stores the split data if only one file is uploaded
        
    # Optuna Study Storage
    if "optuna_study" not in st.session_state:
        st.session_state.optuna_study = None
    
    if "target_feature" not in st.session_state:
        st.session_state.target_features = []
    if "input_features" not in st.session_state:
        st.session_state.input_features = []