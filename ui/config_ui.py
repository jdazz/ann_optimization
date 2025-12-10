# FILE: ui/config_ui.py - REVISED LAYOUT

import streamlit as st
import numpy as np
import copy
from utils.config_utils import load_config, save_config

def render_config_ui(default_config, config_path):
    """
    Renders the Model Configuration and Advanced Settings in a single column.
    File upload and primary column selection are assumed to be handled in app.py.
    Configuration changes are collected and stored in st.session_state.current_ui_config
    IMMEDIATELY upon user interaction.
    """
    # -------------------------------------------------------------------------
    # --- UI Layout: Header and Info ---
    # -------------------------------------------------------------------------
    st.markdown("## Model Configuration")
    st.info("Adjust parameters below. All settings update automatically upon change.") 

    # Load config and handle error
    try:
        current_config = load_config(config_path)
    except FileNotFoundError:
        st.error(f"Error: config.yaml not found.")
        st.stop()

    ui_config = copy.deepcopy(current_config)

    # Initialize new sections if they don't exist (same as before)
    if "variables" not in ui_config:
        ui_config["variables"] = {"input_names": "Feature_1\nFeature_2", "output_names": "Target_1"}
    if "targets" not in ui_config:
         ui_config["targets"] = {"mre_threshold": 25}
    if "cross_validation" not in ui_config: 
        ui_config["cross_validation"] = {}
    
    ui_config["cross_validation"]["test_split_ratio"] = ui_config["cross_validation"].get("test_split_ratio", 0.2)
    
    if "display" not in ui_config:
        ui_config["display"] = {}
    ui_config["display"]["show_prediction_plot"] = ui_config["display"].get("show_prediction_plot", True)
    ui_config["display"]["show_optuna_plots"] = ui_config["display"].get("show_optuna_plots", True)
    
    # -------------------------------------------------------------------------
    # --- Single Column for Core Data/Optimization Settings (Non-Expander) ---
    # -------------------------------------------------------------------------
    
    # --- 1. Number of Trials ---
    hpo_conf = ui_config.get("hyperparameter_search_space", {})
    hpo_conf["n_samples"] = st.number_input(
        "Optuna Trials (n_samples)", 
        min_value=1, 
        value=hpo_conf.get("n_samples", 50),
        key='n_samples_hpo',
        help="The number of optimization trials to run during Hyperparameter Search."
    )
    ui_config["hyperparameter_search_space"] = hpo_conf
    
    cv_conf = ui_config.get("cross_validation", {})
    
    # --- 2. Standardize Features (Moved here) ---
    ui_config["cross_validation"]["standardize_features"] = st.checkbox(
        "Standardize Input Features (Z-Score)",
        value=cv_conf.get("standardize_features", True),
        key='standardize_data_check',
        help="If checked, input features will be normalized to have zero mean and unit variance."
    )
    
    # Keep the current split ratio from config; UI is handled in the upload flow.
    ui_config["cross_validation"]["test_split_ratio"] = cv_conf.get("test_split_ratio", 0.2)
    st.markdown("---")

    
    # -------------------------------------------------------------------------
    # --- Advanced Parameters (Expanders) ---
    # -------------------------------------------------------------------------
    st.markdown("### Advanced Parameters")
    
        
    # --- Network Architecture ---
    with st.expander("Network Architecture", expanded=True):
        net_conf = ui_config.get("network", {})
        
        # Multiselect for Activation Functions
        all_activations = ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU"]
        current_choices = net_conf.get("activation_functions", {}).get("choices", ["ReLU", "Sigmoid", "Tanh"])
        
        selected_activations = st.multiselect(
            "Activation Functions to Sample",
            options=all_activations,
            default=current_choices,
            key='activation_choices',
            help="Select which activation functions Optuna should consider for the hidden layers."
        )
        
        net_conf["activation_functions"] = {"choices": selected_activations}
        
        h_layers = net_conf.get("hidden_layers", {"low": 2, "high": 4})
        hidden_layers_range = st.slider(
            "Hidden Layers Range (Min - Max)",
            min_value=1, max_value=10,
            value=(h_layers.get("low", 2), h_layers.get("high", 4)),
            key='h_layers_range'
        )
        net_conf["hidden_layers"] = {"low": hidden_layers_range[0], "high": hidden_layers_range[1]}

        h_neurons = net_conf.get("hidden_neurons", {"low": 30, "high": 60})
        hidden_neurons_range = st.slider(
            "Hidden Neurons Range (Min - Max)",
            min_value=16, max_value=256,
            value=(h_neurons.get("low", 30), h_neurons.get("high", 60)),
            key='h_neurons_range'
        )
        net_conf["hidden_neurons"] = {"low": hidden_neurons_range[0], "high": hidden_neurons_range[1]}
        ui_config["network"] = net_conf


    # --- Hyperparameter Search Ranges & Choices (Now includes K-fold and MRE) ---
    with st.expander("Hyperparameter Search Ranges & Choices", expanded=False):
        hpo_conf = ui_config.get("hyperparameter_search_space", {})
        
        # --- K-Fold Splits (MOVED HERE) ---
        ui_config["cross_validation"]["kfold"] = st.number_input(
            "K-Fold Splits", 
            min_value=2, 
            value=cv_conf.get("kfold", 5),
            key='kfold_splits_hpo', # Updated key for separation if needed
            help="The number of folds for cross-validation within each Optuna trial."
        )
        
        # --- MRE Threshold (MOVED HERE) ---
        targets_conf = ui_config.get("targets", {})
        ui_config["targets"]["mre_threshold"] = st.number_input(
            "MRE Threshold (%)", 
            min_value=1.0, 
            max_value=100.0,
            value=float(targets_conf.get("mre_threshold", 25.0)),
            step=1.0,
            key='mre_thresh'
        )
        st.markdown("---")
        
        # Multiselect for Optimizers
        all_optimizers = ["Adam", "SGD", "RMSprop", "Adagrad", "AdamW", "LBFGS"]
        current_optimizer_choices = hpo_conf.get("optimizer_name", {}).get("choices", ["Adam"])
        
        selected_optimizers = st.multiselect(
            "Optimizers to Sample",
            options=all_optimizers,
            default=current_optimizer_choices,
            key='optimizer_choices',
            help="Select which optimizers Optuna should consider for training."
        )
        
        hpo_conf["optimizer_name"] = {"choices": selected_optimizers, "type": "categorical"}
        
        lr = hpo_conf.get("learning_rate", {"low": 0.0001, "high": 0.01})
        current_lr_low = np.log10(lr.get("low", 0.0001))
        current_lr_high = np.log10(lr.get("high", 0.01))
        
        lr_exp_low, lr_exp_high = st.slider(
            "Learning Rate Range (10^X)", 
            min_value=-5.0, max_value=-1.0,
            value=(float(current_lr_low), float(current_lr_high)),
            format="10^%.2f",
            key='lr_range'
        )
        hpo_conf["learning_rate"] = {"low": 10**lr_exp_low, "high": 10**lr_exp_high}
        
        bs = hpo_conf.get("batch_size", {"low": 50, "high": 150})
        bs_low, bs_high = st.slider(
            "Batch Size Range", 
            min_value=16, max_value=512, 
            value=(bs.get("low", 50), bs.get("high", 150)),
            key='bs_range'
        )
        hpo_conf["batch_size"] = {"low": bs_low, "high": bs_high}
        
        ep = hpo_conf.get("epochs", {"low": 50, "high": 200})
        epochs_low, epochs_high = st.slider(
            "Number of Epochs Range",
            min_value=10, max_value=1000,
            value=(ep.get("low", 50), ep.get("high", 200)),
            step=10,
            key='epochs_range'
        )
        hpo_conf["epochs"] = {"low": epochs_low, "high": epochs_high}
        ui_config["hyperparameter_search_space"] = hpo_conf

    # -------------------------------------------------------------------------
    # --- Store Configuration ---
    # -------------------------------------------------------------------------
   
    # Store the latest UI configuration into session state for the main app to access
    st.session_state.current_ui_config = ui_config
    
    return ui_config
