import streamlit as st
import os
import torch
import yaml
import io
import contextlib
import copy
import numpy as np
import tempfile # NEW: For creating temporary directories
import shutil # NEW: For cleaning up temporary directories
import matplotlib.pyplot as plt

# --- Import your project's source files ---
# This assumes 'app.py' is in the root folder, and 'src' is a subfolder.
from src.dataset import Dataset
from src.train import find_best_model
from src.model_test import test
from src.model import define_net_regression
from src.plot import make_plot

# --- Constants ---
CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")

# --- Helper Functions ---
@st.cache_data
def load_config(path):
    """Loads the config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config_data, path):
    """Saves the config data to the file."""
    with open(path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

# --- Session State Initialization ---
# 'default_config' holds the original file values to allow reset
if 'default_config' not in st.session_state:
    st.session_state.default_config = load_config(CONFIG_PATH)

if 'training_results' not in st.session_state:
    st.session_state.training_results = None
    
if 'log_output' not in st.session_state:
    st.session_state.log_output = ""

if 'uploaded_train_file' not in st.session_state:
    st.session_state.uploaded_train_file = None

if 'uploaded_test_file' not in st.session_state:
    st.session_state.uploaded_test_file = None


# --- Sidebar UI for Configuration ---
st.sidebar.title("Model Configuration")
st.sidebar.info("Please upload your training and testing data files here and enter the right features and target. Adjust the model parameters as needed, then click 'Save configs' to apply changes before training.")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Load the CURRENT config file for display
try:
    current_config = load_config(CONFIG_PATH)
except FileNotFoundError:
    st.error(f"Error: config.yaml not found at {CONFIG_PATH}")
    st.stop()

# Use a deepcopy to create the UI form, so we don't alter the loaded dict
ui_config = copy.deepcopy(current_config)

# Initialize new sections if they don't exist in the loaded config
if "variables" not in ui_config:
    ui_config["variables"] = {"input_names": "Feature_1\nFeature_2", "output_names": "Target_1"}
if "targets" not in ui_config:
     ui_config["targets"] = {"mre_threshold": 25}

# --- Create UI elements for each config section ---
with st.sidebar.form("config_form"):
    # Data Section (Now uses file uploaders)
    with st.expander("Data Upload", expanded=True):
        st.session_state.uploaded_train_file = st.file_uploader(
            "Upload Training Data (CSV, JSON, etc.)", 
            type=['csv', 'json', 'xls', 'xlsx', 'parquet'],
            key='train_uploader'
        )
        st.session_state.uploaded_test_file = st.file_uploader(
            "Upload Testing Data (CSV, JSON, etc.)", 
            type=['csv', 'json', 'xls', 'xlsx', 'parquet'],
            key='test_uploader'
        )
        # Removed text_input for paths, which are now irrelevant for UI config storage
        # However, we must ensure the keys exist if we don't remove them globally later
        # We will keep them for config compatibility but ignore them during training
        # if "training_path" in ui_config["data"]: del ui_config["data"]["training_path"] 
        # if "testing_path" in ui_config["data"]: del ui_config["data"]["testing_path"]


    # --- Variables & Targets ---
    with st.expander("Variables & Targets", expanded=True):
        # Input/Feature Variables
        var_conf = ui_config.get("variables", {})
        
        input_names_str = var_conf.get("input_names", "Feature_1\nFeature_2")
        ui_config["variables"]["input_names"] = st.text_area(
            "Input/Feature Variables (One per line)",
            input_names_str
        )
        
        # Output/Target Variables
        output_names_str = var_conf.get("output_names", "Target_1")
        ui_config["variables"]["output_names"] = st.text_area(
            "Output/Target Variables (One per line)",
            output_names_str
        )
        
        # MRE Threshold (from targets section)
        targets_conf = ui_config.get("targets", {})
        ui_config["targets"]["mre_threshold"] = st.number_input(
            "MRE Threshold (%)", 
            min_value=1.0, 
            max_value=100.0,
            value=float(targets_conf.get("mre_threshold", 25.0)),
            step=1.0,
            help="Maximum acceptable Mean Relative Error for a sample to be considered accurate."
        )

    # Cross Validation
    with st.expander("Cross Validation"):
        ui_config["cross_validation"]["kfold"] = st.number_input(
            "K-Fold Splits", 
            min_value=2, 
            value=ui_config.get("cross_validation", {}).get("kfold", 5)
        )
    
    # Network Architecture
    with st.expander("Network Architecture"):
        net_conf = ui_config.get("network", {})

    # --- Hidden Layers (range slider) ---
        hidden_layers_low = net_conf.get("hidden_layers", {}).get("low", 2)
        hidden_layers_high = net_conf.get("hidden_layers", {}).get("high", 4)
        hidden_layers_range = st.slider(
        "Hidden Layers Range (Min - Max)",
            min_value=1,
            max_value=10,
            value=(hidden_layers_low, hidden_layers_high),
            help="Select the minimum and maximum number of hidden layers."
        )
        net_conf["hidden_layers"] = {
            "low": hidden_layers_range[0],
            "high": hidden_layers_range[1]
        }

    # --- Hidden Neurons (range slider) ---
        hidden_neurons_low = net_conf.get("hidden_neurons", {}).get("low", 30)
        hidden_neurons_high = net_conf.get("hidden_neurons", {}).get("high", 60)
        hidden_neurons_range = st.slider(
            "Hidden Neurons Range (Min - Max)",
            min_value=16,
            max_value=256,
            value=(hidden_neurons_low, hidden_neurons_high),
            help="Select the minimum and maximum number of neurons per hidden layer."
        )
        net_conf["hidden_neurons"] = {
            "low": hidden_neurons_range[0],
            "high": hidden_neurons_range[1]
        }

    # Update config dictionary
        ui_config["network"] = net_conf

    # Hyperparameter Search
    with st.expander("Hyperparameter Search"):
        hpo_conf = ui_config.get("hyperparameter_search_space", {})
        hpo_conf["n_samples"] = st.number_input(
            "Optuna Trials (n_samples)", 
            min_value=1, 
            value=hpo_conf.get("n_samples", 50)
        )

        # --- Learning Rate (log-scale range slider) ---
        current_lr_low = np.log10(hpo_conf.get("learning_rate", {}).get("low", 0.0001))
        current_lr_high = np.log10(hpo_conf.get("learning_rate", {}).get("high", 0.01))

        lr_exp_low, lr_exp_high = st.slider(
            "Learning Rate Range (10^X)", 
            min_value=-5.0,  # 1e-5
            max_value=-1.0,  # 1e-1
            value=(float(current_lr_low), float(current_lr_high)),
            format="10^%.2f",
            help="Select the range for learning rate in log10 scale (e.g., -4 = 1e-4)"
        )

        # Convert exponents back to linear values
        hpo_conf["learning_rate"] = {
            "low": float(10**lr_exp_low),
            "high": float(10**lr_exp_high)
        }

        # --- Batch Size Range ---
        bs_low, bs_high = st.slider(
            "Batch Size Range", 
            min_value=16, 
            max_value=512, 
            value=(
                hpo_conf.get("batch_size", {}).get("low", 50),
                hpo_conf.get("batch_size", {}).get("high", 150)
            ),
            help="Range of batch sizes to search during optimization."
        )
        hpo_conf["batch_size"] = {"low": bs_low, "high": bs_high}

        # --- Epochs Range (NEW) ---
        epochs_low, epochs_high = st.slider(
            "Number of Epochs Range",
            min_value=10,
            max_value=1000,
            value=(
                hpo_conf.get("epochs", {}).get("low", 50),
                hpo_conf.get("epochs", {}).get("high", 200)
            ),
            step=10,
            help="Select the minimum and maximum number of training epochs to explore."
        )
        hpo_conf["epochs"] = {"low": epochs_low, "high": epochs_high}

        # --- Update dictionary ---
        ui_config["hyperparameter_search_space"] = hpo_conf
        # Display Options
    
    ui_config["display"] = ui_config.get("display", {})
    ui_config["display"]["show_plot"] = st.checkbox(
        "Show Prediction Plot After Training",
        value=ui_config["display"].get("show_plot", True),
        help="If checked, a plot comparing predictions and true values will be shown after training."
    )

    # --- Form Submission Buttons ---
    col1, col2 = st.columns(2)
    
    submitted = col1.form_submit_button(
        "Save configs", 
        help="This will overwrite the config.yaml file with the values above."
    )
    
    reset = col2.form_submit_button(
        "Reset to Defaults", 
        help="Resets config.yaml to the values it had when the app started."
    )

    if submitted:
        try:
            save_config(ui_config, CONFIG_PATH)
            st.sidebar.success("Config saved! Ready to train.")
        except Exception as e:
            st.sidebar.error(f"Error saving config: {e}")
            
    if reset:
        try:
            save_config(st.session_state.default_config, CONFIG_PATH)
            st.sidebar.success("Config reset to defaults.")
            # We don't rerun, just let the user see the success
        except Exception as e:
            st.sidebar.error(f"Error resetting config: {e}")

# --- Main App Interface ---
st.title("ANN Optimization Dashboard")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)
best_model_path = os.path.join("models", "ANN_best_model.pt")

if st.button("Start Training and Testing", type="primary"):
    st.session_state.training_results = None
    st.session_state.log_output = ""
    
    # Check for uploaded files before proceeding
    if not st.session_state.uploaded_train_file or not st.session_state.uploaded_test_file:
        st.error("Please upload both Training and Testing data files in the sidebar.")
        st.stop()
        
    # We need to capture the print() statements from your scripts
    log_stream = io.StringIO()
    temp_dir = None
    
    with st.spinner("Running optimization and testing..."):
        with contextlib.redirect_stdout(log_stream):
            try:
                print("--- Streamlit App: Starting Training ---")
                
                # 1. Create temporary directory to save uploaded files
                temp_dir = tempfile.mkdtemp()
                train_file = st.session_state.uploaded_train_file
                test_file = st.session_state.uploaded_test_file
                
                # 2. Define temporary paths
                train_path = os.path.join(temp_dir, train_file.name)
                test_path = os.path.join(temp_dir, test_file.name)
                
                # 3. Write uploaded file contents to temporary paths
                with open(train_path, "wb") as f:
                    f.write(train_file.getbuffer())
                with open(test_path, "wb") as f:
                    f.write(test_file.getbuffer())


                # 4. Load training dataset using the temporary path
                print(f"Loading training data from temporary file: {train_path}")
                dataset_train = Dataset(train_path)
                
                # 5. Load testing dataset using the temporary path
                print(f"Loading testing data from temporary file: {test_path}")
                dataset_test = Dataset(test_path)

                # 6. Find the best model using the training subset
                print("Starting find_best_model()...")
                best_model, best_param = find_best_model(dataset_train) 
                print("find_best_model() complete.")

                # 7. Test on unseen dataset
                print("Starting test()...")
                test_accuracy, nmae, r2, mre_list, y_pred, y_true = test(dataset_test, best_model_path, best_param)
                print("test() complete.")
                
                st.session_state.training_results = {
                    "test_accuracy": test_accuracy,
                    "nmae": nmae,
                    "r2": r2,
                    "best_param": best_param,
                    "model_structure": str(define_net_regression(
                        best_param, 
                        dataset_train.n_input_params, 
                        dataset_train.n_output_params
                    )),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "mre_list": mre_list
                }

                # 8. Get model structure for display and store results
                model_structure = define_net_regression(
                    best_param, 
                    dataset_train.n_input_params, 
                    dataset_train.n_output_params
                )
                st.session_state.training_results["model_structure"] = str(model_structure)
                print("--- Streamlit App: Training Complete ---")
                

            except Exception as e:
                print(f"\n--- AN ERROR OCCURRED ---")
                print(e)
                st.error(f"An error occurred during training: {e}")
                
            finally:
                # 10. Clean up temporary directory
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                
        # Store the captured log
        st.session_state.log_output = log_stream.getvalue()

# --- Display Results ---
if st.session_state.training_results:
    st.success("Training and Testing Complete!")
    
    results = st.session_state.training_results
   
    
    

        # --- Download Best Model Button ---
    model_path = os.path.join("models", "ANN_best_model.pt")
    if os.path.exists(model_path):
        with open(model_path, "rb") as model_file:
            st.download_button(
                label="📥 Download Best Model",
                data=model_file,
                file_name="ANN_best_model.pt",
                mime="application/octet-stream",
                help="Click to download the trained PyTorch model file."
            )
    else:
        st.warning("Best model file not found in 'models/' folder.")
    
    st.subheader("Final Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy (MRE %)", f"{results['test_accuracy']:.2f}%")
    col2.metric("NMAE", f"{results['nmae']:.4f}")
    col3.metric("R² Score", f"{results['r2']:.4f}")

        # --- Display Results Plot (only if enabled) ---
    
    with st.expander("Prediction Plot", expanded=False):
            if ui_config.get("display", {}).get("show_plot", True):
                fig = make_plot(
                    results['mre_list'], 
                    results['y_pred'], 
                    results['y_true'], 
                    save_path=None  
                )
                st.pyplot(fig)
            else:
                st.info("Plot display is disabled in the configuration.")
    
    with st.expander("Best Hyperparameters"):
        st.json(results['best_param'])
        
    with st.expander("Final Model Architecture"):
        st.text(results['model_structure'])

if st.session_state.log_output:
    with st.expander("Full Training Log"):
        st.text_area("Log", st.session_state.log_output, height=400)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 12px;
        color: #555; /* Dark gray */
    }
    </style>
    <div class="footer">
        This program was developped by the Institute for Dynamic Systems and Control at ETHZ
    </div>
    """,
    unsafe_allow_html=True
)