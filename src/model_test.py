import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.plot import make_plot 
from src.model import define_net_regression 
import os
import yaml

# Load Configuration
config_path = os.path.join(os.getcwd(), "config.yaml")
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Warning: config.yaml not found at {config_path}. Using default settings.")
    config = {}

GENERATE_PLOTS = config.get('generate_plot', False)
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MRE_THRESHOLD = config.get('targets', {}).get('mre_threshold', 25)
TEST_BATCH_SIZE = config.get('testing', {}).get('batch_size', 1)
PLOT_SAVE_PATH = config.get('plots_save_path', None)



def test(dataset, model_path, best_params):
    """
    Evaluate model on unseen dataset (all samples).
    
    Arguments:
        dataset: Your custom Dataset object (must contain full_data and dimensions).
        model_path (str): Path to the saved model weights (.pt file).
        best_params (dict): Dictionary of hyperparameters needed to rebuild the model architecture.
    """
    print("Unseen test begins here")

    test_data = dataset.full_data 
    n_samples = len(test_data)
    n_input = dataset.n_input_params
    n_output = dataset.n_output_params
    
    input_data = test_data[:, :n_input].astype('float32')
    output_data = test_data[:, n_input:n_input + n_output].astype('float32')

    inputs_tensor = torch.from_numpy(input_data).to(Device)
    outputs_tensor = torch.from_numpy(output_data).to(Device)
    test_loader = DataLoader(TensorDataset(inputs_tensor, outputs_tensor),
                             batch_size=TEST_BATCH_SIZE, shuffle=False)

    
    # Instantiate the Model Architecture using the best parameters
    try:
        model = define_net_regression(best_params, n_input, n_output).to(Device)
    except Exception as e:
        print(f"Error instantiating model architecture: {e}")
        return 0, 0, 0 # Return zeros on failure
        
    # Load the model weights

    state_dict = torch.load(model_path, map_location=Device)

    model.load_state_dict(state_dict)
    
    model.eval() 

    # --- Evaluation Loop ---
    print("Calculating metrics for unseen data...")
    mre_list, y_pred_list, y_true_list = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            y_true = y
            
            y_pred_np = y_pred.cpu().numpy().flatten()
            y_true_np = y_true.cpu().numpy().flatten()
            
            rel_error = np.abs((y_pred_np - y_true_np) / (y_true_np + 1e-6))
            
            mre_list.extend(rel_error * 100)
            y_pred_list.extend(y_pred_np)
            y_true_list.extend(y_true_np)


    y_true_array = np.array(y_true_list)
    y_pred_array = np.array(y_pred_list)
    y_mean = np.mean(y_true_array)
    
    # R-squared
    SS_res = np.sum((y_true_array - y_pred_array)**2)
    SS_tot = np.sum((y_true_array - y_mean)**2)
    r2 = 1 - (SS_res / SS_tot)
    
    # Normalized Mean Absolute Error (NMAE)
    nmae = np.sum(np.abs(y_pred_array - y_true_array)) / (len(y_true_array) * y_mean)
    
    # Percentage of samples within MRE_THRESHOLD
    test_accuracy = np.sum(np.array(mre_list) <= MRE_THRESHOLD) / len(y_true_list) * 100

    print(f"R² on unseen data: {r2:.4f}")
    print(f"NMAE on unseen data: {nmae:.4f}")
    print(f"P(error <= {MRE_THRESHOLD}%) on unseen data: {test_accuracy:.2f}%")

    if GENERATE_PLOTS:
       make_plot(mre_list, y_pred_list, y_true_list, save_path=PLOT_SAVE_PATH)

    return test_accuracy, nmae, r2, mre_list, y_pred_list, y_true_list