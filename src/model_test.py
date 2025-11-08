import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.plot import make_plot
import os
import yaml

# Load configuration
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Global settings
GENERATE_PLOTS = config.get('generate_plot', False)
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MRE_THRESHOLD = config.get('targets', {}).get('mre_threshold', 25)
TEST_BATCH_SIZE = config.get('testing', {}).get('batch_size', 1)  # optional section in YAML
PLOT_SAVE_PATH = config.get('plots', {}).get('save_path', None)


def test(dataset, train_subset, model_name, save_plot_path=None):
    """Evaluate model on test subset (data not used during training)."""
    print("Test begins here")

    # Convert train_subset to list for comparison
    train_subset = train_subset.tolist()
    test_data = [list(x) for x in dataset.full_data if list(x) not in train_subset]

    print("Creating matrices for inputs and outputs")
    n_samples = len(test_data)
    input_data = np.zeros((n_samples, dataset.n_input_params), dtype='float32')
    output_data = np.zeros((n_samples, dataset.n_output_params), dtype='float32')
    test_data = np.array(test_data)

    input_data[:] = test_data[:, :dataset.n_input_params]
    output_data[:] = test_data[:, dataset.n_input_params:dataset.n_input_params + dataset.n_output_params]

    # Convert to tensors
    inputs_tensor = torch.from_numpy(input_data)
    outputs_tensor = torch.from_numpy(output_data)
    test_loader = DataLoader(TensorDataset(inputs_tensor, outputs_tensor),
                             batch_size=TEST_BATCH_SIZE, shuffle=False)

    mre_list, y_pred_list, y_true_list = [], [], []

    print("Loading model...")
    model = torch.load(model_name)
    model.eval()

    print("Evaluating model...")
    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            y_true = y
            rel_error = torch.abs((y_pred - y_true) / y_true)
            mean_rel_error = torch.mean(rel_error)

            mre_list.append(mean_rel_error.item() * 100)
            y_pred_list.append(y_pred.item())
            y_true_list.append(y_true.item())

    # Metrics
    y_mean = np.mean(y_true_list)
    sqr = [(y_true_list[i] - y_pred_list[i])**2 for i in range(len(y_true_list))]
    sqt = [(y_true_list[i] - y_mean)**2 for i in range(len(y_true_list))]
    r2 = 1 - (sum(sqr) / sum(sqt))

    nmae = sum([abs(y_pred_list[i] - y_true_list[i]) for i in range(len(y_true_list))]) / (len(y_true_list) * y_mean)
    test_accuracy = sum(1 for err in mre_list if err <= MRE_THRESHOLD) / len(y_true_list) * 100

    print(f"R²: {r2:.4f}")
    print(f"NMAE: {nmae:.4f}")
    print(f"P(error <= {MRE_THRESHOLD}%): {test_accuracy:.2f}%")

    if GENERATE_PLOTS:
        make_plot(mre_list, y_pred_list, y_true_list, save_path=save_plot_path or PLOT_SAVE_PATH)

    return test_accuracy, nmae, r2


def unseen_test(dataset, model_name):
    """Evaluate model on unseen dataset (all samples)."""
    print("Unseen test begins here")

    test_data = np.array([list(x) for x in dataset.full_data])
    n_samples = len(test_data)
    input_data = test_data[:, :dataset.n_input_params].astype('float32')
    output_data = test_data[:, dataset.n_input_params:dataset.n_input_params + dataset.n_output_params].astype('float32')

    # Convert to tensors
    inputs_tensor = torch.from_numpy(input_data).to(Device)
    outputs_tensor = torch.from_numpy(output_data).to(Device)
    test_loader = DataLoader(TensorDataset(inputs_tensor, outputs_tensor),
                             batch_size=TEST_BATCH_SIZE, shuffle=False)

    mre_list, y_pred_list, y_true_list = [], [], []

    model = torch.load(model_name)
    model.eval()

    print("Calculating metrics for unseen data...")
    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            y_true = y
            rel_error = torch.abs((y_pred - y_true) / y_true)
            mean_rel_error = torch.mean(rel_error)

            mre_list.append(mean_rel_error.item() * 100)
            y_pred_list.append(y_pred.item())
            y_true_list.append(y_true.item())

    # Metrics
    y_mean = np.mean(y_true_list)
    sqr = [(y_true_list[i] - y_pred_list[i])**2 for i in range(len(y_true_list))]
    sqt = [(y_true_list[i] - y_mean)**2 for i in range(len(y_true_list))]
    r2 = 1 - (sum(sqr) / sum(sqt))

    nmae = sum([abs(y_pred_list[i] - y_true_list[i]) for i in range(len(y_true_list))]) / (len(y_true_list) * y_mean)
    test_accuracy = sum(1 for err in mre_list if err <= MRE_THRESHOLD) / len(y_true_list) * 100

    print(f"R² on unseen data: {r2:.4f}")
    print(f"NMAE on unseen data: {nmae:.4f}")
    print(f"P(error <= {MRE_THRESHOLD}%) on unseen data: {test_accuracy:.2f}%")

    if GENERATE_PLOTS:
        make_plot(mre_list, y_pred_list, y_true_list, save_path=PLOT_SAVE_PATH)

    return test_accuracy, nmae, r2