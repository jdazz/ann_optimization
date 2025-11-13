import os
import torch
import yaml
from src.dataset import Dataset
from src.train import find_best_model
from src.model_test import test
from src.model import define_net_regression

# Load configuration and get parameters
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_config = config.get("data", {})
training_path = data_config.get("training_path", "data/training.json")
testing_path = data_config.get("testing_path", "data/testing.json")
train_ratio = data_config.get("train_ratio", 0.95)
generate_plot = config.get("generate_plot", False)

os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    print("Creating dataset for training, validation, and test")

    # Load datasets
    dataset_train = Dataset(training_path)
    dataset_test = Dataset(testing_path)

    # Find the best model
    best_model, best_param = find_best_model(dataset_train)

    best_model_path = os.path.join("models", "ANN_best_model.pt")

    # Test on unseen dataset
    print("\n--- Testing Final Model on Unseen Data ---")
    test_accuracy, nmae, r2, rme_list, y_preds, y_true = test(dataset_test, best_model_path, best_param)

    model_structure = define_net_regression(
        best_param, 
        dataset_train.n_input_params, 
        dataset_train.n_output_params
    )
    

    print("\n--- Final Model Architecture and Weights ---")
    print(model_structure)
    print("Model performance on unseen data: test_accuracy = {:.2f}%, NMAE = {:.2f}%, R2 = {:.4f}".format(
    test_accuracy, nmae, r2))

