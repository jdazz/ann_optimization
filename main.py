import os
from src.dataset import Dataset, create_subset
from src.train import find_best_model
from src.model_test import test, unseen_test
import torch

os.makedirs("models", exist_ok=True)

import os
import torch
import yaml
from src.dataset import Dataset, create_subset
from src.train import find_best_model
from src.model_test import test, unseen_test

# Load configuration
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# parameters from config
data_config = config.get("data", {})
training_path = data_config.get("training_path", "data/training.json")
testing_path = data_config.get("testing_path", "data/testing.json")
train_ratio = data_config.get("train_ratio", 0.95)
generate_plot = config.get("generate_plot", False)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    print("Creating dataset for training, validation, and test")

    # Load training dataset
    dataset_train = Dataset(training_path)
    train_subset = create_subset(dataset_train.full_data, train_ratio=train_ratio)

    # Load testing dataset
    dataset_test = Dataset(testing_path)

    # Find the best model using the training subset
    find_best_model(dataset_train, train_subset, dataset_test)

    # Path to save the best model
    best_model_path = os.path.join("models", "ANN_best_model.pt")


    test(dataset_train, train_subset, best_model_path)

    # Test on unseen dataset
    test_accuracy, nmae, r2 = unseen_test(dataset_test, best_model_path)

    # Print the model architecture
    model = torch.load(best_model_path)
    print(model)
print("Model performance on unseen data: test_accuracy = {:.2f}%, NMAE = {:.2f}%, R2 = {:.4f}".format(
    test_accuracy, nmae, r2))

