import os
from src.dataset import Dataset, create_subset
from src.train import find_best_model
from src.model_test import test, unseen_test
import torch

os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    print("Creating dataset for training, validation, and test")
    dataset_1 = Dataset('data/training.json')
    train_subset_1 = create_subset(dataset_1.full_data)
    dataset_2 = Dataset('data/testing.json')

    find_best_model(dataset_1, train_subset_1, dataset_2)

    best_model_path = os.path.join("models", "ANN_best_20Trials.pt")

    test(dataset_1, train_subset_1, best_model_path)
    unseen_test(dataset_2, best_model_path)
    model = torch.load(best_model_path)
    print(model)