import streamlit as st
import os
import torch
import yaml
import pandas as pd
from src.dataset import Dataset, create_subset
from src.train import find_best_model
from src.model_test import test, unseen_test

# --- Helper function to save uploaded files ---
def save_uploaded_file(uploaded_file, save_as):
    with open(save_as, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_as

# --- Helper function to detect file type and load data ---
def load_dataset(file_path):
    if file_path.endswith(".json"):
        return Dataset(file_path)
    elif file_path.endswith(".csv") or file_path.endswith((".xls", ".xlsx")):
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return Dataset(df)  # Assuming Dataset class can accept a DataFrame directly
    else:
        raise ValueError("Unsupported file type!")

# --- Streamlit UI ---
st.title("ANN Optimization Web App")

training_file = st.file_uploader("Upload Training File (.json, .csv, .xlsx)", type=["json", "csv", "xlsx"])
testing_file = st.file_uploader("Upload Testing File (.json, .csv, .xlsx)", type=["json", "csv", "xlsx"])
config_file = st.file_uploader("Upload config.yaml (optional)", type=["yaml", "yml"])

if st.button("Run ANN Optimization"):
    if training_file and testing_file:
        st.info("Saving uploaded files...")
        os.makedirs("models", exist_ok=True)

        # Save uploaded training/testing files
        training_path = save_uploaded_file(training_file, "training_file" + os.path.splitext(training_file.name)[1])
        testing_path = save_uploaded_file(testing_file, "testing_file" + os.path.splitext(testing_file.name)[1])

        # Handle config file
        if config_file:
            config_path = save_uploaded_file(config_file, "config.yaml")
            st.info("Custom config uploaded and saved.")
        else:
            config_path = "config.yaml"
            if not os.path.exists(config_path):
                st.error("No config file found! Please upload a config.yaml or place one in the current directory.")
                st.stop()
            st.info("Using default config.yaml from current directory.")

        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        data_config = config.get("data", {})
        train_ratio = data_config.get("train_ratio", 0.95)

        st.info("Loading datasets...")
        dataset_train = load_dataset(training_path)
        dataset_test = load_dataset(testing_path)
        train_subset = create_subset(dataset_train.full_data, train_ratio=train_ratio)

        st.info("Training and optimizing ANN...")
        find_best_model(dataset_train, train_subset, dataset_test)
        best_model_path = os.path.join("models", "ANN_best_model.pt")

        st.info("Testing on unseen data...")
        test_accuracy, nmae, r2 = unseen_test(dataset_test, best_model_path)

        st.success("Optimization completed!")
        st.write("**Best Model Performance on Unseen Data:**")
        st.write(f"Test Accuracy: {test_accuracy:.2f}%")
        st.write(f"NMAE: {nmae:.2f}%")
        st.write(f"R2 Score: {r2:.4f}")

        st.write("**Best Model Architecture:**")
        model = torch.load(best_model_path)
        st.text(str(model))
    else:
        st.error("Please upload both training and testing files before running.")