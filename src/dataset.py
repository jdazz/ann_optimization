import os
import json
import pandas as pd
import numpy as np
from sklearn.utils import resample
import yaml

# Load configuration
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_config = config.get("data", {})
input_vars = data_config.get("input_variables", [])
output_vars = data_config.get("output_variables", [])
train_ratio = data_config.get("train_ratio", 0.95)


class Dataset:
    def __init__(self, source=None):
        """
        Flexible Dataset loader. Supports:
        - JSON (.json) (list of dicts or dict of lists or list of lists)
        - CSV (.csv)
        - Excel (.xls, .xlsx)
        - Parquet (.parquet)
        - Pandas DataFrame or numpy.ndarray
        """
        self.name = None
        self.dataset = None  # Initialize dataset

        # --- Load from DataFrame ---
        if isinstance(source, pd.DataFrame):
            self.dataset = source.copy()
            self.name = "DataFrame"

        # --- Load from numpy array ---
        elif isinstance(source, np.ndarray):
            self.dataset = pd.DataFrame(source, columns=input_vars + output_vars)
            self.name = "ndarray"

        # --- Load from file ---
        elif isinstance(source, str) and os.path.exists(source):
            self.name = source
            ext = os.path.splitext(source)[1].lower()

            if ext == ".json":
                with open(source, "r") as f:
                    data = json.load(f)

                # Case 1: list of dicts
                if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                    df = pd.DataFrame(data)

                # Case 2: dict of lists
                elif isinstance(data, dict):
                    df = pd.DataFrame(data)

                # Case 3: list of lists
                elif isinstance(data, list) and all(isinstance(d, list) for d in data):
                    # Create temporary column names
                    num_cols = len(data[0])
                    temp_cols = [f"col_{i}" for i in range(num_cols)]
                    df = pd.DataFrame(data, columns=temp_cols)

                else:
                    raise ValueError("Unsupported JSON structure.")

                # Keep only columns specified in config
                selected_cols = input_vars + output_vars
                missing_cols = [c for c in selected_cols if c not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in dataset: {missing_cols}")
                self.dataset = df[selected_cols]

            elif ext == ".csv":
                try:
                    self.dataset = pd.read_csv(source, encoding="utf-8")
                except UnicodeDecodeError:
                    self.dataset = pd.read_csv(source, encoding="latin1")
                self.dataset = self.dataset[input_vars + output_vars]

            elif ext in [".xls", ".xlsx"]:
                self.dataset = pd.read_excel(source)
                self.dataset = self.dataset[input_vars + output_vars]

            elif ext == ".parquet":
                self.dataset = pd.read_parquet(source)
                self.dataset = self.dataset[input_vars + output_vars]

            else:
                raise ValueError(f"Unsupported file type: {ext}")

        else:
            raise ValueError("Unsupported data source type or file not found.")

        # --- Check for missing values ---
        if self.dataset.isnull().values.any():
            missing_cols = self.dataset.columns[self.dataset.isnull().any()].tolist()
            raise ValueError(
                f"Dataset loading failed: Missing values detected. Columns with missing values: {missing_cols}"
            )

        # --- Extract numpy arrays ---
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.input_data = self.dataset[self.input_vars].to_numpy(dtype="float32")
        self.output_data = self.dataset[self.output_vars].to_numpy(dtype="float32")
        self.full_data = np.concatenate([self.input_data, self.output_data], axis=1)
        self.n_input_params = len(self.input_vars)
        self.n_output_params = len(self.output_vars)

    def get_rows(self):
        return len(self.input_data)


def create_subset(full_data, train_ratio=train_ratio, random_state=1):
    """
    Creates a random training subset from full_data.
    """
    sample_size = int(len(full_data) * train_ratio)
    return resample(full_data, replace=False, n_samples=sample_size, random_state=random_state)