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
        - JSON (.json)
        - CSV (.csv)
        - Excel (.xls, .xlsx)
        - Parquet (.parquet)
        - Pandas DataFrame or numpy.ndarray
        """
        self.name = None

        # Load data
        if isinstance(source, pd.DataFrame):
            self.dataset = source.copy()
            self.name = "DataFrame"
        elif isinstance(source, np.ndarray):
            # must match expected input/output layout
            self.dataset = pd.DataFrame(source, columns=input_vars + output_vars)
            self.name = "ndarray"
        elif isinstance(source, str) and os.path.exists(source):
            self.name = source
            ext = os.path.splitext(source)[1].lower()

            if ext == ".json":
                with open(source, "r") as f:
                    data = json.load(f)
                    # Handle both top-level list and dict
                    if isinstance(data, list):
                        if isinstance(data[0], dict):
                            self.dataset = pd.DataFrame(data)
                        else:
                            self.dataset = pd.DataFrame.from_dict(data[0])
                    elif isinstance(data, dict):
                        self.dataset = pd.DataFrame.from_dict(data)
                    else:
                        raise ValueError("Unsupported JSON structure.")
            
            elif ext == ".csv":
                try:
                    self.dataset = pd.read_csv(source, encoding="utf-8")
                except UnicodeDecodeError:
                    self.dataset = pd.read_csv(source, encoding="latin1")
            
            elif ext in [".xls", ".xlsx"]:
                self.dataset = pd.read_excel(source)
            
            elif ext == ".parquet":
                self.dataset = pd.read_parquet(source)
            
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        
        else:
            raise ValueError("Unsupported data source type or file not found.")
        
        # Define input/output columns from config
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