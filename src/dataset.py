import json
import pandas as pd
import numpy as np
from sklearn.utils import resample

class Dataset:
    def __init__(self, text):
        self.name = text
        with open(self.name, "r") as f:
            data = json.load(f)[0]
        self.dataset = pd.DataFrame.from_dict(data)
        self.input_vars = ['distance', 'speed', 'temp_outside', 'AC', 'rain']
        self.output_vars = ['consume']
        self.input_data = self.dataset[self.input_vars].to_numpy()
        self.output_data = self.dataset[self.output_vars].to_numpy()
        self.full_data = np.concatenate([self.input_data, self.output_data], axis=1)
        self.n_input_params = len(self.input_vars)
        self.n_output_params = len(self.output_vars)

    def get_rows(self):
        return len(self.input_data)
import json
import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
import yaml

# Load config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_config = config.get("data", {})
input_vars = data_config.get("input_variables", [])
output_vars = data_config.get("output_variables", [])
train_ratio = data_config.get("train_ratio", 0.95)  # optional, default 95% training

class Dataset:
    def __init__(self, file_path=None):
        """
        Dataset class loads data from JSON and sets input/output matrices
        based on configuration.
        """
        self.name = file_path if file_path else data_config.get("training_path")
        with open(self.name, "r") as f:
            data = json.load(f)[0]

        self.dataset = pd.DataFrame.from_dict(data)

        # Use variables from config.yaml
        self.input_vars = input_vars
        self.output_vars = output_vars

        # Extract input/output matrices
        self.input_data = self.dataset[self.input_vars].to_numpy(dtype="float32")
        self.output_data = self.dataset[self.output_vars].to_numpy(dtype="float32")

        # Concatenate input/output for full_data
        self.full_data = np.concatenate([self.input_data, self.output_data], axis=1)

        # Store number of input/output parameters
        self.n_input_params = len(self.input_vars)
        self.n_output_params = len(self.output_vars)

    def get_rows(self):
        return len(self.input_data)


def create_subset(full_data, train_ratio=train_ratio, random_state=1):
    """
    Creates a subset of full_data for training.
    """
    if train_ratio is None:
        train_ratio = data_config.get("train_ratio", 0.95)
    sample_size = int(len(full_data) * train_ratio)
    return resample(full_data, replace=False, n_samples=sample_size, random_state=random_state)

