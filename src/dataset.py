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


def create_subset(full_data, train_ratio=0.95, random_state=1):
    sample_size = int(len(full_data) * train_ratio)
    return resample(full_data, replace=False, n_samples=sample_size, random_state=random_state)