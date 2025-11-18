import os
import json
import pandas as pd
import numpy as np
from sklearn.utils import resample
import yaml


class Dataset:
    def _parse_var_names(self, var_string):
        """
        Parses a string of variable names (which may contain newlines) 
        into a clean list of strings.
        """
        if isinstance(var_string, str):
            # Split by newline or comma, strip whitespace, and filter out empty strings
            names = [name.strip() for name in var_string.split() if name.strip()]
            return names
        elif isinstance(var_string, list):
             # Already a list, just return it
             return var_string
        return [] # Return empty list if format is unexpected


    def __init__(self, source=None):
        """
        Flexible Dataset loader. Supports:
        - JSON (.json) (list of dicts or dict of lists or list of lists)
        - CSV (.csv)
        - Excel (.xls, .xlsx)
        - Parquet (.parquet)
        - Pandas DataFrame or numpy.ndarray
        """
        # Load configuration
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        data_config = config.get("variables", {})
        
        # --- CRITICAL ADAPTATION ---
        # Parse multi-line string input/output variables into clean lists
        raw_input_vars = data_config.get("input_names", "")
        raw_output_vars = data_config.get("output_names", "")
        
        input_vars = self._parse_var_names(raw_input_vars)
        output_vars = self._parse_var_names(raw_output_vars)
        
        if not input_vars or not output_vars:
            raise ValueError(
                f"Configuration error: Input or output variables are not defined. Input: {input_vars}, Output: {output_vars}"
            )
        # ---------------------------

        self.name = None
        self.dataset = None 

        # --- Load from DataFrame ---
        if isinstance(source, pd.DataFrame):
            self.dataset = source.copy()
            self.name = "DataFrame"

        # --- Load from numpy array ---
        # NOTE: When loading from np.ndarray, column naming needs input_vars and output_vars 
        # to match the dataset's actual structure, which can be tricky without a header.
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

                if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                    df = pd.DataFrame(data)

                elif isinstance(data, dict):
                    df = pd.DataFrame(data)

                elif isinstance(data, list) and all(isinstance(d, list) for d in data):
                    # This branch is complex and assumes data order, which is risky.
                    # It's better to ensure JSON data has headers (dicts).
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
                # Now that input_vars and output_vars are lists, this works
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