import os
import json
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import queue # Required for type hinting/safety, even if not explicitly used in simple put()


class Dataset:
    def _parse_var_names(self, var_string):
        """
        Parses a string of variable names (which may contain newlines) 
        into a clean list of strings.
        """
        if isinstance(var_string, str):
            names = [name.strip() for name in var_string.split() if name.strip()]
            return names
        elif isinstance(var_string, list):
             return var_string
        return []

    # --- MODIFIED: Added update_queue as an optional parameter ---
    def __init__(self, source=None, config=None, update_queue=None):
        """
        Loads data and prepares initial attributes. Scaling is now deferred.
        Accepts optional update_queue for live logging.
        """
        if config is None:
            raise ValueError("Configuration dictionary must be passed to the Dataset constructor.")
        
        # Store the queue for logging
        self.update_queue = update_queue
        
        # Helper function for logging to the queue
        def log_to_queue(message):
            if self.update_queue:
                # Use 'log_messages' key as established in app.py
                self.update_queue.put({'key': 'log_messages', 'value': message})
        
        # --- Retrieve configuration settings ---
        data_config = config.get("variables", {})
        cv_config = config.get("cross_validation", {})
        
        # --- Standardization Flag ---
        self.should_standardize = cv_config.get("standardize_features", False)
        # ---------------------------
        
        raw_input_vars = data_config.get("input_names", "")
        raw_output_vars = data_config.get("output_names", "")
        
        input_vars = self._parse_var_names(raw_input_vars)
        output_vars = self._parse_var_names(raw_output_vars)
        
        if not input_vars or not output_vars:
            raise ValueError(
                f"Configuration error: Input or output variables are not defined. Input: {input_vars}, Output: {output_vars}"
            )

        self.name = None
        self.dataset = None 

        # --- Loading Logic (File handling remains as original) ---
        if isinstance(source, pd.DataFrame):
            self.dataset = source.copy()
            self.name = "DataFrame"
        elif isinstance(source, np.ndarray):
            self.dataset = pd.DataFrame(source, columns=input_vars + output_vars)
            self.name = "ndarray"
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
                else:
                    raise ValueError("Unsupported JSON structure or file type.")

                selected_cols = input_vars + output_vars
                missing_cols = [c for c in selected_cols if c not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in dataset: {missing_cols}")
                self.dataset = df[selected_cols]
            
            elif ext == ".csv":
                try:
                    df = pd.read_csv(source, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(source, encoding="latin1")
                self.dataset = df[input_vars + output_vars]

            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(source)
                self.dataset = df[input_vars + output_vars]

            elif ext == ".parquet":
                df = pd.read_parquet(source)
                self.dataset = df[input_vars + output_vars]

            else:
                raise ValueError(f"Unsupported file type: {ext}")
        else:
            raise ValueError("Unsupported data source type or file not found.")

        if self.dataset is None:
             raise ValueError("Dataset could not be loaded from source.")
        
        # --- Check for missing values (REVISED TO DROP ROWS AND LOG) ---
        initial_rows = len(self.dataset)
        
        if self.dataset.isnull().values.any():
            missing_cols = self.dataset.columns[self.dataset.isnull().any()].tolist()
            log_to_queue(f"⚠️ Missing values detected. Columns: {', '.join(missing_cols)}. Dropping rows...")
            self.dataset.dropna(inplace=True) 
        
        final_rows = len(self.dataset)
        
        if initial_rows != final_rows:
            log_to_queue(f"🗑️ Removed {initial_rows - final_rows} rows due to missing values. "
                         f"Dataset size reduced from {initial_rows} to {final_rows}.")
            
        if final_rows == 0:
            error_message = "❌ Dataset loading failed: All remaining rows were removed due to missing values."
            log_to_queue(error_message)
            raise ValueError(error_message)

        # --- Extract numpy arrays (UNSCALED) ---
        self.input_vars = input_vars
        self.output_vars = output_vars
        
        # Store initial unscaled inputs as a DataFrame (for easy scaling later)
        self.input_df = self.dataset[self.input_vars].copy()
        
        # Initial numpy arrays (unscaled inputs, scaled outputs)
        self.input_data = self.input_df.to_numpy(dtype="float32")
        self.output_data = self.dataset[self.output_vars].to_numpy(dtype="float32")
        
        # Initial full data (unscaled)
        self.full_data = np.concatenate([self.input_data, self.output_data], axis=1)
        self.n_input_params = len(self.input_vars)
        self.n_output_params = len(self.output_vars)
        
        # Store the scaler object (will be None or a fitted StandardScaler)
        self.scaler = None


    def apply_scaler(self, scaler=None, is_fitting=False):
        """
        Applies a fitted StandardScaler to the input features.
        If is_fitting=True, the method performs fit_transform and saves the scaler.
        """
        
        def log_to_queue(message):
            if self.update_queue:
                self.update_queue.put({'key': 'log_messages', 'value': message})
                
        if not self.should_standardize:
            log_to_queue("Standardization disabled by config. Skipping scaling.")
            return
            
        if is_fitting:
            # For TRAINING data: Fit and transform
            if scaler is None:
                scaler = StandardScaler()
            
            input_scaled = scaler.fit_transform(self.input_df)
            self.scaler = scaler # Save the fitted scaler
            log_to_queue("📈 Features standardized successfully (Fitted and Transformed).")
            
        elif scaler is not None:
            # For TESTING data: Only transform using the provided (fitted) scaler
            input_scaled = scaler.transform(self.input_df)
            self.scaler = scaler
            log_to_queue("📈 Features standardized successfully (Transformed using provided scaler).")
            
        else:
            # Case where standardization is True but no scaler is provided for transformation
            error_message = "Standardization is enabled, but no fitted scaler was provided for transformation."
            log_to_queue(f"❌ {error_message}")
            raise ValueError(error_message)

        # Update all internal numpy arrays with the scaled data
        self.input_data = input_scaled.astype("float32")
        self.full_data = np.concatenate([self.input_data, self.output_data], axis=1)


    def get_rows(self):
        return len(self.input_data)