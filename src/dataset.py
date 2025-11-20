import os
import json
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import queue # Required for type hinting/safety


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

    def __init__(self, source=None, config=None, update_queue=None):
        """
        Loads data, handles input variable selection (* wildcard), 
        performs one-hot encoding for categorical variables, and prepares initial attributes. 
        Scaling is deferred to apply_scaler.
        """
        if config is None:
            raise ValueError("Configuration dictionary must be passed to the Dataset constructor.")
        
        self.update_queue = update_queue
        
        def log_to_queue(message):
            if self.update_queue:
                self.update_queue.put({'key': 'log_messages', 'value': message})
        
        # --- Retrieve configuration settings ---
        data_config = config.get("variables", {})
        cv_config = config.get("cross_validation", {})
        
        self.should_standardize = cv_config.get("standardize_features", False)
        
        raw_input_vars = data_config.get("input_names", "")
        raw_output_vars = data_config.get("output_names", "")
        
        # Output variables must be specified
        self.output_vars = self._parse_var_names(raw_output_vars)
        
        if not self.output_vars:
             raise ValueError("Configuration error: Output variables are not defined.")

        self.name = None
        self.dataset = None 

        # --- Loading Logic (REVISED to load ALL potential columns) ---
        
        # The code to load the DataFrame `df` from source is consolidated here 
        # (It must happen before we can check for '*' inputs)
        df = None
        
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            self.name = "DataFrame"
        elif isinstance(source, np.ndarray):
            # If from numpy, we can't reliably determine all column names yet, 
            # so we rely on the config input_vars if it's not '*'
            if raw_input_vars != '*':
                df = pd.DataFrame(source, columns=self._parse_var_names(raw_input_vars) + self.output_vars)
            else:
                 raise ValueError("Using source=np.ndarray requires explicit 'input_names' and cannot use '*'.")
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
                
            elif ext == ".csv":
                try:
                    df = pd.read_csv(source, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(source, encoding="latin1")

            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(source)

            elif ext == ".parquet":
                df = pd.read_parquet(source)

            else:
                raise ValueError(f"Unsupported file type: {ext}")
        else:
            raise ValueError("Unsupported data source type or file not found.")

        if df is None:
             raise ValueError("Dataset could not be loaded from source.")
             
        # --- NEW: Handle '*' wildcard for input variables ---
        if raw_input_vars.strip() == '*':
            # Select all columns EXCEPT the output variables
            input_vars_raw = [col for col in df.columns if col not in self.output_vars]
            log_to_queue(f"Features selected: {', '.join(input_vars_raw)}.")
        else:
            # Use the variables specified in the config
            input_vars_raw = self._parse_var_names(raw_input_vars)
            
        if not input_vars_raw:
             raise ValueError("Configuration error: No input variables were selected.")

        # Final list of columns needed for the dataset
        selected_cols = input_vars_raw + self.output_vars
        missing_cols = [c for c in selected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
            
        self.dataset = df[selected_cols].copy() 
        # --- End of Loading and Selection ---


        # --- 💥 ONE-HOT ENCODING IMPLEMENTATION 💥 ---
        
        # Identify columns that are object type among the current input variables
        categorical_cols = self.dataset[input_vars_raw].select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            log_to_queue(f"Applying One-Hot Encoding to categorical inputs: {', '.join(categorical_cols)}.")
            
            # Apply OHE only to the input columns
            self.dataset = pd.get_dummies(self.dataset, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
            
            # Update the list of input variables to reflect the new dummy columns
            # Get all columns that are NOT output variables (since OHE can change the column count)
            self.input_vars = [col for col in self.dataset.columns if col not in self.output_vars]
            
            log_to_queue(f"Total input features after OHE: {len(self.input_vars)}")
        else:
            self.input_vars = input_vars_raw # No change, keep original list
            log_to_queue("No categorical input features found. Skipping One-Hot Encoding.")
        
        # --- END ONE-HOT ENCODING ---
        
        
        # --- Check for missing values (REVISED TO DROP ROWS AND LOG) ---
        initial_rows = len(self.dataset)
        
        # Use only the columns that are now included in the final dataset (inputs + outputs)
        final_cols = self.input_vars + self.output_vars
        
        if self.dataset[final_cols].isnull().values.any():
            missing_cols = self.dataset[final_cols].columns[self.dataset[final_cols].isnull().any()].tolist()
            log_to_queue(f"⚠️ Missing values detected. Columns: {', '.join(missing_cols)}. Dropping rows...")
            self.dataset.dropna(subset=final_cols, inplace=True) 
        
        final_rows = len(self.dataset)
        
        if initial_rows != final_rows:
            log_to_queue(f"Removed {initial_rows - final_rows} rows due to missing values. "
                         f"Dataset size reduced from {initial_rows} to {final_rows}.")
            
        if final_rows == 0:
            error_message = "Dataset loading failed: All remaining rows were removed due to missing values."
            log_to_queue(error_message)
            raise ValueError(error_message)

        # --- Extract numpy arrays ---
        
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
            log_to_queue("Features standardized successfully (Fitted and Transformed).")
            
        elif scaler is not None:
            # For TESTING data: Only transform using the provided (fitted) scaler
            input_scaled = scaler.transform(self.input_df)
            self.scaler = scaler
            log_to_queue("Features standardized successfully (Transformed using provided scaler).")
            
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