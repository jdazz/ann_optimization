import os
import json
import io

import numpy as np
import pandas as pd
import yaml
import openpyxl  # needed for Excel reading side-effects

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample  # kept if you use it elsewhere
from sklearn.model_selection import train_test_split
import queue  # kept for type hints / external usage


class Dataset:
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _parse_var_names(self, var_value):
        """
        Parse config variable names into a clean list of column names.

        IMPORTANT:
        - Preserves spaces *inside* column names, e.g. "Item Purchased".
        - Supports:
            * Newline-separated strings (typical for your config.yaml)
            * Comma-separated strings
            * Lists of names

        Examples:
            "distance\\nspeed\\ntemp inside"
                -> ["distance", "speed", "temp inside"]

            "distance, speed, temp inside"
                -> ["distance", "speed", "temp inside"]

            ["distance", "speed", "temp inside"]
                -> ["distance", "speed", "temp inside"]
        """
        # Already a list: just strip leading/trailing whitespace but keep inner spaces
        if isinstance(var_value, list):
            cleaned = []
            for v in var_value:
                if v is None:
                    continue
                name = str(v).strip()
                if name:
                    cleaned.append(name)
            return cleaned

        # String case
        if isinstance(var_value, str):
            s = var_value.strip()
            if not s:
                return []

            # Normalize newlines
            s = s.replace("\r\n", "\n").replace("\r", "\n")

            # If there are commas, treat them as separators
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
            else:
                # Default: treat each line as a separate name
                parts = [line.strip() for line in s.split("\n")]

            return [p for p in parts if p]

        # Anything else -> empty list
        return []

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip leading/trailing whitespace from all column names.
        Keeps internal spaces (e.g. 'Item Purchased' stays as is).
        """
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def read_columns_from_source(source):
        """
        Static method to read an uploaded file object (or path) and return
        the list of column names, without initializing the full Dataset object.
        
        Args:
            source: A file path string, a pandas DataFrame, or a Streamlit uploaded
                    file object (BytesIO-like with .name and .getvalue()).
                    
        Returns:
            list[str]: Column names or [] if loading fails / unsupported.
        """
        df = None

        if source is None:
            return []

        try:
            # Case 1: Pandas DataFrame
            if isinstance(source, pd.DataFrame):
                df = source

            # Case 2: Streamlit Uploaded File object (BytesIO-like)
            elif hasattr(source, "name") and hasattr(source, "getvalue"):
                file_name = source.name.lower()
                file_bytes = source.getvalue()

                if file_name.endswith((".csv", ".txt")):
                    df = pd.read_csv(io.BytesIO(file_bytes))
                elif file_name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(io.BytesIO(file_bytes))
                elif file_name.endswith(".json"):
                    df = pd.read_json(io.BytesIO(file_bytes))
                else:
                    # Unsupported type for this utility
                    return []

            # Case 3: File path string
            elif isinstance(source, str) and os.path.exists(source):
                ext = os.path.splitext(source)[1].lower()
                if ext == ".json":
                    df = pd.read_json(source)
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
                    return []  # unsupported extension

            if df is not None:
                # 🔥 Normalize column names here too
                df = Dataset._normalize_columns(df)
                return list(df.columns)
            return []

        except Exception as e:
            print(f"Error in read_columns_from_source: {e}")
            return []

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(self, source=None, config=None, update_queue=None, test_split_ratio=0.0):
        """
        Loads data, handles input variable selection (* wildcard), performs
        one-hot encoding for categorical variables, and prepares attributes.
        If test_split_ratio > 0, performs an internal train/test split.
        Scaling is deferred to apply_scaler().
        """
        if config is None:
            raise ValueError("Configuration dictionary must be passed to the Dataset constructor.")

        self.update_queue = update_queue

        def log_to_queue(message: str):
            if self.update_queue:
                self.update_queue.put({"key": "log_messages", "value": message})

        # ---------------------------------------------------------------------
        # Config
        # ---------------------------------------------------------------------
        data_config = config.get("variables", {})
        cv_config = config.get("cross_validation", {})

        self.should_standardize = cv_config.get("standardize_features", False)
        self.random_seed = cv_config.get("random_seed", 42)

        raw_input_vars = data_config.get("input_names", "")
        raw_output_vars = data_config.get("output_names", "")

        # Outputs
        self.output_vars = self._parse_var_names(raw_output_vars)
        if not self.output_vars:
            raise ValueError("Configuration error: Output variables are not defined.")

        self.name = None
        self.dataset = None

        # ---------------------------------------------------------------------
        # Load DataFrame from source
        # ---------------------------------------------------------------------
        df = None

        # Case 1: DataFrame directly
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            self.name = "DataFrame"

        # Case 2: numpy array
        elif isinstance(source, np.ndarray):
            # Must have explicit input names in config (cannot use '*')
            if isinstance(raw_input_vars, str) and raw_input_vars.strip() == "*":
                raise ValueError(
                    "Using source=np.ndarray requires explicit 'input_names' "
                    "and cannot use '*'."
                )

            input_names = self._parse_var_names(raw_input_vars)
            if not input_names:
                raise ValueError(
                    "Using source=np.ndarray requires valid 'input_names' in config."
                )

            all_cols = input_names + self.output_vars
            if source.shape[1] != len(all_cols):
                raise ValueError(
                    f"Provided ndarray has shape {source.shape}, but expected "
                    f"{len(all_cols)} columns (inputs + outputs)."
                )

            df = pd.DataFrame(source, columns=all_cols)
            self.name = "ndarray"

        # Case 3: string path
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

        # 🔥 Normalize column names right after loading
        df = self._normalize_columns(df)

        # ---------------------------------------------------------------------
        # Resolve input variables (supporting '*' wildcard)
        # ---------------------------------------------------------------------
        if isinstance(raw_input_vars, str) and raw_input_vars.strip() == "*":
            # Use all columns except outputs
            input_vars_raw = [col for col in df.columns if col not in self.output_vars]
            log_to_queue(f"Features selected via '*': {', '.join(input_vars_raw)}.")
        else:
            input_vars_raw = self._parse_var_names(raw_input_vars)

        if not input_vars_raw:
            raise ValueError("Configuration error: No input variables were selected.")

        # Final columns needed
        selected_cols = input_vars_raw + self.output_vars
        missing_cols = [c for c in selected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        self.dataset = df[selected_cols].copy()

        # ---------------------------------------------------------------------
        # One-Hot Encode categorical input columns
        # ---------------------------------------------------------------------
        # NOTE: We only consider the *selected input variables* here.
        categorical_cols = (
            self.dataset[input_vars_raw]
            .select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        if categorical_cols:
            log_to_queue(
                f"Applying One-Hot Encoding to categorical inputs: "
                f"{', '.join(categorical_cols)}."
            )
            original_feature_count = len(input_vars_raw)

            self.dataset = pd.get_dummies(
                self.dataset,
                columns=categorical_cols,
                prefix=categorical_cols,
                dummy_na=False,
            )

            # After OHE, define input_vars as all non-output columns
            self.input_vars = [
                col for col in self.dataset.columns if col not in self.output_vars
            ]
            new_feature_count = len(self.input_vars)
            added = new_feature_count - original_feature_count
            log_to_queue(
                f"WARNING: ⚠️ One-Hot Encoding expanded features from "
                f"{original_feature_count} to {new_feature_count} (+{added}).⚠️"
            )
        else:
            self.input_vars = input_vars_raw
            log_to_queue("No categorical input features found. Skipping One-Hot Encoding.")

        # ---------------------------------------------------------------------
        # Handle missing values
        # ---------------------------------------------------------------------
        initial_rows = len(self.dataset)
        final_cols = self.input_vars + self.output_vars

        if self.dataset[final_cols].isnull().values.any():
            missing_cols = (
                self.dataset[final_cols]
                .columns[self.dataset[final_cols].isnull().any()]
                .tolist()
            )
            log_to_queue(
                f"⚠️ Missing values detected. Columns: {', '.join(missing_cols)}. "
                f"Dropping rows..."
            )
            self.dataset.dropna(subset=final_cols, inplace=True)

        final_rows = len(self.dataset)
        if initial_rows != final_rows:
            log_to_queue(
                f"Removed {initial_rows - final_rows} rows due to missing values. "
                f"Dataset size reduced from {initial_rows} to {final_rows}."
            )

        if final_rows == 0:
            error_message = (
                "Dataset loading failed: All remaining rows were removed "
                "due to missing values."
            )
            log_to_queue(error_message)
            raise ValueError(error_message)

        # ---------------------------------------------------------------------
        # Extract numpy arrays
        # ---------------------------------------------------------------------
        # Keep a copy of inputs as DataFrame for later scaling if needed
        self.input_df = self.dataset[self.input_vars].copy()

        full_input_data = self.input_df.to_numpy(dtype="float32")
        full_output_data = self.dataset[self.output_vars].to_numpy(dtype="float32")

        # ---------------------------------------------------------------------
        # Train/Test Split (if requested)
        # ---------------------------------------------------------------------
        if 0.0 < test_split_ratio < 1.0:
            log_to_queue(
                f"Performing Train/Test split: "
                f"Test Ratio={test_split_ratio:.2f} (Random Seed={self.random_seed})."
            )

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                full_input_data,
                full_output_data,
                test_size=test_split_ratio,
                random_state=self.random_seed,
            )

            log_to_queue(
                f"Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}"
            )

            # Primary data used by training pipeline is the training subset
            self.input_data = self.X_train
            self.output_data = self.y_train
        else:
            # No split: all data considered training
            self.X_train = full_input_data
            self.y_train = full_output_data
            self.X_test = None
            self.y_test = None

            self.input_data = full_input_data
            self.output_data = full_output_data

        # ---------------------------------------------------------------------
        # Final attributes
        # ---------------------------------------------------------------------
        self.full_data = np.concatenate([self.input_data, self.output_data], axis=1)
        self.n_input_params = len(self.input_vars)
        self.n_output_params = len(self.output_vars)

        # Will be set when scaling
        self.scaler = None

    # -------------------------------------------------------------------------
    # Scaling
    # -------------------------------------------------------------------------
    def apply_scaler(self, scaler=None, is_fitting=False):
        """
        Apply feature standardization to self.input_data (training or test).

        - If is_fitting=True:
            * Fit a new or provided scaler on the current self.input_data (training)
            * Transform in-place
        - If is_fitting=False and scaler is provided:
            * Only transform using the provided scaler (e.g., test data)
        """
        def log_to_queue(message: str):
            if self.update_queue:
                self.update_queue.put({"key": "log_messages", "value": message})

        if not self.should_standardize:
            log_to_queue("Standardization disabled by config. Skipping scaling.")
            return

        if is_fitting:
            if scaler is None:
                scaler = StandardScaler()

            # Fit on current input_data
            input_scaled = scaler.fit_transform(self.input_data)
            self.scaler = scaler
            log_to_queue("Features standardized successfully (Fitted and Transformed).")

        elif scaler is not None:
            # Transform using existing scaler
            input_scaled = scaler.transform(self.input_data)
            self.scaler = scaler
            log_to_queue("Features standardized successfully (Transformed using provided scaler).")

        else:
            error_message = (
                "Standardization is enabled, but no fitted scaler was provided for transformation."
            )
            log_to_queue(f"❌ {error_message}")
            raise ValueError(error_message)

        # Update arrays
        self.input_data = input_scaled.astype("float32")
        self.full_data = np.concatenate([self.input_data, self.output_data], axis=1)

    # -------------------------------------------------------------------------
    # Misc
    # -------------------------------------------------------------------------
    def get_rows(self):
        return len(self.input_data)
