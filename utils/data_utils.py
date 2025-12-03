# utils/data_utils.py - FIXED initialize_training_dataset (no double Dataset on single-file)

import streamlit as st
import pandas as pd
import numpy as np

from src.dataset import Dataset


def detect_and_handle_data_input(uploaded_train_file, uploaded_test_file, user_set_split_ratio):
    """
    Detects the data input mode and sets session state variables.
    
    Runs on every script execution, but does not override flags related
    to a currently running training thread.
    """
    if uploaded_train_file is not None and uploaded_test_file is not None:
        # Two separate files: train + test
        st.session_state.data_split_mode = "separate_files"
        st.session_state.test_split_ratio = 0.0
        st.session_state.is_data_ready_for_new_run = True

    elif uploaded_train_file is not None and uploaded_test_file is None:
        # Single file: will be split internally
        st.session_state.data_split_mode = "single_file"
        st.session_state.test_split_ratio = user_set_split_ratio
        st.session_state.is_data_ready_for_new_run = True

        st.warning(
            f"⚠️ Detected: Single File. Data will be automatically split: "
            f"**Train {1 - st.session_state.test_split_ratio:.0%} / "
            f"Test {st.session_state.test_split_ratio:.0%}**."
        )

    else:
        st.session_state.data_split_mode = "none"
        st.session_state.test_split_ratio = 0.0
        st.session_state.is_data_ready_for_new_run = False

    return st.session_state.is_data_ready_for_new_run


def initialize_training_dataset(train_path, test_path):
    """
    Initializes the training Dataset object and stores the test DataFrame
    in st.session_state.test_dataset_df, handling both split modes.

    Args:
        train_path: Source for training data (Streamlit file object or path).
        test_path:  Source for testing data (Streamlit file object or path).

    Returns: 
        dataset_train: Dataset object (for training, processed but not scaled).
    """
    # You might be using current_ui_config or a separate config object.
    # Adjust this line if your config lives somewhere else.
    config = st.session_state.config
    update_queue = st.session_state.update_queue
    split_ratio_to_use = st.session_state.test_split_ratio

    dataset_train = None
    test_df = None

    def log_to_queue(message: str):
        if update_queue:
            update_queue.put({"key": "log_messages", "value": message})

    # -------------------------------------------------------------------------
    # MODE 1: Single file → Dataset handles split ONCE
    # -------------------------------------------------------------------------
    if st.session_state.data_split_mode == "single_file":
        log_to_queue("Loading and splitting single file (one Dataset instance).")

        # 1) Single Dataset instance does:
        #    - load data
        #    - column selection
        #    - one-hot encoding
        #    - missing-value handling
        #    - train/test split
        dataset_train = Dataset(
            source=train_path,
            config=config,
            update_queue=update_queue,
            test_split_ratio=split_ratio_to_use,
        )

        # 2) Build a test DataFrame from the Dataset's X_test / y_test
        if dataset_train.X_test is not None and dataset_train.y_test is not None:
            cols = dataset_train.input_vars + dataset_train.output_vars

            test_arr = np.concatenate(
                [dataset_train.X_test, dataset_train.y_test], axis=1
            )
            test_df = pd.DataFrame(test_arr, columns=cols)

            log_to_queue(
                f"Single-file mode: Test set created with shape {test_df.shape} "
                f"and columns aligned to training features."
            )
        else:
            log_to_queue("Single-file mode: No test split created (X_test/y_test is None).")
            test_df = None

    # -------------------------------------------------------------------------
    # MODE 2: Separate files → one Dataset for train, one for test
    # -------------------------------------------------------------------------
    else:  # st.session_state.data_split_mode == "separate_files"
        log_to_queue("Loading separate training and testing files.")

        # 1) Training Dataset (no internal split)
        dataset_train = Dataset(
            source=train_path,
            config=config,
            update_queue=update_queue,
            test_split_ratio=0.0,
        )

        # 2) Test Dataset (processed separately)
        try:
            log_to_queue("Processing separate testing file to align columns and encoding...")

            temp_test_dataset = Dataset(
                source=test_path,
                config=config,
                update_queue=update_queue,
                test_split_ratio=0.0,
            )

            # Fully processed (OHE, missing handled) test data
            test_df = temp_test_dataset.dataset.copy()

            log_to_queue(
                f"Separate testing file loaded and processed successfully. "
                f"Shape: {test_df.shape}"
            )

        except Exception as e:
            log_to_queue(f"❌ Error loading separate test file for processing: {e}")
            test_df = None

    # -------------------------------------------------------------------------
    # FINAL: Store test_df for later use in the pipeline
    # -------------------------------------------------------------------------
    st.session_state.test_dataset_df = test_df

    return dataset_train, test_df