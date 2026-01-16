# ui/results_panel.py

import os
import copy

import streamlit as st
import optuna.visualization as ov

import pandas as pd  # for isinstance checks if needed

from src.dataset import Dataset
from src.model_test import test
from src.plot import make_plotly_figure
from utils.run_manager import zip_run_dir
from utils.plot_utils import save_plot_with_fallback
from pathlib import Path


def get_test_dataset():
    """
    Builds the test Dataset object for final evaluation.

    IMPORTANT:
    - We avoid double-processing the data.
    - If a preprocessed test DataFrame is available in session_state (test_dataset_df),
      we use that and set input_names='*' so Dataset treats all non-output columns as inputs.
    """
    config_base = st.session_state.config
    update_queue = st.session_state.update_queue
    data_split_mode = st.session_state.data_split_mode

    # Prefer the preprocessed test DataFrame if available
    test_df = st.session_state.get("test_dataset_df", None)

    # -------------------------------------------------------------------------
    # Case 1: We already have a processed test DataFrame (single_file or separate_files)
    # -------------------------------------------------------------------------
    if (
        data_split_mode in ("single_file", "separate_files")
        and isinstance(test_df, pd.DataFrame)
        and not test_df.empty
    ):
        st.session_state.log_messages.append(
            "Creating test dataset from preprocessed test DataFrame."
        )

        # Clone config and override input_names to '*'
        cfg = copy.deepcopy(config_base)
        vars_cfg = cfg.setdefault("variables", {})
        # '*' means: use all columns except the configured output_names as inputs
        vars_cfg["input_names"] = "*"

        return Dataset(
            source=test_df,
            config=cfg,
            update_queue=update_queue,
            test_split_ratio=0.0,  # no further split
        )

    # -------------------------------------------------------------------------
    # Case 2: separate_files mode but no preprocessed DF (fallback)
    # -------------------------------------------------------------------------
    if data_split_mode == "separate_files":
        # Fallback: load from the test file path and process once.
        if not hasattr(st.session_state, "uploaded_test_file") or st.session_state.uploaded_test_file is None:
            st.error("Error: No uploaded test file found in session state.")
            return None

        test_file_name = st.session_state.uploaded_test_file.name
        test_path = f"temp_data/test_data.{test_file_name.split('.')[-1]}"

        st.session_state.log_messages.append(
            f"Creating test dataset from raw test file: {test_path}"
        )

        # Here we can still use the base config, since this is the first processing of that file
        return Dataset(
            source=test_path,
            config=config_base,
            update_queue=update_queue,
            test_split_ratio=0.0,
        )

    # -------------------------------------------------------------------------
    # Case 3: No data available
    # -------------------------------------------------------------------------
    st.error("Error: No test data available for final evaluation.")
    return None


def run_final_test():
    """Executes the final model test and stores results in session state."""
    try:
        dataset_test = get_test_dataset()

        if dataset_test:
            fitted_scaler = st.session_state.fitted_scaler

            if fitted_scaler is not None:
                st.session_state.log_messages.append("Applying fitted scaler to test data.")
                dataset_test.apply_scaler(scaler=fitted_scaler, is_fitting=False)

            test_metrics = test(
                dataset_test,
                st.session_state.final_model_path,
                st.session_state.best_params_so_far,
            )
            st.session_state.test_results = test_metrics
            st.rerun()

    except Exception as e:
        st.error(f"Failed to run test: {e}")


def render_final_download():
    """Offer run archive (zip) and predictions (csv) downloads."""
    run_dir = st.session_state.get("current_run_dir")
    if not run_dir:
        st.info("No run directory found to download artifacts.")
        return

    # Ensure we have a zip archive for the run
    zip_path = st.session_state.get("final_zip_path")
    if not zip_path or not os.path.exists(zip_path):
        zip_path = zip_run_dir(run_dir)
        if zip_path:
            st.session_state.final_zip_path = zip_path

    preds_path = os.path.join(run_dir, "best_predictions.csv")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if zip_path and os.path.exists(zip_path):
            try:
                with open(zip_path, "rb") as f_zip:
                    st.download_button(
                        label="Download run archive (.zip)",
                        data=f_zip.read(),
                        file_name=os.path.basename(zip_path),
                        mime="application/zip",
                        type="primary",
                    )
            except Exception as e:
                st.error(f"Could not read run archive: {e}")
        else:
            st.info("Run archive not available.")

    with dl_col2:
        if preds_path and os.path.exists(preds_path):
            try:
                with open(preds_path, "rb") as f_preds:
                    st.download_button(
                        label="Download predictions (CSV)",
                        data=f_preds.read(),
                        file_name="best_predictions.csv",
                        mime="text/csv",
                        type="secondary",
                    )
            except Exception as e:
                st.error(f"Could not read predictions file: {e}")
        else:
            st.info("Predictions file not available.")


from pathlib import Path
from optuna.importance import FanovaImportanceEvaluator

def render_optuna_plots():
    """Renders the Optuna optimization plots and saves them as PDFs only."""
    config = st.session_state.current_ui_config
    display_config = config.get("display", {})
    run_dir = st.session_state.get("current_run_dir")

    plots_dir = None
    if run_dir:
        plots_dir = Path(run_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

    if not display_config.get("show_optuna_plots", True):
        return

    study = st.session_state.get("optuna_study")
    if study is None:
        st.info("Optuna study results not found in session state.")
        return

    st.subheader("Hyperparameter Optimization Analysis")

    # Fixed-seed evaluator prevents jitter across reruns
    importance_evaluator = FanovaImportanceEvaluator(seed=0)

    # ------------------------------------------------------------------
    # Optimization history
    # ------------------------------------------------------------------
    try:
        fig_history = ov.plot_optimization_history(study)
        st.plotly_chart(fig_history, use_container_width=True)

        # Save ONLY as PDF
        if plots_dir:
            try:
                save_plot_with_fallback(
                    fig_history, plots_dir / "optuna_optimization_history.pdf"
                )
            except Exception as e:
                st.warning(f"Could not save Optimization History plot: {e}")
    except Exception as e:
        st.warning(f"Could not generate Optimization History plot: {e}")

    # ------------------------------------------------------------------
    # Parameter importance
    # ------------------------------------------------------------------
    try:
        fig_importance = ov.plot_param_importances(
            study, evaluator=importance_evaluator
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        # Save ONLY as PDF
        if plots_dir:
            try:
                save_plot_with_fallback(
                    fig_importance, plots_dir / "optuna_param_importance.pdf"
                )
            except Exception as e:
                st.warning(f"Could not save Parameter Importance plot: {e}")
    except Exception as e:
        st.warning(f"Could not generate Parameter Importance plot: {e}")

    st.markdown("---")


def render_final_results():
    """Render the final results section (PDF-only plot saving)."""

    st.divider()
    st.header("Final Results")

    # Condition to trigger the final testing phase
    if st.session_state.final_model_path and st.session_state.data_split_mode != "none":

        # Run final test once training is finished
        if not st.session_state.test_results and not st.session_state.is_running:
            run_final_test()   # triggers rerun on success

        # Once test results exist, display everything
        if st.session_state.test_results:
            st.subheader("Test Metrics")
            results = st.session_state.test_results
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Test NMAE", f"{results['NMAE']:.4f}")
            r_col2.metric("Test R² Score", f"{results['R2']:.4f}")
            r_col3.metric("Test Accuracy", f"{results['Accuracy']:.2f}%")

            # Final Model Download
            render_final_download()

            st.markdown("---")

            # Parity Plot
            config = st.session_state.current_ui_config
            display_config = config.get("display", {})

            if display_config.get("show_prediction_plot", True):
                st.subheader("Final Model Prediction Analysis (Parity Plot)")
                fig_parity = make_plotly_figure(results["y_pred"], results["y_true"])
                st.plotly_chart(fig_parity, use_container_width=True)

                # Save ONLY PDF — no HTML anymore
                run_dir = st.session_state.get("current_run_dir")
                if run_dir:
                    plots_dir = Path(run_dir) / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    pdf_path = plots_dir / "parity_plot.pdf"
                    try:
                        save_plot_with_fallback(
                            fig_parity, pdf_path
                        )
                    except Exception as e:
                        st.warning(f"Could not save parity plot: {e}")

                st.markdown("---")

            # Optuna Plots (only PDF saving)
            try:
                render_optuna_plots()
            except TypeError:
                if "log_messages" in st.session_state:
                    st.session_state.log_messages.appendleft(
                        "Skipped Optuna plots due to a frontend TypeError."
                    )

            # After plots saved, zip the run directory
            run_dir = st.session_state.get("current_run_dir")
            if run_dir:
                zip_path = zip_run_dir(run_dir)
                if zip_path:
                    st.session_state.final_zip_path = zip_path

    # Training manually stopped
    elif st.session_state.was_stopped_manually and os.path.exists(
        st.session_state.best_model_path
    ):
        st.info("Training was manually stopped. View the best intermediate results above.")

    # Training still running
    elif st.session_state.is_running:
        st.info("Waiting for training to complete to display final results...")

    # No data
    else:
        st.warning("Upload a training file and run training to see final results.")
