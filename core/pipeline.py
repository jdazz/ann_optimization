# core/pipeline.py

import os
import json
import torch
import threading
import traceback
import queue
import pandas as pd
import optuna.visualization as ov

from datetime import datetime

from src.train import optimization, train_final_model
from src.model import define_net_regression
from src.plot import make_plotly_figure
from utils.save_onnx import export_to_onnx
from utils.plot_utils import save_plot_with_fallback
from utils.run_manager import (
    derive_dataset_name,
    make_config_hash,
    start_run_folder,
    finalize_run_folder,
    write_summary,
    append_run_log,
    read_value_file,
    zip_run_dir,
)

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training_pipeline(
    dataset_train,
    dataset_test,
    config,
    update_queue: queue.Queue,
    stop_event: threading.Event,
    is_resume: bool = False,
    optuna_study=None,
    dataset_label: str | None = None,
):
    """
    Background training pipeline.

    IMPORTANT:
    - This function runs in a background thread.
    - It MUST NOT call any Streamlit APIs directly (no `st.*`, no SessionStateProxy).
    - All communication with the UI goes through `update_queue`.
    """

    started_at = datetime.utcnow().isoformat()

    def cleanup_in_progress_zip(path: str):
        """Remove any IN_PROGRESS zip for the given path."""
        zip_path = f"{path}.zip"
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass

    def send_update(key, value):
        """Helper to push updates to the main Streamlit thread."""
        payload = {"key": key, "value": value}
        # Mirror critical logs directly to disk so summary.txt keeps growing even
        # if the UI session is reloaded and misses queue processing.
        if key == "log_messages":
            append_run_log(run_dir, str(value))
            payload["logged"] = True
        update_queue.put(payload)

    def generate_plots(target_dir: str, study_obj):
        """Create parity and Optuna plots under target_dir/plots."""
        plots_dir = os.path.join(target_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Parity plot from saved predictions
        preds_path = os.path.join(target_dir, "best_predictions.csv")
        if os.path.exists(preds_path):
            try:
                df = pd.read_csv(preds_path)
                y_true_cols = [c for c in df.columns if c.startswith("y_true_")]
                y_pred_cols = [c for c in df.columns if c.startswith("y_pred_")]
                if y_true_cols and y_pred_cols:
                    y_true = df[y_true_cols].values.flatten()
                    y_pred = df[y_pred_cols].values.flatten()
                    fig = make_plotly_figure(y_pred, y_true)
                    saved_path, _, _ = save_plot_with_fallback(
                        fig, os.path.join(plots_dir, "parity_plot.pdf")
                    )
                    send_update("log_messages", f"Saved parity plot to {saved_path.name}")
            except Exception as e:
                append_run_log(target_dir, f"[plot_error] Parity plot failed: {e}")

        # Optuna plots if study available
        if study_obj is not None:
            try:
                fig_hist = ov.plot_optimization_history(study_obj)
                save_plot_with_fallback(
                    fig_hist, os.path.join(plots_dir, "optuna_optimization_history.pdf")
                )
            except Exception as e:
                append_run_log(target_dir, f"[plot_error] Optuna history plot failed: {e}")

            try:
                fig_imp = ov.plot_param_importances(study_obj)
                save_plot_with_fallback(
                    fig_imp, os.path.join(plots_dir, "optuna_param_importance.pdf")
                )
            except Exception as e:
                append_run_log(target_dir, f"[plot_error] Optuna importance plot failed: {e}")

    def abort_run():
        """Handle abort: rename folder to ABORTED and send state updates."""
        send_update("log_messages", "Stop signal received. Aborting run.")
        send_update("is_running", False)
        send_update("run_status", "ABORTED")
        final_dir = finalize_run_folder(run_dir, "ABORTED")
        send_update("current_run_dir", final_dir)
        write_summary(
            final_dir,
            dataset_name=dataset_name,
            config_hash=config_hash,
            status="ABORTED",
            best_params=None,
            best_loss=None,
            test_metrics=None,
            started_at=started_at,
            finished_at=datetime.utcnow().isoformat(),
            notes="Aborted by user",
        )
        return final_dir

    # ------------------------------------------------------------------
    # Run directory setup (dataset + config hash)
    # ------------------------------------------------------------------
    dataset_name = dataset_label or derive_dataset_name(dataset_train)
    config_hash = make_config_hash(config)
    base_runs = os.path.join(os.getcwd(), "runs")
    run_dir, run_id = start_run_folder(base_runs, dataset_name, config_hash)
    best_model_pt_path = os.path.join(run_dir, "best_model.pt")
    best_model_onnx_path = os.path.join(run_dir, "best_model.onnx")

    send_update("current_run_dir", run_dir)
    send_update("log_messages", f"Run directory initialized at {run_dir}")

    try:
        # Mark run as started
        send_update("is_running", True)
        send_update("log_messages", "Starting HPO optimization...")

        if stop_event.is_set():
            abort_run()
            return

        # 1. Run Hyperparameter Optimization (HPO)
        best_params, study = optimization(
            dataset_train,
            dataset_test,
            config,
            update_queue,
            stop_event,
            is_resume,
            optuna_study,
            run_dir,
        )

        if stop_event.is_set():
            abort_run()
            return

        # Once optimization returns, this run is no longer resumable
        send_update("is_resumable", False)

        if study is not None:
            # Send Optuna study object back to main thread for later visualization / resume
            send_update("optuna_study", study)
            send_update(
                "log_messages",
                "Optimization complete. Optuna study object saved for visualization.",
            )

        if not best_params:
            # No best params found (stopped or failed)
            send_update(
                "log_messages",
                "Optimization stopped or failed to find best parameters.",
            )
            send_update("is_running", False)
            return

        # ------------------------------------------------------------------
        # 2. Prefer the already-trained best intermediate model
        # ------------------------------------------------------------------
        if os.path.exists(best_model_pt_path):
            # Use the already-trained best model as final
            final_model_path = best_model_pt_path

            # Handle ONNX path
            final_onnx_path = best_model_onnx_path
            if not os.path.exists(final_onnx_path):
                # No ONNX yet → try exporting from the saved best model weights
                try:
                    model_for_export = define_net_regression(
                        best_params,
                        dataset_train.n_input_params,
                        dataset_train.n_output_params,
                    ).to(Device)
                    state_dict = torch.load(final_model_path, map_location=Device)
                    model_for_export.load_state_dict(state_dict)

                    final_onnx_path = export_to_onnx(
                        model_for_export,
                        dataset_train,
                        os.path.join(run_dir, "best_model"),
                    )
                except Exception as e:
                    send_update(
                        "log_messages",
                        f"⚠️ Failed to export ONNX from best model: {e}",
                    )
                    final_onnx_path = None
            else:
                final_onnx_path = best_model_onnx_path

            # Inform UI of final paths and params (before finalize)
            send_update(
                "log_messages",
                "Using best intermediate model for final outputs (renamed to final).",
            )
            send_update("best_model_path", final_model_path)
            send_update("best_onnx_path", final_onnx_path)
            send_update("final_model_path", final_model_path)
            send_update("final_onnx_path", final_onnx_path)
            send_update("best_params_so_far", best_params)
            send_update("is_running", False)
            send_update("is_resumable", False)

            # Read best_metrics.json from run_dir (it will move with finalize)
            raw_metrics = read_value_file(run_dir, "best_metrics.json")
            best_metrics = None
            if isinstance(raw_metrics, dict):
                best_metrics = raw_metrics
            elif isinstance(raw_metrics, str):
                try:
                    best_metrics = json.loads(raw_metrics)
                except Exception:
                    best_metrics = None

            metrics_lite = None
            if isinstance(best_metrics, dict):
                metrics_lite = {
                    "NMAE": best_metrics.get("NMAE"),
                    "R2": best_metrics.get("R2"),
                    "Accuracy": best_metrics.get("Accuracy"),
                }

            # Finalize run folder first, then write summary + append final metrics line
            finished_at = datetime.utcnow().isoformat()
            final_dir = finalize_run_folder(run_dir, "DONE")
            cleanup_in_progress_zip(run_dir)
            # Update paths to final_dir
            final_model_path = os.path.join(final_dir, "best_model.pt")
            final_onnx_path = os.path.join(final_dir, "best_model.onnx")

            send_update("best_model_path", final_model_path)
            send_update("final_model_path", final_model_path)
            send_update("run_status", "DONE")
            if os.path.exists(final_onnx_path):
                send_update("best_onnx_path", final_onnx_path)
                send_update("final_onnx_path", final_onnx_path)

            # Write structured summary
            write_summary(
                final_dir,
                dataset_name=dataset_name,
                config_hash=config_hash,
                status="DONE",
                best_params=best_params,
                best_loss=study.best_value if study else None,
                test_metrics=metrics_lite,
                started_at=started_at,
                finished_at=finished_at,
            )

            # Log final metrics to UI and ensure they are the LAST line in summary.txt
            if metrics_lite:
                send_update(
                    "log_messages",
                    "Final test metrics — "
                    f"NMAE: {metrics_lite.get('NMAE', '—')}, "
                    f"R2: {metrics_lite.get('R2', '—')}, "
                    f"Accuracy: {metrics_lite.get('Accuracy', '—')}",
                )
                final_line = (
                    "Final metrics on unseen data: "
                    f"r2score: {metrics_lite.get('R2', '—')}, "
                    f"NMA: {metrics_lite.get('NMAE', '—')}, "
                    f"accuracy: {metrics_lite.get('Accuracy', '—')}"
                )
                # This append is intentionally LAST, so this line is at the bottom.
                append_run_log(final_dir, final_line)

            # Save plots before zipping
            generate_plots(final_dir, study)

            # Zip final DONE folder for download resiliency after reloads
            final_zip = zip_run_dir(final_dir)
            if final_zip:
                send_update("final_zip_path", final_zip)
                send_update(
                    "log_messages",
                    f"Run artifacts zipped (DONE): {os.path.basename(final_zip)}",
                )

            send_update("current_run_dir", final_dir)
            return

        # ------------------------------------------------------------------
        # 3. Fallback: define and train a final model from scratch
        # ------------------------------------------------------------------
        send_update("log_messages", "\n--- Creating Final Model Architecture ---")
        final_model = define_net_regression(
            best_params,
            dataset_train.n_input_params,
            dataset_train.n_output_params,
        ).to(Device)

        send_update("log_messages", "Starting final training on full dataset...")
        final_model = train_final_model(
            final_model,
            dataset_train.full_data,
            best_params,
            dataset_train.n_input_params,
            dataset_train.n_output_params,
            config,
            stop_event,
        )

        # 4. Save the Final Model
        os.makedirs(run_dir, exist_ok=True)
        final_model_path = best_model_pt_path
        torch.save(final_model.state_dict(), final_model_path)

        # 5. Export Model to ONNX Format
        try:
            final_onnx_path = export_to_onnx(
                final_model,
                dataset_train,
                os.path.join(run_dir, "best_model"),
            )
        except Exception as e:
            send_update(
                "log_messages",
                f"⚠️ Failed to export final ONNX model: {e}",
            )
            final_onnx_path = None

        # 6. Send Final Status Updates to Streamlit (before finalize)
        send_update("final_model_path", final_model_path)
        send_update("final_onnx_path", final_onnx_path)
        send_update("best_params_so_far", best_params)
        send_update(
            "log_messages",
            f"Final best model saved to: {final_model_path}",
        )
        send_update("is_running", False)

        # Finalize run folder and summary
        finished_at = datetime.utcnow().isoformat()
        final_dir = finalize_run_folder(run_dir, "DONE")
        cleanup_in_progress_zip(run_dir)
        final_model_path = os.path.join(final_dir, "best_model.pt")
        final_onnx_path = os.path.join(final_dir, "best_model.onnx")
        send_update("current_run_dir", final_dir)
        send_update("final_model_path", final_model_path)
        if os.path.exists(final_onnx_path):
            send_update("final_onnx_path", final_onnx_path)
            send_update("best_onnx_path", final_onnx_path)
        send_update("best_model_path", final_model_path)

        # Load metrics from best_metrics.json (now in final_dir)
        raw_metrics = read_value_file(final_dir, "best_metrics.json")
        best_metrics = None
        if isinstance(raw_metrics, dict):
            best_metrics = raw_metrics
        elif isinstance(raw_metrics, str):
            try:
                best_metrics = json.loads(raw_metrics)
            except Exception:
                best_metrics = None

        metrics_lite = None
        if isinstance(best_metrics, dict):
            metrics_lite = {
                "NMAE": best_metrics.get("NMAE"),
                "R2": best_metrics.get("R2"),
                "Accuracy": best_metrics.get("Accuracy"),
            }

        write_summary(
            final_dir,
            dataset_name=dataset_name,
            config_hash=config_hash,
            status="DONE",
            best_params=best_params,
            best_loss=study.best_value if study else None,
            test_metrics=metrics_lite,
            started_at=started_at,
            finished_at=finished_at,
        )

        if metrics_lite:
            print('______________________________________________________________________________')
            send_update(
                "log_messages",
                "Final test metrics — "
                f"NMAE: {metrics_lite.get('NMAE', '—')}, "
                f"R2: {metrics_lite.get('R2', '—')}, "
                f"Accuracy: {metrics_lite.get('Accuracy', '—')}",
            )
            final_line = (
                "Final metrics on unseen data: "
                f"r2score: {metrics_lite.get('R2', '—')}, "
                f"NMA: {metrics_lite.get('NMAE', '—')}, "
                f"accuracy: {metrics_lite.get('Accuracy', '—')}"
            )
            # Again, this append is LAST so it becomes the final line.
            append_run_log(final_dir, final_line)

        # Save plots before zipping
        generate_plots(final_dir, study)

        # Zip final DONE folder for download resiliency after reloads
        final_zip = zip_run_dir(final_dir)
        if final_zip:
            send_update("final_zip_path", final_zip)
            send_update(
                "log_messages",
                f"Run artifacts zipped (DONE): {os.path.basename(final_zip)}",
            )

    except Exception as e:
        # Use traceback to get better error detail
        full_traceback = traceback.format_exc()
        error_msg = f"❌ An error occurred in the training pipeline: {e}\n{full_traceback}"
        send_update("log_messages", error_msg)
        send_update("is_running", False)
        finished_at = datetime.utcnow().isoformat()
        write_summary(
            run_dir,
            dataset_name=dataset_name,
            config_hash=config_hash,
            status="FAILED",
            best_params=None,
            best_loss=None,
            test_metrics=None,
            started_at=started_at,
            finished_at=finished_at,
            notes=str(e),
        )
        finalize_run_folder(run_dir, "FAILED")
        cleanup_in_progress_zip(run_dir)

    finally:
        send_update("log_messages", "--- Training thread finished ---")
