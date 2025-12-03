# utils/queue_utils.py

import streamlit as st
import queue
import os

from utils.run_manager import append_run_log, write_value_file


def process_queue_updates() -> bool:
    """
    Process all pending messages from st.session_state.update_queue and
    apply them to st.session_state.

    Returns
    -------
    bool
        True if at least one update was processed, False otherwise.
    """
    ss = st.session_state
    q = ss.get("update_queue", None)

    if q is None:
        # No queue set yet – nothing to process
        return False

    updates_processed = 0

    while True:
        try:
            update = q.get_nowait()
        except queue.Empty:
            break
        except Exception as e:
            st.error(f"Error processing queue update: {e}")
            break

        if not isinstance(update, dict):
            # Ignore malformed messages
            q.task_done()
            continue

        key = update.get("key")
        value = update.get("value")

        # ---------------------------------------------------------------------
        # Special-case handling for known keys
        # ---------------------------------------------------------------------
        if key == "log_messages":
            # Ensure list exists
            if "log_messages" not in ss:
                ss.log_messages = []
            ss.log_messages.append(value)
            # Mirror live log into summary.txt of current run, if known
            run_dir = ss.get("current_run_dir")
            append_run_log(run_dir, str(value))

        elif key == "is_running":
            ss.is_running = bool(value)

        elif key == "optuna_study":
            # Store the Optuna Study object so render_optuna_plots() can use it
            ss.optuna_study = value
        elif key == "live_best_test_metrics":
            ss.live_best_test_metrics = value
            run_dir = ss.get("current_run_dir")
            if value:
                append_run_log(run_dir, f"[best_metrics] {value}")
                write_value_file(run_dir, "best_metrics.json", value)

        elif key == "best_loss_so_far":
            try:
                ss.best_loss_so_far = float(value)
            except Exception:
                ss.best_loss_so_far = value
            run_dir = ss.get("current_run_dir")
            append_run_log(run_dir, f"[best_cv_loss] {value}")
            write_value_file(run_dir, "best_cv_loss.txt", value)

        elif key == "best_params_so_far":
            ss.best_params_so_far = value
            run_dir = ss.get("current_run_dir")
            append_run_log(run_dir, f"[best_params] {value}")
            write_value_file(run_dir, "best_params.json", value)

        elif key in ("best_intermediate_r2", "best_intermediate_nmae", "best_intermediate_accuracy"):
            ss[key] = value
            run_dir = ss.get("current_run_dir")
            append_run_log(run_dir, f"[{key}] {value}")
            filename = f"{key}.txt"
            write_value_file(run_dir, filename, value)

        # Add other explicit keys here if you like, e.g. best_loss_so_far, etc.
        # elif key == "best_loss_so_far":
        #     ss.best_loss_so_far = float(value)
        # elif key == "best_params_so_far":
        #     ss.best_params_so_far = value

        else:
            # Generic fallback: store under its key in session_state
            if key is not None:
                ss[key] = value

        q.task_done()
        updates_processed += 1

    return updates_processed > 0
