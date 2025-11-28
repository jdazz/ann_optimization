# utils/queue_utils.py

import streamlit as st
import queue


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

        elif key == "is_running":
            ss.is_running = bool(value)

        elif key == "optuna_study":
            # Store the Optuna Study object so render_optuna_plots() can use it
            ss.optuna_study = value
        elif key == "live_best_test_metrics":
            ss.live_best_test_metrics = value

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