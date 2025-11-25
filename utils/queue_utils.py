import streamlit as st
import queue

def process_queue_updates():
    q = st.session_state.update_queue
    updates_processed = 0

    while not q.empty():
        try:
            update = q.get_nowait()
            key = update['key']
            value = update['value']

            if key == 'log_messages':
                st.session_state.log_messages.append(value)
            elif key == 'is_running':
                st.session_state.is_running = value
            # The thread object and stop event are handled in the main loop and initialization
            # We don't need to overwrite them based on queue updates unless signaling a full cleanup
            else:
                st.session_state[key] = value

            q.task_done()
            updates_processed += 1

        except queue.Empty:
            break
        except Exception as e:
            st.error(f"Error processing queue update: {e}")

    return updates_processed > 0