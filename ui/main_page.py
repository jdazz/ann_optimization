# FILE: ui/main_page.py

import streamlit as st
import os
from src.plot import make_plot # Import your plotting function

def render_results(results, plot_config):
    """
    Displays the final metrics, download button, and result plot.
    """
    st.success("Training and Testing Complete!")
    
    # --- Download Best Model Button ---
    model_path = os.path.join("models", "ANN_best_model.pt")
    if os.path.exists(model_path):
        with open(model_path, "rb") as model_file:
            st.download_button(
                label="📥 Download Best Model",
                data=model_file,
                file_name="ANN_best_model.pt",
                mime="application/octet-stream",
                help="Click to download the trained PyTorch model file."
            )
    else:
        st.warning("Best model file not found in 'models/' folder.")
    
    # --- Metrics ---
    st.subheader("Final Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy (MRE %)", f"{results['test_accuracy']:.2f}%")
    col2.metric("NMAE", f"{results['nmae']:.4f}")
    col3.metric("R² Score", f"{results['r2']:.4f}")

    # --- Display Results Plot ---
    with st.expander("Prediction Plot", expanded=False):
        if plot_config.get("show_plot", True):
            fig = make_plot(
                results['mre_list'], 
                results['y_pred'], 
                results['y_true'], 
                save_path=None  # Don't save, just display
            )
            st.pyplot(fig)
        else:
            st.info("Plot display is disabled in the configuration.")
    
    # --- Hyperparameters ---
    with st.expander("Best Hyperparameters"):
        st.json(results['best_param'])
        
    # --- Model Architecture ---
    with st.expander("Final Model Architecture"):
        st.text(results['model_structure'])

def render_log(log_output):
    """
    Displays the captured training log in an expander.
    """
    if log_output:
        with st.expander("Full Training Log"):
            st.text_area("Log", log_output, height=400)

def render_footer():
    """
    Displays the footer.
    """
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 12px;
            color: #555; /* Dark gray */
        }
        </style>
        <div class="footer">
            This program was developped by the Institute for Dynamic Systems and Control at ETHZ
        </div>
        """,
        unsafe_allow_html=True
    )