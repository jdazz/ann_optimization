# plot.py - Fixed for Streamlit Thread Safety

import matplotlib.pyplot as plt
import numpy as np # Import numpy for safe calculations

def make_plot(y_pred_plot, y_true_plot):
    """
    Generates 2 plots:
      1. Predictions vs. Labels (by sample index)
      2. Predictions vs. Labels (scatter with ±3% error lines)
    
    Args:
        y_pred_plot (list/np.array): Predicted output values
        y_true_plot (list/np.array): True output values
        
    Returns:
        matplotlib.figure.Figure: The created Matplotlib Figure object.
    """
    print("Generating plots...")
    
    # 1. Convert inputs to NumPy arrays for safe calculation
    y_true = np.array(y_true_plot)
    y_pred = np.array(y_pred_plot)
    
    # CRITICAL FIX: Use plt.subplots() to create the Figure (fig) and Axes (axs) explicitly.
    # This prevents using the global figure object.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6)) # Create a 1 row, 2 column subplot structure

    font = {'family': 'serif', 'color': 'darkred', 'size': 14} # Adjusted font size for clarity

    # --- Plot 1: Predictions vs True (sample index) ---
    ax1 = axs[0] # Access the first Axes object
    
    ax1.plot(range(len(y_true)), y_pred, 'o', color='blue', alpha=0.7, label='Predictions')
    ax1.plot(range(len(y_true)), y_true, 'o', color='red', alpha=0.5, label='Labels')
    
    # Set labels and title using the Axes object methods (ax.set_...)
    ax1.set_title('Predictions vs. Labels (Sample Index)', fontdict=font)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Output Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)


    # --- Plot 2: Predictions vs True (scatter with ±3% error lines) ---
    ax2 = axs[1] # Access the second Axes object
    
    # Scatter plot
    ax2.plot(y_true, y_pred, 'o', color='blue', alpha=0.5, label='Predictions')
    
    # Perfect prediction line (y=x)
    ax2.plot(y_true, y_true, color='red', linewidth=2, label='Perfect Prediction (y=x)')
    
    # ±3% error lines
    ax2.plot(y_true, y_true * 1.03, '--', color='gray', label='±3% Error Bound')
    ax2.plot(y_true, y_true * 0.97, '--', color='gray') # No label for the second line
    
    # Set labels and title using the Axes object methods
    ax2.set_title('Error Plot: Predictions vs. True', fontdict=font)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout()

    # CRITICAL FIX: Always return the figure object for st.pyplot(fig)
    return fig

# NOTE: The save_path logic was removed, as Streamlit handles display via fig.
# If you need saving, that logic should be added back, but it's not needed for st.pyplot.