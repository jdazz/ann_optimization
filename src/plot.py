# plot.py - Enhanced with Seaborn for Aesthetics

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns # 👈 NEW: Import Seaborn for better styling
import io # Not strictly needed here, but good practice for plots

def make_plot(y_pred_plot, y_true_plot):
    """
    Generates 2 aesthetic plots using Seaborn styling:
      1. Predictions vs. Labels (by sample index)
      2. Residual Scatter Plot (Error vs. True Value)
    
    Args:
        y_pred_plot (list/np.array): Predicted output values
        y_true_plot (list/np.array): True output values
        
    Returns:
        matplotlib.figure.Figure: The created Matplotlib Figure object (with Seaborn style).
    """
    
    # 1. Convert inputs to NumPy arrays
    y_true = np.array(y_true_plot).flatten()
    y_pred = np.array(y_pred_plot).flatten()
    
    # Calculate Residuals and MRE
    residuals = y_pred - y_true
    # Calculate Mean Relative Error (MRE) for display
    # Use small epsilon to prevent division by zero for true values near zero
    epsilon = 1e-6
    mre = np.mean(np.abs(residuals / (y_true + epsilon))) * 100
    
    # --- Apply Seaborn Style ---
    sns.set_theme(style="whitegrid") 
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 7)) # Slightly wider figure

    # --- Plot 1: Predictions vs True (Line Plot by Sample Index) ---
    ax1 = axs[0]
    
    # Use the seaborn plot style
    sns.lineplot(x=range(len(y_true)), y=y_true, ax=ax1, label='True Values', color='darkred', linewidth=1.5)
    sns.scatterplot(x=range(len(y_true)), y=y_pred, ax=ax1, label='Predictions', color='steelblue', alpha=0.7, s=40)
    
    ax1.set_title(f'Predictions vs. True Values (MRE: {mre:.2f}%)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Output Value', fontsize=12)
    ax1.legend(loc='upper right', frameon=True)


    # --- Plot 2: Residual Scatter Plot (Error vs. True Value) ---
    ax2 = axs[1]
    
    # Scatter plot of residuals vs. true values
    sns.scatterplot(x=y_true, y=residuals, ax=ax2, color='#2ca02c', alpha=0.6, s=40)
    
    # Zero error line
    ax2.axhline(0, color='red', linestyle='-', linewidth=1.5, label='Zero Error')

    # ±3% error bounds (using actual error values)
    error_bound_plus = y_true * 0.03
    error_bound_minus = -y_true * 0.03
    
    # Plotting the bounds as shaded area for better aesthetics
    ax2.fill_between(y_true, error_bound_minus, error_bound_plus, color='gray', alpha=0.15, label='±3% Error Band')
    
    ax2.set_title('Residual Error vs. True Values', fontsize=16, fontweight='bold')
    ax2.set_xlabel('True Values', fontsize=12)
    ax2.set_ylabel('Residual Error (Predicted - True)', fontsize=12)
    ax2.legend(loc='lower right', frameon=True)
    ax2.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on residual axis


    plt.tight_layout()
    # Reset style for other parts of the application if necessary, though Streamlit handles figure isolation
    # sns.reset_orig() 

    return fig


def make_plotly_figure(y_pred_plot, y_true_plot, error_pct=3.0):
    """
    Generates an interactive Plotly Parity Plot (True Values vs. Predicted Values).
    
    Args:
        y_pred_plot (list/np.array): Predicted output values.
        y_true_plot (list/np.array): True output values.
        error_pct (float): The percentage threshold for error boundaries (e.g., 3.0 for +/- 3%).
        
    Returns:
        plotly.graph_objects.Figure: The created Plotly Figure object.
    """
    y_true = np.array(y_true_plot).flatten()
    y_pred = np.array(y_pred_plot).flatten()
    residuals = y_pred - y_true
    
    # Calculate MRE for title
    epsilon = 1e-6
    mre = np.mean(np.abs(residuals / (y_true + epsilon))) * 100

    # Determine the range for the diagonal line and error bounds
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    # Add a small buffer to the plot limits
    plot_range = [min_val * 0.95, max_val * 1.05]
    
    # 1. Create a single Plotly Figure
    fig = go.Figure()

    # --- 1. Ideal Prediction Line (y=x) ---
    fig.add_trace(go.Scatter(
        x=plot_range, 
        y=plot_range, 
        mode='lines', 
        name='Ideal Prediction (y=x)', 
        line=dict(color='red', width=2, dash='dash'),
        hoverinfo='skip' # Don't show hover data for the diagonal line
    ))

    # --- 2. Error Bounds (e.g., +/- 3% lines) ---
    error_factor = error_pct / 100.0
    
    # Upper Bound
    fig.add_trace(go.Scatter(
        x=plot_range, 
        y=[x * (1 + error_factor) for x in plot_range], 
        mode='lines', 
        name=f'+{error_pct}% Error Bound', 
        line=dict(color='gray', width=1, dash='dot'),
        hoverinfo='skip'
    ))
    
    # Lower Bound
    fig.add_trace(go.Scatter(
        x=plot_range, 
        y=[x * (1 - error_factor) for x in plot_range], 
        mode='lines', 
        name=f'-{error_pct}% Error Bound', 
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False, # Only show one label for the error band
        hoverinfo='skip'
    ))

    # --- 3. Scatter Markers (Actual Data Points) ---
    fig.add_trace(go.Scatter(
        x=y_true, 
        y=y_pred, 
        mode='markers', 
        name='Predictions', 
        marker=dict(color='steelblue', size=6, opacity=0.7),
        # Add residual to hover text for quick inspection
        hovertemplate='<b>True:</b> %{x:.2f}<br><b>Predicted:</b> %{y:.2f}<br><b>Residual:</b> %{customdata:.2f}<extra></extra>',
        customdata=residuals
    ))
    
    # Update axes titles and layout
    fig.update_xaxes(
        title_text="True Values (Target)",
        range=plot_range,
        constrain='domain'
    )
    fig.update_yaxes(
        title_text="Predicted Values",
        scaleanchor="x", # Ensures Y-axis uses the same scale as X
        scaleratio=1,     # Ensures the ratio is 1:1 (square plot area)
        range=plot_range,
        constrain='domain'
    )

    fig.update_layout(
        height=600, 
        showlegend=True, 
        title_text=f"Parity Plot: Predicted vs. True Values (MRE: {mre:.2f}%)",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
                      
    return fig