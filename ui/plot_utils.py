# FILE: ui/plot_utils.py

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna.visualization

def plot_optimization_history(study):
    """Generates the Optuna optimization history plot."""
    try:
        # Requires 'plotly' backend
        return optuna.visualization.plot_optimization_history(study)
    except Exception as e:
        print(f"Error generating optimization history plot: {e}")
        return None

def plot_true_vs_predicted(y_true, y_pred, nmae):
    """
    Generates a scatter plot showing true vs. predicted values.
    """
    if y_true is None or y_pred is None:
        return None
        
    fig = go.Figure()
    
    # 1. Scatter Plot of Predictions
    fig.add_trace(go.Scatter(
        x=y_true, 
        y=y_pred,
        mode='markers',
        name='Predicted Data Points',
        marker=dict(size=5, opacity=0.6, color='blue')
    ))
    
    # 2. Ideal Line (True = Predicted)
    y_max = max(y_true.max(), y_pred.max())
    y_min = min(y_true.min(), y_pred.min())
    
    fig.add_trace(go.Scatter(
        x=[y_min, y_max], 
        y=[y_min, y_max],
        mode='lines',
        name='Ideal Prediction (y=x)',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=f'True vs. Predicted Values (Test Set) - NMAE: {nmae:.4f}',
        xaxis_title="True Value ($y_{true}$)",
        yaxis_title="Predicted Value ($\hat{y}$)",
        hovermode="closest",
        showlegend=True
    )
    return fig

# You may add other plotting functions here (e.g., plot_feature_importance)