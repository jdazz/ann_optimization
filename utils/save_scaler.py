import json
import numpy as np

def save_scaler_to_json(scaler, input_vars):
    """
    Extracts mean and scale from a fitted StandardScaler and formats it as JSON.
    """
    if scaler is None or not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        return None 

    scaler_data = {}
    
    if isinstance(scaler.mean_, np.ndarray):
        n_features = len(scaler.mean_)
    else:
        return None 

    loop_len = min(n_features, len(input_vars))

    if loop_len == 0:
        return json.dumps({}, indent=4)

    for i in range(loop_len):
        var_name = input_vars[i]
        scaler_data[var_name] = {
            "mean": scaler.mean_[i].item(),
            "std": scaler.scale_[i].item()
        }

    return json.dumps(scaler_data, indent=4)