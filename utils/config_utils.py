# FILE: utils/config_utils.py

import yaml
import os

def load_config(path):
    """Loads the config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config_data, path):
    """Saves the config data to the file."""
    with open(path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)