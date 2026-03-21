import os
import pandas as pd
import yaml
import logging

def load_results_csv(output_dir: str):
    csv_path = os.path.join(output_dir, "reports", "all.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            logging.error(f"Failed to read {csv_path}: {e}")
    return None

def load_config_yaml(output_dir: str):
    config_path = os.path.join(output_dir, "krkn-ai.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to read {config_path}: {e}")
    return None
