import os
import pandas as pd
import yaml
import logging
import glob
import streamlit as st

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

@st.cache_data(ttl=5)
def load_detailed_scenarios_data(output_dir: str):
    yaml_pattern = os.path.join(output_dir, "yaml", "generation_*", "scenario_*.yaml")
    yaml_files = glob.glob(yaml_pattern)
    
    rows = []
    for filepath in yaml_files:
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
                
            scen_id = data.get("scenario_id")
            start_time_str = data.get("start_time")
            if not start_time_str or scen_id is None:
                continue
                
            start_dt = pd.to_datetime(start_time_str)
            hc_results = data.get("health_check_results", {})
            
            for url, req_list in hc_results.items():
                if not isinstance(req_list, list):
                    continue
                for req in req_list:
                    req_dt = pd.to_datetime(req.get("timestamp"))
                    seconds_into = (req_dt - start_dt).total_seconds()
                    
                    rows.append({
                        "scenario_id": str(scen_id),
                        "service": req.get("name", "unknown"),
                        "timestamp": req.get("timestamp"),
                        "seconds_into_scenario": seconds_into,
                        "response_time": req.get("response_time"),
                        "status_code": req.get("status_code"),
                        "success": req.get("success"),
                        "error": str(req.get("error")) if req.get("error") is not None else "None"
                    })
        except Exception as e:
            logging.error(f"Failed to parse {filepath}: {e}")
            
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(by="seconds_into_scenario")
        return df
    return pd.DataFrame()

def load_health_check_csv(output_dir: str):
    csv_path = os.path.join(output_dir, "reports", "health_check_report.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def load_best_scenarios_yaml(output_dir: str):
    yaml_path = os.path.join(output_dir, "reports", "best_scenarios.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load best_scenarios.yaml: {e}")
    return None
