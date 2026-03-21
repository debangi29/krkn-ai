import os
import sys
import argparse
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from krkn_ai.dashboard.data_loader import load_results_csv, load_config_yaml


def get_output_dir():
    """Retrieve output directory from the config file written by cmd.py."""
    config_path = os.path.join(os.path.dirname(__file__), ".krkn-monitor")
    try:
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                data = json.load(f)
                return data.get("output_dir", "./")
    except Exception as e:
        print(f"Error reading config: {e}")
    return "./"


def render_summary(df):
    st.header("Experiment Summary")
    if df is None or df.empty:
        st.warning("Results data not yet available. Waiting for Krkn-AI engine...")
        return

    # stats directly from CSV data
    generations_completed = int(df["generation_id"].max() + 1) if "generation_id" in df.columns else 0
    scenarios_executed = len(df)
    best_fitness = df["fitness_score"].max() if "fitness_score" in df.columns else 0.0
    avg_fitness = df["fitness_score"].mean() if "fitness_score" in df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations Completed", generations_completed)
    col2.metric("Scenarios Executed", scenarios_executed)
    col3.metric("Best Fitness Score", f"{best_fitness:.4f}")
    col4.metric("Avg Fitness Score", f"{avg_fitness:.4f}")


def render_fitness_evolution(df):
    st.header("Fitness Score Evolution")
    if df is None or df.empty or "generation_id" not in df.columns:
        st.write("Not enough data to plot fitness evolution.")
        return

    # Grouping CSV by generation to plot Best vs Average
    grouped = df.groupby("generation_id")["fitness_score"].agg(['mean', 'max']).reset_index()
    grouped.rename(columns={"mean": "Average Fitness", "max": "Best Fitness"}, inplace=True)
    grouped["generation_id"] = grouped["generation_id"] + 1  
    
    if not grouped.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grouped["generation_id"], y=grouped["Average Fitness"], mode='lines+markers', name='Average Fitness'))
        fig.add_trace(go.Scatter(x=grouped["generation_id"], y=grouped["Best Fitness"], mode='lines+markers', name='Best Fitness'))
        
        fig.update_layout(
            title="Fitness Performance Over Generations",
            xaxis_title="Generation",
            yaxis_title="Fitness Score",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fitness scores recorded yet.")

def render_generation_details(df):
    st.header("Generation & Scenario Details")
    if df is None or df.empty or "generation_id" not in df.columns:
        st.write("No scenario details available yet.")
        return

    # Extract all unique generation numbers for the dropdown
    gen_nums = sorted(df["generation_id"].unique().tolist())
    display_gens = [g + 1 for g in gen_nums]
    selected_gen_disp = st.selectbox("Select Generation to view executed scenarios:", options=display_gens)
    selected_gen_raw = selected_gen_disp - 1

    st.subheader(f"Results for Generation {selected_gen_disp}")
    
    gen_scenarios = df[df["generation_id"] == selected_gen_raw].copy()
    
    if not gen_scenarios.empty:
        # Re-arrange and rename columns for display
        display_cols = ['scenario_id', 'scenario', 'fitness_score', 'parameters']
        available_cols = [c for c in display_cols if c in gen_scenarios.columns]
        
        # We can expand parameters into individual columns if needed, or just display raw
        st.dataframe(gen_scenarios[available_cols], use_container_width=True)
    else:
        st.write("No testing details available for this specific generation.")


def render_config(config_data):
    st.header("Krkn-AI Configuration")
    if config_data:
        st.json(config_data)
    else:
        st.write("Configuration file not found.")

def render_graphs(output_dir):
    st.header("Generated Graphs")
    graphs_dir = os.path.join(output_dir, "reports", "graphs")
    
    if not os.path.exists(graphs_dir):
        st.write("No graphs directory found yet.")
        return
        
    import glob
    image_files = glob.glob(os.path.join(graphs_dir, "*.png"))
    
    if not image_files:
        st.write("No graphs generated yet...")
        return
        
    # Sort files by modification time (newest first)
    image_files.sort(key=os.path.getmtime, reverse=True)
    
    for img_path in image_files:
        st.subheader(os.path.basename(img_path))
        try:
            st.image(img_path)
        except OSError as e:
            if "truncated" in str(e).lower():
                st.warning(f"Image is currently being generated. Retrying in the next refresh... ({os.path.basename(img_path)})")
            else:
                st.error(f"Error loading image: {e}")
        except Exception as e:
            st.error(f"Error loading image: {e}")

def main():
    st.set_page_config(page_title="Krkn-AI Monitor", layout="wide")
    st.title("🦑 Krkn-AI Execution Monitor")

    output_dir = get_output_dir()

    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    
    if auto_refresh:
        st.sidebar.text("Polling active...")

    # Load data
    df_results = load_results_csv(output_dir)
    config_data = load_config_yaml(output_dir)

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Configuration", "Graphs"])

    with tab1:
        if df_results is None or df_results.empty:
            st.warning(f"Waiting for scenario reports in `{output_dir}/reports/all.csv`...")
        else:
            render_summary(df_results)
            st.divider()
            render_fitness_evolution(df_results)
            st.divider()
            render_generation_details(df_results)

    with tab2:
        render_config(config_data)

    with tab3:
        render_graphs(output_dir)

    # Refresh mechanism (as of now)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
