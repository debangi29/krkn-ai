import os
import sys
import argparse
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from krkn_ai.dashboard.data_loader import load_results_csv, load_config_yaml, load_health_check_csv, load_detailed_scenarios_data, load_best_scenarios_yaml


def get_monitor_config():
    """Retrieve monitor config from command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./")
    try:
        args, _ = parser.parse_known_args()
        return {"output_dir": args.output_dir}
    except SystemExit:
        return {"output_dir": "./"}


def is_execution_running(output_dir: str) -> bool:
    """Detect if krkn-ai is currently running by checking lockfile."""
    return os.path.exists(os.path.join(output_dir, ".krkn-running"))


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
            hovermode="x unified",
            xaxis={"tickmode": 'linear', "tick0": 1, "dtick": 1}
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No fitness scores recorded yet.")

def render_scenario_distribution(df):
    st.header("Scenario Distribution")
    if df is None or df.empty or "scenario" not in df.columns:
        st.write("Not enough data to plot distribution.")
        return

    fig = px.histogram(df, x="scenario", title="Executed Scenarios Frequency", color="scenario")
    fig.update_layout(xaxis_title="Scenario Name", yaxis_title="Execution Count")
    st.plotly_chart(fig, width='stretch')

def render_scenario_fitness_variation(df):
    st.header("Scenario-wise Fitness Variation")
    if df is None or df.empty or "generation_id" not in df.columns or "scenario" not in df.columns:
        st.write("Not enough data to plot scenario fitness variation.")
        return

    # Group by scenario and generation
    grouped = df.groupby(["generation_id", "scenario"])["fitness_score"].max().reset_index()
    grouped["generation_id"] = grouped["generation_id"] + 1

    if not grouped.empty:
        fig = px.line(grouped, x="generation_id", y="fitness_score", color="scenario", markers=True, 
                      title="Best Fitness Variation by Scenario")
        fig.update_layout(
            xaxis_title="Generation",
            yaxis_title="Best Fitness Score",
            hovermode="x unified",
            xaxis={"tickmode": 'linear', "tick0": 1, "dtick": 1}
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Not enough data points yet.")

def render_generation_details(df):
    st.header("Generation & Scenario Details")
    if df is None or df.empty or "generation_id" not in df.columns:
        st.write("No scenario details available yet.")
        return

    # Extract all unique generation numbers for the dropdown
    gen_nums = sorted(df["generation_id"].unique().tolist())
    display_gens = ["All"] + [g + 1 for g in gen_nums]
    selected_gen_disp = st.selectbox("Select Generation to view executed scenarios:", options=display_gens)
    
    if selected_gen_disp == "All":
        st.subheader("Results for All Generations")
        gen_scenarios = df.copy()
    else:
        st.subheader(f"Results for Generation {selected_gen_disp}")
        selected_gen_raw = selected_gen_disp - 1
        gen_scenarios = df[df["generation_id"] == selected_gen_raw].copy()
    
    if not gen_scenarios.empty:
        sort_order = st.radio(
            "Sort Fitness Score by:", 
            ["Descending (Best First)", "Ascending (Worst First)"], 
            horizontal=True
        )
        is_asc = sort_order == "Ascending (Worst First)"
        
        
        gen_scenarios = gen_scenarios.sort_values(by="fitness_score", ascending=is_asc)
        
        display_cols = ['generation_id', 'scenario_id', 'scenario', 'fitness_score', 'parameters']
        available_cols = [c for c in display_cols if c in gen_scenarios.columns]
        
        st.table(gen_scenarios[available_cols])
    else:
        st.write("No testing details available for this specific generation.")


def render_health_checks(df):
    st.header("Service Health Checks")
    if df is None or df.empty:
        st.warning("Health check data not yet available.")
        return

    if "failure_rate" not in df.columns:
        df["failure_rate"] = df["failure_count"] / (df["success_count"] + df["failure_count"]).clip(lower=1)
    if "variance" not in df.columns:
        df["variance"] = (df["max_response_time"] - df["min_response_time"]) / df["average_response_time"].clip(lower=0.0001)

    scenarios = ["All"] + sorted(df["scenario_id"].unique().tolist())
    
    st.subheader("Interactive Heatmap")
    metric_col = st.selectbox("Select Metric:", ["average_response_time", "max_response_time", "min_response_time"])
    top_k_heat = st.number_input("Top K Slowest Services:", min_value=1, value=10, max_value=50)

    heat_df = df.groupby(["component_name", "scenario_id"])[metric_col].mean().reset_index()
    top_comps = df.groupby("component_name")[metric_col].mean().nlargest(top_k_heat).index
    heat_df = heat_df[heat_df["component_name"].isin(top_comps)]
    heat_df["scenario_id"] = heat_df["scenario_id"].astype(str)
    
    fig = px.density_heatmap(heat_df, x="component_name", y="scenario_id", z=metric_col,
                             histfunc="avg", title=f"{metric_col} Heatmap",
                             color_continuous_scale="RdYlGn_r")
    fig.update_layout(
        xaxis_title="Component", 
        yaxis_title="Scenario ID",
        yaxis={"type": "category"}
    )
    fig.update_traces(xgap=3, ygap=3)
    st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    #scenario trends line chart
    st.subheader("Scenario Trends")
    line_metric = st.selectbox("Trend Metric:", ["average_response_time", "max_response_time", "min_response_time"], key="line_metric")
    
    trend_comps = sorted(df["component_name"].unique().tolist())
    selected_trend_comps = st.multiselect("Filter Services (leave empty for all):", trend_comps, default=[], key="trend_comps")
    
    target_trend_comps = selected_trend_comps if selected_trend_comps else trend_comps
    line_df = df[df["component_name"].isin(target_trend_comps)].sort_values("scenario_id")
    line_df["scenario_id"] = line_df["scenario_id"].astype(str)
    fig2 = px.line(line_df, x="scenario_id", y=line_metric, color="component_name", markers=True,
                   title=f"{line_metric} Trends")
    fig2.update_layout(xaxis={"type": "category"})
    st.plotly_chart(fig2, width='stretch')

    st.divider()

    st.divider()

    st.subheader("Success vs Failure")
    bar_scene = st.selectbox("Select Scenario for Stacked Bar:", scenarios, key="bar_scene")
    
    bar_comps = sorted(df["component_name"].unique().tolist())
    selected_bar_comps = st.multiselect("Filter Services (leave empty for all):", bar_comps, default=[], key="bar_comps")
    
    target_bar_comps = selected_bar_comps if selected_bar_comps else bar_comps
    bar_base_df = df[df["component_name"].isin(target_bar_comps)].copy()
    if bar_scene == "All":
        bar_df = bar_base_df.groupby("component_name")[["success_count", "failure_count"]].sum().reset_index()
    else:
        bar_df = bar_base_df[bar_base_df["scenario_id"] == bar_scene].copy()
        
    melt_bar = bar_df.melt(id_vars=["component_name"], value_vars=["success_count", "failure_count"],
                           var_name="Status", value_name="Count")
    fig3 = px.bar(melt_bar, x="component_name", y="Count", color="Status", title="Success vs Failure Counts",
                  barmode="stack", color_discrete_map={"success_count": "#28a745", "failure_count": "#dc3545"})
    st.plotly_chart(fig3, width='stretch')

    st.divider()

    st.subheader("Resilience Radar Chart")
    radar_scene = st.selectbox("Select Scenario for Radar:", scenarios, key="radar_scene")
    
    radar_df = df.copy()
    if radar_scene != "All":
        radar_df = radar_df[radar_df["scenario_id"] == radar_scene]
        title_text = f"Resilience Profile (Scenario {radar_scene})"
    else:
        title_text = "Resilience Profile (All Scenarios)"
        radar_df["scenario_id"] = radar_df["scenario_id"].astype(str)
        
    if not radar_df.empty:
        radar_df["score"] = 1 / radar_df["average_response_time"].clip(lower=0.0001)
        
        color_arg = "scenario_id" if radar_scene == "All" else None
        
        fig4 = px.line_polar(radar_df, r='score', theta='component_name', line_close=True,
                             color=color_arg, title=title_text)
        fig4.update_traces(fill='toself', opacity=0.5 if radar_scene == "All" else 0.8)
        st.plotly_chart(fig4, width='stretch')
    else:
        st.info("No data for radar chart.")

    st.divider()

    st.subheader("Response Range Plot (Min-Max)")
    range_scene = st.selectbox("Select Scenario for Range plot:", scenarios, key="range_scene")
    if range_scene == "All":
        range_df = df.groupby("component_name").agg({"min_response_time": "min", "max_response_time": "max"}).reset_index()
    else:
        range_df = df[df["scenario_id"] == range_scene].copy()
        
    fig5 = go.Figure()
    for _, row in range_df.iterrows():
        fig5.add_trace(go.Scatter(
            x=[row["component_name"], row["component_name"]],
            y=[row["min_response_time"], row["max_response_time"]],
            mode='lines+markers',
            name=row["component_name"],
            showlegend=False,
            marker={"symbol": "line-ew", "size": 15}
        ))
    fig5.update_layout(title="Min/Max Range per Component", xaxis_title="Component", yaxis_title="Response Time Range")
    st.plotly_chart(fig5, width='stretch')

    st.divider()

    st.subheader("Top-K Worst Components Table")
    sort_by = st.selectbox("Sort Table By (Descending):", ["average_response_time", "failure_count", "failure_rate", "variance"])
    worst_k = st.number_input("Top K Worst Components:", min_value=1, value=10, max_value=50, key="worst_k")
    worst_table = df.sort_values(by=sort_by, ascending=False).head(worst_k)
    st.dataframe(worst_table)


def render_best_scenarios_summary(df_best):
    if not df_best:
        return
        
    st.subheader("Best Scenarios Overview (Per Generation)")
    
    best_rows = []
    for item in df_best:
        best_rows.append({
            "Generation": item.get("generation_id", "N/A"),
            "Scenario ID": item.get("scenario_id", "N/A"),
            "Scenario Name": item.get("scenario", {}).get("name", "N/A"),
            "Fitness Score": item.get("fitness_result", {}).get("fitness_score", 0.0),
            "Duration (s)": round(item.get("duration_seconds", 0.0), 2)
        })
        
    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df = best_df.sort_values(by="Generation")
        st.dataframe(best_df, width='stretch')
        
    st.divider()

def render_detailed_scenarios(df_details):
    st.header("Detailed Scenarios Runtime Tracking")
    if df_details is None or df_details.empty:
        st.warning("No detailed scenario YAML telemetry available.")
        return
        
    scenarios = ["All"] + sorted(df_details["scenario_id"].unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
    services = sorted(df_details["service"].unique().tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_scen = st.selectbox("Select Scenario:", scenarios, key="det_scen")
    with col2:
        selected_serv = st.multiselect("Select Target Services (leave empty for all):", services, default=[], key="det_serv")
        
    target_df = df_details.copy()
    if selected_scen != "All":
        target_df = target_df[target_df["scenario_id"] == selected_scen]
    
    if selected_serv:
        target_df = target_df[target_df["service"].isin(selected_serv)]
        
    if target_df.empty:
        st.info("No data available for the selected filters.")
        return
        
    fig = go.Figure()
    
    for scen in target_df["scenario_id"].unique():
        for srv in target_df[target_df["scenario_id"] == scen]["service"].unique():
            srv_df = target_df[(target_df["scenario_id"] == scen) & (target_df["service"] == srv)]
            
            fig.add_trace(go.Scatter(
                x=srv_df["seconds_into_scenario"],
                y=srv_df["response_time"],
                mode="lines+markers",
                name=f"{srv} (Scen {scen})",
                customdata=srv_df[["timestamp", "status_code", "error"]],
                hovertemplate="Service: " + srv + "<br>Scenario: " + str(scen) + "<br>Time: %{customdata[0]}<br>Seconds: %{x:.2f}s<br>Response Time: %{y:.4f}s<br>Status: %{customdata[1]}<br>Error: %{customdata[2]}<extra></extra>",
                marker=dict(size=6)
            ))
                
    fig.update_layout(
        title="Runtime Telemetry: Response Time vs Scenario Execution Time",
        xaxis_title="Seconds into Scenario (s)",
        yaxis_title="Response Time (s)",
        hovermode="closest"
    )
    
    st.plotly_chart(fig, width='stretch')

    st.divider()

    st.subheader("Success per Service Over Time")
    success_scenarios = [s for s in scenarios if s != "All"]
    success_scen = st.selectbox("Select Scenario for Success Plot:", success_scenarios, key="succ_scen")
    
    succ_df = df_details.copy()
    if success_scen != "All":
        succ_df = succ_df[succ_df["scenario_id"] == success_scen]
    if selected_serv:
        succ_df = succ_df[succ_df["service"].isin(selected_serv)]
        
    if not succ_df.empty:
        succ_df["time_sec"] = succ_df["seconds_into_scenario"].astype(int)
        agg_df = succ_df.groupby(["service", "time_sec"])["success"].min().reset_index()
        agg_df["success_int"] = agg_df["success"].astype(int)
        
        pivot_df = agg_df.pivot(index="service", columns="time_sec", values="success_int")
        
        fig_succ = px.imshow(
            pivot_df,
            color_continuous_scale=[[0.0, "red"], [1.0, "green"]],
            zmin=0, zmax=1,
            labels=dict(x="Seconds into Scenario (s)", y="Application", color="Status"),
            aspect="auto",
            title=f"Success Timeline (Scenario: {success_scen})"
        )
        fig_succ.update_layout(coloraxis_showscale=False)
        fig_succ.update_traces(xgap=1, ygap=1, hovertemplate="Application: %{y}<br>Seconds: %{x}s<br>Status (1=Success, 0=Fail): %{z}<extra></extra>")
        st.plotly_chart(fig_succ, width='stretch')
    else:
        st.info("No data available for Success Plot.")

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
    st.title("Krkn-AI Execution Monitor")

    monitor_config = get_monitor_config()
    output_dir = monitor_config.get("output_dir", "./")
    
    # Detect state purely from lockfile (reliable across st.rerun() cycles)
    running = is_execution_running(output_dir)

    st.sidebar.header("Controls")
    if running:
        st.sidebar.info("⏳ Execution in progress...")
        auto_refresh = True
    else:
        st.sidebar.success("Execution completed!")
        auto_refresh = False

    # Load data
    df_results = load_results_csv(output_dir)
    config_data = load_config_yaml(output_dir)
    df_health = load_health_check_csv(output_dir)
    df_details = load_detailed_scenarios_data(output_dir)
    df_best = load_best_scenarios_yaml(output_dir)

    filter_type = "All"
    k_value, p_value = None, None
    
    if df_results is not None and not df_results.empty:
        st.sidebar.header("Filters")
        
        all_scenarios = sorted(df_results["scenario"].unique().tolist())
        selected_scenarios = st.sidebar.multiselect(
            "Filter by Scenario:",
            options=all_scenarios,
            default=all_scenarios
        )
        # Apply scenario filter
        if selected_scenarios:
            df_results = df_results[df_results["scenario"].isin(selected_scenarios)]
            
        st.sidebar.subheader("Best Iterations Scope")
        filter_type = st.sidebar.radio("Filter Generator Rows:", ["All", "Top K", "Top P (%)"])
        
        if filter_type == "Top K":
            k_value = st.sidebar.number_input("Top K count:", min_value=1, value=3, step=1)
            df_results = df_results.sort_values(by="fitness_score", ascending=False).head(k_value)
        elif filter_type == "Top P (%)":
            p_value = st.sidebar.slider("Top Percentage (%):", min_value=1, max_value=100, value=25)
            cutoff = max(1, int(len(df_results) * (p_value / 100.0)))
            df_results = df_results.sort_values(by="fitness_score", ascending=False).head(cutoff)
            
        # Apply identical scenario filters over to the Health Checks dataframe
        if df_health is not None and not df_health.empty:
            valid_scenarios = df_results["scenario_id"].unique().tolist()
            df_health = df_health[df_health["scenario_id"].isin(valid_scenarios)]
            
        # Apply identical scenario filters over to the Detailed Scenarios dataframe
        if df_details is not None and not df_details.empty:
            valid_str_scenarios = [str(x) for x in df_results["scenario_id"].unique().tolist()]
            df_details = df_details[df_details["scenario_id"].isin(valid_str_scenarios)]


    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Health Checks", "Detailed Scenarios", "Configuration", "Graphs"])

    with tab1:
        if df_results is None or df_results.empty:
            st.warning(f"Waiting for scenario reports in `{output_dir}/reports/all.csv`...")
        else:
            render_summary(df_results)
            st.divider()
            
            # Additional visual metric charts
            colA, colB = st.columns(2)
            with colA:
                render_scenario_distribution(df_results)
            with colB:
                render_scenario_fitness_variation(df_results)
            
            st.divider()
            render_fitness_evolution(df_results)
            st.divider()
            render_generation_details(df_results)

    with tab2:
        render_health_checks(df_health)

    with tab3:
        render_best_scenarios_summary(df_best)
        render_detailed_scenarios(df_details)

    with tab4:
        render_config(config_data)

    with tab5:
        render_graphs(output_dir)

    # Refresh mechanism (as of now)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
