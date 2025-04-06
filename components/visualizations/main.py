# components/visualizations/main.py

import streamlit as st
import pandas as pd

from .anomaly_heatmap import create_attack_type_heatmap
from .time_series import render_time_based_visualizations, render_day_of_week_analysis
from .distribution_plots import (
    create_scatter_plot, 
    create_density_heatmap, 
    create_violin_plot, 
    render_distribution_explanation
)
from .event_analysis import render_event_drill_down, show_high_anomaly_events

def render_anomaly_scores_dashboard(df_clean, threshold):
  
    # Create a flag column based on the threshold
    df_clean["Above Threshold"] = df_clean["Anomaly Scores"] > threshold
    
    st.subheader("Anomaly Scores Dashboard")
    
    # Top row - two main visualizations side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap visualization
        st.subheader("Anomaly Scores Distribution by Attack Type")
        create_attack_type_heatmap(df_clean, threshold)
    
    with col2:
        # Time-based Activity Visualization
        st.subheader("Time-based Anomaly Detection")
        render_time_based_visualizations(df_clean)
    
    # Second row - more visualizations
    st.subheader("Day of Week Analysis")
    if "Day_of_Week" in df_clean.columns:
        render_day_of_week_analysis(df_clean)
    else:
        st.info("Day of Week data not available. Cannot create analysis.")
    
    # Third row - Additional visualization options
    st.subheader("Packet Length vs. Anomaly Scores")
    
    # Render the enhanced scatter plot / visualization options
    if "Packet Length" in df_clean.columns:
        # Visualization type selector
        viz_col1, viz_col2 = st.columns([1, 3])
        
        with viz_col1:
            scatter_type = st.radio(
                "Select Visualization Type",
                ["Scatter Plot", "Density Heatmap", "Violin Plot"]
            )
            
            color_by = "Above Threshold"
            if "Attack Type" in df_clean.columns:
                color_options = ["Above Threshold", "Attack Type", "Protocol"]
                color_by = st.selectbox("Color points by:", color_options)
                
            if scatter_type == "Violin Plot" and "Attack Type" in df_clean.columns:
                group_by = st.selectbox(
                    "Group by:", 
                    ["Attack Type", "Protocol", "Action Taken"]
                )
        
        with viz_col2:
            if scatter_type == "Scatter Plot":
                scatter_fig = create_scatter_plot(df_clean, threshold, color_by)
                st.plotly_chart(scatter_fig)
                
            elif scatter_type == "Density Heatmap":
                heat_fig = create_density_heatmap(df_clean, threshold)
                st.plotly_chart(heat_fig)
                
            elif scatter_type == "Violin Plot":
                if "Attack Type" in df_clean.columns:
                    violin_fig = create_violin_plot(df_clean, threshold, group_by, color_by)
                    if violin_fig:
                        st.plotly_chart(violin_fig)
                    else:
                        st.warning(f"Column '{group_by}' not found for violin plot.")
                else:
                    st.warning("Categorical data like 'Attack Type' not available for violin plot")
            
            # Add explanation for the current visualization
            with st.expander("Understanding this visualization"):
                render_distribution_explanation(scatter_type)
    else:
        st.info("Packet Length column not found. Cannot create visualization.")
    
    # Fourth row - High anomaly events table
    st.subheader("High Anomaly Events (Above Threshold)")
    high_anomaly = show_high_anomaly_events(df_clean, threshold)
    
    # Fifth row - Event drill-down
    st.subheader("Event Drill-Down Analysis")
    render_event_drill_down(high_anomaly)

def render_visualization_section(df_clean):
  
    st.header("Interactive Visualizations")
    
    if "Anomaly Scores" in df_clean.columns:
        # Allow threshold adjustment
        min_score = float(df_clean["Anomaly Scores"].min())
        max_score = float(df_clean["Anomaly Scores"].max())
        default_threshold = float(df_clean["Anomaly Scores"].mean())
        threshold = st.slider(
            "Set Anomaly Score Threshold", 
            min_value=min_score, 
            max_value=max_score, 
            value=default_threshold
        )
        
        render_anomaly_scores_dashboard(df_clean, threshold)
    else:
        st.info("Column 'Anomaly Scores' not found in the dataset. Skipping visualization.")