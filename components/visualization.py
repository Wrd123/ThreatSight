# components/visualization.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime

def create_attack_type_heatmap(df_clean, threshold):
  
    # Check if Attack Type column exists
    if "Attack Type" not in df_clean.columns:
        st.warning("Attack Type column not found. Cannot create heatmap.")
        return
    
    # Create score bins for better visualization
    score_bins = [0, 20, 40, 60, 80, 100]
    bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    df_clean['Score Range'] = pd.cut(df_clean['Anomaly Scores'], bins=score_bins, labels=bin_labels)
    
    # Get the unique attack types
    attack_types = df_clean['Attack Type'].unique()
    
    # Create a cross-tabulation of attack types vs. score ranges
    heatmap_data = pd.crosstab(
        df_clean['Attack Type'], 
        df_clean['Score Range'],
        normalize='index'  # Normalize by row (attack type) to show percentage distribution
    ) * 100  # Convert to percentage
    
    # Create the heatmap visualization
    # Handle the annotation text carefully to avoid type errors
    annotations = []
    for i in range(heatmap_data.shape[0]):
        row = []
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data.values[i, j]
            row.append(f"{value:.1f}%")
        annotations.append(row)
    
    fig = ff.create_annotated_heatmap(
        z=heatmap_data.values,
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index),
        annotation_text=annotations,
        colorscale='Reds',
        showscale=True
    )
    
    # Update layout for better visualization
    fig.update_layout(
        title='',
        xaxis=dict(title='Anomaly Score Range'),
        yaxis=dict(title='Attack Type'),
        height=400 + (len(attack_types) * 25),  # Adjust height based on number of attack types
        margin=dict(l=150, r=50, b=50, t=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Add a vertical line to indicate the threshold (approximate position)
    threshold_bin = min(int(threshold // 20), 4)  # Map threshold to bin index (0-4)
    threshold_position = threshold_bin + 0.5  # Center of the bin
    
    fig.add_shape(
        type="line",
        x0=threshold_position,
        y0=-0.5,
        x1=threshold_position,
        y1=len(attack_types) - 0.5,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add an annotation for the threshold
    fig.add_annotation(
        x=threshold_position,
        y=-0.5,
        text=f"Threshold: {threshold:.1f}",
        showarrow=False,
        yshift=-30,
        font=dict(color="red"),
    )
    
    st.plotly_chart(fig)
    
    # Add explanation
    st.markdown("""
    **Understanding this heatmap:**
    - Each cell shows the percentage of events of a specific attack type that fall within an anomaly score range
    - Darker colors indicate higher concentrations of events
    - The red dashed line represents the current anomaly score threshold
    - This visualization helps identify which attack types tend to generate higher or lower anomaly scores
    """)

def render_anomaly_scores_dashboard(df_clean, threshold):
   
    # Create a flag column based on the threshold
    df_clean["Above Threshold"] = df_clean["Anomaly Scores"] > threshold
    
    st.subheader("Anomaly Scores Dashboard")
    
    # Top row - two main visualizations side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap visualization
        st.subheader("Anomaly Scores Distribution by Attack Type")
        
        # Check if Attack Type column exists
        if "Attack Type" not in df_clean.columns:
            st.warning("Attack Type column not found. Cannot create heatmap.")
        else:
            # Create score bins for better visualization
            score_bins = [0, 20, 40, 60, 80, 100]
            bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
            df_clean['Score Range'] = pd.cut(df_clean['Anomaly Scores'], bins=score_bins, labels=bin_labels)
            
            # Get the unique attack types
            attack_types = df_clean['Attack Type'].unique()
            
            # Create a cross-tabulation of attack types vs. score ranges
            heatmap_data = pd.crosstab(
                df_clean['Attack Type'], 
                df_clean['Score Range'],
                normalize='index'  # Normalize by row (attack type) to show percentage distribution
            ) * 100  # Convert to percentage
            
            # Create the heatmap visualization
            # Handle the annotation text carefully to avoid type errors
            annotations = []
            for i in range(heatmap_data.shape[0]):
                row = []
                for j in range(heatmap_data.shape[1]):
                    value = heatmap_data.values[i, j]
                    row.append(f"{value:.1f}%")
                annotations.append(row)
            
            fig = ff.create_annotated_heatmap(
                z=heatmap_data.values,
                x=list(heatmap_data.columns),
                y=list(heatmap_data.index),
                annotation_text=annotations,
                colorscale='Reds',
                showscale=True
            )
            
            # Update layout for better visualization
            fig.update_layout(
                title='',
                xaxis=dict(title='Anomaly Score Range'),
                yaxis=dict(title='Attack Type'),
                height=400 + (len(attack_types) * 25),  # Adjust height based on number of attack types
                margin=dict(l=150, r=50, b=50, t=50),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Add a vertical line to indicate the threshold (approximate position)
            threshold_bin = min(int(threshold // 20), 4)  # Map threshold to bin index (0-4)
            threshold_position = threshold_bin + 0.5  # Center of the bin
            
            fig.add_shape(
                type="line",
                x0=threshold_position,
                y0=-0.5,
                x1=threshold_position,
                y1=len(attack_types) - 0.5,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add an annotation for the threshold
            fig.add_annotation(
                x=threshold_position,
                y=-0.5,
                text=f"Threshold: {threshold:.1f}",
                showarrow=False,
                yshift=-30,
                font=dict(color="red"),
            )
            
            st.plotly_chart(fig)
    
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
                # Add some jitter to y-values to show density better
                df_plot = df_clean.copy()
                if "Attack Type" in df_clean.columns:
                    # Make marker size vary by attack type to better visualize patterns
                    df_plot['marker_size'] = 6
                    # Scale marker size based on anomaly score to highlight higher-risk points
                    df_plot.loc[df_plot['Anomaly Scores'] > threshold, 'marker_size'] = 10
                
                scatter_fig = px.scatter(
                    df_plot,
                    x="Packet Length",
                    y="Anomaly Scores",
                    color=color_by,
                    size='marker_size' if 'marker_size' in df_plot.columns else None,
                    size_max=12,
                    opacity=0.7,  # Add transparency to see overlapping points
                    hover_data=["Protocol", "Attack Type"] if "Attack Type" in df_clean.columns else None
                )
                
                # Add a horizontal line for the threshold
                scatter_fig.add_hline(
                    y=threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Threshold: {threshold:.1f}",
                    annotation_position="right"
                )
                
                scatter_fig.update_layout(
                    xaxis_title="Packet Length",
                    yaxis_title="Anomaly Scores",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(scatter_fig)
                
            elif scatter_type == "Density Heatmap":
                # Create a 2D histogram/heatmap to show density
                heat_fig = px.density_heatmap(
                    df_clean,
                    x="Packet Length",
                    y="Anomaly Scores",
                    marginal_x="histogram",
                )
                
                # Add a horizontal line for the threshold
                heat_fig.add_hline(
                    y=threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Threshold: {threshold:.1f}",
                    annotation_position="right"
                )
                
                heat_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(heat_fig)
                
            elif scatter_type == "Violin Plot":
                # If we have categorical data, create a violin plot
                if "Attack Type" in df_clean.columns:
                    violin_fig = px.violin(
                        df_clean,
                        y="Anomaly Scores",
                        x=group_by,
                        color=color_by,
                        box=True,  # include box plot inside the violin
                        points="all",  # show all points
                        title=f"Distribution of Anomaly Scores by {group_by}"
                    )
                    
                    # Add a horizontal line for the threshold
                    violin_fig.add_hline(
                        y=threshold, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Threshold: {threshold:.1f}",
                        annotation_position="right"
                    )
                    
                    violin_fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(violin_fig)
                else:
                    st.warning("Categorical data like 'Attack Type' not available for violin plot")
            
            # Add explanation for the current visualization
            with st.expander("Understanding this visualization"):
                if scatter_type == "Scatter Plot":
                    st.markdown("""
                    **Understanding this scatter plot:**
                    - Each point represents a security event
                    - Points are colored by the selected dimension
                    - Points above the red dashed line exceed the anomaly threshold
                    - Larger points (if present) indicate higher severity or risk
                    """)
                elif scatter_type == "Density Heatmap":
                    st.markdown("""
                    **Understanding this density heatmap:**
                    - Darker areas indicate higher concentration of events
                    - The histograms show the distribution of events across each axis
                    - The red dashed line represents the current anomaly threshold
                    - This visualization helps identify clusters and patterns in the data
                    """)
                elif scatter_type == "Violin Plot":
                    st.markdown("""
                    **Understanding this violin plot:**
                    - The width of each "violin" shows the density distribution
                    - Wider sections represent more events at that anomaly score level
                    - The box plots show quartiles and median values
                    - Points represent individual events
                    - The red dashed line represents the current anomaly threshold
                    """)
    else:
        st.info("Packet Length column not found. Cannot create visualization.")
    
    # Fourth row - High anomaly events table
    st.subheader("High Anomaly Events (Above Threshold)")
    high_anomaly = df_clean[df_clean["Anomaly Scores"] > threshold]
    st.write(f"Number of events above threshold: {len(high_anomaly)}")
    st.dataframe(high_anomaly)
    
    # Fifth row - Event drill-down
    st.subheader("Event Drill-Down Analysis")
    render_event_drill_down(high_anomaly)

def render_time_based_visualizations(df_clean):
    """
    Renders time-based visualizations if time data is available.
    
    Args:
        df_clean (pd.DataFrame): The cleaned dataframe
    """
    # Process the timestamp column for visualization (use Hour that was already created)
    if "Hour" in df_clean.columns and "Above Threshold" in df_clean.columns:
        # Group by hour and count anomalies above threshold
        hourly_data = df_clean.groupby("Hour").agg(
            total_events=pd.NamedAgg(column="Anomaly Scores", aggfunc="count"),
            high_anomalies=pd.NamedAgg(column="Above Threshold", aggfunc="sum")
        ).reset_index()
        
        # Calculate percentage of high anomalies
        hourly_data["anomaly_percentage"] = (hourly_data["high_anomalies"] / hourly_data["total_events"] * 100).round(1)
        
        # Create the time-based activity plot
        fig_time = px.bar(hourly_data, x="Hour", y=["total_events", "high_anomalies"],
                        barmode="overlay",
                        labels={"value": "Count", "variable": "Event Type"},
                        title="Hourly Distribution of Security Events")
        
        # Add a line for the anomaly percentage
        fig_time.add_scatter(x=hourly_data["Hour"], y=hourly_data["anomaly_percentage"],
                            mode="lines+markers", name="% High Anomalies",
                            yaxis="y2", line=dict(color="red", width=2))
        
        # Update layout for dual y-axis
        fig_time.update_layout(
            yaxis2=dict(
                title="Anomaly Percentage (%)",
                overlaying="y",
                side="right",
                range=[0, max(hourly_data["anomaly_percentage"]) * 1.2 if len(hourly_data) > 0 else 100]  # Dynamic range
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_time)
        
        # REMOVED: Removed the call to render_day_of_week_analysis from here to prevent duplication
    else:
        st.error("Could not create time-based visualization. Required data is not available.")

def render_day_of_week_analysis(df_clean):
 
    if "Day_of_Week" in df_clean.columns and "Above Threshold" in df_clean.columns:
        # Map numeric day of week to names
        day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 
                    4: "Friday", 5: "Saturday", 6: "Sunday"}
        df_clean["Day_Name"] = df_clean["Day_of_Week"].map(day_names)
        
        daily_data = df_clean.groupby("Day_Name").agg(
            total_events=pd.NamedAgg(column="Anomaly Scores", aggfunc="count"),
            high_anomalies=pd.NamedAgg(column="Above Threshold", aggfunc="sum")
        ).reset_index()
        
        # Ensure days are in correct order
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_data["Day_Name"] = pd.Categorical(daily_data["Day_Name"], categories=day_order, ordered=True)
        daily_data = daily_data.sort_values("Day_Name")
        
        # Calculate percentage
        daily_data["anomaly_percentage"] = (daily_data["high_anomalies"] / daily_data["total_events"] * 100).round(1)
        
        # Create day-of-week plot
        fig_dow = px.bar(daily_data, x="Day_Name", y=["total_events", "high_anomalies"],
                        barmode="overlay",
                        labels={"value": "Count", "variable": "Event Type", "Day_Name": "Day of Week"},
                        title="Distribution of Security Events by Day of Week")
        
        fig_dow.add_scatter(x=daily_data["Day_Name"], y=daily_data["anomaly_percentage"],
                        mode="lines+markers", name="% High Anomalies",
                        yaxis="y2", line=dict(color="red", width=2))
        
        fig_dow.update_layout(
            yaxis2=dict(
                title="Anomaly Percentage (%)",
                overlaying="y",
                side="right",
                range=[0, max(daily_data["anomaly_percentage"]) * 1.2 if len(daily_data) > 0 else 100]  # Dynamic range
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_dow)

def render_event_drill_down(high_anomaly):
  
    if not high_anomaly.empty:
        selected_index = st.selectbox("Select event index for drill-down analysis", high_anomaly.index)
        if selected_index is not None:
            event_details = high_anomaly.loc[selected_index]
            st.markdown("### Detailed Event Data")
            st.json(event_details.to_dict())

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