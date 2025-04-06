# components/visualizations/distribution_plots.py
"""
Functions for creating various distribution plots (scatter, density, violin).
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from .utils import apply_dark_theme, add_threshold_line

def create_scatter_plot(df_clean, threshold, color_by="Above Threshold"):
    """
    Create a scatter plot of Packet Length vs Anomaly Scores.
    
    Args:
        df_clean (pd.DataFrame): The cleaned dataframe
        threshold (float): The anomaly score threshold
        color_by (str): Column to use for color coding points
        
    Returns:
        fig: The plotly figure object
    """
    # Add some jitter to y-values to show density better
    df_plot = df_clean.copy()
    
    # Make marker size vary by attack type to better visualize patterns
    if "Attack Type" in df_clean.columns:
        df_plot['marker_size'] = 6
        # Scale marker size based on anomaly score to highlight higher-risk points
        df_plot.loc[df_plot['Anomaly Scores'] > threshold, 'marker_size'] = 10
    
    hover_data = ["Protocol", "Attack Type"] if "Attack Type" in df_clean.columns else None
    
    scatter_fig = px.scatter(
        df_plot,
        x="Packet Length",
        y="Anomaly Scores",
        color=color_by,
        size='marker_size' if 'marker_size' in df_plot.columns else None,
        size_max=12,
        opacity=0.7,  # Add transparency to see overlapping points
        hover_data=hover_data
    )
    
    scatter_fig.update_layout(
        xaxis_title="Packet Length",
        yaxis_title="Anomaly Scores",
    )
    
    # Add threshold line
    add_threshold_line(scatter_fig, threshold)
    
    # Apply dark theme
    apply_dark_theme(scatter_fig)
    
    return scatter_fig

def create_density_heatmap(df_clean, threshold):
    """
    Create a density heatmap of Packet Length vs Anomaly Scores.
    
    Args:
        df_clean (pd.DataFrame): The cleaned dataframe
        threshold (float): The anomaly score threshold
        
    Returns:
        fig: The plotly figure object
    """
    # Create a 2D histogram/heatmap to show density
    heat_fig = px.density_heatmap(
        df_clean,
        x="Packet Length",
        y="Anomaly Scores",
        marginal_x="histogram",
    )
    
    # Add threshold line
    add_threshold_line(heat_fig, threshold)
    
    # Apply dark theme
    apply_dark_theme(heat_fig)
    
    return heat_fig

def create_violin_plot(df_clean, threshold, group_by, color_by="Above Threshold"):
    """
    Create a violin plot of Anomaly Scores grouped by a categorical variable.
    
    Args:
        df_clean (pd.DataFrame): The cleaned dataframe
        threshold (float): The anomaly score threshold
        group_by (str): Column to group by on x-axis
        color_by (str): Column to use for color coding
        
    Returns:
        fig: The plotly figure object or None if required data is not available
    """
    if group_by not in df_clean.columns:
        return None
        
    violin_fig = px.violin(
        df_clean,
        y="Anomaly Scores",
        x=group_by,
        color=color_by,
        box=True,  # include box plot inside the violin
        points="all",  # show all points
        title=f"Distribution of Anomaly Scores by {group_by}"
    )
    
    # Add threshold line
    add_threshold_line(violin_fig, threshold)
    
    # Apply dark theme
    apply_dark_theme(violin_fig)
    
    return violin_fig

def render_distribution_explanation(visualization_type):
    """
    Render an explanation for a specific type of distribution visualization.
    
    Args:
        visualization_type (str): Type of visualization ('scatter', 'density', or 'violin')
    """
    if visualization_type == "Scatter Plot":
        st.markdown("""
        **Understanding this scatter plot:**
        - Each point represents a security event
        - Points are colored by the selected dimension
        - Points above the red dashed line exceed the anomaly threshold
        - Larger points (if present) indicate higher severity or risk
        """)
    elif visualization_type == "Density Heatmap":
        st.markdown("""
        **Understanding this density heatmap:**
        - Darker areas indicate higher concentration of events
        - The histograms show the distribution of events across each axis
        - The red dashed line represents the current anomaly threshold
        - This visualization helps identify clusters and patterns in the data
        """)
    elif visualization_type == "Violin Plot":
        st.markdown("""
        **Understanding this violin plot:**
        - The width of each "violin" shows the density distribution
        - Wider sections represent more events at that anomaly score level
        - The box plots show quartiles and median values
        - Points represent individual events
        - The red dashed line represents the current anomaly threshold
        """)