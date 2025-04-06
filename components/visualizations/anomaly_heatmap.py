# components/visualizations/anomaly_heatmap.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

from .utils import apply_dark_theme

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
    )
    
    # Apply dark theme
    apply_dark_theme(fig)
    
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