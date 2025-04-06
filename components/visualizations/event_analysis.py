# components/visualizations/event_analysis.py
"""
Functions for analyzing and displaying detailed event information.
"""

import streamlit as st
import pandas as pd

def render_event_drill_down(high_anomaly):
    """
    Renders event drill-down analysis for selected high anomaly events.
    
    Args:
        high_anomaly (pd.DataFrame): DataFrame containing high anomaly events
    """
    if not high_anomaly.empty:
        selected_index = st.selectbox("Select event index for drill-down analysis", high_anomaly.index)
        if selected_index is not None:
            event_details = high_anomaly.loc[selected_index]
            st.markdown("### Detailed Event Data")
            st.json(event_details.to_dict())
    else:
        st.info("No events above threshold to display.")

def show_high_anomaly_events(df_clean, threshold):
    """
    Displays a table of events above the anomaly threshold.
    
    Args:
        df_clean (pd.DataFrame): The cleaned dataframe
        threshold (float): The anomaly score threshold
    """
    high_anomaly = df_clean[df_clean["Anomaly Scores"] > threshold]
    st.write(f"Number of events above threshold: {len(high_anomaly)}")
    st.dataframe(high_anomaly)
    return high_anomaly