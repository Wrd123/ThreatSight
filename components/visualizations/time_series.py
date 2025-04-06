# components/visualizations/time_series.py

import streamlit as st
import pandas as pd
import plotly.express as px

from .utils import apply_dark_theme, map_days_of_week, get_day_order

def render_time_based_visualizations(df_clean):

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
        )
        
        # Apply dark theme
        apply_dark_theme(fig_time)
        
        st.plotly_chart(fig_time)
    else:
        st.error("Could not create time-based visualization. Required data is not available.")

def render_day_of_week_analysis(df_clean):

    if "Day_of_Week" in df_clean.columns and "Above Threshold" in df_clean.columns:
        # Map numeric day of week to names
        day_names = map_days_of_week()
        df_clean["Day_Name"] = df_clean["Day_of_Week"].map(day_names)
        
        daily_data = df_clean.groupby("Day_Name").agg(
            total_events=pd.NamedAgg(column="Anomaly Scores", aggfunc="count"),
            high_anomalies=pd.NamedAgg(column="Above Threshold", aggfunc="sum")
        ).reset_index()
        
        # Ensure days are in correct order
        day_order = get_day_order()
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
        )
        
        # Apply dark theme
        apply_dark_theme(fig_dow)
        
        st.plotly_chart(fig_dow)
    else:
        st.info("Day of Week data not available. Cannot create analysis.")