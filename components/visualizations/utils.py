# components/visualizations/utils.py

import pandas as pd
import plotly.graph_objects as go

def apply_dark_theme(fig):
  
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def add_threshold_line(fig, threshold, is_vertical=False, annotation_position="right"):
  
    if is_vertical:
        fig.add_shape(
            type="line",
            x0=threshold,
            y0=0,
            x1=threshold,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=threshold,
            y=0,
            text=f"Threshold: {threshold:.1f}",
            showarrow=False,
            yshift=-30,
            font=dict(color="red"),
        )
    else:
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold: {threshold:.1f}",
            annotation_position=annotation_position
        )
    
    return fig

def map_days_of_week():

    return {
        0: "Monday", 
        1: "Tuesday", 
        2: "Wednesday", 
        3: "Thursday", 
        4: "Friday", 
        5: "Saturday", 
        6: "Sunday"
    }

def get_day_order():
 
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]