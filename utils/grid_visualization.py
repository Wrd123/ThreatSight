# threatsight/utils/grid_visualization.py

import base64
from io import BytesIO
import streamlit.components.v1 as components
import pandas as pd
import os

def create_grid_visualization(figures_data):
    """
    Create an interactive grid visualization with expandable components.
    
    Parameters:
    -----------
    figures_data : dict
        Dictionary containing the data for the grid visualization
        Keys: 'feature_importance', 'cv_results', 'confusion_matrix', 'classification_report'
        Values: dict with 'image' or 'html' and 'title'
    
    Returns:
    --------
    None
    """
    # Get the current directory to properly locate static files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # Load CSS from file
    css_path = os.path.join(project_root, 'static/css/grid_style.css')
    with open(css_path, 'r') as css_file:
        css_content = css_file.read()
    
    # Load JavaScript from file
    js_path = os.path.join(project_root, 'static/js/grid_interaction.js')
    with open(js_path, 'r') as js_file:
        js_content = js_file.read()
    
    # Create HTML structure
    html_content = f"""
    <style>
    {css_content}
    </style>
    
    <div class="overlay" id="overlay"></div>
    
    <div class="grid-container">
        <div class="grid-item" id="feature-importance">
            <h3>{figures_data['feature_importance']['title']}</h3>
            {f'<img src="data:image/png;base64,{figures_data["feature_importance"]["image"]}" alt="Feature Importance" />' 
              if 'image' in figures_data['feature_importance'] else 
              figures_data['feature_importance'].get('html', '<p>Data not available.</p>')}
        </div>
        
        <div class="grid-item" id="cv-results">
            <h3>{figures_data['cv_results']['title']}</h3>
            {figures_data['cv_results'].get('html', '<p>Data not available.</p>')}
        </div>
        
        <div class="grid-item" id="confusion-matrix">
            <h3>{figures_data['confusion_matrix']['title']}</h3>
            {f'<img src="data:image/png;base64,{figures_data["confusion_matrix"]["image"]}" alt="Confusion Matrix" />' 
              if 'image' in figures_data['confusion_matrix'] else 
              figures_data['confusion_matrix'].get('html', '<p>Data not available.</p>')}
        </div>
        
        <div class="grid-item" id="classification-report">
            <h3>{figures_data['classification_report']['title']}</h3>
            <div style="max-height: 220px; overflow: auto;">
                {figures_data['classification_report'].get('html', '<p>Data not available.</p>')}
            </div>
        </div>
    </div>
    
    <script>
    {js_content}
    </script>
    """
    
    # Display the custom HTML
    components.html(html_content, height=650)


def figure_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def dataframe_to_html(df, classes='data-table'):
    """Convert pandas DataFrame to HTML table with CSS classes"""
    return df.to_html(index=False, classes=classes)