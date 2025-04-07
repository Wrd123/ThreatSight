# utils/grid_visualization.py
import streamlit as st
import base64
from io import BytesIO
import streamlit.components.v1 as components
import pandas as pd
import os

def create_grid_visualization(figures_data):

    # Get the current directory to properly locate static files
    script_dir = os.path.dirname(os.path.abspath(__file__))  # utils folder
    project_root = os.path.dirname(script_dir)  # ThreatSight folder
    
    # Load CSS from file with explicit UTF-8 encoding
    css_path = os.path.join(project_root, 'static', 'css', 'grid_style.css')
    
    # Load JavaScript from file with explicit UTF-8 encoding
    js_path = os.path.join(project_root, 'static', 'js', 'grid_interaction.js')
    
    # Check if files exist and load them with UTF-8 encoding
    try:
        with open(css_path, 'r', encoding='utf-8') as css_file:
            css_content = css_file.read()
        with open(js_path, 'r', encoding='utf-8') as js_file:
            js_content = js_file.read()
    except FileNotFoundError as e:
        st.error(f"Error loading static files: {e}")
        st.info(f"Expected path for CSS: {css_path}")
        st.info(f"Expected path for JS: {js_path}")
        st.info("Please create the necessary directories and files, or update the paths in grid_visualization.py")
        return
    except UnicodeDecodeError as e:
        st.error(f"Error decoding file: {e}")
        st.info("Please save your CSS and JS files with UTF-8 encoding")
        return
    
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