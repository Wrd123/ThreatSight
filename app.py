# app.py

import streamlit as st
import pandas as pd

# Import component modules
from components.data_section import render_data_section
from components.visualizations import render_visualization_section
from components.modeling import render_modeling_section

# Set Streamlit page configuration
st.set_page_config(page_title="Cybersecurity Threat Detection Dashboard", layout="wide")

# Main application layout
def main():
    st.title("Cybersecurity Threat Detection Dashboard")
    
    # Create a session state to share data between components
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    
    # Render the data section and get processed dataframe
    df_clean = render_data_section()
    
    # Only render visualization and modeling sections if data is loaded
    if df_clean is not None:
        st.session_state.df_clean = df_clean
        
        # Render visualization section
        render_visualization_section(df_clean)
        
        # Render modeling section
        render_modeling_section(df_clean)

if __name__ == "__main__":
    main()