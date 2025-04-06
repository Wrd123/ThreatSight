# components/data_section.py

import streamlit as st
import pandas as pd
from datetime import datetime

# Importing our custom modules
from ingestion_processing import DataIngestion, DataCleaning, DataProcessing
from utils.helpers import get_categorical_features, get_numerical_features

def render_data_section():
   
    st.header("Data Ingestion and Processing")
    
    uploaded_file = st.file_uploader("Upload CSV file with cybersecurity attack data", type=["csv"])
    
    if uploaded_file is None:
        return None
    
    # Read CSV data from the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None
    
    # Define columns to drop (as per our modular code)
    drop_columns = [
        "Payload Data", "Source Port", "Destination Port",
        "IDS/IPS Alerts", "Source IP Address", "Destination IP Address",
        "User Information", "Device Information", "Geo-location Data",
        "Firewall Logs", "Proxy Information", "Log Source"
    ]
    
    # Data Cleaning
    cleaning = DataCleaning(drop_columns)
    df_clean = cleaning.clean_data(df)
    
    # Process timestamp column and extract useful features
    timestamp_features = []
    if "Timestamp" in df_clean.columns:
        try:
            # Convert Timestamp to datetime
            df_clean["datetime"] = pd.to_datetime(df_clean["Timestamp"])
            
            # Extract time-based features
            df_clean["Hour"] = df_clean["datetime"].dt.hour
            df_clean["Day_of_Week"] = df_clean["datetime"].dt.dayofweek
            
            # Add these as new features
            timestamp_features = ["Hour", "Day_of_Week"]
            
            # Keep the Timestamp column for visualization but NOT for modeling
        except Exception as e:
            st.warning(f"Could not process Timestamp column: {e}")
    
    # Data Preview with filtering options
    st.subheader("Data Preview")
    search_query = st.text_input("Search records by keyword", value="")
    if search_query:
        # Filter rows that contain the search query in any column
        df_filtered = df_clean[df_clean.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
        st.dataframe(df_filtered.head(100))
    else:
        st.dataframe(df_clean.head(100))
    
    # Build the processing pipeline
    categorical_features = get_categorical_features()
    numerical_features = get_numerical_features()
    
    # Add timestamp-derived features to appropriate lists
    if "Hour" in df_clean.columns:
        numerical_features.extend(["Hour", "Day_of_Week"])
    
    processing = DataProcessing(categorical_features, numerical_features)
    preprocessor = processing.build_preprocessor()
    
    return df_clean