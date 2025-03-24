# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Importing modules from our modular files
from ingestion_processing import DataIngestion, DataCleaning, DataProcessing
from model_visualization import ModelTraining, Visualization

# Set Streamlit page configuration
st.set_page_config(page_title="Cybersecurity Threat Detection Dashboard", layout="wide")

# ========================================
# Section 1: Data Ingestion and Processing
# ========================================
st.title("Cybersecurity Threat Detection Dashboard")
st.header("Data Ingestion and Processing")

uploaded_file = st.file_uploader("Upload CSV file with cybersecurity attack data", type=["csv"])

if uploaded_file is not None:
    # Read CSV data from the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
    
    # Define columns to drop (as per our modular code)
    drop_columns = [
        "Timestamp", "Payload Data", "Source Port", "Destination Port",
        "IDS/IPS Alerts", "Source IP Address", "Destination IP Address",
        "User Information", "Device Information", "Geo-location Data",
        "Firewall Logs", "Proxy Information", "Log Source"
    ]
    
    # Data Cleaning
    cleaning = DataCleaning(drop_columns)
    df_clean = cleaning.clean_data(df)
    
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
    # Define feature lists (ensure these columns exist in your CSV)
    categorical_features = [
        "Protocol", "Packet Type", "Traffic Type", "Malware Indicators",
        "Attack Type", "Attack Signature", "Action Taken", "Network Segment",
        "Alerts/Warnings", "Severity Level"
    ]
    numerical_features = ["Packet Length"]
    
    processing = DataProcessing(categorical_features, numerical_features)
    preprocessor = processing.build_preprocessor()
    
    # ========================================
    # Section 2: Interactive Visualizations
    # ========================================
    st.header("Interactive Visualizations")
    if "Anomaly Scores" in df_clean.columns:
        # Allow threshold adjustment
        min_score = float(df_clean["Anomaly Scores"].min())
        max_score = float(df_clean["Anomaly Scores"].max())
        default_threshold = float(df_clean["Anomaly Scores"].mean())
        threshold = st.slider("Set Anomaly Score Threshold", min_value=min_score, max_value=max_score, value=default_threshold)
        
        # Create a flag column based on the threshold
        df_clean["Above Threshold"] = df_clean["Anomaly Scores"] > threshold
        
        # Box Plot of Anomaly Scores
        st.subheader("Box Plot of Anomaly Scores")
        fig_box, ax_box = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df_clean["Anomaly Scores"], ax=ax_box)
        ax_box.axvline(threshold, color="red", linestyle="--", label="Threshold")
        ax_box.legend()
        st.pyplot(fig_box)
        
        # Histogram of Anomaly Scores
        st.subheader("Histogram of Anomaly Scores")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        sns.histplot(df_clean["Anomaly Scores"], bins=20, kde=True, ax=ax_hist)
        ax_hist.axvline(threshold, color="red", linestyle="--", label="Threshold")
        ax_hist.legend()
        st.pyplot(fig_hist)
        
        # List high anomaly events
        st.subheader("High Anomaly Events (Above Threshold)")
        high_anomaly = df_clean[df_clean["Anomaly Scores"] > threshold]
        st.write(f"Number of events above threshold: {len(high_anomaly)}")
        st.dataframe(high_anomaly)
        
        # Drill-Down Analysis: Allow selection of an event to view details
        if not high_anomaly.empty:
            selected_index = st.selectbox("Select event index for drill-down analysis", high_anomaly.index)
            if selected_index is not None:
                event_details = high_anomaly.loc[selected_index]
                st.markdown("### Detailed Event Data")
                st.json(event_details.to_dict())
    else:
        st.info("Column 'Anomaly Scores' not found in the dataset. Skipping visualization.")

    # ========================================
    # Section 3: Model Interaction and Feedback
    # ========================================
    st.header("Model Interaction and Feedback")
    if "Severity Level" in df_clean.columns:
        # Allow hyperparameter tuning
        n_estimators = st.slider("Number of trees in RandomForest", min_value=50, max_value=200, value=100, step=10)
        if st.button("Retrain Model"):
            # Define features and target for model training
            # Using 'Severity Level' as target; features include the defined categorical and numerical features.
            features_for_model = categorical_features + (["Packet Length", "Anomaly Scores"] if "Anomaly Scores" in df_clean.columns else ["Packet Length"])
            X = df_clean[features_for_model]
            y = df_clean["Severity Level"]
            
            # Build processing pipeline (again) for model training
            processing_model = DataProcessing(categorical_features, (["Packet Length", "Anomaly Scores"] if "Anomaly Scores" in df_clean.columns else ["Packet Length"]))
            preprocessor_model = processing_model.build_preprocessor()
            
            # Initialize the ModelTraining module with 'classification'
            model_training = ModelTraining(model_type="classification")
            # Build pipeline; adjust the model hyperparameter for n_estimators
            pipeline_model = model_training.build_pipeline(preprocessor_model)
            pipeline_model["model"].n_estimators = n_estimators  # update hyperparameter
            
            # Split the data for training and testing
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            trained_pipeline = model_training.train(X_train, y_train)
            
            # Evaluate the model
            y_pred = trained_pipeline.predict(X_test)
            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            
            st.success("Model retrained successfully!")
            st.write(f"**Model Accuracy:** {accuracy:.2f}")
            st.write(f"**Model F1 Score:** {f1:.2f}")
            
            # Display feature importance if available
            if hasattr(trained_pipeline["model"], "feature_importances_"):
                importances = trained_pipeline["model"].feature_importances_
                # Retrieve feature names using the preprocessor (for simplicity, create placeholder names)
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                st.subheader("Feature Importance")
                Visualization.plot_feature_importance(feature_names, importances)
    else:
        st.info("Column 'Severity Level' not found in the dataset. Model training and evaluation skipped.")
