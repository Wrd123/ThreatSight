# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.express as px

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Importing modules 
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
        col1, col2, col3 = st.columns([1, 4, 1])  # Creates margins on the sides
        with col2:
            st.pyplot(fig_box)
        
        # Histogram of Anomaly Scores
        st.subheader("Histogram of Anomaly Scores")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        sns.histplot(df_clean["Anomaly Scores"], bins=20, kde=True, ax=ax_hist)
        ax_hist.axvline(threshold, color="red", linestyle="--", label="Threshold")
        ax_hist.legend()
        col1, col2, col3 = st.columns([1, 4, 1])  # Creates margins on the sides
        with col2:
            st.pyplot(fig_hist)

         # --- Add Interactive Scatter Plot ---
        if "Packet Length" in df_clean.columns:
            st.subheader("Interactive Scatter Plot: Packet Length vs. Anomaly Scores")
            scatter_fig = px.scatter(
                df_clean,
                x="Packet Length",
                y="Anomaly Scores",
                color="Above Threshold",
                hover_data=df_clean.columns,
                title="Packet Length vs. Anomaly Scores"
            )
            st.plotly_chart(scatter_fig)
        
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
       # Hyperparameter slider for tuning
        n_estimators = st.slider("Number of trees in RandomForest", min_value=50, max_value=200, value=100, step=10)

        # Retrain Model block
        if st.button("Retrain Model", key="retrain_button"):
            # Debug: Output timestamp to confirm retraining execution
            st.write("Retraining triggered at:", datetime.now())
            
            # Define features and target for model training
            if "Anomaly Scores" in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean["Anomaly Scores"]):
                features_for_model = categorical_features + ["Packet Length", "Anomaly Scores"]
            else:
                features_for_model = categorical_features + ["Packet Length"]
            X = df_clean[features_for_model]
            y = df_clean["Severity Level"]
            
            # Create a new processing pipeline instance to ensure no cached objects are used
            processing_model = DataProcessing(categorical_features, (["Packet Length", "Anomaly Scores"] if "Anomaly Scores" in df_clean.columns else ["Packet Length"]))
            preprocessor_model = processing_model.build_preprocessor()
            
            # Initialize the ModelTraining module with 'classification'
            model_training = ModelTraining(model_type="classification")
            # Build pipeline
            pipeline_model = model_training.build_pipeline(preprocessor_model)
            # Update hyperparameter based on slider value
            pipeline_model.named_steps["model"].n_estimators = n_estimators
            
            # Split the data for training and testing (removed fixed random_state for variability)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Train the model
            trained_pipeline = model_training.train(X_train, y_train)
            
            # Evaluate the model
            y_pred = trained_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            
            st.success("Model retrained successfully!")
            st.write(f"**Model Accuracy:** {accuracy:.2f}")
            st.write(f"**Model F1 Score:** {f1:.2f}")
            
            # Debug: Output a timestamp to confirm retraining completion
            st.write("Retraining complete at:", datetime.now())
            
            # Display feature importance if available
            model_step = trained_pipeline.named_steps.get("model")
            if model_step is not None and hasattr(model_step, "feature_importances_"):
                importances = model_step.feature_importances_
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                st.subheader("Feature Importance")
                fig = Visualization.plot_feature_importance(feature_names, importances)
                st.pyplot(fig)
            else:
                st.info("Feature importances are not available for the current model.")