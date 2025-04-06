# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Importing our custom modules
from ingestion_processing import DataIngestion, DataCleaning, DataProcessing
from model_visualization import ModelTraining, Visualization, ModelEvaluation

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
    #st.write("Columns in the cleaned data:", df_clean.columns)
    
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
        "Alerts/Warnings"
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
        # Hyperparameter tuning: sliders for model parameters
        st.subheader("Hyperparameter Tuning")
        n_estimators = st.slider("Number of trees in RandomForest", min_value=50, max_value=500, value=100, step=10)
        max_depth = st.slider("Maximum depth of trees", min_value=5, max_value=30, value=10, step=1)
        min_samples_split = st.slider("Minimum samples required to split", min_value=2, max_value=20, value=2, step=1)
        min_samples_leaf = st.slider("Minimum samples required in leaf", min_value=1, max_value=20, value=1, step=1)
        max_features_options = ['sqrt', 'log2', None]
        max_features = st.selectbox("Maximum features to consider when splitting", options=max_features_options, index=0)

        # Define initial feature sets before selection
        if "Anomaly Scores" in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean["Anomaly Scores"]):
            available_features = categorical_features + ["Packet Length", "Anomaly Scores"]
        else:
            available_features = categorical_features + ["Packet Length"]

        # Feature selection options
        st.subheader("Feature Selection")
        feature_selection_method = st.radio(
            "Feature selection method", 
            options=["Use all features", "Manual selection"]
        )
        
        # Initialize selected features to all available features
        selected_features = available_features
        
        # If manual selection, show multiselect widget
        if feature_selection_method == "Manual selection":
            selected_features = st.multiselect(
                "Select features to use", 
                options=available_features,
                default=available_features
            )
            
            if len(selected_features) < 1:
                st.warning("Please select at least one feature")
                selected_features = available_features

        # Retrain Model block
        if st.button("Retrain Model", key="retrain_button"):
            st.write("Retraining triggered at:", datetime.now())
            
            # Use the selected features
            features_for_model = selected_features
            
            # Debug: Print features to check for leakage
            st.write("Features being used for model training:")
            st.write(features_for_model)
            st.write("Target variable:")
            st.write("Severity Level")
            
            X = df_clean[features_for_model]
            y = df_clean["Severity Level"]
            
            # Create a new processing pipeline instance
            numerical_features_selected = [f for f in numerical_features if f in features_for_model]
            categorical_features_selected = [f for f in categorical_features if f in features_for_model]
            
            processing_model = DataProcessing(
                categorical_features_selected, 
                numerical_features_selected
            )
            preprocessor_model = processing_model.build_preprocessor()
            
            # Initialize the ModelTraining module with classification
            model_training = ModelTraining(model_type="classification")
            pipeline_model = model_training.build_pipeline(preprocessor_model)
            
            # Update all hyperparameters based on slider values
            pipeline_model.named_steps["model"].n_estimators = n_estimators
            pipeline_model.named_steps["model"].max_depth = max_depth
            pipeline_model.named_steps["model"].min_samples_split = min_samples_split
            pipeline_model.named_steps["model"].min_samples_leaf = min_samples_leaf
            pipeline_model.named_steps["model"].max_features = max_features
            
            # Split the data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            trained_pipeline = model_training.train(X_train, y_train)
            
            # Evaluate the model on test data
            y_pred = trained_pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            
            st.success("Model retrained successfully!")
            st.write(f"**Model Accuracy:** {acc:.2f}")
            st.write(f"**Model F1 Score:** {f1:.2f}")
            st.write("Retraining complete at:", datetime.now())
            
            # Display feature importance if available
            model_step = trained_pipeline.named_steps.get("model")
            if model_step is not None and hasattr(model_step, "feature_importances_"):
                importances = model_step.feature_importances_
                # Retrieve and clean feature names from the preprocessor using ModelEvaluation utility
                feature_names = ModelEvaluation.get_clean_feature_names(preprocessor_model)
                st.subheader("Feature Importance")
                fig_feat = Visualization.plot_feature_importance(feature_names, importances)
                st.pyplot(fig_feat)
            else:
                st.info("Feature importances are not available for the current model.")
            
            # ===========================
            # Evaluation Section
            # ===========================
            st.subheader("Model Evaluation")
            
            # Cross-Validation Evaluation for Classification
            cv_results = ModelEvaluation.evaluate_classification(trained_pipeline, X_train, y_train)
            st.write("Cross-Validation Results:")
            st.write(cv_results)
            
            # Plot Confusion Matrix
            st.subheader("Confusion Matrix")
            fig_cm = ModelEvaluation.plot_confusion_matrix(y_test, y_pred, class_labels=['Low', 'Medium', 'High'])
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.pyplot(fig_cm)
            
            # Generate and Display Classification Report
            st.subheader("Classification Report")
            report_df = ModelEvaluation.generate_classification_report(y_test, y_pred, output_csv="test_res.csv")
            st.dataframe(report_df)