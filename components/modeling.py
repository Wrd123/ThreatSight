# components/modeling.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Importing our custom modules
from ingestion_processing import DataProcessing
from model_visualization import ModelTraining, Visualization, ModelEvaluation
from utils.helpers import get_categorical_features, get_numerical_features

def render_hyperparameter_section():
   
    st.subheader("Hyperparameter Tuning")
    
    # Add explanation and slider for n_estimators
    st.markdown("**Number of trees in RandomForest**")
    st.markdown("*Controls the number of trees in the ensemble. More trees generally improve performance but increase computation time. Higher values reduce variance but may lead to overfitting.*")
    n_estimators = st.slider("-", min_value=50, max_value=500, value=100, step=10)
    
    # Add explanation and slider for max_depth
    st.markdown("**Maximum depth of trees**")
    st.markdown("*Defines how deep each decision tree can grow. Deeper trees can model more complex patterns but are more prone to overfitting. Shallower trees are more generalizable.*")
    max_depth = st.slider("-", min_value=5, max_value=30, value=10, step=1)
    
    # Add explanation and slider for min_samples_split
    st.markdown("**Minimum samples required to split**")
    st.markdown("*The minimum number of samples required to split an internal node. Higher values prevent creating nodes that might only capture noise in the training data.*")
    min_samples_split = st.slider("-", min_value=2, max_value=20, value=2, step=1)
    
    # Add explanation and slider for min_samples_leaf
    st.markdown("**Minimum samples required in leaf**")
    st.markdown("*The minimum number of samples required to be at a leaf node. Higher values create more conservative trees that are less likely to overfit.*")
    min_samples_leaf = st.slider("-", min_value=1, max_value=20, value=1, step=1)
    
    # Add explanation and dropdown for max_features
    st.markdown("**Maximum features to consider when splitting**")
    st.markdown("*The number of features to consider when looking for the best split. 'sqrt' uses square root of total features, 'log2' uses log base 2, and None uses all features. Controls the randomness in feature selection.*")
    max_features_options = ['sqrt', 'log2', None]
    max_features = st.selectbox("-", options=max_features_options, index=0)
    
    hyperparameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features
    }
    
    return hyperparameters

def render_feature_selection(df_clean, categorical_features, numerical_features):
 
    st.subheader("Feature Selection")
    
    # Define initial feature sets before selection
    if "Anomaly Scores" in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean["Anomaly Scores"]):
        available_features = categorical_features + ["Packet Length", "Anomaly Scores"]
    else:
        available_features = categorical_features + ["Packet Length"]
    
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
    
    return selected_features

def train_and_evaluate_model(df_clean, selected_features, hyperparameters):
 
    st.write("Retraining triggered at:", datetime.now())
    
    # Debug: Print features to check for leakage
    st.write("Features being used for model training:")
    st.write(selected_features)
    st.write("Target variable:")
    st.write("Severity Level")
    
    X = df_clean[selected_features]
    y = df_clean["Severity Level"]
    
    # Get feature lists from utils
    all_categorical_features = get_categorical_features()
    all_numerical_features = get_numerical_features()
    
    # Create a new processing pipeline instance
    numerical_features_selected = [f for f in all_numerical_features if f in selected_features]
    categorical_features_selected = [f for f in all_categorical_features if f in selected_features]
    
    processing_model = DataProcessing(
        categorical_features_selected, 
        numerical_features_selected
    )
    preprocessor_model = processing_model.build_preprocessor()
    
    # Initialize the ModelTraining module with classification
    model_training = ModelTraining(model_type="classification")
    pipeline_model = model_training.build_pipeline(preprocessor_model)
    
    # Update all hyperparameters based on slider values
    pipeline_model.named_steps["model"].n_estimators = hyperparameters["n_estimators"]
    pipeline_model.named_steps["model"].max_depth = hyperparameters["max_depth"]
    pipeline_model.named_steps["model"].min_samples_split = hyperparameters["min_samples_split"]
    pipeline_model.named_steps["model"].min_samples_leaf = hyperparameters["min_samples_leaf"]
    pipeline_model.named_steps["model"].max_features = hyperparameters["max_features"]
    
    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    trained_pipeline = model_training.train(X_train, y_train)
    
    # Evaluate the model on test data
    y_pred = trained_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    # Retrieve feature names
    feature_names = ModelEvaluation.get_clean_feature_names(preprocessor_model)
    
    return trained_pipeline, X_train, X_test, y_train, y_test, y_pred, feature_names, {"accuracy": acc, "f1": f1}

def render_evaluation_section(trained_pipeline, X_train, X_test, y_train, y_test, y_pred, feature_names):

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

def render_modeling_section(df_clean):
  
    st.header("Model Interaction and Feedback")
    
    if "Severity Level" not in df_clean.columns:
        st.info("Column 'Severity Level' not found in the dataset. Skipping modeling section.")
        return
    
    # Get hyperparameter selections
    hyperparameters = render_hyperparameter_section()
    
    # Get feature selections
    categorical_features = get_categorical_features()
    numerical_features = get_numerical_features()
    selected_features = render_feature_selection(df_clean, categorical_features, numerical_features)
    
    # Retrain Model block
    if st.button("Retrain Model", key="retrain_button"):
        # Train and evaluate the model
        trained_pipeline, X_train, X_test, y_train, y_test, y_pred, feature_names, metrics = train_and_evaluate_model(
            df_clean, selected_features, hyperparameters
        )
        
        # Display training results
        st.success("Model retrained successfully!")
        st.write(f"**Model Accuracy:** {metrics['accuracy']:.2f}")
        st.write(f"**Model F1 Score:** {metrics['f1']:.2f}")
        st.write("Retraining complete at:", datetime.now())
        
        # Display feature importance if available
        model_step = trained_pipeline.named_steps.get("model")
        if model_step is not None and hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            # Use the feature names from the function
            st.subheader("Feature Importance")
            fig_feat = Visualization.plot_feature_importance(feature_names, importances)
            st.pyplot(fig_feat)
        else:
            st.info("Feature importances are not available for the current model.")
        
        # Render evaluation section
        render_evaluation_section(trained_pipeline, X_train, X_test, y_train, y_test, y_pred, feature_names)