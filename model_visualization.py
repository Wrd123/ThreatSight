# model_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
    precision_score, recall_score, make_scorer
)

# ---------------- Model Training Module ----------------
class ModelTraining:
    def __init__(self, model_type: str = "classification") -> None:
        self.model_type = model_type
        self.pipeline = None

    def build_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        # Choose the model based on type
        if self.model_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be either 'classification' or 'regression'")

        # Build the full pipeline with preprocessor and model
        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        print(f"{self.model_type.capitalize()} pipeline built successfully.")
        return self.pipeline

    def train(self, X, y) -> Pipeline:
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        self.pipeline.fit(X, y)
        print("Model training complete.")
        return self.pipeline

# ---------------- Visualization Module ----------------
class Visualization:
    @staticmethod
    def plot_boxplot(data, column: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=data[column], ax=ax)
        ax.set_title(f"Distribution of {column} (Box Plot)")
        ax.set_xlabel(column)
        return fig

    @staticmethod
    def plot_histogram(data, column: str, bins: int = 20):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=bins, ax=ax)
        ax.set_title(f"Distribution of {column} (Histogram)")
        ax.set_xlabel(column)
        return fig

    @staticmethod
    def plot_scatter(df, x, y, color_column=None, title="Scatter Plot"):
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color_column,
            hover_data=df.columns,
            title=title
        )
        return fig

    @staticmethod
    def plot_feature_importance(feature_names, importances):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.barh(feature_names, importances)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance")
        return fig

# ---------------- Model Evaluation Module ----------------
class ModelEvaluation:
    @staticmethod
    def evaluate_regression(pipeline, X_train, y_train, cv=5, scoring='r2', return_train_score=True):
        """
        Performs cross-validation for a regression pipeline.
        """
        results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=return_train_score)
        print("Regression Cross-Validation Results:")
        print(results)
        return results

    @staticmethod
    def evaluate_classification(pipeline, X_train, y_train, cv=None, scoring=None, return_train_score=True):
        """
        Performs cross-validation for a classification pipeline.
        By default uses StratifiedKFold and multiple scoring metrics.
        """
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if scoring is None:
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1_micro': make_scorer(f1_score, average='micro'),
                'f1_macro': make_scorer(f1_score, average='macro'),
                'precision_macro': make_scorer(precision_score, average='macro'),
                'recall_macro': make_scorer(recall_score, average='macro'),
            }
        results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=return_train_score)
        print("Classification Cross-Validation Results:")
        print(results)
        return results

    @staticmethod
    def get_clean_feature_names(preprocessor: ColumnTransformer):
        """
        Retrieves and cleans feature names from the preprocessor.
        """
        try:
            ft_names = preprocessor.get_feature_names_out()
        except AttributeError:
            ft_names = []
        cleaned_ft_names = [name.replace("cat__", "").replace("num__", "") for name in ft_names]
        return cleaned_ft_names

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_labels=None, cmap='Blues'):
        """
        Plots the confusion matrix using seaborn heatmap and returns the figure.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
        if class_labels is not None:
            ax.set_yticks(np.arange(len(class_labels)) + 0.5)
            ax.set_yticklabels(class_labels, rotation=0, va='center')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix')
        return fig

    @staticmethod
    def generate_classification_report(y_true, y_pred, output_csv=None):
        """
        Generates and prints a classification report.
        Optionally writes the report to a CSV file.
        """
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        print("Classification Report:")
        print(report_df)
        if output_csv is not None:
            report_df.to_csv(output_csv)
        return report_df
