import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px


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
    def plot_boxplot(data, column: str) -> None:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f"Distribution of {column} (Box Plot)")
        plt.xlabel(column)
        plt.show()

    @staticmethod
    def plot_histogram(data, column: str, bins: int = 20) -> None:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=bins)
        plt.title(f"Distribution of {column} (Histogram)")
        plt.xlabel(column)
        plt.show()

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