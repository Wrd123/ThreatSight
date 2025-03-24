import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------- Data Ingestion Module ----------------
class DataIngestion:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        df = pd.read_csv(self.file_path)
        print("Data loaded successfully.")
        return df

# ---------------- Data Cleaning Module ----------------
class DataCleaning:
    def __init__(self, drop_columns: list) -> None:
        self.drop_columns = drop_columns

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop irrelevant columns (ignoring errors if column is missing)
        df.drop(columns=self.drop_columns, inplace=True, errors='ignore')
        # Fill missing values for specific columns if they exist
        if "Malware Indicators" in df.columns:
            df["Malware Indicators"].fillna("None Detected", inplace=True)
        if "Alerts/Warnings" in df.columns:
            df["Alerts/Warnings"].fillna("No Alert", inplace=True)
        print("Data cleaned successfully.")
        return df

# ---------------- Data Processing Module ----------------
class DataProcessing:
    def __init__(self, categorical_features: list, numerical_features: list) -> None:
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.preprocessor = None

    def build_preprocessor(self) -> ColumnTransformer:
        # Define transformers for categorical and numerical features
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        numerical_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, self.categorical_features),
                ("num", numerical_transformer, self.numerical_features)
            ]
        )
        print("Preprocessor built successfully.")
        return self.preprocessor

    def transform_data(self, X: pd.DataFrame) -> any:
        if self.preprocessor is None:
            self.build_preprocessor()
        X_transformed = self.preprocessor.fit_transform(X)
        print("Data transformed successfully.")
        return X_transformed
