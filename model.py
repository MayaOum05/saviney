import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class OutageModel:
    def __init__(self):
        self.reg = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        self.pipeline = None

    def preprocess(self, df):
        # Datetime features
        df["Datetime Event Began"] = pd.to_datetime(df["Datetime Event Began"])
        df["hour"] = df["Datetime Event Began"].dt.hour
        df["weekday"] = df["Datetime Event Began"].dt.weekday
        df["month"] = df["Datetime Event Began"].dt.month

        # Categorical and numeric features
        categorical = ["state_event", "Event Type", "state", "county"]
        numeric = ["duration", "min_customers", "max_customers", "hour", "weekday", "month"]

        # Clean numeric columns
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Target variable
        df["mean_customers"] = pd.to_numeric(df["mean_customers"], errors="coerce").fillna(0)

        X = df[categorical + numeric]
        y = df["mean_customers"]

        return X, y, categorical, numeric

    def train(self, df):
        X, y, categorical, numeric = self.preprocess(df)

        # Build preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", "passthrough", numeric)
            ]
        )

        # Create full pipeline
        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", self.reg)
        ])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.pipeline.fit(X_train, y_train)
        return X_test, y_test

    def predict(self, df):
        X, _, _, _ = self.preprocess(df)
        if self.pipeline is None:
            raise ValueError("Model is not trained yet. Call train() first.")
        return self.pipeline.predict(X)
