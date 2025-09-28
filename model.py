import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class OutageModel:
    def __init__(self):
        self.reg = RandomForestRegressor(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df, training=True):
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

        # Encode categorical columns
        for col in categorical:
            if training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
                if "Unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "Unknown")
                df[col] = le.transform(df[col])

        # Scale numeric features
        if training:
            df[numeric] = self.scaler.fit_transform(df[numeric])
        else:
            df[numeric] = self.scaler.transform(df[numeric])

        X = df[categorical + numeric]

        # Target variable
        df["mean_customers"] = pd.to_numeric(df["mean_customers"], errors="coerce").fillna(0)
        y_reg = df["mean_customers"]

        return X, y_reg

    def train(self, df):
        X, y_reg = self.preprocess(df, training=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        self.reg.fit(X_train, y_train)
        return X_test, y_test

    def predict(self, df):
        X, _ = self.preprocess(df, training=False)
        predicted_customers = self.reg.predict(X)
        return predicted_customers
