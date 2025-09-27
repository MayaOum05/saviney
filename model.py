import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class OutageModel:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=200, random_state=42)
        self.reg = RandomForestRegressor(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df, training=True):
        """
        Returns (X, y_class, y_reg) for modeling.
        Works on a copy of df so original is untouched.
        """
        dfc = df.copy() 

        categorical = ["Geographic Areas", "NERC Region", "Tags", "Event Description"]
        numeric = ["Year", "Demand Loss (MW)"]
        dfc["Number of Customers Affected"] = (
            pd.to_numeric(
                dfc["Number of Customers Affected"].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            .fillna(0)
        )

        for col in categorical:
            dfc[col] = dfc[col].astype(str).fillna("Unknown").replace(["nan", "None"], "Unknown")

            if training:
                le = LabelEncoder()
                dfc[col] = le.fit_transform(dfc[col])
                self.label_encoders[col] = le
            else:
                if col not in self.label_encoders:
                    raise ValueError(f"Label encoder for '{col}' not found. Train model first.")

                le = self.label_encoders[col]
                vals = dfc[col].astype(str)
                vals = vals.apply(lambda x: x if x in le.classes_ else "Unknown")

                if "Unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "Unknown")

                dfc[col] = le.transform(vals)

        for col in numeric:
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce").fillna(0)

        if training:
            dfc[numeric] = self.scaler.fit_transform(dfc[numeric])
        else:
            dfc[numeric] = self.scaler.transform(dfc[numeric])

        y_class = (dfc["Number of Customers Affected"] > 0).astype(int)
        y_reg = dfc["Number of Customers Affected"]

        X = dfc[categorical + numeric]

        return X, y_class, y_reg

    def train(self, df):
        """
        Train both the classifier (outage/no-outage) and the regressor (severity).
        Returns test splits for inspection: X_test, y_class_test, y_reg_test
        """
        X, y_class, y_reg = self.preprocess(df, training=True)

        X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )

        self.clf.fit(X_train, yc_train)
        self.reg.fit(X_train, yr_train)

        return X_test, yc_test, yr_test

    def predict(self, df):
        """
        Given a DataFrame, return (outage_probs, predicted_customers).
        Does not mutate the original df.
        """
        X, _, _ = self.preprocess(df, training=False)
        outage_probs = self.clf.predict_proba(X)[:, 1]
        predicted_customers = self.reg.predict(X)
        return outage_probs, predicted_customers
