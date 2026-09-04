import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from train_23L2572 import load_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

RANDOM_STATE = 42


def build_features(df):
    df = df.copy()
    df["Date Changed"] = pd.to_datetime(df["Date Changed"], format="%d %b %y")

    future = df["Date Changed"] > pd.Timestamp.now()
    df.loc[future, "Date Changed"] = df.loc[future, "Date Changed"] - pd.DateOffset(years=100)

    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df.dropna(subset=["Date Changed", "Rate"]).sort_values("Date Changed")

    df["year"] = df["Date Changed"].dt.year
    df["month"] = df["Date Changed"].dt.month
    df["day"] = df["Date Changed"].dt.day
    df["day_of_year"] = df["Date Changed"].dt.dayofyear
    df["prev_rate"] = df["Rate"].shift(1)
    df = df.dropna(subset=["prev_rate"])

    feature_cols = ["year", "month", "day", "day_of_year", "prev_rate"]
    return df[feature_cols], df["Rate"]


def main():
    print("Loading data...")
    data = load_data(DATA_PATH)
    if data is None:
        raise SystemExit("Failed to load data. Check that data/data.csv exists.")
    print(f"Data loaded successfully! Shape: {data.shape}")

    X, y = build_features(data)
    print(f"Feature matrix: {X.shape}, target: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Test MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"Test R^2: {r2_score(y_test, predictions):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
