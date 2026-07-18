import os
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
POLY_FEATURES = ["distance_km_log", "hour_sin", "hour_cos", "distance_x_rush_hour"]
CATEGORICAL_FEATURES = ["month", "passenger_count"]
CYCLIC_AND_FLAGS = ["dow_sin", "dow_cos", "is_rush_hour", "is_weekend"]

# single source of truth for both train.py and predict.py
TRAIN_FEATURES = CATEGORICAL_FEATURES + CYCLIC_AND_FLAGS + POLY_FEATURES

MODEL_PATH = os.path.join("model", "ridge_model.pkl")


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c


def prepare_data(df):
    """Cleans the raw dataframe and builds every engineered feature in place.
    Used by both train.py (on train/val) and predict.py (on the held-out test
    set), so both go through the exact same processing.

    Handles two cases:
      - trip_duration column present (train/val, and the current test.csv used
        for grading): outlier filtering on duration + target column created.
      - trip_duration column absent (real deployment / true unseen data):
        those two steps are skipped, everything else still runs the same way.
    """
    df.drop(columns=["id"], inplace=True, errors="ignore")

    # --- basic cleaning ---
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.query("passenger_count >= 1 and passenger_count <= 6", inplace=True)

    has_target = "trip_duration" in df.columns
    if has_target:
        df.query("trip_duration >= 60 and trip_duration <= 7200", inplace=True)
        df["log_trip_duration"] = np.log1p(df["trip_duration"])

    # --- date/time features ---
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofyear"] = df["pickup_datetime"].dt.dayofyear

    # --- distance feature ---
    df["distance_km"] = haversine_distance(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"],
    )
    df.query("distance_km > 0.05", inplace=True)  # drop near-zero-distance noise
    df["distance_km_log"] = np.log1p(df["distance_km"])

    # --- flags ---
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # --- cyclic encoding: hour 23 and hour 0 are actually close, raw ints don't capture that ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # --- interaction: same distance takes longer in rush-hour traffic ---
    df["distance_x_rush_hour"] = df["distance_km_log"] * df["is_rush_hour"]


def predict_eval(model, data, train_features, name):
    y_pred = model.predict(data[train_features])
    rmse = mean_squared_error(data.log_trip_duration, y_pred) ** 0.5
    r2 = r2_score(data.log_trip_duration, y_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def build_pipeline():

    column_transformer = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("scaling", StandardScaler(), CYCLIC_AND_FLAGS),
        ("poly", Pipeline([
            ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale_poly", StandardScaler()),
        ]), POLY_FEATURES),
    ], remainder="drop")

    pipeline = Pipeline(steps=[
        ("preprocessing", column_transformer),
        ("regression", Ridge(alpha=1)),
    ])
    return pipeline


def train_model(train, val):
    pipeline = build_pipeline()
    model = pipeline.fit(train[TRAIN_FEATURES], train.log_trip_duration)

    predict_eval(model, train, TRAIN_FEATURES, "train")
    predict_eval(model, val, TRAIN_FEATURES, "val")

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    # Use "Data/split_sample" for quick iteration, "Data/split" for the full run
    root_dir = os.path.join("Data", "split")
    train = pd.read_csv(os.path.join(root_dir, "train.csv"))
    val = pd.read_csv(os.path.join(root_dir, "val.csv"))

    prepare_data(train)
    prepare_data(val)

    train_model(train, val)