import sys
import pickle
import numpy as np
import pandas as pd
from src.train import MODEL_PATH, TRAIN_FEATURES, predict_eval, prepare_data


def main(csv_path: str):
    print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Loading data from {csv_path} ...")
    data = pd.read_csv(csv_path)
    has_target = "trip_duration" in data.columns

    # exact same cleaning/feature engineering used during training
    prepare_data(data)

    if has_target:
        # ground truth available (e.g. current test.csv) -> report the metrics
        predict_eval(model, data, TRAIN_FEATURES, "test")
    else:
        # true deployment case: no ground truth, just produce predictions
        log_preds = model.predict(data[TRAIN_FEATURES])
        data["predicted_trip_duration"] = np.expm1(log_preds)

        output_path = "predictions.csv"
        data[["predicted_trip_duration"]].to_csv(output_path, index=False)
        print(f"No 'trip_duration' column found - wrote predictions to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict path/to/data.csv")
        sys.exit(1)

    main(sys.argv[1])