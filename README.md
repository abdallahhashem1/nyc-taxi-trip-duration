# NYC Taxi Trip Duration Prediction

Predicting NYC taxi trip duration using Ridge Regression, with a focus on
feature engineering (this project follows the "80% of tabular ML is about
the data" philosophy — the model is intentionally kept simple and fixed,
so performance gains come entirely from understanding and engineering the
data well).

## Project Structure
```
Project_trip/
├── Data/
│   └── split/
│       ├── train_val_data.zip   # compressed train.csv + val.csv (~46MB)
│       └── (unzip here → train.csv, val.csv)
|       └── test.csv # For final test
│   └── split_sample/
│       ├── train.csv
│       └── val.csv
|       └── test.csv
├── model/
│   └── ridge_model.pkl          # trained model, ready to use
├── notebooks/
│   └── 01_EDA_and_Visualization.ipynb
|   └── 02_Modeling_Ridge_for_test.ipynb
├── src/
│   ├── __init__.py
│   ├── train.py                 # cleans data, builds features, trains, saves model
│   └── predict.py               # loads saved model, evaluates on any CSV
└── README.md
```

## Setup

1. Clone the repository and open it in PyCharm.
2. Create a virtual environment and install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Unzip the dataset:
   ```
   Data/split/train_val_data.zip → extract into Data/split/
   ```
   This produces `Data/split/train.csv` and `Data/split/val.csv`.

## Model

- **Algorithm:** Ridge Regression, `alpha` fixed to `1` (no hyperparameter
  search — the goal of this project is to demonstrate that most of the
  performance gain comes from feature engineering, not model tuning).
- **Target:** `np.log1p(trip_duration)` (log-transformed to handle the
  right-skewed duration distribution).

### Engineered features
- **`distance_km_log`** — haversine distance between pickup/dropoff
  coordinates, log-transformed (right-skewed like the target).
- **Cyclic time encoding** (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`) —
  hour-of-day and day-of-week are circular, not linear (hour 23 and hour 0
  are actually close together), so raw integers were replaced with
  sine/cosine encoding.
- **`is_rush_hour`, `is_weekend`** — binary flags for known traffic
  patterns.
- **`distance_x_rush_hour`** — interaction term: the same distance takes
  longer during rush-hour traffic.
- **Polynomial features (degree 2)** on the four features above, to let
  the linear model capture some curvature/interaction it otherwise
  couldn't.
- Outlier removal: trips under 1 minute or over 2 hours, near-zero-distance
  trips (GPS noise), and passenger counts outside [1, 6].

## Usage

**Train the model:**
```
python -m src.train
```
Cleans the data, engineers features, trains `Ridge(alpha=1)`, prints
train/val RMSE and R², and saves the fitted pipeline to
`model/ridge_model.pkl`.

**Evaluate on any CSV:**
```
python -m src.predict path/to/data.csv
```
- If the CSV has a `trip_duration` column, it prints RMSE/R² against
  that ground truth.
- If it doesn't (true unseen data), it writes predictions to
  `predictions.csv` instead.

## Results

| Split | RMSE (log scale) | R² |
|---|---|---|
| Train |  0.4078 | 0.6844  |
| Val   | 0.4075  | 0.6857  |
| Test  | 0.4079  | 0.6856  |

## Notes

- All cleaning/feature-engineering logic lives in `src/train.py`'s
  `prepare_data()` function and is reused by `predict.py`, so train,
  validation, and test data are always processed identically.

