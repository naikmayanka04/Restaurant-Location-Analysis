"""Supervised learning: predict a restaurant's rating from its features.

Compares a Linear Regression baseline against a Random Forest Regressor,
using a proper train/test split and standard regression metrics (MAE, RMSE,
R^2). Also reports the Random Forest's feature importances so we can see
which factors — including the DBSCAN hotspot cluster from clustering.py —
actually drive rating.

    python src/predict_rating.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from clustering import ncr_dbscan
from constants import OUTPUTS_DIR
from load_data import load_clean

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N_CITIES = 15  # cities kept as individual one-hot categories; rest -> "Other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the feature table used for rating prediction."""
    df = df.copy()

    # Only restaurants that have actually been rated (rating 0 == "Not rated").
    df = df[df["Aggregate rating"] > 0]

    # Cuisine count: how many cuisines a restaurant lists.
    df["cuisine_count"] = df["Cuisines"].fillna("").apply(
        lambda s: len(s.split(", ")) if s else 0
    )

    # Binary flags.
    df["has_online_delivery"] = (df["Has Online delivery"] == "Yes").astype(int)
    df["has_table_booking"] = (df["Has Table booking"] == "Yes").astype(int)

    # Bucket rare cities so the model doesn't overfit to one-off cities.
    top_cities = df["City"].value_counts().head(TOP_N_CITIES).index
    df["city_bucketed"] = df["City"].where(df["City"].isin(top_cities), "Other")

    # DBSCAN hotspot cluster as a location feature for Indian restaurants;
    # -2 (not -1) marks restaurants outside India entirely, so it's never
    # confused with DBSCAN's own noise label (-1).
    india_clustered = ncr_dbscan(df[df["Country"] == "India"])
    df = df.merge(
        india_clustered[["Restaurant ID", "hotspot_cluster"]], on="Restaurant ID", how="left"
    )
    df["hotspot_cluster"] = df["hotspot_cluster"].fillna(-2).astype(int).astype(str)

    return df


FEATURE_COLUMNS_NUMERIC = ["Votes", "Average Cost for two", "Price range", "cuisine_count", "has_online_delivery", "has_table_booking"]
FEATURE_COLUMNS_CATEGORICAL = ["city_bucketed", "hotspot_cluster"]
TARGET_COLUMN = "Aggregate rating"


def make_design_matrix(df: pd.DataFrame):
    """One-hot encode categoricals and assemble the final X, y, feature names."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = encoder.fit_transform(df[FEATURE_COLUMNS_CATEGORICAL])
    cat_names = encoder.get_feature_names_out(FEATURE_COLUMNS_CATEGORICAL)

    X_numeric = df[FEATURE_COLUMNS_NUMERIC].to_numpy()
    X = np.hstack([X_numeric, cat_encoded])
    feature_names = FEATURE_COLUMNS_NUMERIC + list(cat_names)
    y = df[TARGET_COLUMN].to_numpy()
    return X, y, feature_names


def evaluate(model, X_train, X_test, y_train, y_test) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "train_mae": mean_absolute_error(y_train, train_pred),
        "train_rmse": mean_squared_error(y_train, train_pred) ** 0.5,
        "train_r2": r2_score(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_rmse": mean_squared_error(y_test, test_pred) ** 0.5,
        "test_r2": r2_score(y_test, test_pred),
    }


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_clean()
    features_df = build_features(df)
    print(f"Training on {len(features_df):,} rated restaurants "
          f"({len(df) - len(features_df):,} unrated rows excluded).")

    X, y, feature_names = make_design_matrix(features_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate(lr, X_train, X_test, y_train, y_test)
    lr_metrics["model"] = "Linear Regression"
    results.append(lr_metrics)

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_train, X_test, y_train, y_test)
    rf_metrics["model"] = "Random Forest"
    results.append(rf_metrics)

    results_df = pd.DataFrame(results)[
        ["model", "train_mae", "train_rmse", "train_r2", "test_mae", "test_rmse", "test_r2"]
    ]
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (rating is on a 0-5 scale)")
    print("=" * 60)
    print(results_df.round(3).to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 15 Random Forest feature importances:")
    print(importances.head(15).round(4))
    importances.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        os.path.join(OUTPUTS_DIR, "feature_importances.csv"), index=False
    )

    # Predicted vs. actual on the test set, for the report's scatter plot.
    test_pred = rf.predict(X_test)
    pd.DataFrame({"actual": y_test, "predicted": test_pred}).to_csv(
        os.path.join(OUTPUTS_DIR, "test_predictions.csv"), index=False
    )

    print(f"\nSaved model comparison, feature importances, and test predictions to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
