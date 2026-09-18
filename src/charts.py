"""Generate all chart images used in the report.

Run directly to (re)generate every PNG into images/:

    python src/charts.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import IMAGES_DIR, NCR_CITIES, OUTPUTS_DIR
from load_data import load_clean, load_geo

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#444"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def _save(fig, name: str) -> str:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, f"{name}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def world_map(geo: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    countries = geo["Country"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
    for country, color in zip(countries, colors):
        sub = geo[geo["Country"] == country]
        ax.scatter(sub.Longitude, sub.Latitude, s=10, alpha=0.6, label=country, color=color)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Global Distribution of Restaurants in Dataset (colored by country)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    return _save(fig, "world_map")


def india_map(geo: pd.DataFrame) -> str:
    india = geo[geo["Country"] == "India"]
    fig, ax = plt.subplots(figsize=(8, 7))
    for city in NCR_CITIES:
        sub = india[india["City"] == city]
        ax.scatter(sub.Longitude, sub.Latitude, s=8, alpha=0.5, label=city)
    other = india[~india["City"].isin(NCR_CITIES)]
    ax.scatter(other.Longitude, other.Latitude, s=8, alpha=0.4, label="Other Indian cities", color="gray")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("India / NCR Restaurant Density (83% of all restaurants)")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, "india_map")


def top_cities_count(stats: pd.DataFrame) -> str:
    top15 = stats.sort_values("restaurant_count", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(top15["City"][::-1], top15["restaurant_count"][::-1], color="#2E86AB")
    ax.set_xlabel("Number of Restaurants")
    ax.set_title("Top 15 Cities by Restaurant Count")
    for b in bars:
        ax.text(b.get_width() + 20, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}", va="center", fontsize=8)
    return _save(fig, "top_cities_count")


def rating_vs_cost(stats: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = stats[stats["restaurant_count"] >= 15]
    sc = ax.scatter(
        plot_data.avg_cost_for_two,
        plot_data.avg_rating,
        s=plot_data.restaurant_count / 3,
        alpha=0.6,
        c=plot_data.avg_price_range,
        cmap="viridis",
    )
    for _, row in plot_data.sort_values("restaurant_count", ascending=False).head(12).iterrows():
        ax.annotate(row.City, (row.avg_cost_for_two, row.avg_rating), fontsize=7, alpha=0.8)
    ax.set_xlabel("Average Cost for Two (local currency)")
    ax.set_ylabel("Average Rating")
    ax.set_title("City-level: Avg Rating vs Avg Cost (bubble size = # restaurants)")
    plt.colorbar(sc, label="Avg Price Range")
    ax.set_xscale("log")
    return _save(fig, "rating_vs_cost")


def top_cuisines(df: pd.DataFrame) -> str:
    counts = df["Cuisines"].dropna().str.split(", ").explode().value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#E07A5F")
    ax.set_xlabel("Number of Restaurants Offering Cuisine")
    ax.set_title("Top 15 Cuisines Across the Dataset")
    return _save(fig, "top_cuisines")


def price_range(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    dist = df["Price range"].value_counts(normalize=True).sort_index() * 100
    ax.bar(["1 (Budget)", "2", "3", "4 (Premium)"], dist.values, color="#3D5A80")
    ax.set_ylabel("% of Restaurants")
    ax.set_title("Price Range Distribution (All Restaurants)")
    for i, v in enumerate(dist.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    return _save(fig, "price_range")


# --- ML charts -------------------------------------------------------------

def silhouette_elbow(scores_df: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(scores_df["k"], scores_df["silhouette_score"], "o-", color="#2E86AB", label="Silhouette score")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Silhouette score", color="#2E86AB")
    ax1.tick_params(axis="y", labelcolor="#2E86AB")
    best_k = scores_df.loc[scores_df["silhouette_score"].idxmax(), "k"]
    ax1.axvline(best_k, color="#2E86AB", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(scores_df["k"], scores_df["inertia"], "s-", color="#E07A5F", alpha=0.7, label="Inertia")
    ax2.set_ylabel("Inertia (elbow)", color="#E07A5F")
    ax2.tick_params(axis="y", labelcolor="#E07A5F")
    ax2.grid(False)

    ax1.set_title(f"KMeans Model Selection: Silhouette Score & Inertia by k (best k={int(best_k)})")
    return _save(fig, "kmeans_selection")


def global_cluster_map(geo_clustered: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    clusters = sorted(geo_clustered["region_cluster"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(clusters)))
    for c, color in zip(clusters, colors):
        sub = geo_clustered[geo_clustered["region_cluster"] == c]
        label = f"Cluster {c} ({sub['Country'].mode().iloc[0]}, n={len(sub)})"
        ax.scatter(sub.Longitude, sub.Latitude, s=10, alpha=0.6, label=label, color=color)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("KMeans Regional Clusters (unsupervised, on raw lat/lon)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    return _save(fig, "kmeans_region_clusters")


def dbscan_hotspot_map(india: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 7))
    noise = india[india["hotspot_cluster"] == -1]
    ax.scatter(noise.Longitude, noise.Latitude, s=8, alpha=0.35, color="lightgray", label=f"Noise / sparse ({len(noise)})")

    clustered = india[india["hotspot_cluster"] != -1]
    top_clusters = clustered["hotspot_cluster"].value_counts().head(10).index
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_clusters)))
    for c, color in zip(top_clusters, colors):
        sub = clustered[clustered["hotspot_cluster"] == c]
        city = sub["City"].mode().iloc[0]
        ax.scatter(sub.Longitude, sub.Latitude, s=12, alpha=0.8, color=color, label=f"{city} hotspot (n={len(sub)})")
    other = clustered[~clustered["hotspot_cluster"].isin(top_clusters)]
    ax.scatter(other.Longitude, other.Latitude, s=10, alpha=0.5, color="darkgray", label="Smaller hotspots")

    # Almost every hotspot sits in Delhi NCR; zoom there so the clusters are
    # actually distinguishable instead of collapsing into one dot at
    # India's national scale. Individual far-flung restaurants (noise
    # points elsewhere in India) fall outside this window by design.
    ax.set_xlim(76.6, 77.6)
    ax.set_ylim(28.3, 28.85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("DBSCAN Density Hotspots within Delhi NCR (top 10 labeled, zoomed)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return _save(fig, "dbscan_hotspots")


def feature_importance_chart(importances: pd.DataFrame, top_n: int = 12) -> str:
    top = importances.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#3D5A80")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("What Predicts a Restaurant's Rating?")
    return _save(fig, "feature_importance")


def predicted_vs_actual(preds: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(preds["actual"], preds["predicted"], s=12, alpha=0.4, color="#2E86AB")
    lims = [0, 5]
    ax.plot(lims, lims, "--", color="#E07A5F", label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title("Random Forest: Predicted vs. Actual Rating (test set)")
    ax.legend(frameon=False)
    return _save(fig, "predicted_vs_actual")


def generate_all() -> None:
    df = load_clean()
    geo = load_geo()
    stats_path = os.path.join(OUTPUTS_DIR, "city_stats.csv")
    if os.path.exists(stats_path):
        stats = pd.read_csv(stats_path)
    else:
        from analysis import city_level_stats

        stats = city_level_stats(df)

    paths = [
        world_map(geo),
        india_map(geo),
        top_cities_count(stats),
        rating_vs_cost(stats),
        top_cuisines(df),
        price_range(df),
    ]

    # ML charts: only built if the corresponding pipeline step has been run
    # (src/clustering.py and src/predict_rating.py write these CSVs).
    silhouette_path = os.path.join(OUTPUTS_DIR, "kmeans_silhouette_scores.csv")
    region_path = os.path.join(OUTPUTS_DIR, "global_region_clusters.csv")
    hotspot_labels_path = os.path.join(OUTPUTS_DIR, "india_hotspot_labels.csv")
    importances_path = os.path.join(OUTPUTS_DIR, "feature_importances.csv")
    predictions_path = os.path.join(OUTPUTS_DIR, "test_predictions.csv")

    if os.path.exists(silhouette_path):
        paths.append(silhouette_elbow(pd.read_csv(silhouette_path)))
    if os.path.exists(region_path):
        paths.append(global_cluster_map(pd.read_csv(region_path)))
    if os.path.exists(hotspot_labels_path):
        paths.append(dbscan_hotspot_map(pd.read_csv(hotspot_labels_path)))
    if os.path.exists(importances_path):
        paths.append(feature_importance_chart(pd.read_csv(importances_path)))
    if os.path.exists(predictions_path):
        paths.append(predicted_vs_actual(pd.read_csv(predictions_path)))

    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    generate_all()
