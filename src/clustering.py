"""Unsupervised learning: discover restaurant location hotspots.

Two techniques, at two scales:

1. KMeans over ALL restaurants (global lat/lon) — picks k via silhouette
   score, producing broad regional clusters. At global scale this mostly
   rediscovers "country/metro area" groupings, which is a useful sanity
   check that the clustering is behaving sensibly.
2. DBSCAN within India (where 90%+ of the data lives) — a density-based
   method that finds tight, high-density micro-hotspots (e.g. a stretch of
   a single street with dozens of restaurants) without needing a
   pre-specified number of clusters, and explicitly labels sparse outliers
   as noise (-1) instead of forcing them into a cluster.

    python src/clustering.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from constants import OUTPUTS_DIR
from load_data import load_geo

RANDOM_STATE = 42


def choose_k_via_silhouette(X: np.ndarray, k_range=range(3, 11)) -> tuple[int, pd.DataFrame]:
    """Fit KMeans for each k in k_range, score with silhouette, return the best k."""
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append({"k": k, "silhouette_score": score, "inertia": km.inertia_})
    scores_df = pd.DataFrame(scores)
    best_k = int(scores_df.loc[scores_df["silhouette_score"].idxmax(), "k"])
    return best_k, scores_df


def global_kmeans(geo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Cluster every restaurant's (lat, lon) into regional groups with KMeans."""
    X = geo[["Latitude", "Longitude"]].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    best_k, scores_df = choose_k_via_silhouette(X_scaled)
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    geo = geo.copy()
    geo["region_cluster"] = km.fit_predict(X_scaled)
    return geo, scores_df, best_k


def ncr_dbscan(geo: pd.DataFrame, eps_km: float = 1.0, min_samples: int = 8) -> pd.DataFrame:
    """Density-based hotspot detection within India using DBSCAN.

    eps is converted from kilometers to (approximate) degrees for the Delhi
    NCR latitude band, since DBSCAN needs a distance threshold in the same
    units as the input.
    """
    india = geo[geo["Country"] == "India"].copy()
    X = india[["Latitude", "Longitude"]].to_numpy()

    km_per_degree = 111.0  # ~constant for latitude; a reasonable approximation for a small area
    eps_deg = eps_km / km_per_degree

    db = DBSCAN(eps=eps_deg, min_samples=min_samples)
    india["hotspot_cluster"] = db.fit_predict(X)
    return india


def summarize_hotspots(india: pd.DataFrame) -> pd.DataFrame:
    """One row per DBSCAN hotspot (excluding noise, label -1), sorted by size."""
    clustered = india[india["hotspot_cluster"] != -1]
    summary = (
        clustered.groupby("hotspot_cluster")
        .agg(
            restaurant_count=("Restaurant ID", "count"),
            avg_rating=("Aggregate rating", "mean"),
            avg_cost_for_two=("Average Cost for two", "mean"),
            center_lat=("Latitude", "mean"),
            center_lon=("Longitude", "mean"),
            top_city=("City", lambda s: s.mode().iloc[0]),
            top_locality=("Locality", lambda s: s.mode().iloc[0]) if "Locality" in clustered.columns else ("City", "first"),
        )
        .reset_index()
        .sort_values("restaurant_count", ascending=False)
    )
    return summary


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    geo = load_geo()

    print("=" * 60)
    print("GLOBAL KMEANS (k chosen by silhouette score)")
    print("=" * 60)
    geo_clustered, scores_df, best_k = global_kmeans(geo)
    print(scores_df.round(3).to_string(index=False))
    print(f"\nBest k = {best_k}")
    print(geo_clustered.groupby("region_cluster")["Country"].agg(lambda s: s.mode().iloc[0]))
    scores_df.to_csv(os.path.join(OUTPUTS_DIR, "kmeans_silhouette_scores.csv"), index=False)
    geo_clustered[["Restaurant ID", "City", "Country", "Latitude", "Longitude", "region_cluster"]].to_csv(
        os.path.join(OUTPUTS_DIR, "global_region_clusters.csv"), index=False
    )

    print("\n" + "=" * 60)
    print("DBSCAN HOTSPOTS WITHIN INDIA")
    print("=" * 60)
    india = ncr_dbscan(geo)
    n_hotspots = india["hotspot_cluster"].nunique() - (1 if -1 in india["hotspot_cluster"].values else 0)
    n_noise = (india["hotspot_cluster"] == -1).sum()
    print(f"Found {n_hotspots} dense hotspots among {len(india):,} Indian restaurants")
    print(f"{n_noise:,} restaurants ({n_noise / len(india):.1%}) are sparse outliers (not in any hotspot)")

    hotspot_summary = summarize_hotspots(india)
    print("\nTop 10 hotspots by restaurant count:")
    print(hotspot_summary.head(10).round(2).to_string(index=False))
    hotspot_summary.to_csv(os.path.join(OUTPUTS_DIR, "dbscan_hotspots.csv"), index=False)
    india[["Restaurant ID", "City", "Locality", "Latitude", "Longitude", "hotspot_cluster"]].to_csv(
        os.path.join(OUTPUTS_DIR, "india_hotspot_labels.csv"), index=False
    )
    print(f"\nSaved cluster outputs to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
