"""Core geographic and statistical analysis of the restaurant dataset.

Run directly to print a full summary to stdout and write
outputs/city_stats.csv:

    python src/analysis.py
"""

import os

import pandas as pd

from constants import NCR_CITIES, OUTPUTS_DIR
from load_data import load_clean, load_geo


def city_level_stats(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """Aggregate rating, cost, price range, and cuisine diversity per city."""
    stats = (
        df.groupby("City")
        .agg(
            restaurant_count=("Restaurant ID", "count"),
            avg_rating=("Aggregate rating", "mean"),
            avg_votes=("Votes", "mean"),
            avg_cost_for_two=("Average Cost for two", "mean"),
            avg_price_range=("Price range", "mean"),
        )
        .reset_index()
    )
    stats = stats[stats["restaurant_count"] >= min_count]

    diversity = (
        df.groupby("City")["Cuisines"]
        .apply(lambda g: len({c for row in g.dropna().str.split(", ") for c in row}))
        .reset_index(name="unique_cuisines")
    )
    stats = stats.merge(diversity, on="City", how="left")
    return stats.sort_values("restaurant_count", ascending=False)


def cuisine_counts(df: pd.DataFrame) -> pd.Series:
    """Count restaurants per individual cuisine (a restaurant can list several)."""
    return df["Cuisines"].dropna().str.split(", ").explode().value_counts()


def delivery_and_booking_by_city(df: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    """Percent of restaurants offering online delivery / table booking, by city."""
    subset = df[df["City"].isin(cities)]
    return subset.groupby("City").agg(
        pct_online_delivery=("Has Online delivery", lambda x: (x == "Yes").mean() * 100),
        pct_table_booking=("Has Table booking", lambda x: (x == "Yes").mean() * 100),
    ).round(1)


def run_summary() -> pd.DataFrame:
    """Print the full analysis summary and return the city-level stats table."""
    df = load_clean()
    geo = load_geo()

    print("=" * 60)
    print("COUNTRY DISTRIBUTION")
    print("=" * 60)
    print(df["Country"].value_counts())

    stats = city_level_stats(df)

    print("\n" + "=" * 60)
    print("TOP 15 CITIES BY RESTAURANT COUNT")
    print("=" * 60)
    print(stats.head(15).round(2).to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 10 RATED CITIES (>=10 restaurants)")
    print("=" * 60)
    print(stats.sort_values("avg_rating", ascending=False).head(10).round(2).to_string(index=False))

    print("\n" + "=" * 60)
    print("LOWEST 10 RATED CITIES (>=10 restaurants)")
    print("=" * 60)
    print(stats.sort_values("avg_rating", ascending=True).head(10).round(2).to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 15 CUISINES OVERALL")
    print("=" * 60)
    print(cuisine_counts(df).head(15))

    ncr_df = df[df["City"].isin(NCR_CITIES)]
    print("\n" + "=" * 60)
    print(f"NCR REGION: {len(ncr_df):,} restaurants ({len(ncr_df) / len(df):.1%} of dataset)")
    print("=" * 60)
    print(
        ncr_df.groupby("City")
        .agg(count=("Restaurant ID", "count"), avg_rating=("Aggregate rating", "mean"))
        .round(2)
    )

    top10_cities = stats.head(10)["City"].tolist()
    print("\n" + "=" * 60)
    print("ONLINE DELIVERY / TABLE BOOKING, TOP 10 CITIES")
    print("=" * 60)
    print(delivery_and_booking_by_city(df, top10_cities))

    print(f"\nRows with missing (0,0) coordinates: {len(df) - len(geo):,}")

    return stats


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stats = run_summary()
    out_path = os.path.join(OUTPUTS_DIR, "city_stats.csv")
    stats.to_csv(out_path, index=False)
    print(f"\nSaved city-level stats to {out_path}")


if __name__ == "__main__":
    main()
