"""Build the standalone HTML report (images embedded as base64) from
outputs/city_stats.csv and images/*.png.

    python src/generate_report.py
"""

import base64
import os

import pandas as pd

from constants import IMAGES_DIR, OUTPUTS_DIR, REPORTS_DIR

CHART_NAMES = [
    "world_map",
    "india_map",
    "top_cities_count",
    "rating_vs_cost",
    "top_cuisines",
    "price_range",
    "kmeans_selection",
    "kmeans_region_clusters",
    "dbscan_hotspots",
    "feature_importance",
    "predicted_vs_actual",
]


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _rows(df: pd.DataFrame, cols: list[str]) -> str:
    lines = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{r[c]}</td>" for c in cols)
        lines.append(f"<tr>{cells}</tr>")
    return "\n".join(lines)


def build_report() -> str:
    imgs = {name: _b64(os.path.join(IMAGES_DIR, f"{name}.png")) for name in CHART_NAMES}
    stats = pd.read_csv(os.path.join(OUTPUTS_DIR, "city_stats.csv"))

    top15 = stats.sort_values("restaurant_count", ascending=False).head(15).round(2)
    top_rated = (
        stats[stats.restaurant_count >= 10].sort_values("avg_rating", ascending=False).head(10).round(2)
    )

    top15_rows = _rows(top15, ["City", "restaurant_count", "avg_rating", "avg_cost_for_two", "avg_price_range", "unique_cuisines"])
    top_rated_rows = _rows(top_rated, ["City", "restaurant_count", "avg_rating", "avg_votes", "avg_cost_for_two"])

    # ML outputs (from src/clustering.py and src/predict_rating.py)
    hotspots = pd.read_csv(os.path.join(OUTPUTS_DIR, "dbscan_hotspots.csv")).round(2)
    hotspot_rows = _rows(
        hotspots.head(8)[["hotspot_cluster", "restaurant_count", "avg_rating", "avg_cost_for_two", "top_city", "top_locality"]],
        ["hotspot_cluster", "restaurant_count", "avg_rating", "avg_cost_for_two", "top_city", "top_locality"],
    )
    model_comparison = pd.read_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv")).round(3)
    model_rows = _rows(model_comparison, ["model", "test_mae", "test_rmse", "test_r2", "train_r2"])
    india_labels = pd.read_csv(os.path.join(OUTPUTS_DIR, "india_hotspot_labels.csv"))
    n_hotspots = india_labels[india_labels.hotspot_cluster != -1]["hotspot_cluster"].nunique()
    n_noise = (india_labels.hotspot_cluster == -1).sum()
    pct_noise = n_noise / len(india_labels) * 100
    best_rf_r2 = model_comparison.loc[model_comparison.model == "Random Forest", "test_r2"].iloc[0]
    best_rf_mae = model_comparison.loc[model_comparison.model == "Random Forest", "test_mae"].iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Restaurant Location Analysis</title>
<style>
  :root {{
    --bg: #fbfaf8; --fg: #1f2328; --muted: #5b6470; --card: #ffffff;
    --border: #e6e2da; --accent: #b5502c; --accent2: #2e6f6f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #17181a; --fg: #e9e6df; --muted: #a3a9b2; --card: #1f2124; --border: #33363a;
      --accent: #e08a5f; --accent2: #6ec4c4;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #17181a; --fg: #e9e6df; --muted: #a3a9b2; --card: #1f2124; --border: #33363a;
    --accent: #e08a5f; --accent2: #6ec4c4;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--fg); margin:0; padding: 0 0 4rem 0; font-family: Georgia, 'Times New Roman', serif; line-height: 1.6; }}
  header {{ padding: 3rem 1.5rem 2rem; max-width: 880px; margin: 0 auto; }}
  header h1 {{ font-size: 2.1rem; margin: 0 0 .4rem; font-weight: 600; letter-spacing: -0.01em; }}
  header p.subtitle {{ color: var(--muted); font-size: 1rem; margin: 0; }}
  main {{ max-width: 880px; margin: 0 auto; padding: 0 1.5rem; }}
  section {{ margin: 2.6rem 0; }}
  h2 {{ font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 1.05rem; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: .5rem; }}
  p {{ font-size: 1.02rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem 1.4rem; margin: 1.2rem 0; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; text-align: center; }}
  .stat .num {{ font-size: 1.6rem; font-weight: 700; color: var(--accent2); font-family: -apple-system, Helvetica, Arial, sans-serif;}}
  .stat .label {{ font-size: .8rem; color: var(--muted); margin-top: .2rem; font-family: -apple-system, Helvetica, Arial, sans-serif;}}
  img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); display:block; margin: 1rem auto; }}
  figcaption {{ text-align:center; color: var(--muted); font-size: .85rem; margin-top:-.5rem; font-family: -apple-system, Helvetica, Arial, sans-serif;}}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; font-family: -apple-system, Helvetica, Arial, sans-serif; }}
  th, td {{ text-align: left; padding: .5rem .7rem; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th {{ color: var(--muted); font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
  .insight {{ border-left: 3px solid var(--accent); padding: .3rem 0 .3rem 1rem; margin: 1rem 0; color: var(--fg); }}
  footer {{ text-align:center; color: var(--muted); font-size: .8rem; margin-top: 3rem; font-family: -apple-system, Helvetica, Arial, sans-serif; }}
</style>
</head>
<body>
<header>
  <h1>Where the Restaurants Are: A Geographic Analysis</h1>
  <p class="subtitle">9,551 restaurants across 15 countries &middot; EDA, KMeans/DBSCAN clustering, and a rating-prediction model</p>
</header>
<main>

<section>
  <h2>Snapshot</h2>
  <div class="stat-grid">
    <div class="stat"><div class="num">9,551</div><div class="label">Restaurants</div></div>
    <div class="stat"><div class="num">15</div><div class="label">Countries</div></div>
    <div class="stat"><div class="num">83.2%</div><div class="label">In the Delhi NCR region</div></div>
    <div class="stat"><div class="num">{n_hotspots}</div><div class="label">DBSCAN hotspots found (India)</div></div>
    <div class="stat"><div class="num">{best_rf_r2:.0%}</div><div class="label">Rating variance explained (RF, test R&sup2;)</div></div>
  </div>
  <p>The dataset is a global sample, but it is overwhelmingly an <strong>India dataset</strong>: 8,652 of 9,551 restaurants (90.6%) are in India, and within India, restaurants cluster almost entirely in the Delhi National Capital Region (New Delhi, Gurgaon, Noida, Faridabad, Ghaziabad). Every other country contributes a flat, even sample &mdash; almost exactly 20 restaurants each &mdash; which looks like a deliberate "one city, 20 restaurants" benchmark set layered on top of a much deeper India/NCR dataset. 499 rows have (0,0) coordinates and were excluded from the maps below.</p>
</section>

<section>
  <h2>1. Geographic Distribution</h2>
  <figure>
    <img src="data:image/png;base64,{imgs['world_map']}" alt="World scatter map of restaurants by country">
    <figcaption>Every restaurant with valid coordinates, colored by country. India dominates entirely; the other 14 countries form small, tight clusters around single cities each.</figcaption>
  </figure>
  <p>Because India accounts for over 90% of the rows, zooming into it reveals the real structure of the data:</p>
  <figure>
    <img src="data:image/png;base64,{imgs['india_map']}" alt="Zoomed map of Delhi NCR restaurant density">
    <figcaption>Delhi NCR restaurants (New Delhi, Gurgaon, Noida, Faridabad, Ghaziabad) vs. the rest of India. The five NCR cities alone account for 83% of the entire dataset.</figcaption>
  </figure>
  <div class="insight">New Delhi alone contributes 57% of all rows in the dataset (5,473 of 9,551) &mdash; more than every other city and country combined except the rest of NCR.</div>
</section>

<section>
  <h2>2. Concentration by City</h2>
  <figure>
    <img src="data:image/png;base64,{imgs['top_cities_count']}" alt="Bar chart of top 15 cities by restaurant count">
    <figcaption>Top 15 cities by number of restaurants. The drop-off after the four NCR cities is steep &mdash; the 5th-largest city (Ghaziabad) has 25 restaurants vs. Faridabad's 251.</figcaption>
  </figure>
  <div class="table-wrap">
  <table>
    <tr><th>City</th><th>Restaurants</th><th>Avg Rating</th><th>Avg Cost for Two</th><th>Avg Price Range</th><th>Unique Cuisines</th></tr>
    {top15_rows}
  </table>
  </div>
</section>

<section>
  <h2>3. Ratings, Cost, and Price by Location</h2>
  <figure>
    <img src="data:image/png;base64,{imgs['rating_vs_cost']}" alt="Scatter plot of average rating vs average cost by city">
    <figcaption>Each bubble is a city (&ge;15 restaurants). Bubble size = restaurant count, color = average price range. Cost axis is log-scaled because currencies aren't normalized across countries.</figcaption>
  </figure>
  <p>The clearest pattern: the four dominant NCR cities (New Delhi, Gurgaon, Noida, Faridabad) sit in the <strong>lower-left</strong> &mdash; high volume, but the lowest average ratings in the whole dataset. Smaller, less saturated markets rate much higher on average:</p>
  <div class="table-wrap">
  <table>
    <tr><th>City</th><th>Restaurants</th><th>Avg Rating</th><th>Avg Votes</th><th>Avg Cost for Two</th></tr>
    {top_rated_rows}
  </table>
  </div>
  <div class="insight">Faridabad has both the lowest average rating (1.87) and the fewest average votes (26) among cities with 10+ restaurants &mdash; consistent with an under-reviewed, saturated market rather than genuinely bad food.</div>
</section>

<section>
  <h2>4. Cuisines by Location</h2>
  <figure>
    <img src="data:image/png;base64,{imgs['top_cuisines']}" alt="Bar chart of top 15 cuisines overall">
    <figcaption>North Indian and Chinese dominate overall &mdash; a direct reflection of the dataset's India/NCR skew rather than a global cuisine trend.</figcaption>
  </figure>
  <p>Cuisine variety tracks city size closely: New Delhi offers 81 distinct cuisine types, Gurgaon 71, Noida 49 &mdash; while most 20-restaurant sample cities outside India offer 15&ndash;25. Delivery infrastructure also varies sharply by city: <strong>38% of Gurgaon restaurants</strong> offer online delivery vs. <strong>0%</strong> in Amritsar, Bhubaneshwar, Lucknow, and Guwahati in this sample.</p>
</section>

<section>
  <h2>5. Price Range</h2>
  <figure>
    <img src="data:image/png;base64,{imgs['price_range']}" alt="Bar chart of price range distribution">
    <figcaption>Price range 1 (cheapest) accounts for nearly half of all restaurants; premium (4) is a small minority.</figcaption>
  </figure>
</section>

<section>
  <h2>6. Machine Learning: Discovering Location Hotspots (Unsupervised)</h2>
  <p>Rather than hand-picking "NCR cities" as the unit of analysis, two unsupervised methods were applied directly to (latitude, longitude):</p>
  <p><strong>KMeans</strong> on all 9,052 geolocated restaurants, with <em>k</em> chosen by silhouette score across k=3&ndash;10 rather than picked arbitrarily:</p>
  <figure>
    <img src="data:image/png;base64,{imgs['kmeans_selection']}" alt="Silhouette score and inertia by k">
    <figcaption>Silhouette score peaks at k=4, so that's the k used below. Inertia (inside the elbow curve) keeps dropping past k=4, which is expected &mdash; silhouette, not the elbow alone, is what decides cluster count here.</figcaption>
  </figure>
  <figure>
    <img src="data:image/png;base64,{imgs['kmeans_region_clusters']}" alt="KMeans regional clusters map">
    <figcaption>The 4 clusters KMeans finds on raw coordinates align almost exactly with country boundaries &mdash; a sanity check that the clustering is picking up real geographic structure, not noise.</figcaption>
  </figure>
  <p><strong>DBSCAN</strong> was then run within India specifically (eps &asymp; 1km, min_samples=8) to find dense micro-hotspots at street/neighborhood scale, which KMeans can't do (KMeans assumes round, evenly-sized clusters; DBSCAN finds arbitrary-shaped dense regions and explicitly flags sparse restaurants as noise instead of forcing them into a cluster):</p>
  <figure>
    <img src="data:image/png;base64,{imgs['dbscan_hotspots']}" alt="DBSCAN hotspot clusters within Delhi NCR">
    <figcaption>DBSCAN found {n_hotspots} distinct hotspots among 8,156 Indian restaurants; {n_noise} restaurants ({pct_noise:.1f}%) are sparse outliers not part of any hotspot.</figcaption>
  </figure>
  <div class="table-wrap">
  <table>
    <tr><th>Cluster ID</th><th>Restaurants</th><th>Avg Rating</th><th>Avg Cost for Two</th><th>City</th><th>Representative Locality</th></tr>
    {hotspot_rows}
  </table>
  </div>
  <div class="insight">The single biggest hotspot (2,512 restaurants, centered on Connaught Place, New Delhi) rates 2.85 on average &mdash; but the small 102-restaurant Mahipalpur hotspot (near the airport) rates just 1.43, the worst of any hotspot found. Density alone doesn't predict quality; DBSCAN surfaces specific problem neighborhoods that a city-level average would hide.</div>
</section>

<section>
  <h2>7. Machine Learning: Predicting Rating (Supervised)</h2>
  <p>A Random Forest Regressor and a Linear Regression baseline were trained to predict <code>Aggregate rating</code> from Votes, Average Cost for two, Price range, cuisine count, delivery/booking flags, city, and the DBSCAN hotspot cluster &mdash; on a proper 80/20 train/test split (2,148 "Not rated" restaurants excluded from training entirely, since a 0 there means "no rating," not "bad rating").</p>
  <div class="table-wrap">
  <table>
    <tr><th>Model</th><th>Test MAE</th><th>Test RMSE</th><th>Test R&sup2;</th><th>Train R&sup2;</th></tr>
    {model_rows}
  </table>
  </div>
  <figure>
    <img src="data:image/png;base64,{imgs['predicted_vs_actual']}" alt="Predicted vs actual rating scatter plot">
    <figcaption>Random Forest predictions vs. actual test-set ratings. Points near the dashed line are accurate; the model does reasonably at mid-range ratings and is less confident at the extremes, which is typical when few examples exist there.</figcaption>
  </figure>
  <figure>
    <img src="data:image/png;base64,{imgs['feature_importance']}" alt="Random Forest feature importance bar chart">
    <figcaption>Vote count dominates: how many people have rated a restaurant is a far stronger predictor of its rating than cost, price range, or location. This likely reflects a popularity/visibility bias in how ratings accumulate, more than a location effect.</figcaption>
  </figure>
  <div class="insight">The Random Forest explains {best_rf_r2:.0%} of the variance in test-set ratings (MAE &asymp; {best_rf_mae:.2f} stars) and clearly outperforms plain Linear Regression &mdash; but <strong>Votes alone</strong> carries most of that signal. Location features (city, hotspot cluster) matter, but far less than a restaurant's popularity. This is a useful, honest finding: geography helps explain <em>where</em> restaurants cluster, but it's a weak predictor of <em>quality</em> on its own.</div>
</section>

<section>
  <h2>Key Insights</h2>
  <div class="card">
  <ul>
    <li><strong>Extreme geographic skew:</strong> This isn't really a "global" restaurant dataset &mdash; it's a deep Delhi NCR dataset (India) with a shallow, uniform 20-restaurants-per-city sample layered on top from 14 other countries, likely added for benchmarking coverage rather than depth.</li>
    <li><strong>Saturation correlates with lower ratings:</strong> The four highest-volume cities (New Delhi, Gurgaon, Noida, Faridabad) have the four lowest average ratings among all cities with 10+ restaurants. More competition doesn't translate to higher quality scores in this data &mdash; it may reflect harsher/more numerous reviewers, market maturity, or genuine oversaturation.</li>
    <li><strong>Small markets rate higher, with fewer votes:</strong> Cities like London, Orlando, Tampa Bay, and Abu Dhabi (20 restaurants each) post average ratings above 4.3, but this is a much smaller, likely curated sample &mdash; worth treating as directional, not definitive.</li>
    <li><strong>Cuisine diversity scales with restaurant density</strong>, not geography or country wealth &mdash; it's simply a function of how many restaurants a city has in the sample.</li>
    <li><strong>Delivery/booking infrastructure is city-specific</strong>, not nationally uniform: within the same country (India), online delivery adoption ranges from 0% to 52% depending on the city.</li>
    <li><strong>Data quality note:</strong> 499 restaurants (5.2%) have (0,0) placeholder coordinates and were excluded from all maps; a handful of city names (e.g. Bras&iacute;lia, S&atilde;o Paulo) show character-encoding corruption from the source file and should be cleaned before any production use.</li>
    <li><strong>Unsupervised clustering confirms the geographic skew quantitatively:</strong> KMeans' silhouette-optimal split of the entire dataset is just k=4, and those 4 clusters map almost one-to-one onto country borders &mdash; there's so little geographic variety outside India that a global algorithm can't find more structure than that.</li>
    <li><strong>DBSCAN finds quality problems city-level stats hide:</strong> the worst-rated hotspot (Mahipalpur, 1.43 avg rating) is a small, dense cluster near Delhi airport &mdash; a pattern invisible in a city-wide "New Delhi" average but obvious once density-based clustering isolates it.</li>
    <li><strong>Location is a weak rating predictor on its own:</strong> a Random Forest using votes, cost, cuisine, delivery flags, city, and hotspot cluster explains {best_rf_r2:.0%} of rating variance (MAE {best_rf_mae:.2f}) &mdash; but vote count alone accounts for most of that. Where a restaurant is matters far less to its rating than how many people have reviewed it.</li>
  </ul>
  </div>
</section>

<footer>Restaurant_Dataset.xlsx &middot; 9,551 rows &middot; analysis generated with pandas &amp; matplotlib</footer>

</main>
</body>
</html>
"""
    return html


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    html = build_report()
    out_path = os.path.join(REPORTS_DIR, "restaurant_location_analysis.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
