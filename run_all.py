"""Run the full pipeline: analysis -> charts -> HTML report.

    python run_all.py
"""

import subprocess
import sys

STEPS = [
    "src/analysis.py",
    "src/clustering.py",
    "src/predict_rating.py",
    "src/charts.py",
    "src/generate_report.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n>>> Running {step}")
        result = subprocess.run([sys.executable, step])
        if result.returncode != 0:
            sys.exit(result.returncode)
    print("\nDone. See outputs/city_stats.csv, images/, and reports/restaurant_location_analysis.html")


if __name__ == "__main__":
    main()
