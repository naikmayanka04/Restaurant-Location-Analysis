"""Shared constants for the restaurant location analysis."""

# The dataset stores countries as numeric codes (Zomato dataset convention).
COUNTRY_CODE_MAP = {
    1: "India",
    14: "Australia",
    30: "Brazil",
    37: "Canada",
    94: "Indonesia",
    148: "New Zealand",
    162: "Philippines",
    166: "Qatar",
    184: "Singapore",
    189: "South Africa",
    191: "Sri Lanka",
    208: "Turkey",
    214: "UAE",
    215: "United Kingdom",
    216: "United States",
}

# Delhi National Capital Region cities, treated as one metro area in parts
# of the analysis since they dominate the dataset (~83% of all rows).
NCR_CITIES = ["New Delhi", "Gurgaon", "Noida", "Faridabad", "Ghaziabad"]

DATA_PATH = "data/Restaurant_Dataset.xlsx"
OUTPUTS_DIR = "outputs"
IMAGES_DIR = "images"
REPORTS_DIR = "reports"
