import pandas as pd

from constants import COUNTRY_CODE_MAP, DATA_PATH

def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Excel file into a DataFrame."""
    return pd.read_excel(path)

def add_country_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map numeric Country Code -> readable Country name."""
    df = df.copy()
    df["Country"] = (
        df["Country Code"].map(COUNTRY_CODE_MAP).fillna(df["Country Code"].astype(str))
    )
    return df

def load_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and attach a readable Country column."""
    return add_country_names(load_raw(path))

def load_geo(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset, keeping only rows with valid (non-zero) coordinates."""
    df = load_clean(path)
    return df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()

if __name__ == "__main__":
    df = load_clean()
    print(f"Loaded {len(df):,} restaurants across {df['Country'].nunique()} countries.")
    print(f"Rows with valid coordinates: {len(load_geo()):,}")
