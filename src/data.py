"""Dataset acquisition, validation, and demo-data generation."""
from pathlib import Path
import io, zipfile, urllib.request
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
DATA_PATH = DATA_DIR / "online_shoppers.csv"
URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"

def make_demo_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic rows solely to smoke-test the pipeline offline."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Administrative": rng.poisson(2, n), "Administrative_Duration": rng.gamma(1.5, 35, n),
        "Informational": rng.poisson(0.5, n), "Informational_Duration": rng.gamma(1.2, 20, n),
        "ProductRelated": rng.poisson(18, n), "ProductRelated_Duration": rng.gamma(2, 90, n),
        "BounceRates": rng.beta(1.2, 12, n), "ExitRates": rng.beta(1.5, 10, n),
        "PageValues": rng.gamma(1.0, 4, n), "SpecialDay": rng.choice([0, .2, .4, .6, .8, 1], n),
        "Month": rng.choice(["Jan","Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"], n),
        "OperatingSystems": rng.integers(1, 9, n), "Browser": rng.integers(1, 14, n),
        "Region": rng.integers(1, 10, n), "TrafficType": rng.integers(1, 21, n),
        "VisitorType": rng.choice(["Returning_Visitor","New_Visitor","Other"], n, p=[.7,.28,.02]),
        "Weekend": rng.choice([True, False], n),
    })
    logit = -2.8 + .04*df.ProductRelated + .012*df.ProductRelated_Duration - 7*df.BounceRates - 5*df.ExitRates + .10*df.PageValues + .7*(df.VisitorType == "Returning_Visitor")
    p = 1/(1+np.exp(-np.clip(logit, -10, 10)))
    df["Revenue"] = rng.random(n) < p
    df.attrs["synthetic"] = True
    return df

def load_data(allow_demo: bool = True) -> pd.DataFrame:
    """Load cached data, download official UCI archive, or create demo fallback."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df.attrs["synthetic"] = False
        return validate_data(df)
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "SWYNEX-Student-Project/1.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(archive.open(csv_name))
        df.to_csv(DATA_PATH, index=False)
        df.attrs["synthetic"] = False
        return validate_data(df)
    except Exception as exc:
        if not allow_demo:
            raise RuntimeError(f"Could not download UCI dataset: {exc}") from exc
        print(f"WARNING: UCI download failed ({exc}). Using synthetic demo data; do not report its metrics as real.")
        df = make_demo_data()
        df.attrs["synthetic"] = True
        return validate_data(df)

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Dataset is empty")
    if "Revenue" not in df.columns:
        raise ValueError("Expected target column 'Revenue' was not found")
    df = df.drop_duplicates().copy()
    df["Revenue"] = df["Revenue"].map(_to_bool)
    df = df.dropna(subset=["Revenue"])
    if df["Revenue"].nunique() < 2:
        raise ValueError("Target must contain both purchase and non-purchase sessions")
    return df

def _to_bool(value):
    if isinstance(value, (bool, np.bool_)): return int(value)
    if str(value).strip().lower() in {"true", "1", "yes"}: return 1
    if str(value).strip().lower() in {"false", "0", "no"}: return 0
    return np.nan
