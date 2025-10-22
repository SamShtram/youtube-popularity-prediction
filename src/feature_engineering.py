import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

# === Load cleaned datasets ===
SCRAPED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_clean.csv")
API_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_clean.csv")

df_scraped = pd.read_csv(SCRAPED_PATH)
df_api = pd.read_csv(API_PATH)

# === Helper 1: Convert ISO 8601 duration (PT5M33S → minutes) ===
def convert_duration(duration_str):
    if not isinstance(duration_str, str) or not duration_str.startswith("PT"):
        return np.nan
    m = re.search(r"(\d+)M", duration_str)
    s = re.search(r"(\d+)S", duration_str)
    minutes = int(m.group(1)) if m else 0
    seconds = int(s.group(1)) if s else 0
    return round(minutes + seconds / 60, 2)

# === Helper 2: Safe datetime conversion ===
def to_datetime_safe(val):
    try:
        dt = pd.to_datetime(val, errors="coerce", utc=True)
        if dt is not pd.NaT:
            return dt
    except Exception:
        pass
    return pd.NaT

# === Helper 3: Keyword density ===
def keyword_density(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b[a-zA-Z]{3,}\b", text))

# === Apply transformations ===
for name, df in [("Scraped", df_scraped), ("API", df_api)]:
    print(f"Processing {name} dataset...")

    # Duration conversion
    if "duration" in df.columns:
        df["duration_mins"] = df["duration"].apply(convert_duration)

    # Upload date and days since upload
    if "upload_date" in df.columns:
        df["upload_date"] = df["upload_date"].apply(to_datetime_safe)

        # Normalize to timezone-naive consistently (for arithmetic)
        df["upload_date"] = df["upload_date"].dt.tz_localize(None)

        # Always compare naive vs. naive timestamps
        now = pd.Timestamp.now(tz=None)
        df["days_since_upload"] = (now - df["upload_date"]).dt.days

    # Engagement rate
    if all(col in df.columns for col in ["likes", "comments", "views"]):
        df["engagement_rate"] = (
            (pd.to_numeric(df["likes"], errors="coerce") + pd.to_numeric(df["comments"], errors="coerce"))
            / pd.to_numeric(df["views"], errors="coerce")
        ).replace([np.inf, -np.inf, np.nan], 0)

    # Keyword counts
    if "title" in df.columns:
        df["title_keyword_count"] = df["title"].apply(keyword_density)
    if "description" in df.columns:
        df["desc_keyword_count"] = df["description"].apply(keyword_density)

print("Feature engineering complete.")

# === Save engineered versions ===
FE_SCRAPED = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_features.csv")
FE_API = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_features.csv")
df_scraped.to_csv(FE_SCRAPED, index=False)
df_api.to_csv(FE_API, index=False)

print("Feature-engineered datasets saved to /data/")
