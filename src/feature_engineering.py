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
    if pd.isna(duration_str):
        return np.nan
    if isinstance(duration_str, (int, float)):
        return duration_str
    duration_str = str(duration_str).strip()
    # Handle API format (PT5M33S) or normal MM:SS
    if duration_str.startswith("PT"):
        m = re.search(r"(\d+)M", duration_str)
        s = re.search(r"(\d+)S", duration_str)
        minutes = int(m.group(1)) if m else 0
        seconds = int(s.group(1)) if s else 0
        return round(minutes + seconds / 60, 2)
    elif ":" in duration_str:
        parts = duration_str.split(":")
        try:
            minutes, seconds = int(parts[0]), int(parts[1])
            return round(minutes + seconds / 60, 2)
        except:
            return np.nan
    return np.nan

# === Helper 2: Safe datetime conversion ===
def to_datetime_safe(val):
    try:
        return pd.to_datetime(val, errors="coerce", utc=True).tz_localize(None)
    except Exception:
        return pd.NaT

# === Helper 3: Keyword density ===
def keyword_density(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b[a-zA-Z]{3,}\b", text))

# === Helper 4: Title feature extraction ===
def title_features(df):
    df["title_length"] = df["title"].astype(str).apply(len)
    df["word_count_title"] = df["title"].astype(str).apply(lambda x: len(x.split()))
    df["has_music_keyword"] = df["title"].str.contains("music|official|video|remix", case=False, regex=True).astype(int)
    return df

# === Helper 5: Temporal features ===
def temporal_features(df):
    if "upload_date" in df.columns:
        df["upload_date"] = df["upload_date"].apply(to_datetime_safe)
        now = pd.Timestamp.now()
        df["days_since_upload"] = (now - df["upload_date"]).dt.days
        df["upload_year"] = df["upload_date"].dt.year
        df["upload_month"] = df["upload_date"].dt.month
        df["upload_weekday"] = df["upload_date"].dt.weekday
    return df

# === Helper 6: Engagement ===
def engagement_features(df):
    for col in ["likes", "comments", "views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if all(c in df.columns for c in ["likes", "comments", "views"]):
        df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(0, np.nan)
        df["engagement_rate"] = df["engagement_rate"].replace([np.inf, -np.inf, np.nan], 0)
    return df

# === Main processing loop ===
for name, df in [("Scraped", df_scraped), ("API", df_api)]:
    print(f"Processing {name} dataset...")

    df["duration_mins"] = df["duration"].apply(convert_duration)
    df = title_features(df)
    df = temporal_features(df)
    df = engagement_features(df)

    if "description" in df.columns:
        df["desc_keyword_count"] = df["description"].apply(keyword_density)
    if "tags" in df.columns:
        df["tag_count"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")) if "|" in x else len(x.split(",")))

print("Feature engineering complete.")

# === Save engineered versions ===
FE_SCRAPED = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_features.csv")
FE_API = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_features.csv")

df_scraped.to_csv(FE_SCRAPED, index=False)
df_api.to_csv(FE_API, index=False)

print("✅ Feature-engineered datasets saved to /data/")
