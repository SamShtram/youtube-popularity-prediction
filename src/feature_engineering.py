import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

# === Paths ===
SCRAPED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_clean.csv")
API_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_clean.csv")

df_scraped = pd.read_csv(SCRAPED_PATH)
df_api = pd.read_csv(API_PATH)

# -------------------------------------------------------
# 🧹 Utility functions
# -------------------------------------------------------
def convert_duration(duration_str):
    """Convert PT#M#S or MM:SS to float minutes."""
    if pd.isna(duration_str):
        return np.nan
    s = str(duration_str).strip()
    if s.startswith("PT"):
        m = re.search(r"(\d+)M", s)
        sec = re.search(r"(\d+)S", s)
        return (int(m.group(1)) if m else 0) + (int(sec.group(1)) if sec else 0) / 60
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60
        except:
            return np.nan
    return np.nan

def to_datetime_safe(val):
    try:
        return pd.to_datetime(val, errors="coerce").tz_localize(None)
    except Exception:
        return pd.NaT

def keyword_density(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b[a-zA-Z]{3,}\b", text))

# -------------------------------------------------------
# ⚙️ Feature functions
# -------------------------------------------------------
def basic_text_features(df):
    df["title_length"] = df["title"].astype(str).apply(len)
    df["word_count_title"] = df["title"].astype(str).apply(lambda x: len(x.split()))
    df["has_music_keyword"] = df["title"].str.contains("music|official|video|remix", case=False, regex=True).astype(int)
    if "description" in df.columns:
        df["desc_keyword_count"] = df["description"].apply(keyword_density)
    return df

def time_features(df):
    if "upload_date" in df.columns:
        df["upload_date"] = df["upload_date"].apply(to_datetime_safe)
        now = pd.Timestamp.now()
        df["days_since_upload"] = (now - df["upload_date"]).dt.days
        df["upload_year"] = df["upload_date"].dt.year
        df["upload_month"] = df["upload_date"].dt.month
        df["upload_weekday"] = df["upload_date"].dt.weekday
    return df

def engagement_features(df):
    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if all(c in df.columns for c in ["likes", "comments", "views"]):
        df["engagement_rate"] = (df["likes"] + df["comments"]) / (df["views"] + 1)
        df["likes_to_views"] = df["likes"] / (df["views"] + 1)
        df["comments_to_views"] = df["comments"] / (df["views"] + 1)
        df["likes_to_comments"] = df["likes"] / (df["comments"] + 1)
    return df

def log_and_ratio_features(df):
    for col in ["views", "likes", "comments", "days_since_upload"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    return df

# -------------------------------------------------------
# 🚀 Apply to both datasets
# -------------------------------------------------------
for name, df in [("Scraped", df_scraped), ("API", df_api)]:
    print(f"Processing {name} dataset...")
    if "duration" in df.columns:
        df["duration_mins"] = df["duration"].apply(convert_duration)
    df = basic_text_features(df)
    df = time_features(df)
    df = engagement_features(df)
    df = log_and_ratio_features(df)
    if "tags" in df.columns:
        df["tag_count"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")) if "|" in x else len(x.split(",")))

print("✅ Feature engineering complete.")

# === Save engineered datasets ===
FE_SCRAPED = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_features.csv")
FE_API = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_features.csv")
df_scraped.to_csv(FE_SCRAPED, index=False)
df_api.to_csv(FE_API, index=False)

print("Feature-engineered datasets saved to /data/")
