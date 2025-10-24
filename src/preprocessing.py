import os
import pandas as pd
import numpy as np
import re
import unicodedata

# === Load raw CSVs ===
SCRAPED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_raw.csv")
API_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_raw.csv")

if not os.path.exists(SCRAPED_PATH) or not os.path.exists(API_PATH):
    raise FileNotFoundError("Raw CSV files not found. Make sure both data/youtube_scraped_raw.csv and data/youtube_api_raw.csv exist.")

df_scraped = pd.read_csv(SCRAPED_PATH)
df_api = pd.read_csv(API_PATH)

# === Clean column names ===
df_scraped.columns = df_scraped.columns.str.strip().str.lower()
df_api.columns = df_api.columns.str.strip().str.lower()

# ----------------------------------------------------------
#  Helper: clean YouTube-style numbers
# ----------------------------------------------------------
def clean_views(value):
    """Convert '76,924,840 views' → 76924840"""
    if pd.isna(value):
        return np.nan
    s = str(value).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("views", "").replace(",", "").strip()
    digits = re.findall(r"\d+", s)
    if digits:
        return float(digits[0])
    return np.nan

# ----------------------------------------------------------
#  Apply cleaning to scraped dataset
# ----------------------------------------------------------
for col in ["views", "likes", "comments"]:
    if col in df_scraped.columns:
        df_scraped[col] = df_scraped[col].apply(clean_views).fillna(0)

# For API dataset (already numeric), we can safely convert:
for col in ["views", "likes", "comments"]:
    if col in df_api.columns:
        df_api[col] = pd.to_numeric(df_api[col], errors="coerce").fillna(0)

# === Fill missing non-numeric fields ===
fill_defaults = {
    "title": "Unknown Title",
    "channel": "Unknown Channel",
    "category": "Unknown",
    "upload_date": pd.NaT,
    "duration": "PT0S",
    "tags": ""
}
df_scraped = df_scraped.fillna(fill_defaults)
df_api = df_api.fillna(fill_defaults)

# === Save cleaned versions ===
CLEAN_SCRAPED = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_clean.csv")
CLEAN_API = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_clean.csv")

df_scraped.to_csv(CLEAN_SCRAPED, index=False)
df_api.to_csv(CLEAN_API, index=False)

print(" Preprocessing complete. Cleaned CSVs saved to /data/")
print(df_scraped[["views"]].head())
