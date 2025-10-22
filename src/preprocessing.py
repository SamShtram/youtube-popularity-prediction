import os
import pandas as pd
import numpy as np

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

# === Convert numeric columns ===
numeric_cols = ["views", "likes", "comments"]
for name, df in [("Scraped", df_scraped), ("API", df_api)]:
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    print(f"{name} dataset: standardized numeric columns {', '.join([c for c in numeric_cols if c in df.columns])}")

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

print("Preprocessing complete. Cleaned CSVs saved to /data/")
