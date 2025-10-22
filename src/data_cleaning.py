import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# === Load feature-engineered datasets ===
SCRAPED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_features.csv")
API_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_features.csv")

df_scraped = pd.read_csv(SCRAPED_PATH)
df_api = pd.read_csv(API_PATH)

# === Function to clean and normalize ===
def preprocess_and_normalize(df, dataset_name):
    print(f"Cleaning and normalizing {dataset_name} dataset...")

    df = df.drop_duplicates(subset=["title", "channel"], keep="first").reset_index(drop=True)

    # Numeric columns
    numeric_cols = ["views", "likes", "comments", "duration_mins", "days_since_upload", "engagement_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[:, col] = df[col].fillna(df[col].median())

    # Text fields
    for col in ["title", "description", "channel"]:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna("Unknown")

    # Log-transform large numeric features
    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Standardize numeric features
    scale_cols = [c for c in ["log_views", "duration_mins", "days_since_upload", "engagement_rate"] if c in df.columns]
    if scale_cols:
        scaler = StandardScaler()
        df[[f"{c}_scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    print(f"{dataset_name} preprocessing complete. Rows: {len(df)}, Columns: {len(df.columns)}")
    return df

# === Apply to both datasets ===
df_scraped = preprocess_and_normalize(df_scraped, "Scraped")
df_api = preprocess_and_normalize(df_api, "API")

# === Save cleaned outputs ===
FINAL_SCRAPED = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_ready.csv")
FINAL_API = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_api_ready.csv")

df_scraped.to_csv(FINAL_SCRAPED, index=False)
df_api.to_csv(FINAL_API, index=False)

print("Final cleaned and normalized datasets saved to /data/")
