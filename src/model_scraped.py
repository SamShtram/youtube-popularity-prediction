# ===============================================================
# YOUTUBE POPULARITY PREDICTION (SCRAPED DATASET MODEL)
# ===============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Load cleaned scraped dataset
# ---------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_clean.csv")
df_scraped = pd.read_csv(DATA_PATH)

print(f"Loaded scraped dataset: {df_scraped.shape[0]} rows, {df_scraped.shape[1]} columns")

# ---------------------------------------------------------------
# 2. Clean numeric-looking columns and prepare features
# ---------------------------------------------------------------
def prepare_features(df, target_col="views"):
    df = df.copy()

    # Try to convert all potentially numeric columns
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+\.?\d*)")[0]
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns that are fully empty after conversion
    df = df.dropna(axis=1, how="all")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found after cleaning.")

    y = df[target_col].replace([np.inf, -np.inf], np.nan).dropna()
    X = df.loc[y.index].drop(columns=[target_col])
    X = X.fillna(0)

    return X, y


X_scraped, y_scraped = prepare_features(df_scraped, "views")
print(f"Numeric features: {X_scraped.shape[1]} | Target samples: {len(y_scraped)}")

# ---------------------------------------------------------------
# 3. Log-transform target and split data
# ---------------------------------------------------------------
y_scraped_log = np.log1p(y_scraped)
X_train, X_test, y_train, y_test = train_test_split(
    X_scraped, y_scraped_log, test_size=0.2, random_state=42
)

print(f"Train/Test split → {X_train.shape}, {X_test.shape}")

# ---------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10
