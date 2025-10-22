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
# 2. Prepare numeric features
# ---------------------------------------------------------------
def prepare_features(df, target_col="views"):
    df_num = df.select_dtypes(include=["number"]).copy()

    # Ensure target column is numeric
    if target_col not in df_num.columns and target_col in df.columns:
        df_num[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    y = df_num[target_col].replace([np.inf, -np.inf, np.nan], np.nan).dropna()
    X = df_num.loc[y.index].drop(columns=[target_col])
    return X, y

X_scraped, y_scraped = prepare_features(df_scraped, "views")
print(f"Numeric features: {X_scraped.shape[1]} | Target samples: {len(y_scraped)}")

# ---------------------------------------------------------------
# 3. Clean & log-transform target variable
# ---------------------------------------------------------------
y_scraped_clean = y_scraped.replace([np.inf, -np.inf], np.nan).dropna()
valid_idx = y_scraped_clean.index
X_scraped_clean = X_scraped.loc[valid_idx]
y_scraped_log = np.log1p(y_scraped_clean)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scraped_clean, y_scraped_log, test_size=0.2, random_state=42
)

print(f"Target range: min={y_scraped_clean.min():.0f}, max={y_scraped_clean.max():.0f}")
print(f"Train/Test split → {X_train.shape}, {X_test.shape}")

# ---------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
xgb = XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.1, max_depth=6)

# ---------------------------------------------------------------
# 5. Train & evaluate function
# ---------------------------------------------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds_log = model.predict(X_test)

    # Replace invalids before reversing log transform
    preds_log = np.nan_to_num(preds_log, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.3f}")
    return preds, rmse, r2

# ---------------------------------------------------------------
# 6. Train models
# ---------------------------------------------------------------
print("Training models on scraped data...")
train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest (Scraped)")
train_and_evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost (Scraped)")
