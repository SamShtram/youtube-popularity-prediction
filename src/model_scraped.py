# ==========================================================
# 📊 YOUTUBE POPULARITY PREDICTION (SCRAPED DATASET MODEL)
# ==========================================================
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# 1️⃣ Load cleaned scraped dataset
# ----------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_clean.csv")
df = pd.read_csv(DATA_PATH)

print(f"✅ Loaded scraped dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------
# 2️⃣ Convert columns to numeric where possible
# ----------------------------------------------------------
def clean_views(value):
    """Convert view strings like '1,234,567' or '1 234 567' safely to int."""
    if pd.isna(value):
        return np.nan
    value = str(value)
    # Replace all known separators and invisible spaces
    for ch in [",", " ", "\u202f", "\xa0", "\u00a0"]:
        value = value.replace(ch, "")
    # Keep only digits
    value = re.sub(r"[^0-9]", "", value)
    if not value.isdigit():
        return np.nan
    return int(value)
    if "views" in df.columns:
        df["views"] = df["views"].apply(clean_views)
        print("\n🔍 Sample of cleaned 'views' column:")
        print(df["views"].head(10).tolist())





def duration_to_minutes(value):
    """Convert duration '3:59' or '1:02:45' to minutes."""
    if not isinstance(value, str) or ":" not in value:
        return np.nan
    parts = list(map(int, value.split(":")))
    if len(parts) == 3:
        return parts[0] * 60 + parts[1] + parts[2] / 60
    elif len(parts) == 2:
        return parts[0] + parts[1] / 60
    return np.nan

# Apply conversions
if "views" in df.columns:
    df["views"] = df["views"].apply(clean_views)

if "duration" in df.columns:
    df["duration_mins"] = df["duration"].apply(duration_to_minutes)

# Keep only numeric columns (views + duration_mins)
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if not numeric_cols:
    raise ValueError("❌ No numeric columns found after cleaning!")

print(f"✅ Numeric columns detected: {numeric_cols}")

# ----------------------------------------------------------
# 3️⃣ Prepare features and target
# ----------------------------------------------------------
target_col = "views"
if target_col not in df.columns:
    raise ValueError("❌ 'views' column not found in dataset!")

df = df.dropna(subset=[target_col])
y = df[target_col]
X = df[[col for col in numeric_cols if col != target_col]]

if X.empty:
    raise ValueError("❌ No numeric feature columns available for training!")

print(f"✅ Numeric features: {X.shape[1]} | Target samples: {len(y)}")

# ----------------------------------------------------------
# 4️⃣ Train/test split
# ----------------------------------------------------------
y_log = np.log1p(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

print(f"✅ Target range: min={y.min():,.0f}, max={y.max():,.0f}")
print(f"✅ Train/Test split → {X_train.shape}, {X_test.shape}")

# ----------------------------------------------------------
# 5️⃣ Define models
# ----------------------------------------------------------
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
xgb = XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.1, max_depth=6)

# ----------------------------------------------------------
# 6️⃣ Train & evaluate
# ----------------------------------------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds_log = model.predict(X_test)
    preds_log = np.nan_to_num(preds_log, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"📊 {name} → RMSE: {rmse:,.0f}, R²: {r2:.3f}")
    return preds, rmse, r2

print("\n🚀 Training models on scraped data...")
train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest (Scraped)")
train_and_evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost (Scraped)")
