# ==========================================================
# 📊 YOUTUBE POPULARITY PREDICTION (SCRAPED DATASET MODEL)
# ==========================================================
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# 1️⃣ Load dataset
# ----------------------------------------------------------
path_scraped = "data/youtube_scraped_clean.csv"
df = pd.read_csv(path_scraped)
print(f"✅ Loaded scraped dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------
# 2️⃣ Clean and convert
# ----------------------------------------------------------
def clean_views(value):
    if pd.isna(value):
        return np.nan
    s = str(value)
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\d]", "", s)
    return int(s) if s.isdigit() else np.nan

def convert_duration(dur):
    if not isinstance(dur, str):
        return np.nan
    m = re.match(r"(\d+):(\d+)", dur)
    if not m:
        return np.nan
    minutes, seconds = int(m.group(1)), int(m.group(2))
    return round(minutes + seconds / 60, 2)

df["views"] = df["views"].apply(clean_views)
df["duration_mins"] = df["duration"].apply(convert_duration)

print("\n🔍 After cleaning:")
print(df[["views", "duration_mins"]].head(10))
print("✅ Views dtype:", df["views"].dtype)

# ----------------------------------------------------------
# 3️⃣ Drop NA and prep
# ----------------------------------------------------------
df = df.dropna(subset=["views", "duration_mins"])
y = df["views"]
X = df[["duration_mins"]]

if y.nunique() <= 1 or X.isna().all().all():
    raise ValueError("❌ Cleaning failed: target or features not numeric.")

print(f"\n✅ Final numeric features: {X.shape[1]} | Samples: {len(y)}")
print(f"✅ Target range: min={y.min():,.0f}, max={y.max():,.0f}")

# ----------------------------------------------------------
# 4️⃣ Train/test split
# ----------------------------------------------------------
y_log = np.log1p(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"✅ Train/Test split → {X_train.shape}, {X_test.shape}")

# ----------------------------------------------------------
# 5️⃣ Models
# ----------------------------------------------------------
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
xgb = XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.1, max_depth=6)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"📊 {name} → RMSE: {rmse:,.0f}, R²: {r2:.3f}")
    return preds, rmse, r2

# ----------------------------------------------------------
# 6️⃣ Train models
# ----------------------------------------------------------
print("\n🚀 Training models on scraped data...")
train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest (Scraped)")
train_and_evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost (Scraped)")
