# ==========================================================
#  YOUTUBE POPULARITY PREDICTION (SCRAPED DATASET MODEL)
# ==========================================================
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import unicodedata

# ----------------------------------------------------------
# 1️ Load cleaned scraped dataset
# ----------------------------------------------------------
path_scraped = "data/youtube_scraped_clean.csv"
df = pd.read_csv(path_scraped)

print(f" Loaded scraped dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------
# 2️ Clean numeric and duration columns
# ----------------------------------------------------------
def clean_views(value):
    if pd.isna(value):
        return np.nan
    s = str(value)
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^0-9]", "", s)
    return int(s) if s.isdigit() else np.nan

def convert_duration(dur):
    if not isinstance(dur, str):
        return np.nan
    match = re.match(r"(\d+):(\d+)", dur)
    if not match:
        return np.nan
    m, s = int(match.group(1)), int(match.group(2))
    return round(m + s / 60, 2)

# Apply cleaning
if "views" in df.columns:
    df["views"] = df["views"].apply(clean_views)
if "duration" in df.columns:
    df["duration_mins"] = df["duration"].apply(convert_duration)

# ----------------------------------------------------------
# 3️ Prepare numeric features
# ----------------------------------------------------------
df_num = df.select_dtypes(include=["number"]).copy()
if "views" not in df_num.columns:
    raise ValueError(" 'views' column not found or not numeric!")

y = df_num["views"].replace([np.inf, -np.inf], np.nan).dropna()
X = df_num.loc[y.index].drop(columns=["views"])

print(f" Numeric features: {X.shape[1]} | Target samples: {len(y)}")

# ----------------------------------------------------------
# 4️ Log-transform target and split
# ----------------------------------------------------------
y_log = np.log1p(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

print(f" Target range: min={y.min():,.0f}, max={y.max():,.0f}")
print(f" Train/Test split → {X_train.shape}, {X_test.shape}")

# ----------------------------------------------------------
# 5️ Define models
# ----------------------------------------------------------
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
xgb = XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.1, max_depth=6)

# ----------------------------------------------------------
# 6️ Train & evaluate
# ----------------------------------------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f" {name} → RMSE: {rmse:,.0f}, R²: {r2:.3f}")
    return preds, rmse, r2

print("\n Training models on scraped data...")
train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest (Scraped)")
train_and_evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost (Scraped)")
