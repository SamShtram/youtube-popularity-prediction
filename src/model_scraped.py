import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
#  Load enhanced dataset
# -------------------------------------------------------
path = "data/youtube_scraped_features.csv"
df = pd.read_csv(path)
print(f" Loaded feature dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------------------------------
#  Clean + select useful features
# -------------------------------------------------------
df = df.dropna(subset=["views"])
df = df[df["views"] > 0]

# Cap extreme outliers (top 1%)
upper_cap = df["views"].quantile(0.99)
df = df[df["views"] <= upper_cap]

# Select numeric features only
X = df.select_dtypes(include=[np.number]).drop(columns=["views"], errors="ignore")
y = df["views"]

# Log transform target
y_log = np.log1p(y)

print(f"Final numeric features: {X.shape[1]} | Samples: {len(y)}")

# -------------------------------------------------------
#  Train/Test split + scale
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------------
#  Define tuned models
# -------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.8,
    random_state=42,
    n_jobs=-1
)

# -------------------------------------------------------
#  Train & evaluate
# -------------------------------------------------------
def evaluate(model, name):
    model.fit(X_train, y_train)
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f" {name}  RMSE: {rmse:,.0f}, R²: {r2:.3f}")
    return rmse, r2

print("\n Training tuned models on enhanced scraped data...\n")
evaluate(rf, "Random Forest (Tuned)")
evaluate(xgb, "XGBoost (Tuned)")
