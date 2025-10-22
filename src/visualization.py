import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# === Load dataset ===
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_features.csv")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["views"])
df = df[df["views"] > 0]
df = df[df["views"] <= df["views"].quantile(0.99)]  # cap top 1%

X = df.select_dtypes(include=[np.number]).drop(columns=["views"], errors="ignore")
y = np.log1p(df["views"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Models ===
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

# === Train and evaluate ===
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

def evaluate(model, name):
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    return {"Model": name, "RMSE": rmse, "R2": r2}

results = [evaluate(rf, "Random Forest"), evaluate(xgb, "XGBoost")]
results_df = pd.DataFrame(results)
print(results_df)

# === 1. Model comparison ===
plt.figure(figsize=(6, 4))
plt.bar(results_df["Model"], results_df["R2"], color=["steelblue", "darkorange"])
plt.ylabel("R² Score")
plt.title("Model Comparison: Random Forest vs XGBoost")
plt.tight_layout()
plt.savefig(os.path.join("data", "model_comparison.png"))
plt.close()

# === 2. Feature importance (XGBoost) ===
importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
importance.plot(kind="bar")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join("data", "feature_importance.png"))
plt.close()

# === 3. Views vs Duration visualization ===
if "duration_mins" in df.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df["duration_mins"], df["views"], alpha=0.5)
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Views")
    plt.title("Video Duration vs Views")
    plt.tight_layout()
    plt.savefig(os.path.join("data", "views_vs_duration.png"))
    plt.close()

print("Visualization complete. Graphs saved to /data/")
