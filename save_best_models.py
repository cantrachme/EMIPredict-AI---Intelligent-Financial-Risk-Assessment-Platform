import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("emi_prediction_dataset.csv")

# =========================
# ENCODE CATEGORICAL DATA
# =========================
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

# =========================
# CLASSIFICATION (BEST MODEL)
# =========================
X_cls = df.drop(["emi_eligibility", "max_monthly_emi"], axis=1)
y_cls = df["emi_eligibility"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42
)

best_classifier = XGBClassifier(
    n_estimators=100, random_state=42, eval_metric="logloss"
)
best_classifier.fit(Xc_train, yc_train)

# =========================
# REGRESSION (BEST MODEL)
# (Lower n_estimators to avoid memory crash)
# =========================
Xr = df.drop(columns=["max_monthly_emi"])
yr = df["max_monthly_emi"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.25, random_state=42
)

best_regressor = RandomForestRegressor(
    n_estimators=50,  # 🔥 reduced to avoid MemoryError
    random_state=42,
    n_jobs=-1,
)
best_regressor.fit(Xr_train, yr_train)

# =========================
# SAVE MODELS
# =========================
os.makedirs("models", exist_ok=True)

with open("models/emi_eligibility_model.pkl", "wb") as f:
    pickle.dump(best_classifier, f)

with open("models/max_emi_model.pkl", "wb") as f:
    pickle.dump(best_regressor, f)

print("✅ Pickle files saved successfully!")
