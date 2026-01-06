import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    r"C:\Users\TINA\OneDrive\Desktop\EMIPredict AI\emi_prediction_dataset.csv",
    low_memory=False,
)

# =========================
# ENCODE CATEGORICAL DATA
# =========================
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# =========================
# HANDLE NaN & INF
# =========================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

# =========================
# CLASSIFICATION DATA
# =========================
X_cls = df.drop(["emi_eligibility", "max_monthly_emi"], axis=1)
y_cls = df["emi_eligibility"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42
)


# =========================
# REGRESSION DATASET
# Target: max_monthly_emi
# =========================
target_reg = "max_monthly_emi"

Xr = df.drop(columns=[target_reg])
yr = df[target_reg]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.25, random_state=42
)

# =========================
# MLFLOW SETUP
# =========================
mlflow.set_experiment("EMIPredict_AI")

# =========================
# LINEAR REGRESSION
# =========================
with mlflow.start_run(run_name="Linear_Regression"):
    lr = LinearRegression()
    lr.fit(Xr_train, yr_train)

    preds = lr.predict(Xr_test)

    mlflow.log_metric("MAE", mean_absolute_error(yr_test, preds))
    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(yr_test, preds)))
    mlflow.log_metric("R2", r2_score(yr_test, preds))

    mlflow.sklearn.log_model(lr, "linear_regression")

# =========================
# RANDOM FOREST REGRESSOR
# =========================
with mlflow.start_run(run_name="Random_Forest_Regressor"):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xr_train, yr_train)

    preds = rf.predict(Xr_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(yr_test, preds)))
    mlflow.log_metric("R2", r2_score(yr_test, preds))


# =========================
# XGBOOST REGRESSOR
# =========================
with mlflow.start_run(run_name="XGBoost_Regressor"):
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
    )
    xgb.fit(Xr_train, yr_train)

    preds = xgb.predict(Xr_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(yr_test, preds)))
    mlflow.log_metric("R2", r2_score(yr_test, preds))

    mlflow.sklearn.log_model(xgb, "xgboost_regressor")

print("✅ MLflow runs completed successfully!")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# LOGISTIC REGRESSION (CLASSIFICATION)
# =========================
with mlflow.start_run(run_name="Logistic_Regression_Classifier"):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(Xc_train, yc_train)

    y_pred = log_reg.predict(Xc_test)

    mlflow.log_metric("accuracy", accuracy_score(yc_test, y_pred))
    mlflow.log_metric("precision", precision_score(yc_test, y_pred, average="weighted"))
    mlflow.log_metric("recall", recall_score(yc_test, y_pred, average="weighted"))
    mlflow.log_metric("f1_score", f1_score(yc_test, y_pred, average="weighted"))

    mlflow.sklearn.log_model(log_reg, "logistic_regression_classifier")


# =========================
# RANDOM FOREST CLASSIFIER
# =========================
with mlflow.start_run(run_name="Random_Forest_Classifier"):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(Xc_train, yc_train)

    y_pred = rf_clf.predict(Xc_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score(yc_test, y_pred))
    mlflow.log_metric("precision", precision_score(yc_test, y_pred, average="weighted"))
    mlflow.log_metric("recall", recall_score(yc_test, y_pred, average="weighted"))
    mlflow.log_metric("f1_score", f1_score(yc_test, y_pred, average="weighted"))

    mlflow.sklearn.log_model(rf_clf, "random_forest_classifier")


# =========================
# XGBOOST CLASSIFIER
# =========================
with mlflow.start_run(run_name="XGBoost_Classifier"):
    xgb_clf = XGBClassifier(
        objective="multi:softmax",
        random_state=42,
        eval_metric="mlogloss",
        n_estimators=100,
    )
    xgb_clf.fit(Xc_train, yc_train)

    y_pred = xgb_clf.predict(Xc_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score(yc_test, y_pred))
    mlflow.log_metric("precision", precision_score(yc_test, y_pred, average="weighted"))
    mlflow.log_metric("recall", recall_score(yc_test, y_pred, average="weighted"))
    mlflow.log_metric("f1_score", f1_score(yc_test, y_pred, average="weighted"))

    mlflow.sklearn.log_model(xgb_clf, "xgboost_classifier")

print("✅ All classification models logged to MLflow!")

# =========================
# SAVE BEST MODELS (PICKLE)
# =========================
import pickle
import os

os.makedirs("models", exist_ok=True)

# Best Classification Model (XGBoost)
with open("models/emi_eligibility_model.pkl", "wb") as f:
    pickle.dump(xgb_clf, f, protocol=pickle.HIGHEST_PROTOCOL)

# Best Regression Model (Linear Regression)
with open("models/max_emi_model.pkl", "wb") as f:
    pickle.dump(lr, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Pickle files created successfully!")
