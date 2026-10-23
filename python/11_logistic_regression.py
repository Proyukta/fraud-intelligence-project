
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# 1. Load the behavioural feature dataset
df = pd.read_csv(
    "Data/Processed/fraudTrain_features.csv"
)

df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

df = df.sort_values(
    "trans_date_trans_time"
).reset_index(drop=True)

# 2. Select features that can be known at transaction time
features = [
    "amt",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_late_night",
    "minutes_since_previous",
    "transactions_previous_hour",
]

X = df[features].copy()
y = df["is_fraud"].astype(int)

# Reduce skew in transaction amounts and elapsed time
X["amt"] = np.log1p(X["amt"].clip(lower=0))
X["minutes_since_previous"] = np.log1p(
    X["minutes_since_previous"].clip(lower=0)
)

# 3. Split chronologically: earlier 80% for training,
#    later 20% for evaluation
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training transactions:", len(X_train))
print("Evaluation transactions:", len(X_test))
print("Fraud cases in training:", int(y_train.sum()))
print("Fraud cases in evaluation:", int(y_test.sum()))

# 4. Build a preprocessing and modelling pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# 5. Train using earlier transactions only
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)
joblib.dump(
    model,
    "Data/Processed/logistic_regression_model.joblib"
)
print("Trained model saved successfully.")

# 6. Predict on the later portion
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 7. Evaluate performance
print("\n--- Logistic Regression Results ---")
print(
    "Precision:",
    round(precision_score(
        y_test, predictions, zero_division=0
    ), 4)
)
print(
    "Recall:",
    round(recall_score(
        y_test, predictions, zero_division=0
    ), 4)
)
print(
    "F1 score:",
    round(f1_score(
        y_test, predictions, zero_division=0
    ), 4)
)
print(
    "Average precision:",
    round(average_precision_score(
        y_test, probabilities
    ), 4)
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# 8. Save evaluation results
results = pd.DataFrame({
    "actual_fraud": y_test.to_numpy(),
    "predicted_fraud": predictions,
    "fraud_probability": probabilities
})

results.to_csv(
    "Data/Processed/logistic_regression_results.csv",
    index=False
)

print(
    "\nSaved: Data/Processed/logistic_regression_results.csv"
)
print("Machine-learning baseline completed.")
