
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Load features and model predictions
df = pd.read_csv("Data/Processed/fraudTrain_features.csv")
results = pd.read_csv(
    "Data/Processed/logistic_regression_results.csv"
)

# Recreate the same chronological holdout
split_index = int(len(df) * 0.8)
test = df.iloc[split_index:].reset_index(drop=True)

# Confirm both datasets line up
if len(test) != len(results):
    raise ValueError("Test data and prediction rows do not match.")

actual = results["actual_fraud"].astype(int)
probabilities = results["fraud_probability"]

# Rule-based baseline: use the amount score from Script 07
baseline = pd.read_csv("Data/Processed/fraudTrain_scored.csv")
baseline = baseline.sort_values(
    ["customer_id", "trans_date_trans_time"]
).reset_index(drop=True)

# Script 07 also sorted by customer and time.
# Match its chronological holdout to the same time split.
baseline = baseline.sort_values(
    "trans_date_trans_time"
).reset_index(drop=True)

baseline_test = baseline.iloc[split_index:].reset_index(drop=True)

if len(baseline_test) != len(actual):
    raise ValueError("Baseline and model test rows do not match.")

baseline_scores = baseline_test["amount_risk_score"].fillna(0)
baseline_predictions = (baseline_scores >= 70).astype(int)

# Evaluate baseline and Logistic Regression at threshold 0.5
model_predictions = (probabilities >= 0.5).astype(int)

def report(name, y_true, y_pred, scores):
    print(f"\n{name}")
    print("Precision:", round(
        precision_score(y_true, y_pred, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_true, y_pred, zero_division=0), 4
    ))
    print("F1 score:", round(
        f1_score(y_true, y_pred, zero_division=0), 4
    ))
    print("Average precision:", round(
        average_precision_score(y_true, scores), 4
    ))

report(
    "Amount-based baseline",
    actual,
    baseline_predictions,
    baseline_scores
)

report(
    "Logistic Regression",
    actual,
    model_predictions,
    probabilities
)

print("\nComparison completed.")
