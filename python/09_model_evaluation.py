
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("Data/Processed/fraudTrain_scored.csv")

# Exclude transactions with insufficient history
df = df.dropna(subset=["amount_risk_score"])

# Treat scores of 70 or above as flagged
df["predicted_fraud"] = (
    df["amount_risk_score"] >= 70
).astype(int)

actual = df["is_fraud"].astype(int)
predicted = df["predicted_fraud"]

tn, fp, fn, tp = confusion_matrix(
    actual, predicted, labels=[0, 1]
).ravel()

print("\n--- Risk Score Evaluation ---")
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)

print("\nPrecision:", round(
    precision_score(actual, predicted, zero_division=0), 4
))
print("Recall:", round(
    recall_score(actual, predicted, zero_division=0), 4
))
print("F1 score:", round(
    f1_score(actual, predicted, zero_division=0), 4
))

print("\nEvaluation completed.")
