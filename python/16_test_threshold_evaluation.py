
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Load predictions from the separate test dataset
df = pd.read_csv(
    "Data/Processed/unseen_test_results.csv"
)

actual = df["actual_fraud"].astype(int)
probabilities = df["fraud_probability"]

# Apply the provisional threshold chosen using validation data
threshold = 0.7
predicted = (probabilities >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    actual, predicted, labels=[0, 1]
).ravel()

print("\n--- Test Evaluation at Threshold 0.7 ---")
print("Threshold:", threshold)
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)

print(
    "Precision:",
    round(precision_score(actual, predicted, zero_division=0), 4)
)
print(
    "Recall:",
    round(recall_score(actual, predicted, zero_division=0), 4)
)
print(
    "F1 score:",
    round(f1_score(actual, predicted, zero_division=0), 4)
)

print("\nTest threshold evaluation completed.")
