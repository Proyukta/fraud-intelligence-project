
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load predictions from the unseen test dataset
df = pd.read_csv(
    "Data/Processed/logistic_regression_results.csv"
)

actual = df["actual_fraud"].astype(int)
probabilities = df["fraud_probability"]

# Compare different probability thresholds
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

results = []

for threshold in thresholds:
    predicted = (probabilities >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1_score": f1_score(
            actual, predicted, zero_division=0
        ),
        "flagged_transactions": int(predicted.sum())
    })

summary = pd.DataFrame(results)

print("\nThreshold comparison:")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    "Data/Processed/validation_threshold_analysis.csv",
    index=False
)

print("\nSaved: Data/Processed/threshold_analysis.csv")
