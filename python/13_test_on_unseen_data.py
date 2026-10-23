
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# 1. Load the saved model
model = joblib.load(
    "Data/Processed/logistic_regression_model.joblib"
)

# 2. Load the original training and test datasets
train = pd.read_csv("Data/Raw/fraudTrain.csv")
test = pd.read_csv("Data/Raw/fraudTest.csv")

# Keep track of which rows belong to the test dataset
train["is_test_row"] = False
test["is_test_row"] = True

# Combine rows so test features can use earlier training history
df = pd.concat([train, test], ignore_index=True)

# 3. Prepare timestamps and customer IDs
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

df = df.sort_values(
    ["cc_num", "trans_date_trans_time"]
).reset_index(drop=True)

# 4. Create time-based features
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_late_night"] = (df["hour"] < 5).astype(int)

# 5. Calculate minutes since the previous customer transaction
previous_time = (
    df.groupby("cc_num")["trans_date_trans_time"].shift(1)
)

df["minutes_since_previous"] = (
    df["trans_date_trans_time"] - previous_time
).dt.total_seconds() / 60

# 6. Count earlier transactions in the preceding 60 minutes
df["transactions_previous_hour"] = 0

for _, group in df.groupby("cc_num", sort=False):
    indices = group.index.to_numpy()

    times = (
        group["trans_date_trans_time"]
        .astype("int64")
        .to_numpy() // 1_000_000_000
    )

    left = 0
    counts = np.zeros(len(times), dtype=np.int32)

    for i in range(len(times)):
        while times[i] - times[left] > 3600:
            left += 1

        counts[i] = i - left

    df.loc[indices, "transactions_previous_hour"] = counts

# 7. Keep only test rows for evaluation
test_df = df[df["is_test_row"]].copy()

features = [
    "amt",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_late_night",
    "minutes_since_previous",
    "transactions_previous_hour",
]

X_test = test_df[features].copy()
y_test = test_df["is_fraud"].astype(int)

# Apply the same transformations used during training
X_test["amt"] = np.log1p(X_test["amt"].clip(lower=0))

X_test["minutes_since_previous"] = np.log1p(
    X_test["minutes_since_previous"].clip(lower=0)
)

# 8. Predict using the already-trained model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 9. Evaluate predictions
print("\n--- Unseen Test Data Evaluation ---")
print("Test transactions:", len(test_df))
print("Actual fraud cases:", int(y_test.sum()))

print(
    "Precision:",
    round(precision_score(y_test, predictions, zero_division=0), 4)
)
print(
    "Recall:",
    round(recall_score(y_test, predictions, zero_division=0), 4)
)
print(
    "F1 score:",
    round(f1_score(y_test, predictions, zero_division=0), 4)
)
print(
    "Average precision:",
    round(average_precision_score(y_test, probabilities), 4)
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# 10. Save predictions for further analysis
results = pd.DataFrame({
    "transaction_time": test_df["trans_date_trans_time"].to_numpy(),
    "actual_fraud": y_test.to_numpy(),
    "predicted_fraud": predictions,
    "fraud_probability": probabilities,
})

results.to_csv(
    "Data/Processed/unseen_test_results.csv",
    index=False
)

print("\nSaved: Data/Processed/unseen_test_results.csv")
print("Unseen-data evaluation completed.")
