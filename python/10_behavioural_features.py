
import pandas as pd
import numpy as np

# 1. Load the cleaned transactions
df = pd.read_csv("Data/Processed/fraudTrain_cleaned.csv")

df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

# 2. Sort transactions in chronological order
df = df.sort_values(
    ["customer_id", "trans_date_trans_time"]
).reset_index(drop=True)

# 3. Calculate time since the customer's previous transaction
previous_time = (
    df.groupby("customer_id")["trans_date_trans_time"].shift(1)
)

df["minutes_since_previous"] = (
    df["trans_date_trans_time"] - previous_time
).dt.total_seconds() / 60

# 4. Count earlier transactions in the previous 60 minutes
df["transactions_previous_hour"] = 0

for _, group in df.groupby("customer_id", sort=False):
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

# 5. Create a simple unusual-time indicator
df["is_late_night"] = (
    (df["trans_date_trans_time"].dt.hour < 5)
).astype(int)

# 6. Save the new feature dataset
output_path = "Data/Processed/fraudTrain_features.csv"

df.to_csv(output_path, index=False)

# 7. Inspect the results
print("\nNew behavioural features:")
print(
    df[
        [
            "customer_id",
            "trans_date_trans_time",
            "amt",
            "minutes_since_previous",
            "transactions_previous_hour",
            "is_late_night",
            "is_fraud",
        ]
    ].head(15).to_string(index=False)
)

print("\nFeature summary:")
print(
    df[
        [
            "minutes_since_previous",
            "transactions_previous_hour",
            "is_late_night",
        ]
    ].describe().round(2)
)

print(f"\nSaved feature dataset to: {output_path}")
print("Behavioural feature engineering completed.")
