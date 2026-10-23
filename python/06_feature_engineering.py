
import pandas as pd

# Load the cleaned transaction dataset
df = pd.read_csv(
    "Data/Processed/fraudTrain_cleaned.csv"
)

# Convert the timestamp to datetime
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

# Sort transactions in chronological order
df = df.sort_values(
    ["customer_id", "trans_date_trans_time"]
).reset_index(drop=True)

# Calculate each customer's average transaction amount
df["customer_avg_amount"] = (
    df.groupby("customer_id")["amt"]
    .transform("mean")
)

# Calculate each customer's maximum transaction amount
df["customer_max_amount"] = (
    df.groupby("customer_id")["amt"]
    .transform("max")
)

# Compare each transaction with the customer's average
df["amount_vs_customer_avg"] = (
    df["amt"] / df["customer_avg_amount"]
)

# Calculate the time since the customer's previous transaction
df["previous_transaction_time"] = (
    df.groupby("customer_id")["trans_date_trans_time"]
    .shift(1)
)

df["minutes_since_previous_transaction"] = (
    (
        df["trans_date_trans_time"]
        - df["previous_transaction_time"]
    ).dt.total_seconds() / 60
)

# Display a sample of the new features
print("New behavioral features:")
print(
    df[
        [
            "customer_id",
            "amt",
            "customer_avg_amount",
            "amount_vs_customer_avg",
            "minutes_since_previous_transaction"
        ]
    ].head(10)
)

print("\nFeature engineering completed.")
