
import pandas as pd
import numpy as np

# 1. Load cleaned transaction data
df = pd.read_csv("Data/Processed/fraudTrain_cleaned.csv")

df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

# 2. Sort transactions chronologically for each customer
df = df.sort_values(
    ["customer_id", "trans_date_trans_time"]
).reset_index(drop=True)

# 3. Calculate historical transaction behaviour
customer_groups = df.groupby("customer_id")["amt"]

df["previous_amount_sum"] = (
    customer_groups.cumsum() - df["amt"]
)

df["previous_transaction_count"] = (
    df.groupby("customer_id").cumcount()
)

df["historical_avg_amount"] = (
    df["previous_amount_sum"]
    / df["previous_transaction_count"].replace(0, np.nan)
)

# 4. Compare the current transaction with past behaviour
df["amount_vs_history"] = (
    df["amt"] / df["historical_avg_amount"]
)

# 5. Assign an experimental amount-anomaly score
df["amount_risk_score"] = np.select(
    [
        df["amount_vs_history"] >= 5,
        df["amount_vs_history"] >= 3,
        df["amount_vs_history"] >= 2,
    ],
    [100, 70, 40],
    default=10
)

# 6. Do not assign a score when history is unavailable
df.loc[
    df["historical_avg_amount"].isna(),
    "amount_risk_score"
] = np.nan

# 7. Create understandable risk bands
df["risk_band"] = np.select(
    [
        df["amount_risk_score"].isna(),
        df["amount_risk_score"] >= 100,
        df["amount_risk_score"] >= 70,
        df["amount_risk_score"] >= 40,
    ],
    [
        "Insufficient history",
        "High",
        "Elevated",
        "Moderate",
    ],
    default="Low"
)

# 8. Summarise results for an initial review
print("\nRisk band distribution:")
print(df["risk_band"].value_counts(dropna=False))

print("\nFraud rate by risk band (%):")
fraud_rates = (
    df.groupby("risk_band", observed=True)["is_fraud"]
    .agg(["count", "sum", "mean"])
)

fraud_rates["fraud_rate_percent"] = (
    fraud_rates["mean"] * 100
)

print(
    fraud_rates[
        ["count", "sum", "fraud_rate_percent"]
    ].round(3)
)

# 9. Save the scored dataset
output_path = "Data/Processed/fraudTrain_scored.csv"
df.to_csv(output_path, index=False)

print(f"\nScored data saved to: {output_path}")
print("Risk-scoring script completed.")

