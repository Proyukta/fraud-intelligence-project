
import pandas as pd

# Load the scored transactions
df = pd.read_csv("Data/Processed/fraudTrain_scored.csv")

# Check the dataset
print("Dataset shape:", df.shape)

print("\nTransactions in each risk band:")
print(df["risk_band"].value_counts(dropna=False))

# Calculate fraud rate for each risk band
summary = (
    df.groupby("risk_band", dropna=False)["is_fraud"]
    .agg(
        total_transactions="count",
        fraud_cases="sum",
        fraud_rate="mean"
    )
    .reset_index()
)

summary["fraud_rate_percent"] = (
    summary["fraud_rate"] * 100
)

print("\nFraud rate by risk band:")
print(
    summary[
        [
            "risk_band",
            "total_transactions",
            "fraud_cases",
            "fraud_rate_percent"
        ]
    ].round(3).to_string(index=False)
)

# Save the summary
summary.to_csv(
    "Data/Processed/risk_band_validation.csv",
    index=False
)

print("\nSaved: Data/Processed/risk_band_validation.csv")
