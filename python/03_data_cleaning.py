import pandas as pd

# Load the raw dataset
df = pd.read_csv("Data/Raw/fraudTrain.csv")

print("Original shape:", df.shape)

# Remove unnecessary exported index column
df = df.drop(columns=["Unnamed: 0"])

print("After removing index:", df.shape)
# Convert transaction time from text to datetime
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

print("Transaction time type:")
print(df["trans_date_trans_time"].dtype)
# Check for duplicate transaction IDs
duplicates = df["trans_num"].duplicated().sum()

print("Duplicate transaction IDs:", duplicates)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check invalid transaction amounts
zero_amounts = (df["amt"] == 0).sum()
negative_amounts = (df["amt"] < 0).sum()

print("\nZero transaction amounts:", zero_amounts)
print("Negative transaction amounts:", negative_amounts)

# Check fraud label
print("\nFraud label distribution:")
print(df["is_fraud"].value_counts())

# Check unique identifiers
print("\nUnique transaction IDs:", df["trans_num"].nunique())
print("Unique customers:", df["cc_num"].nunique())
print("Unique merchants:", df["merchant"].nunique())
print("Unique categories:", df["category"].nunique())

# Check transaction date range
print("\nTransaction date range:")
print("Start:", df["trans_date_trans_time"].min())
print("End:", df["trans_date_trans_time"].max())

# Sort transactions chronologically
df = df.sort_values(
    "trans_date_trans_time"
).reset_index(drop=True)

print("\nTransactions sorted chronologically.")

# Create time features
df["date"] = df["trans_date_trans_time"].dt.date
df["year"] = df["trans_date_trans_time"].dt.year
df["month"] = df["trans_date_trans_time"].dt.month
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

df["is_weekend"] = df["day_of_week"] >= 5

df["is_night"] = (
    (df["hour"] < 6) |
    (df["hour"] >= 22)
)

# Rename identifiers
df = df.rename(columns={
    "trans_num": "transaction_id",
    "cc_num": "customer_id"
})

# Final validation
print("\nFinal shape:", df.shape)

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nFinal columns:")
print(df.columns.tolist())

# Save cleaned dataset
df.to_csv(
    "Data/Processed/fraudTrain_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")