import pandas as pd

# Load cleaned data
df = pd.read_csv("Data/Processed/fraudTrain_cleaned.csv")

print("Cleaned data loaded.")
print("Shape:", df.shape)

# Create customer table
customers = df[
    [
        "customer_id",
        "first",
        "last",
        "gender",
        "dob",
        "city",
        "state",
        "zip",
        "job"
    ]
].drop_duplicates("customer_id")

print("\nNumber of customers:", len(customers))

# Create merchant mapping
merchant_mapping = pd.DataFrame({
    "merchant": df["merchant"].unique()
})

merchant_mapping["merchant_id"] = range(
    1,
    len(merchant_mapping) + 1
)

# Add merchant ID to transactions
df = df.merge(
    merchant_mapping,
    on="merchant",
    how="left"
)

# Create merchant table
merchants = df[
    [
        "merchant_id",
        "merchant",
        "merch_lat",
        "merch_long"
    ]
].drop_duplicates("merchant_id")

print("Number of merchants:", len(merchants))

# Create category mapping
category_mapping = pd.DataFrame({
    "category": df["category"].unique()
})

category_mapping["category_id"] = range(
    1,
    len(category_mapping) + 1
)

# Add category ID to transactions
df = df.merge(
    category_mapping,
    on="category",
    how="left"
)

# Create category table
categories = category_mapping[
    [
        "category_id",
        "category"
    ]
]

print("Number of categories:", len(categories))

# Create transaction table
transactions = df[
    [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "category_id",
        "trans_date_trans_time",
        "amt",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "year",
        "month",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        "is_fraud"
    ]
]

# Save tables
customers.to_csv(
    "Data/Processed/customers.csv",
    index=False
)

merchants.to_csv(
    "Data/Processed/merchants.csv",
    index=False
)

categories.to_csv(
    "Data/Processed/categories.csv",
    index=False
)

transactions.to_csv(
    "Data/Processed/transactions.csv",
    index=False
)

print("\nAll tables saved successfully.")

print("\nFINAL SUMMARY")
print("Customers:", len(customers))
print("Merchants:", len(merchants))
print("Categories:", len(categories))
print("Transactions:", len(transactions))
