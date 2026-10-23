import pandas as pd

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

file_path = "Data/Raw/fraudTrain.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)


# --------------------------------------------------
# 1. DATASET SIZE
# --------------------------------------------------

print("\n1. DATASET SIZE")
print("-" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# 2. COLUMN INFORMATION
# --------------------------------------------------

print("\n2. COLUMN INFORMATION")
print("-" * 60)

info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

print(info.to_string(index=False))


# --------------------------------------------------
# 3. DUPLICATES
# --------------------------------------------------

print("\n3. DUPLICATES")
print("-" * 60)

print("Duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 4. TRANSACTION ID CHECK
# --------------------------------------------------

print("\n4. TRANSACTION ID CHECK")
print("-" * 60)

print("Unique transaction numbers:", df["trans_num"].nunique())
print("Total rows:", len(df))


# --------------------------------------------------
# 5. CUSTOMER / CARD CHECK
# --------------------------------------------------

print("\n5. CUSTOMER / CARD CHECK")
print("-" * 60)

print("Unique card numbers:", df["cc_num"].nunique())


# --------------------------------------------------
# 6. MERCHANT CHECK
# --------------------------------------------------

print("\n6. MERCHANT CHECK")
print("-" * 60)

print("Unique merchants:", df["merchant"].nunique())


# --------------------------------------------------
# 7. CATEGORY CHECK
# --------------------------------------------------

print("\n7. CATEGORY CHECK")
print("-" * 60)

print("Unique categories:", df["category"].nunique())

print("\nCategories:")
print(df["category"].value_counts())


# --------------------------------------------------
# 8. FRAUD DISTRIBUTION
# --------------------------------------------------

print("\n8. FRAUD DISTRIBUTION")
print("-" * 60)

fraud_counts = df["is_fraud"].value_counts()

print(fraud_counts)

print("\nFraud percentage:")
print(
    (df["is_fraud"].value_counts(normalize=True) * 100)
    .round(3)
)


# --------------------------------------------------
# 9. DATE RANGE
# --------------------------------------------------

print("\n9. DATE RANGE")
print("-" * 60)

df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

print("Start:", df["trans_date_trans_time"].min())
print("End:", df["trans_date_trans_time"].max())


# --------------------------------------------------
# 10. TRANSACTION AMOUNT CHECK
# --------------------------------------------------

print("\n10. TRANSACTION AMOUNT")
print("-" * 60)

print(df["amt"].describe())

print("\nNegative amounts:", (df["amt"] < 0).sum())
print("Zero amounts:", (df["amt"] == 0).sum())


# --------------------------------------------------
# 11. FRAUD BY CATEGORY
# --------------------------------------------------

print("\n11. FRAUD BY CATEGORY")
print("-" * 60)

fraud_by_category = (
    df.groupby("category")["is_fraud"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

fraud_by_category["fraud_rate_percent"] = (
    fraud_by_category["mean"] * 100
).round(3)

print(fraud_by_category)


print("\n" + "=" * 60)
print("END OF DATA QUALITY REPORT")
print("=" * 60)