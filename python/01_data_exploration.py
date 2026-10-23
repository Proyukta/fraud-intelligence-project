import pandas as pd

# Load the dataset
file_path = "Data/Raw/fraudTrain.csv"
df = pd.read_csv(file_path)

# --------------------------------------------------
# 1. BASIC INFORMATION
# --------------------------------------------------

print("=" * 50)
print("BASIC INFORMATION")
print("=" * 50)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# --------------------------------------------------
# 2. COLUMN NAMES
# --------------------------------------------------

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)

for column in df.columns:
    print(column)


# --------------------------------------------------
# 3. DATA TYPES
# --------------------------------------------------

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)


# --------------------------------------------------
# 4. MISSING VALUES
# --------------------------------------------------

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


# --------------------------------------------------
# 5. DUPLICATE ROWS
# --------------------------------------------------

print("\n" + "=" * 50)
print("DUPLICATES")
print("=" * 50)

print("Duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 6. FRAUD DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 50)
print("FRAUD DISTRIBUTION")
print("=" * 50)

print(df["is_fraud"].value_counts())


# --------------------------------------------------
# 7. FRAUD PERCENTAGE
# --------------------------------------------------

print("\n" + "=" * 50)
print("FRAUD PERCENTAGE")
print("=" * 50)

fraud_percentage = df["is_fraud"].value_counts(normalize=True) * 100

print(fraud_percentage)


# --------------------------------------------------
# 8. TRANSACTION AMOUNT
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRANSACTION AMOUNT")
print("=" * 50)

print(df["amt"].describe())


# --------------------------------------------------
# 9. FIRST 5 ROWS
# --------------------------------------------------

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)

print(df.head())