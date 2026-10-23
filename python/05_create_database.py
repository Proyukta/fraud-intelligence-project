import pandas as pd
import sqlite3

# Load processed tables
customers = pd.read_csv("Data/Processed/customers.csv")
merchants = pd.read_csv("Data/Processed/merchants.csv")
categories = pd.read_csv("Data/Processed/categories.csv")
transactions = pd.read_csv("Data/Processed/transactions.csv")

# Create SQLite database
connection = sqlite3.connect(
    "database/fraud_database.db"
)

# Load tables into SQLite
customers.to_sql(
    "customers",
    connection,
    if_exists="replace",
    index=False
)

merchants.to_sql(
    "merchants",
    connection,
    if_exists="replace",
    index=False
)

categories.to_sql(
    "categories",
    connection,
    if_exists="replace",
    index=False
)

transactions.to_sql(
    "transactions",
    connection,
    if_exists="replace",
    index=False
)

print("Database created successfully.")

# Close connection
connection.close()