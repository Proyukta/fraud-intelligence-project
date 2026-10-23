SELECT
    COUNT(*) AS total_transactions,
    ROUND(SUM(amt), 2) AS total_transaction_value,
    ROUND(AVG(amt), 2) AS average_transaction_amount
FROM transactions;
SELECT
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(is_fraud) / COUNT(*),
        4
    ) AS fraud_percentage
FROM transactions;