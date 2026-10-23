SELECT
    customer_id,
    COUNT(*) AS total_transactions,
    ROUND(AVG(amt), 2) AS average_transaction_amount,
    ROUND(MAX(amt), 2) AS maximum_transaction_amount,
    SUM(is_fraud) AS fraud_transactions
FROM transactions
GROUP BY customer_id
ORDER BY total_transactions DESC
LIMIT 20;