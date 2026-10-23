SELECT
    m.merchant AS merchant_name,
    COUNT(*) AS total_transactions,
    SUM(t.is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(t.is_fraud) / COUNT(*),
        4
    ) AS fraud_rate_percentage,
    ROUND(SUM(t.amt), 2) AS total_transaction_value
FROM transactions AS t
JOIN merchants AS m
    ON t.merchant_id = m.merchant_id
GROUP BY m.merchant_id, m.merchant
HAVING COUNT(*) >= 100
ORDER BY fraud_rate_percentage DESC;