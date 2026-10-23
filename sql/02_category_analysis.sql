SELECT
    category_id,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(is_fraud) / COUNT(*),
        4
    ) AS fraud_rate_percentage
FROM transactions
GROUP BY category_id
ORDER BY fraud_rate_percentage DESC;
SELECT
    c.category AS category_name,
    COUNT(*) AS total_transactions,
    SUM(t.is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(t.is_fraud) / COUNT(*),
        4
    ) AS fraud_rate_percentage
FROM transactions AS t
JOIN categories AS c
    ON t.category_id = c.category_id
GROUP BY c.category_id, c.category
ORDER BY fraud_rate_percentage DESC;