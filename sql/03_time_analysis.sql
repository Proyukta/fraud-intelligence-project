SELECT
    hour,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(is_fraud) / COUNT(*),
        4
    ) AS fraud_rate_percentage
FROM transactions
GROUP BY hour
ORDER BY fraud_rate_percentage DESC;
SELECT
    day_of_week,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(is_fraud) / COUNT(*),
        4
    ) AS fraud_rate_percentage
FROM transactions
GROUP BY day_of_week
ORDER BY day_of_week;
SELECT
    year,
    month,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(is_fraud) / COUNT(*),
        4
    ) AS fraud_rate_percentage
FROM transactions
GROUP BY year, month
ORDER BY year, month;