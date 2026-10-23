WITH merchant_metrics AS (
    SELECT
        m.merchant AS merchant_name,
        COUNT(*) AS total_transactions,
        SUM(t.is_fraud) AS fraud_transactions,
        100.0 * SUM(t.is_fraud) / COUNT(*) AS fraud_rate
    FROM transactions AS t
    JOIN merchants AS m
        ON t.merchant_id = m.merchant_id
    GROUP BY m.merchant_id, m.merchant
    HAVING COUNT(*) >= 100
)
SELECT
    merchant_name,
    total_transactions,
    fraud_transactions,
    ROUND(fraud_rate, 4) AS fraud_rate_percentage,
    RANK() OVER (ORDER BY fraud_rate DESC) AS fraud_rate_rank
FROM merchant_metrics
ORDER BY fraud_rate_rank;
WITH monthly_metrics AS (
    SELECT
        year,
        month,
        COUNT(*) AS total_transactions,
        SUM(is_fraud) AS fraud_transactions,
        100.0 * SUM(is_fraud) / COUNT(*) AS fraud_rate
    FROM transactions
    GROUP BY year, month
)
SELECT
    year,
    month,
    total_transactions,
    ROUND(fraud_rate, 4) AS fraud_rate_percentage,
    ROUND(
        LAG(fraud_rate) OVER (ORDER BY year, month),
        4
    ) AS previous_month_fraud_rate
FROM monthly_metrics
ORDER BY year, month;