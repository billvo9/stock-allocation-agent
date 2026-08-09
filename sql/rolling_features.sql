WITH daily_returns AS (
    SELECT
        date,
        symbol,
        adjusted_close,
        adjusted_close
            / LAG(adjusted_close) OVER (
                PARTITION BY symbol
                ORDER BY date
            )
            - 1 AS daily_return
    FROM read_parquet('data/raw/prices.parquet')
)

SELECT
    date,
    symbol,
    adjusted_close,
    daily_return,
    adjusted_close / LAG(adjusted_close, 20) OVER (PARTITION BY symbol ORDER BY date) -1 as momentum_20d,
    STDDEV_SAMP(daily_return) OVER (
        PARTITION BY symbol
        ORDER BY date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS volatility_20d
FROM daily_returns
ORDER BY symbol, date;