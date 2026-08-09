-- SELECT
--     date, symbol, adjusted_close,
--     LAG(adjusted_close, 1, NULL) OVER (PARTITION BY symbol ORDER BY date) AS previous_adjusted_close,
--     adjusted_close/LAG(adjusted_close, 1, NULL) OVER (PARTITION BY symbol ORDER BY date) -1 as daily_return
-- FROM read_parquet('data/raw/prices.parquet')
-- ORDER BY symbol, date;

WITH lagged_prices AS(
    SELECT date, symbol, adjusted_close, 
    LAG(adjusted_close, 1, NULL) OVER (PARTITION BY symbol ORDER BY date) AS previous_adjusted_close
    from read_parquet('data/raw/prices.parquet')
)
SELECT date, symbol, adjusted_close, previous_adjusted_close,
adjusted_close/previous_adjusted_close -1 as daily_return
FROM lagged_prices
ORDER BY symbol, date;