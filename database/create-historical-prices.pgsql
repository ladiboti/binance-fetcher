DROP TABLE IF EXISTS historical_prices_part CASCADE;

CREATE TABLE historical_prices_part (
    id BIGSERIAL,
    currency_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    cluster_id INTEGER NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    sma_10 DECIMAL(20, 8),
    ema_10 DECIMAL(20, 8),
    rsi_14 DECIMAL(20, 8),
    bollinger_upper DECIMAL(20, 8),
    bollinger_middle DECIMAL(20, 8),
    bollinger_lower DECIMAL(20, 8),
    macd DECIMAL(20, 8),
    macd_signal DECIMAL(20, 8),
    atr DECIMAL(20, 8),
    obv DECIMAL(20, 8),
    best_bid_price DECIMAL(20, 8),
    best_ask_price DECIMAL(20, 8),
    best_bid_volume DECIMAL(20, 8),
    best_ask_volume DECIMAL(20, 8),
    bid_ask_spread DECIMAL(20, 8),
    order_book_imbalance DECIMAL(20, 8),
    bid_volume_depth DECIMAL(20, 8),
    ask_volume_depth DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (currency_id, timestamp, id),
    FOREIGN KEY (currency_id) REFERENCES currencies (id)
) PARTITION BY RANGE (timestamp);

-- TODO: add dinamic intervals!!!
CREATE TABLE historical_prices PARTITION OF historical_prices_part
FOR VALUES FROM ('1970-01-01') TO ('9999-12-31');

CREATE INDEX idx_historical_currency_id ON historical_prices_part (currency_id);
CREATE INDEX idx_historical_timestamp ON historical_prices_part (timestamp);
CREATE INDEX idx_historical_cluster_id ON historical_prices_part (cluster_id);
CREATE INDEX idx_historical_combined ON historical_prices_part (currency_id, timestamp, cluster_id);
