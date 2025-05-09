-- DROP TABLE IF EXISTS snapshots;

CREATE TABLE snapshots (
    id BIGSERIAL PRIMARY KEY,
    currency_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    cluster_id INTEGER NOT NULL,
    snapshot_type VARCHAR(20) DEFAULT 'automatic', 
    snapshot_reason VARCHAR(255) DEFAULT 'Periodic snapshot',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (currency_id) REFERENCES currencies (id)
);

CREATE INDEX idx_snapshots_currency_timestamp ON snapshots (currency_id, timestamp);
CREATE INDEX idx_snapshots_cluster_id ON snapshots (cluster_id);
CREATE INDEX idx_snapshots_created_at ON snapshots (created_at);