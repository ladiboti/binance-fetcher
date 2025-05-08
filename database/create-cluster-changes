-- DROP TABLE IF EXISTS cluster_changes;

CREATE TABLE cluster_changes (
    id BIGSERIAL PRIMARY KEY,
    currency_id INTEGER NOT NULL,
    old_cluster_id INTEGER NOT NULL,
    new_cluster_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (currency_id) REFERENCES currencies (id)
);

CREATE INDEX idx_cluster_changes_currency ON cluster_changes (currency_id);
CREATE INDEX idx_cluster_changes_timestamp ON cluster_changes (timestamp);
CREATE INDEX idx_cluster_changes_old_new ON cluster_changes (old_cluster_id, new_cluster_id);