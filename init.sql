CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS equipment_events (
    time TIMESTAMPTZ NOT NULL,
    track_id INT NOT NULL,
    session_id TEXT,
    class_name TEXT,
    state TEXT,
    activity TEXT,
    confidence REAL,
    bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
    active_sec REAL, idle_sec REAL, total_sec REAL, util_pct REAL
);

SELECT create_hypertable('equipment_events', 'time', if_not_exists => TRUE);

-- Indexes for fast dashboard queries
CREATE INDEX IF NOT EXISTS idx_events_track_time ON equipment_events (track_id, time DESC);