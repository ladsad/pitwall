-- Pitwall Supabase Schema
-- Run this in Supabase SQL Editor (https://supabase.com → Project → SQL Editor)

-- ── PREDICTIONS TABLE ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predictions (
    id                 BIGSERIAL PRIMARY KEY,
    driver             TEXT NOT NULL,
    team               TEXT,
    event              TEXT NOT NULL,
    round              INT NOT NULL,
    season             INT NOT NULL,
    model_version      TEXT NOT NULL,
    predicted_position INT,
    win_probability    REAL,
    uncertainty        REAL,
    sessions           JSONB,
    trend              JSONB,
    feature_importance JSONB,
    generated_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(season, event, driver, model_version)
);


-- ── ROW LEVEL SECURITY ──────────────────────────────────────────────────────
-- Public read (anon key), service-role write (service key)

ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read predictions" ON predictions FOR SELECT USING (true);
CREATE POLICY "Service write predictions" ON predictions FOR ALL USING (true) WITH CHECK (true);


-- ── INDEXES ──────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_pred_latest ON predictions(season, event, model_version);
