from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS batch (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS material (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    video_path TEXT,
    srt_path TEXT,
    transcript TEXT,
    has_face INTEGER,
    category TEXT NOT NULL DEFAULT 'shapewear',
    status TEXT NOT NULL DEFAULT 'uploaded',
    duration_ms INTEGER NOT NULL DEFAULT 0,
    transcript_source TEXT,
    asr_model TEXT,
    asr_error TEXT,
    asr_request_id TEXT,
    word_count INTEGER NOT NULL DEFAULT 0,
    semantic_status TEXT,
    semantic_error TEXT,
    visual_status TEXT,
    visual_error TEXT,
    rewrite_status TEXT,
    rewrite_error TEXT,
    render_status TEXT,
    render_output_path TEXT,
    render_error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS finding (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    title TEXT NOT NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    modality TEXT NOT NULL,
    quote TEXT NOT NULL,
    rationale TEXT NOT NULL,
    start_ms INTEGER,
    end_ms INTEGER,
    confidence REAL NOT NULL DEFAULT 0.8,
    needs_manual_time INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'rule',
    evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS suggestion (
    id TEXT PRIMARY KEY,
    finding_id TEXT NOT NULL,
    before_text TEXT NOT NULL,
    after_text TEXT NOT NULL,
    method TEXT NOT NULL,
    abc_class TEXT NOT NULL,
    estimated_duration_ms INTEGER,
    original_duration_ms INTEGER,
    duration_warning INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.8,
    status TEXT NOT NULL DEFAULT 'pending',
    action TEXT,
    edit_instruction TEXT,
    kept_selling_point TEXT,
    alternatives_json TEXT,
    rewrite_status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recheck (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    status TEXT NOT NULL,
    remaining_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    suggestion_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    material_id TEXT,
    batch_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT,
    progress INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_material_batch ON material(batch_id);
CREATE INDEX IF NOT EXISTS idx_finding_material ON finding(material_id);
CREATE INDEX IF NOT EXISTS idx_suggestion_finding ON suggestion(finding_id);
CREATE TABLE IF NOT EXISTS audit_case (
    id TEXT PRIMARY KEY,
    api_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    adv TEXT,
    adv_name TEXT,
    material_id TEXT,
    video_id TEXT,
    filename TEXT,
    platform TEXT,
    template TEXT,
    detail TEXT,
    quotes_json TEXT,
    first_seen TEXT,
    last_seen TEXT,
    synced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_sync (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_synced_at TEXT,
    total INTEGER DEFAULT 0,
    inserted INTEGER DEFAULT 0,
    updated INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_audit_case_kind ON audit_case(kind);
CREATE INDEX IF NOT EXISTS idx_audit_case_template ON audit_case(template);
CREATE INDEX IF NOT EXISTS idx_job_status ON job(status);
CREATE INDEX IF NOT EXISTS idx_job_material ON job(material_id);
CREATE INDEX IF NOT EXISTS idx_job_batch ON job(batch_id);
"""

MATERIAL_MIGRATIONS = {
    "transcript_source": "TEXT",
    "asr_model": "TEXT",
    "asr_error": "TEXT",
    "asr_request_id": "TEXT",
    "word_count": "INTEGER NOT NULL DEFAULT 0",
    "semantic_status": "TEXT",
    "semantic_error": "TEXT",
    "visual_status": "TEXT",
    "visual_error": "TEXT",
    "rewrite_status": "TEXT",
    "rewrite_error": "TEXT",
    "render_status": "TEXT",
    "render_output_path": "TEXT",
    "render_error": "TEXT",
}

FINDING_MIGRATIONS = {
    "source": "TEXT NOT NULL DEFAULT 'rule'",
    "evidence_json": "TEXT",
}

SUGGESTION_MIGRATIONS = {
    "action": "TEXT",
    "edit_instruction": "TEXT",
    "kept_selling_point": "TEXT",
    "alternatives_json": "TEXT",
    "rewrite_status": "TEXT",
}


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(material)").fetchall()}
    for name, ddl in MATERIAL_MIGRATIONS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE material ADD COLUMN {name} {ddl}")
    conn.execute(
        "UPDATE material SET transcript_source = 'srt' "
        "WHERE transcript_source IS NULL AND srt_path IS NOT NULL"
    )
    conn.execute(
        "UPDATE material SET transcript_source = 'script' "
        "WHERE transcript_source IS NULL AND transcript IS NOT NULL AND TRIM(transcript) <> ''"
    )
    finding_columns = {row[1] for row in conn.execute("PRAGMA table_info(finding)").fetchall()}
    for name, ddl in FINDING_MIGRATIONS.items():
        if name not in finding_columns:
            conn.execute(f"ALTER TABLE finding ADD COLUMN {name} {ddl}")
    suggestion_columns = {row[1] for row in conn.execute("PRAGMA table_info(suggestion)").fetchall()}
    for name, ddl in SUGGESTION_MIGRATIONS.items():
        if name not in suggestion_columns:
            conn.execute(f"ALTER TABLE suggestion ADD COLUMN {name} {ddl}")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
