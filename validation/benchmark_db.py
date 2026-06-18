"""SQLite-backed store for a benchmark run."""

import sqlite3

from validation.ned_score import NedResult, TokenEvent

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id        TEXT PRIMARY KEY,
    kern_text        TEXT,
    actual_text      TEXT,
    ned              REAL,
    distance         INTEGER,
    kern_len         INTEGER,
    xml_len          INTEGER,
    rhythm_ned       REAL,
    pitch_ned        REAL,
    lift_ned         REAL,
    articulation_ned REAL,
    slur_ned         REAL,
    error            TEXT
);

CREATE TABLE IF NOT EXISTS token_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id        TEXT NOT NULL,
    staff            TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    exp_rhythm       TEXT,
    exp_pitch        TEXT,
    exp_lift         TEXT,
    exp_articulation TEXT,
    exp_slur         TEXT,
    act_rhythm       TEXT,
    act_pitch        TEXT,
    act_lift         TEXT,
    act_articulation TEXT,
    act_slur         TEXT
);

CREATE INDEX IF NOT EXISTS ix_te_sample      ON token_events(sample_id);
CREATE INDEX IF NOT EXISTS ix_te_event       ON token_events(event_type);
CREATE INDEX IF NOT EXISTS ix_te_exp_rhythm  ON token_events(exp_rhythm);
CREATE INDEX IF NOT EXISTS ix_te_exp_pitch   ON token_events(exp_pitch);
CREATE INDEX IF NOT EXISTS ix_te_act_pitch   ON token_events(act_pitch);
CREATE INDEX IF NOT EXISTS ix_te_exp_lift    ON token_events(exp_lift);
CREATE INDEX IF NOT EXISTS ix_te_act_lift    ON token_events(act_lift);
"""

_INSERT_SAMPLE = """
INSERT OR REPLACE INTO samples
    (sample_id, kern_text, actual_text, ned, distance, kern_len, xml_len,
     rhythm_ned, pitch_ned, lift_ned, articulation_ned, slur_ned, error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_FAILURE = """
INSERT OR REPLACE INTO samples (sample_id, kern_text, actual_text, error)
VALUES (?, ?, ?, ?)
"""

_INSERT_EVENT = """
INSERT INTO token_events
    (sample_id, staff, event_type,
     exp_rhythm, exp_pitch, exp_lift, exp_articulation, exp_slur,
     act_rhythm, act_pitch, act_lift, act_articulation, act_slur)
VALUES
    (:sample_id, :staff, :event_type,
     :exp_rhythm, :exp_pitch, :exp_lift, :exp_articulation, :exp_slur,
     :act_rhythm, :act_pitch, :act_lift, :act_articulation, :act_slur)
"""

_DELETE_EVENTS = "DELETE FROM token_events WHERE sample_id=?"


class BenchmarkDB:
    """SQLite-backed store for a benchmark run."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE_SCHEMA)
        self._conn.commit()

    def already_processed_ids(self) -> set[str]:
        return {row[0] for row in self._conn.execute("SELECT sample_id FROM samples")}

    def write_success(
        self,
        sample_id: str,
        kern_text: str,
        actual_text: str,
        result: NedResult,
        events: list[TokenEvent],
    ) -> None:
        self._conn.execute(
            _INSERT_SAMPLE,
            (
                sample_id,
                kern_text,
                actual_text,
                result.ned,
                result.distance,
                result.kern_len,
                result.xml_len,
                result.rhythm_ned,
                result.pitch_ned,
                result.lift_ned,
                result.articulation_ned,
                result.slur_ned,
                None,
            ),
        )
        self._conn.executemany(
            _INSERT_EVENT,
            [{"sample_id": sample_id, **e} for e in events],
        )
        self._conn.commit()

    def write_failure(
        self, sample_id: str, kern_text: str, error: str, actual_text: str | None = None
    ) -> None:
        self._conn.execute(_INSERT_FAILURE, (sample_id, kern_text, actual_text, error))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
