"""Every inspection result, in SQLite.

SQLite because the alternative on a Pi is a database server nobody will
maintain. It is a single file, needs no daemon, survives power loss with WAL
enabled, and the dashboard reads it with the same library that writes it.

The one thing to be careful about is that this is a hot loop: at 25 FPS a naive
implementation opens a connection, commits, and fsyncs 25 times a second, and
the SD card will not thank you. The write path here is deliberately cheap --
see `_configure`.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL    NOT NULL,          -- unix seconds, UTC
    verdict        TEXT    NOT NULL,          -- PASS or FAIL
    reason         TEXT    NOT NULL,          -- why, in words, for the operator
    defect_class   TEXT,                      -- highest-confidence class, or NULL
    confidence     REAL,
    n_defects      INTEGER NOT NULL DEFAULT 0,
    anomaly_score  REAL,                      -- NULL when PatchCore did not run
    latency_ms     REAL    NOT NULL,
    ejected        INTEGER NOT NULL DEFAULT 0,
    image_path     TEXT                       -- saved frame for failures
);

-- The dashboard's two queries are "most recent N" and "counts since T".
-- Both are index scans with this in place, and full table scans without it.
CREATE INDEX IF NOT EXISTS idx_inspections_ts ON inspections(ts);
CREATE INDEX IF NOT EXISTS idx_inspections_verdict_ts ON inspections(verdict, ts);
"""


@dataclass
class InspectionRecord:
    ts: float
    verdict: str
    reason: str
    defect_class: str | None
    confidence: float | None
    n_defects: int
    anomaly_score: float | None
    latency_ms: float
    ejected: bool
    image_path: str | None = None


class Storage:
    def __init__(self, db_path: str | Path, commit_every: int = 25):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False because the Flask dashboard reads from a
        # different thread than the pipeline writes from. Every access is
        # guarded by the lock below, so this is safe -- unguarded, it is not.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._pending = 0
        self.commit_every = commit_every

        self._configure()
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def _configure(self) -> None:
        """Pragmas that decide whether this survives on an SD card.

        WAL lets the dashboard read while the pipeline writes, instead of the
        two blocking each other. synchronous=NORMAL skips an fsync per commit;
        combined with WAL the worst case on power loss is losing the last few
        records, never a corrupt database. For inspection logging that trade is
        obviously right -- and batching commits means we are not doing 25 disk
        syncs a second on flash storage with finite write cycles.
        """
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def insert(self, record: InspectionRecord) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO inspections (ts, verdict, reason, defect_class, "
                "confidence, n_defects, anomaly_score, latency_ms, ejected, image_path) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (record.ts, record.verdict, record.reason, record.defect_class,
                 record.confidence, record.n_defects, record.anomaly_score,
                 record.latency_ms, int(record.ejected), record.image_path),
            )
            self._pending += 1
            # A FAIL commits immediately. Losing a pass record on power loss is
            # a statistic; losing the record of a part that was physically
            # ejected is a traceability hole.
            if record.verdict == "FAIL" or self._pending >= self.commit_every:
                self.conn.commit()
                self._pending = 0
            return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self, since: float | None = None) -> dict:
        clause, params = ("WHERE ts >= ?", (since,)) if since else ("", ())
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) AS total, "
                f"SUM(verdict = 'FAIL') AS fails, "
                f"SUM(ejected) AS ejected, "
                f"AVG(latency_ms) AS avg_latency "
                f"FROM inspections {clause}", params
            ).fetchone()
            by_class = self.conn.execute(
                f"SELECT defect_class, COUNT(*) AS n FROM inspections "
                f"{clause + ' AND' if clause else 'WHERE'} defect_class IS NOT NULL "
                f"GROUP BY defect_class ORDER BY n DESC", params
            ).fetchall()

        total = row["total"] or 0
        fails = row["fails"] or 0
        return {
            "total": total,
            "fails": fails,
            "passes": total - fails,
            "ejected": row["ejected"] or 0,
            "reject_rate": (fails / total) if total else 0.0,
            "avg_latency_ms": row["avg_latency"] or 0.0,
            "by_class": {r["defect_class"]: r["n"] for r in by_class},
        }

    def latency_percentile(self, p: float = 0.95, since: float | None = None) -> float:
        """The p-th percentile latency, in milliseconds.

        p95 rather than the mean because a conveyor is sized by its worst case:
        one slow frame misses a part, and a good average does not help that part.
        reports/benchmark.md makes the same argument about the offline numbers;
        this is the live equivalent.

        SQLite has no PERCENTILE(). Pulling every row into Python to sort it
        would mean loading the whole table to answer one number, so instead the
        database does it: skip p*N rows in sorted order and take the next one.
        Always call this with `since` -- the sort is over the filtered set, which
        is cheap for a window and a full table sort without one.
        """
        clause, params = ("WHERE ts >= ?", [since]) if since else ("", [])
        with self._lock:
            n = self.conn.execute(
                f"SELECT COUNT(*) FROM inspections {clause}", params
            ).fetchone()[0]
            if not n:
                return 0.0
            # min() guards p=1.0, which would otherwise offset past the last row.
            offset = min(int(n * p), n - 1)
            row = self.conn.execute(
                f"SELECT latency_ms FROM inspections {clause} "
                f"ORDER BY latency_ms LIMIT 1 OFFSET ?", params + [offset]
            ).fetchone()
        return float(row[0]) if row else 0.0

    def timeseries(self, since: float, bucket_s: float = 10.0) -> list[dict]:
        """Throughput and reject count per time bucket, oldest first.

        Grouped in SQL rather than in Python: the chart needs a few dozen points
        and the table holds hundreds of thousands of rows. Integer division on
        ts is what buckets them, and the ts index carries the WHERE.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT CAST(ts / ? AS INTEGER) * ? AS bucket, "
                "COUNT(*) AS n, "
                "SUM(verdict = 'FAIL') AS fails, "
                "AVG(latency_ms) AS avg_latency "
                "FROM inspections WHERE ts >= ? "
                "GROUP BY bucket ORDER BY bucket", (bucket_s, bucket_s, since),
            ).fetchall()
        return [{"t": r["bucket"], "n": r["n"], "fails": r["fails"] or 0,
                 "avg_latency_ms": r["avg_latency"] or 0.0} for r in rows]

    def flush(self) -> None:
        with self._lock:
            if self._pending:
                self.conn.commit()
                self._pending = 0

    def close(self) -> None:
        self.flush()
        with self._lock:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def demo() -> None:
    """Self-check against a throwaway database."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "selfcheck.db"
        store = Storage(db, commit_every=100)

        now = time.time()
        store.insert(InspectionRecord(now, "PASS", "no defects", None, None,
                                      0, None, 18.2, False))
        store.insert(InspectionRecord(now + 1, "FAIL", "scratches 0.91", "scratches",
                                      0.91, 1, None, 19.4, True, "reports/f1.jpg"))
        store.insert(InspectionRecord(now + 2, "FAIL", "anomaly 3.10", None,
                                      None, 0, 3.10, 44.0, True))

        recent = store.recent(10)
        assert len(recent) == 3, recent
        assert recent[0]["verdict"] == "FAIL", "recent must be newest first"
        assert recent[0]["anomaly_score"] == 3.10

        s = store.summary()
        assert s["total"] == 3 and s["fails"] == 2 and s["passes"] == 1, s
        assert s["ejected"] == 2, s
        assert abs(s["reject_rate"] - 2 / 3) < 1e-9, s
        assert s["by_class"] == {"scratches": 1}, s

        # A FAIL must be committed the moment it is written, not batched, so a
        # second connection sees it immediately.
        other = sqlite3.connect(str(db))
        n = other.execute("SELECT COUNT(*) FROM inspections WHERE verdict='FAIL'").fetchone()[0]
        other.close()
        assert n == 2, f"FAIL rows should commit immediately, saw {n}"

        since = store.summary(since=now + 1.5)
        assert since["total"] == 1, since

        # Percentiles. Latencies are 18.2, 19.4, 44.0 -- with three rows the
        # p95 offset lands on the last, and p50 on the middle one.
        assert store.latency_percentile(0.95) == 44.0, store.latency_percentile(0.95)
        assert store.latency_percentile(0.50) == 19.4, store.latency_percentile(0.50)
        # p=1.0 must not run off the end of the table.
        assert store.latency_percentile(1.0) == 44.0
        # A window with nothing in it is 0.0, not a crash and not a stale value.
        assert store.latency_percentile(0.95, since=now + 99) == 0.0

        # One record per second, bucketed at one second, so three buckets.
        series = store.timeseries(since=now - 1, bucket_s=1.0)
        assert len(series) == 3, series
        assert sum(b["n"] for b in series) == 3, series
        assert sum(b["fails"] for b in series) == 2, series
        assert series == sorted(series, key=lambda b: b["t"]), "must be oldest first"
        # Coarser buckets must collapse, not duplicate.
        assert len(store.timeseries(since=now - 1, bucket_s=3600.0)) <= 2

        store.close()
        print("storage self-check passed: 3 records, FAIL commits immediately, "
              "summary, filtering, percentiles and bucketing correct")


if __name__ == "__main__":
    demo()
