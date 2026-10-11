# 15 — SQLite and Result Storage

**Prerequisites:** [02 — Python environment and tooling](02_python_env_tooling.md).

---

## 1. Concept

SQLite is a relational database that is not a server. There is no process to
start, no port, no username. The entire database is **one file**, and the
"database engine" is a C library linked into your program. Python ships it in
the standard library — `import sqlite3` and you are done.

It is the most deployed database in existence: every Android phone, every
browser, every iOS app.

---

## 2. Why it is needed

The pipeline makes a decision every 27 ms and then throws it away. Three
questions cannot be answered without persistence:

**Traceability.** A customer returns a batch. Which parts did we inspect, when,
and what did we decide? In regulated manufacturing this is a legal requirement,
not a feature.

**Process control.** Scratches jumped from 5% to 30% of rejects this morning.
That is not a vision problem — a roller is failing. **The inspection system's
most valuable output is often the trend, not the individual verdict.**

**Tuning evidence.** Was lowering `conf_threshold` to 0.30 a good idea? Only the
before-and-after reject rates can say.

A log file could do this. A database can *query* it: "reject rate per class per
hour for the last week" is one statement, not a parsing script.

---

## 3. Real-world example

Automotive suppliers operate under IATF 16949, which requires traceability from
finished component back to raw material batch. A vision system that rejects a
part must record what it saw, when, and under which model version — because when
a defect reaches the customer, the audit works backwards through exactly these
records.

The industrial norm is a central SQL Server or historian. **On an edge device
the same discipline applies at smaller scale**, and SQLite is the standard
answer: it survives power loss, needs no administration, and the whole database
copies with `scp`.

---

## 4. How it works

### The schema

```sql
CREATE TABLE IF NOT EXISTS inspections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,      -- unix epoch, float seconds
    verdict     TEXT    NOT NULL,      -- 'PASS' or 'FAIL'
    reason      TEXT,                  -- human-readable cause
    top_class   TEXT,                  -- highest-confidence defect
    confidence  REAL,
    anomaly     REAL,                  -- NULL when the gate did not fire
    latency_ms  REAL    NOT NULL,
    image_path  TEXT                   -- NULL for PASS
);
CREATE INDEX IF NOT EXISTS idx_inspections_ts
    ON inspections(ts);
CREATE INDEX IF NOT EXISTS idx_inspections_verdict_ts
    ON inspections(verdict, ts);
```

Design choices worth defending:

**`ts` is a REAL epoch, not a formatted string.** Sorting a string date works
only for ISO-8601, comparison is lexical, and timezone handling becomes your
problem. A float sorts numerically, subtracts to give durations, and is what
`time.time()` already returns.

**`anomaly` is nullable and that is meaningful.** `NULL` means the gate never
fired — the detector was confident. `0.0` would be a lie: it would claim
PatchCore ran and found nothing unusual.

**`image_path` only for FAIL.** Saving every frame fills the card in hours. A
rejected part is the one someone will want to look at.

### Why the second index

The dashboard's hot query is *"the last 50 failures"*:

```sql
SELECT * FROM inspections WHERE verdict='FAIL' ORDER BY ts DESC LIMIT 50;
```

With only `idx_inspections_ts`, SQLite walks the timestamp index backwards
discarding PASS rows. At a 10% reject rate it reads ~500 rows to return 50; at
1%, 5,000. The composite index `(verdict, ts)` stores FAIL rows already sorted
by time, so it reads exactly 50.

**Index order matters.** `(verdict, ts)` works because equality on the first
column then range on the second is exactly what a B-tree supports. `(ts,
verdict)` would not help this query at all.

### WAL mode

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
```

Default SQLite uses a **rollback journal**: to modify the database it first
copies the original pages to a journal file, then writes in place. Two disk
writes per transaction, and readers are blocked during writes.

**Write-Ahead Logging** inverts this. Changes append to a `-wal` file and the
main database is updated later in bulk. Appending is sequential — much faster on
an SD card, whose random-write performance is poor — and **readers never block
writers**. The Flask dashboard reads while the pipeline writes, with no
contention.

`synchronous=NORMAL` skips an `fsync` on every commit. The honest trade: on
sudden power loss you may lose the last few transactions, but the database
cannot corrupt. `FULL` guarantees every committed row survives at the cost of an
fsync per commit — roughly 10 ms on an SD card, which at 36 FPS is impossible.

---

## 5. Mathematical intuition

### Why an index is not optional

A table scan is $O(n)$. A B-tree index lookup is $O(\log_B n)$ where $B$ is the
number of keys per page — for SQLite's 4 KB pages, typically several hundred.

For one million rows:

| | comparisons |
|---|---|
| Full scan | 1,000,000 |
| B-tree, B=256 | $\log_{256}(10^6) \approx 2.5$ pages |

Six orders of magnitude, for the cost of one `CREATE INDEX`.

Indexes are not free: each one is a second B-tree that must be updated on every
insert. Two indexes on a table that is written 36 times a second and read a few
times a minute is a deliberate choice — it costs microseconds per insert and
saves milliseconds per query, and the queries are what a human waits for.

### Why batching commits matters

Each `COMMIT` is a durability boundary with fixed overhead. At 36 inserts per
second:

- commit per row: 36 × (transaction overhead + WAL sync)
- commit per 20 rows: 1.8 × the same overhead

Roughly a 20× reduction in transaction overhead. The data still reaches the WAL
either way; what changes is how often SQLite pays the fixed cost.

---

## 6. Software implementation

`src/inspector/storage.py`.

```python
def insert(self, record):
    with self._lock:
        self._conn.execute(INSERT_SQL, record.as_row())
        self._pending += 1
        # A FAIL row is committed immediately, whatever the batch state.
        if record.verdict == "FAIL" or self._pending >= self.batch_size:
            self._conn.commit()
            self._pending = 0
```

**FAIL rows commit immediately, PASS rows batch.** This asymmetry is the most
important line in the file. If the power fails, losing the record of nineteen
good parts is an inconvenience. Losing the record of a part that was
*physically ejected* is a traceability hole — the part is in the reject bin and
the database says it never existed.

Durability should be proportional to consequence, not uniform.

```python
self._conn = sqlite3.connect(path, check_same_thread=False)
self._lock = threading.Lock()
```

`check_same_thread=False` disables Python's guard against cross-thread use, and
is necessary because the capture thread writes while Flask reads. Disabling a
safety check obliges you to replace it — hence the explicit `threading.Lock`
around every access. **Never disable that flag without adding the lock.**

### Reading

```python
def summary(self, since=None):
    ...
    SELECT verdict, COUNT(*), AVG(latency_ms) FROM inspections
    WHERE ts >= ? GROUP BY verdict
```

Aggregation happens in SQL, not Python. Pulling 100,000 rows into a list to
count them wastes memory and time; `COUNT(*)` uses the index.

---

## 7. Hardware implementation

**SD cards wear out.** Flash has a finite erase-cycle count per block, and a
consumer card's controller does limited wear levelling. A naive design that
fsyncs a small write 36 times a second writes the same few blocks continuously.

Three mitigations, all present:

1. **WAL** turns random in-place writes into sequential appends.
2. **Batched commits** reduce write count roughly 20×.
3. **Only FAIL images** are written, not every frame.

For a production line, move the database to a USB SSD or NVMe HAT and point
`storage.path` at it. One config line; no code change.

**Power loss is normal, not exceptional.** Industrial machines are switched off
at the mains. WAL plus `synchronous=NORMAL` means the database always opens
cleanly — SQLite replays or discards the WAL on the next open automatically.

**Disk exhaustion is the real risk.** At 36 FPS with a 10% reject rate that is
~3.6 JPEGs per second. Left running, it will fill any card. Retention — delete
images older than N days, `VACUUM` monthly — is not implemented and is a known
gap.

---

## 8. Common mistakes

**Committing every row.** The most common performance bug. Correct default is to
batch; the exception must be justified, as the FAIL case is here.

**`check_same_thread=False` without a lock.** Produces `database is locked`
errors, interleaved transactions, and occasionally corruption. Intermittent and
load-dependent, so it survives testing and fails in the demo.

**No index, then blaming SQLite.** A dashboard that takes four seconds to load
is almost always a missing index, not a slow database.

**String timestamps.** `"09/08/2026 14:03"` sorts wrongly, cannot be subtracted,
and hides timezone bugs. Store epoch floats; format at display time.

**Storing images as BLOBs.** Tempting for atomicity, but it bloats the file,
makes every backup enormous, and prevents looking at a rejected part with an
image viewer. Store the path.

**Assuming `AUTOINCREMENT` counts parts.** It counts *rows*. Restart the service
and the sequence continues; delete rows and numbers are skipped. If you need a
part serial number, that is a separate column.

**Forgetting `IF NOT EXISTS`.** Works on the first run, crashes on the second.

---

## 9. Interview questions

**Q: Why SQLite and not MySQL or PostgreSQL?**
No server process, no configuration, no network. On a Pi doing single-writer
edge logging, a client-server database adds an administration burden, a startup
dependency and memory pressure for capabilities — concurrent writers, network
access, user management — that the workload does not need. If results ever need
to reach a factory-wide system, the right design is to keep SQLite local and
sync upward, so inspection continues when the network fails.

**Q: What is WAL mode and why use it here?**
Write-Ahead Logging appends changes to a separate file rather than modifying the
database in place. Two benefits that matter: sequential appends suit SD cards,
whose random-write performance is poor, and readers do not block writers, so the
dashboard can query while the pipeline inserts.

**Q: You commit FAIL rows immediately but batch PASS rows. Why?**
Because the consequences differ. A lost PASS row is a gap in a statistic; a lost
FAIL row means a part was physically ejected with no record of why. Durability
should match consequence.

**Q: Why two indexes?**
`ts` serves time-range queries. `(verdict, ts)` serves the dashboard's hot query
— recent failures — by storing FAIL rows pre-sorted by time, so it reads 50 rows
instead of scanning hundreds. Column order matters: equality first, range
second, which is what a B-tree supports.

**Q: What happens on power loss mid-write?**
SQLite either replays a complete WAL frame or discards an incomplete one; the
database always opens consistent. With `synchronous=NORMAL` the last few
uncommitted transactions may be lost, which is why FAIL commits immediately.

**Q: How would you scale this to 50 machines?**
Keep SQLite as the local buffer of record, and add a sync process pushing rows
to a central store. Never make inspection depend on network availability — a
line that stops because a database server rebooted is worse than one with a
delayed report.

---

## 10. Summary

- SQLite is a **library, not a server**: one file, zero administration, in the
  Python standard library.
- Stored for three reasons: **traceability, process control, tuning evidence.**
  The trend is often worth more than the individual verdict.
- Timestamps are **epoch floats**. `anomaly` is nullable and `NULL` means "the
  gate never fired", which is different from zero.
- **WAL + `synchronous=NORMAL`**: sequential appends suit SD cards, readers
  never block writers, and the database cannot corrupt on power loss.
- **Two indexes**, including `(verdict, ts)` for the dashboard's hot query.
  Column order is not arbitrary.
- **FAIL commits immediately, PASS batches.** Durability proportional to
  consequence.
- `check_same_thread=False` **requires** an explicit lock. Never one without the
  other.
- Known gap: no retention policy. Images will eventually fill the card.

Next: [16 — The Flask dashboard](16_flask_dashboard.md).
