"""SQLite storage: schema, connection helper and demo seed data."""

import json
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from frontdesk import config

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clinic.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    dob TEXT NOT NULL,              -- YYYY-MM-DD
    phone TEXT,
    insurance TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),   -- NULL = slot held by a patient outside the demo
    provider_id INTEGER NOT NULL,
    start TEXT NOT NULL,            -- local clinic time, YYYY-MM-DDTHH:MM
    duration_min INTEGER NOT NULL,
    visit_type TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'booked',        -- booked | cancelled
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    kind TEXT NOT NULL,             -- refill | callback | staff_transfer | emergency
    details TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,          -- webrtc | phone | simulator
    started_at TEXT NOT NULL,
    ended_at TEXT,
    outcome TEXT,
    flags TEXT NOT NULL DEFAULT '[]',
    events TEXT NOT NULL DEFAULT '[]',   -- redacted transcript + tool calls
    metrics TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_appt_provider_start ON appointments(provider_id, start);
"""

# Named demo patients. `appt` = (provider_id, days_from_anchor, "HH:MM", visit_type, reason).
# Several of these are referenced by the caller simulator personas.
DEMO_PATIENTS = [
    {"first": "Maria", "last": "Gonzalez", "dob": "1985-03-14", "phone": "512-555-0142",
     "ins": "Aetna", "appt": None},
    {"first": "Robert", "last": "Miller", "dob": "1952-07-02", "phone": "512-555-0178",
     "ins": "Medicare", "appt": (1, 2, "14:00", "follow_up", "blood pressure check")},
    {"first": "James", "last": "Carter", "dob": "1979-11-30", "phone": "512-555-0110",
     "ins": "Cigna", "appt": (1, 1, "09:30", "follow_up", "lab results review")},
    {"first": "David", "last": "Kim", "dob": "1970-05-21", "phone": "512-555-0199",
     "ins": "UnitedHealthcare", "appt": (2, 6, "10:00", "annual_physical", "annual physical")},
    {"first": "Emma", "last": "Wilson", "dob": "2018-09-09", "phone": "512-555-0133",
     "ins": "Blue Cross Blue Shield", "appt": None},
    {"first": "Kevin", "last": "Brooks", "dob": "1993-04-12", "phone": "512-555-0121",
     "ins": "Aetna", "appt": None},
    {"first": "Aisha", "last": "Rahman", "dob": "1990-01-25", "phone": "512-555-0187",
     "ins": "Cigna", "appt": (3, 3, "11:00", "follow_up", "thyroid follow-up")},
    {"first": "Carlos", "last": "Mendoza", "dob": "1968-04-04", "phone": "512-555-0165",
     "ins": "Humana", "appt": None},
    {"first": "Tom", "last": "Becker", "dob": "1975-08-19", "phone": "512-555-0154",
     "ins": "UnitedHealthcare", "appt": None},
    {"first": "Hannah", "last": "Lee", "dob": "1995-12-03", "phone": "512-555-0116",
     "ins": "Blue Cross Blue Shield", "appt": None},
    {"first": "Gloria", "last": "Stevens", "dob": "1960-06-10", "phone": "512-555-0148",
     "ins": "Medicare", "appt": None},
    {"first": "Frank", "last": "Russo", "dob": "1958-02-17", "phone": "512-555-0172",
     "ins": "Medicare", "appt": None},
]


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def next_weekday(d: date, n: int) -> date:
    """The date `n` business days after `d` (n=0 returns d itself, even on a weekend)."""
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def init_db(conn: sqlite3.Connection, anchor: datetime, fill_ratio: float = 0.55, seed: int = 7) -> None:
    """Drop everything and seed demo data relative to `anchor` (clinic-local "now")."""
    conn.executescript(
        "DROP TABLE IF EXISTS appointments; DROP TABLE IF EXISTS tasks; "
        "DROP TABLE IF EXISTS patients; DROP TABLE IF EXISTS calls;"
    )
    conn.executescript(SCHEMA)
    created = anchor.isoformat(timespec="minutes")
    today = anchor.date()

    for p in DEMO_PATIENTS:
        cur = conn.execute(
            "INSERT INTO patients(first_name, last_name, dob, phone, insurance, created_at) VALUES (?,?,?,?,?,?)",
            (p["first"], p["last"], p["dob"], p["phone"], p["ins"], created),
        )
        if p["appt"]:
            provider_id, bdays, hhmm, vtype, reason = p["appt"]
            day = next_weekday(today, bdays)
            conn.execute(
                "INSERT INTO appointments(patient_id, provider_id, start, duration_min, visit_type, reason, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (cur.lastrowid, provider_id, f"{day.isoformat()}T{hhmm}",
                 config.VISIT_TYPES[vtype][0], vtype, reason, created),
            )

    # Fill part of the next three weeks with appointments for patients outside the demo,
    # so availability looks like a real, partly-booked clinic.
    rng = random.Random(seed)
    taken = {(r["provider_id"], r["start"]) for r in conn.execute("SELECT provider_id, start FROM appointments")}
    for offset in range(0, 22):
        day = today + timedelta(days=offset)
        for prov in config.PROVIDERS:
            if day.weekday() not in prov["weekdays"]:
                continue
            for start in day_slots(day):
                key = (prov["id"], start.isoformat(timespec="minutes"))
                if key in taken or rng.random() > fill_ratio:
                    continue
                conn.execute(
                    "INSERT INTO appointments(patient_id, provider_id, start, duration_min, visit_type, created_at)"
                    " VALUES (NULL,?,?,?,?,?)",
                    (prov["id"], key[1], config.SLOT_MINUTES, "follow_up", created),
                )
    conn.commit()


def day_slots(day: date) -> list[datetime]:
    """All bookable slot start times on a day (ignores provider schedules)."""
    slots = []
    t = datetime.combine(day, config.OPEN_TIME)
    end = datetime.combine(day, config.CLOSE_TIME)
    lunch_start = datetime.combine(day, config.LUNCH_START)
    lunch_end = datetime.combine(day, config.LUNCH_END)
    while t < end:
        if not (lunch_start <= t < lunch_end):
            slots.append(t)
        t += timedelta(minutes=config.SLOT_MINUTES)
    return slots


def save_call(conn: sqlite3.Connection, record: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO calls(id, channel, started_at, ended_at, outcome, flags, events, metrics)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            record["id"], record["channel"], record["started_at"], record.get("ended_at"),
            record.get("outcome"), json.dumps(record.get("flags", [])),
            json.dumps(record.get("events", [])), json.dumps(record.get("metrics", {})),
        ),
    )
    conn.commit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reset the demo clinic database.")
    parser.add_argument("--path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    Path(args.path).parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(config.CLINIC_TZ).replace(tzinfo=None, second=0, microsecond=0)
    c = connect(args.path)
    init_db(c, now)
    n = c.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
    print(f"Seeded {args.path} anchored at {now:%a %Y-%m-%d %H:%M} ({n} appointments).")
