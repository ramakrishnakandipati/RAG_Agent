"""Mock PMS/CRM backed by SQLite: Guest360 lookup, live ancillary inventory, folio postings, outbox, HITL queue."""
from __future__ import annotations
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.config import get_settings
from src.schemas.domain import Guest360, StayContext
from src.utils.logger import get_logger

log = get_logger("pms")
_init_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS guests(guest_id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, email TEXT, segment TEXT,
  loyalty_tier TEXT, country TEXT, total_stays INT, ota_stays INT, direct_stays INT, ltv_usd REAL, avg_adr REAL,
  opt_in_marketing INT, avg_sentiment REAL);
CREATE TABLE IF NOT EXISTS reservations(reservation_id TEXT PRIMARY KEY, guest_id TEXT, channel TEXT, channel_group TEXT,
  check_in TEXT, check_out TEXT, nights INT, party_size INT, room_type TEXT, adr REAL, lead_days INT, status TEXT);
CREATE TABLE IF NOT EXISTS inventory(date TEXT, sku TEXT, capacity INT, booked INT, PRIMARY KEY(date, sku));
CREATE TABLE IF NOT EXISTS folio(id INTEGER PRIMARY KEY AUTOINCREMENT, reservation_id TEXT, sku TEXT, qty INT, amount REAL,
  service_date TEXT, ts TEXT, UNIQUE(reservation_id, sku));
CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT, reservation_id TEXT, guest_id TEXT,
  ts TEXT, channel TEXT, subject TEXT, body TEXT, sku TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS hitl_queue(thread_id TEXT PRIMARY KEY, reservation_id TEXT, guest_id TEXT, reasons TEXT, draft TEXT,
  offer TEXT, created TEXT, status TEXT, decision TEXT);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, thread_id TEXT, reservation_id TEXT,
  guest_id TEXT, action TEXT, sku TEXT, amount REAL, status TEXT, detail TEXT);
"""


@contextmanager
def connect():
    s = get_settings()
    con = sqlite3.connect(s.db_path, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
    finally:
        con.close()


def init_pms(force: bool = False) -> None:
    """Load raw CSVs into SQLite (idempotent). `force` rebuilds everything."""
    s = get_settings()
    with _init_lock:
        if force and s.db_path.exists():
            s.db_path.unlink(missing_ok=True)
        with connect() as con:
            con.executescript(SCHEMA)
            if con.execute("SELECT COUNT(*) FROM guests").fetchone()[0] > 0:
                return
            g = pd.read_csv(s.raw_dir / "guests.csv")
            g["opt_in_marketing"] = g["opt_in_marketing"].astype(int)
            r = pd.read_csv(s.raw_dir / "reservations.csv").drop(columns=["room_revenue"])
            inv = pd.read_csv(s.raw_dir / "inventory.csv")
            g.to_sql("guests", con, if_exists="append", index=False, method="multi", chunksize=500)
            r.to_sql("reservations", con, if_exists="append", index=False, method="multi", chunksize=500)
            inv.to_sql("inventory", con, if_exists="append", index=False, method="multi", chunksize=500)
        log.info("pms_initialised", guests=len(g), reservations=len(r), inventory=len(inv))


def reset_runtime_tables(reload_inventory: bool = True) -> None:
    """Clear folio/outbox/hitl/audit and restore inventory to the generated baseline."""
    s = get_settings()
    with connect() as con:
        for t in ("folio", "outbox", "hitl_queue", "audit"):
            con.execute(f"DELETE FROM {t}")
        if reload_inventory:
            con.execute("DELETE FROM inventory")
            pd.read_csv(s.raw_dir / "inventory.csv").to_sql("inventory", con, if_exists="append", index=False, method="multi", chunksize=500)


# ------------------------------------------------------------------------------------------------- reads
def get_guest_360(guest_id: str) -> Optional[dict]:
    with connect() as con:
        row = con.execute("SELECT * FROM guests WHERE guest_id=?", (guest_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["opt_in_marketing"] = bool(d["opt_in_marketing"])
    return Guest360(**d).model_dump()


def get_stay(reservation_id: str) -> Optional[dict]:
    with connect() as con:
        row = con.execute("SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
    if not row:
        return None
    return StayContext(**dict(row)).model_dump(mode="json")


def list_upcoming_reservations(limit: int = 200, min_days: int = 1, max_days: int = 10) -> list[dict]:
    from data.synthetic_generator import TODAY
    lo, hi = (TODAY + timedelta(days=min_days)).isoformat(), (TODAY + timedelta(days=max_days)).isoformat()
    with connect() as con:
        rows = con.execute("SELECT r.reservation_id, r.guest_id, r.check_in, r.channel_group, g.loyalty_tier, g.segment "
                           "FROM reservations r JOIN guests g USING(guest_id) WHERE r.status='confirmed' "
                           "AND r.check_in BETWEEN ? AND ? ORDER BY r.check_in LIMIT ?", (lo, hi, limit)).fetchall()
    return [dict(r) for r in rows]


def get_availability(sku: str, service_date: str) -> int:
    with connect() as con:
        row = con.execute("SELECT capacity-booked AS free FROM inventory WHERE sku=? AND date=?", (sku, service_date)).fetchone()
    return int(max(row["free"], 0)) if row else 0


def recent_message_count(guest_id: str, days: int = 7) -> int:
    since = (datetime.now() - timedelta(days=days)).isoformat()
    with connect() as con:
        return int(con.execute("SELECT COUNT(*) FROM outbox WHERE guest_id=? AND ts>=? AND status='SENT'",
                               (guest_id, since)).fetchone()[0])


# ------------------------------------------------------------------------------------------------- writes
def book_ancillary_item(reservation_id: str, sku: str, price: float, service_date: str, qty: int = 1) -> dict:
    """Atomic real-time inventory check + decrement + folio posting. Idempotent per (reservation, sku)."""
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            dup = con.execute("SELECT id, amount FROM folio WHERE reservation_id=? AND sku=?", (reservation_id, sku)).fetchone()
            if dup:
                con.execute("COMMIT")
                return {"ok": True, "already_booked": True, "folio_id": dup["id"], "amount": dup["amount"]}
            row = con.execute("SELECT capacity-booked AS free FROM inventory WHERE sku=? AND date=?", (sku, service_date)).fetchone()
            if not row or row["free"] < qty:
                con.execute("ROLLBACK")
                return {"ok": False, "reason": "sold_out"}
            con.execute("UPDATE inventory SET booked=booked+? WHERE sku=? AND date=?", (qty, sku, service_date))
            cur = con.execute("INSERT INTO folio(reservation_id, sku, qty, amount, service_date, ts) VALUES(?,?,?,?,?,?)",
                              (reservation_id, sku, qty, round(price * qty, 2), service_date, datetime.now().isoformat()))
            con.execute("COMMIT")
            return {"ok": True, "folio_id": cur.lastrowid, "amount": round(price * qty, 2), "already_booked": False}
        except Exception:
            con.execute("ROLLBACK")
            raise


def record_message(thread_id: str, reservation_id: str, guest_id: str, subject: str, body: str,
                   sku: Optional[str], status: str = "SENT", channel: str = "email") -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO outbox(thread_id, reservation_id, guest_id, ts, channel, subject, body, sku, status) "
                          "VALUES(?,?,?,?,?,?,?,?,?)", (thread_id, reservation_id, guest_id, datetime.now().isoformat(),
                                                        channel, subject, body, sku, status))
        return int(cur.lastrowid)


def write_audit(thread_id: str, reservation_id: str, guest_id: str, action: str, status: str,
                sku: Optional[str] = None, amount: Optional[float] = None, detail: Optional[dict] = None) -> None:
    with connect() as con:
        con.execute("INSERT INTO audit(ts, thread_id, reservation_id, guest_id, action, sku, amount, status, detail) "
                    "VALUES(?,?,?,?,?,?,?,?,?)", (datetime.now().isoformat(), thread_id, reservation_id, guest_id, action,
                                                  sku, amount, status, json.dumps(detail or {}, default=str)))


def get_folio(reservation_id: str) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM folio WHERE reservation_id=?", (reservation_id,))]


# ------------------------------------------------------------------------------------------------- HITL queue
def enqueue_hitl(thread_id: str, reservation_id: str, guest_id: str, reasons: list[str], draft: dict, offer: Optional[dict]) -> None:
    with connect() as con:
        con.execute("INSERT INTO hitl_queue(thread_id, reservation_id, guest_id, reasons, draft, offer, created, status) "
                    "VALUES(?,?,?,?,?,?,?, 'PENDING') ON CONFLICT(thread_id) DO UPDATE SET reasons=excluded.reasons, "
                    "draft=excluded.draft, offer=excluded.offer WHERE status='PENDING'",
                    (thread_id, reservation_id, guest_id, json.dumps(reasons), json.dumps(draft, default=str),
                     json.dumps(offer, default=str), datetime.now().isoformat()))


def list_hitl(status: str = "PENDING") -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM hitl_queue WHERE status=? ORDER BY created", (status,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("reasons", "draft", "offer", "decision"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        out.append(d)
    return out


def resolve_hitl(thread_id: str, decision: dict) -> None:
    with connect() as con:
        con.execute("UPDATE hitl_queue SET status=?, decision=? WHERE thread_id=?",
                    ("APPROVED" if decision.get("decision") == "approve" else "REJECTED", json.dumps(decision), thread_id))
