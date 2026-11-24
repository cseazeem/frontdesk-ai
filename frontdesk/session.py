"""Per-call state and the tools the voice agent can use.

Business rules live here, in code, not in the prompt. The model cannot book,
cancel or read appointment details until the caller has verified their identity,
cannot cancel inside 24 hours without the caller agreeing to the fee, and cannot
request a controlled-substance refill. Whatever the model says, the backend enforces it.
"""

import sqlite3
import time
import uuid
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher

from frontdesk import config, db, safety

MAX_VERIFY_ATTEMPTS = 3
SEARCH_DAYS = 21


def human_time(dt: datetime) -> str:
    """'Tuesday, November 25 at 2:30 PM', the way a receptionist would say it."""
    hour = dt.strftime("%I").lstrip("0")
    minute = "" if dt.minute == 0 else f":{dt.minute:02d}"
    return f"{dt.strftime('%A, %B')} {dt.day} at {hour}{minute} {dt.strftime('%p')}"


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalpha())


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_start(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.strip().replace(" ", "T"))
    except (ValueError, AttributeError):
        return None


class CallSession:
    """One phone call: the verified caller, an event log, and the tool implementations."""

    def __init__(self, conn: sqlite3.Connection, channel: str = "webrtc", now: datetime | None = None):
        self.conn = conn
        self.channel = channel
        self.call_id = uuid.uuid4().hex[:12]
        self._fixed_now = now
        self.started_at = self.now()
        self.verified_patient_id: int | None = None
        self.failed_verifications = 0
        self.outcome: str | None = None
        self.flags: list[dict] = []
        self.events: list[dict] = []
        self.metrics: dict = {"latencies": []}
        self._t0 = time.monotonic()

    # ---- clock & logging ---------------------------------------------------------

    def now(self) -> datetime:
        if self._fixed_now:
            return self._fixed_now
        return datetime.now(config.CLINIC_TZ).replace(tzinfo=None, second=0, microsecond=0)

    def log(self, event_type: str, **data) -> None:
        self.events.append({"t": round(time.monotonic() - self._t0, 2), "type": event_type, **data})

    def flag(self, kind: str, detail: str) -> None:
        if not any(f["kind"] == kind for f in self.flags):
            self.flags.append({"kind": kind, "detail": detail})
            self.log("flag", kind=kind, detail=detail)

    def record_user(self, text: str) -> str | None:
        """Log a caller utterance and run the emergency detector on it."""
        self.log("user", text=safety.redact_text(text))
        hit = safety.detect_emergency(text)
        if hit:
            self.flag("possible_emergency", f'Caller said "{hit}"')
        return hit

    def record_agent(self, text: str) -> None:
        self.log("agent", text=safety.redact_text(text))

    def to_record(self) -> dict:
        called = {e["name"] for e in self.events if e["type"] == "tool_call"}
        if any(f["kind"] == "possible_emergency" for f in self.flags) and "escalate_emergency" not in called:
            self.flag("missed_emergency", "Emergency language detected but the agent never escalated")
        lat = self.metrics.get("latencies", [])
        metrics = dict(self.metrics)
        if lat:
            metrics["latency_avg"] = round(sum(lat) / len(lat), 3)
            metrics["latency_p95"] = round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 3)
        metrics["duration_secs"] = round(time.monotonic() - self._t0, 1)
        metrics["turns"] = sum(1 for e in self.events if e["type"] == "user")
        return {
            "id": self.call_id,
            "channel": self.channel,
            "started_at": self.started_at.isoformat(timespec="minutes"),
            "ended_at": self.now().isoformat(timespec="minutes"),
            "outcome": self.outcome or "caller_hung_up",
            "flags": self.flags,
            "events": self.events,
            "metrics": metrics,
        }

    def save(self) -> dict:
        record = self.to_record()
        db.save_call(self.conn, record)
        return record

    # ---- tool dispatch -----------------------------------------------------------

    def call_tool(self, name: str, args: dict) -> dict:
        """Run a tool by name, logging the call and result (with PHI redacted)."""
        fn = getattr(self, f"tool_{name}", None)
        self.log("tool_call", name=name, args=safety.redact_args(args))
        if fn is None:
            result = {"error": f"Unknown tool {name}"}
        else:
            try:
                result = fn(**(args or {}))
            except TypeError as e:
                result = {"error": f"Bad arguments: {e}"}
        self.log("tool_result", name=name, result=_redact_result(result))
        return result

    # ---- helpers -----------------------------------------------------------------

    def _patient(self, pid: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()

    def _require_verified(self) -> dict | None:
        if self.verified_patient_id is None:
            return {"error": "NOT_VERIFIED", "message": "Verify the caller's identity with verify_patient "
                    "(full name and date of birth) before doing this."}
        return None

    def _provider_by_name(self, name: str | None) -> dict | None:
        if not name:
            return None
        n = name.lower().replace("dr.", "").replace("doctor", "").strip()
        for p in config.PROVIDERS:
            first = p["name"].replace("Dr. ", "").split()[0].lower()
            last = p["name"].split()[-1].lower()
            if n and (last in n or first in n or n in p["name"].lower()):
                return p
        if "nurse" in n or "np" in n.split():
            return config.PROVIDERS[2]
        return None

    def _provider(self, pid: int) -> dict:
        return next(p for p in config.PROVIDERS if p["id"] == pid)

    def _is_free(self, provider_id: int, start: datetime, minutes: int, ignore_appt: int | None = None) -> bool:
        end = start + timedelta(minutes=minutes)
        rows = self.conn.execute(
            "SELECT id, start, duration_min FROM appointments WHERE provider_id=? AND status='booked' "
            "AND start LIKE ?",
            (provider_id, start.date().isoformat() + "%"),
        ).fetchall()
        for r in rows:
            if r["id"] == ignore_appt:
                continue
            s = datetime.fromisoformat(r["start"])
            if s < end and start < s + timedelta(minutes=r["duration_min"]):
                return False
        return True

    def _slot_problem(self, provider: dict, visit_type: str, start: datetime, ignore_appt: int | None = None) -> str | None:
        """Why this slot can't be booked, or None if it can."""
        minutes = config.VISIT_TYPES[visit_type][0]
        end = start + timedelta(minutes=minutes)
        day = start.date()
        if visit_type not in provider["visit_types"]:
            return f"{provider['name']} does not do {config.VISIT_TYPES[visit_type][1]}s."
        if day.weekday() not in provider["weekdays"]:
            return f"{provider['name']} does not work on {start:%A}s."
        if start not in db.day_slots(day):
            return "That is not a valid appointment time. Times are on the hour or half hour, 9 to 5, closed noon to 1."
        lunch_s = datetime.combine(day, config.LUNCH_START)
        lunch_e = datetime.combine(day, config.LUNCH_END)
        if start < lunch_e and lunch_s < end:
            return "That visit would run into the lunch closure."
        if end > datetime.combine(day, config.CLOSE_TIME):
            return "That visit would run past closing time."
        earliest = self.now() + timedelta(hours=1)
        if visit_type != "sick_visit":
            earliest = max(earliest, datetime.combine(self.now().date() + timedelta(days=1), config.OPEN_TIME))
        if start < earliest:
            return "That time is too soon. Only sick visits can be booked for today."
        if not self._is_free(provider["id"], start, minutes, ignore_appt):
            return "That time is already taken."
        return None

    def _patient_conflict(self, start: datetime, minutes: int, ignore_appt: int | None = None) -> bool:
        end = start + timedelta(minutes=minutes)
        rows = self.conn.execute(
            "SELECT id, start, duration_min FROM appointments WHERE patient_id=? AND status='booked'",
            (self.verified_patient_id,),
        ).fetchall()
        for r in rows:
            if r["id"] == ignore_appt:
                continue
            s = datetime.fromisoformat(r["start"])
            if s < end and start < s + timedelta(minutes=r["duration_min"]):
                return True
        return False

    def _appt_view(self, r: sqlite3.Row) -> dict:
        start = datetime.fromisoformat(r["start"])
        return {
            "appointment_id": r["id"],
            "when": human_time(start),
            "start": r["start"],
            "provider": self._provider(r["provider_id"])["name"],
            "visit_type": config.VISIT_TYPES[r["visit_type"]][1],
            "reason": r["reason"],
        }

    def _own_appointment(self, appointment_id) -> sqlite3.Row | None:
        try:
            aid = int(appointment_id)
        except (TypeError, ValueError):
            return None
        return self.conn.execute(
            "SELECT * FROM appointments WHERE id=? AND patient_id=? AND status='booked'",
            (aid, self.verified_patient_id),
        ).fetchone()

    # ---- tools -------------------------------------------------------------------

    def tool_verify_patient(self, first_name: str, last_name: str, date_of_birth: str) -> dict:
        if self.failed_verifications >= MAX_VERIFY_ATTEMPTS:
            return {"verified": False, "locked": True, "message": "Too many failed attempts. Do not try again. "
                    "Offer to transfer the caller to front-desk staff."}
        dob = _parse_date(date_of_birth)
        if dob is None:
            return {"verified": False, "message": "Date of birth must be YYYY-MM-DD. Ask the caller again."}
        candidates = self.conn.execute("SELECT * FROM patients WHERE dob=?", (dob.isoformat(),)).fetchall()
        best, best_score = None, 0.0
        for c in candidates:
            # Speech-to-text often mangles spellings (Jon/John, Smyth/Smith), so match names loosely
            # but require the exact date of birth.
            score = min(_name_similarity(first_name, c["first_name"]), _name_similarity(last_name, c["last_name"]))
            if score > best_score:
                best, best_score = c, score
        if best is None or best_score < 0.75:
            self.failed_verifications += 1
            left = MAX_VERIFY_ATTEMPTS - self.failed_verifications
            return {"verified": False, "attempts_left": left,
                    "message": "No match. Do not say whether this person is a patient. Ask the caller to "
                    "spell their name and repeat their date of birth." if left else
                    "No match and no attempts left. Offer to transfer to front-desk staff."}
        self.verified_patient_id = best["id"]
        age = (self.now().date() - dob).days // 365
        return {"verified": True, "first_name": best["first_name"], "last_name": best["last_name"],
                "age": age, "insurance_on_file": best["insurance"]}

    def tool_register_new_patient(self, first_name: str, last_name: str, date_of_birth: str,
                                  phone: str, insurance: str = "") -> dict:
        dob = _parse_date(date_of_birth)
        if dob is None or dob >= self.now().date() or dob.year < 1900:
            return {"error": "Invalid date of birth. Confirm it with the caller as month, day and year."}
        digits = "".join(ch for ch in phone if ch.isdigit())
        if len(digits) not in (10, 11):
            return {"error": "Phone number should be 10 digits. Ask the caller to repeat it."}
        existing = self.conn.execute("SELECT * FROM patients WHERE dob=?", (dob.isoformat(),)).fetchall()
        for c in existing:
            if _name_similarity(first_name, c["first_name"]) > 0.8 and _name_similarity(last_name, c["last_name"]) > 0.8:
                self.verified_patient_id = c["id"]
                return {"already_registered": True, "message": "This caller is already a patient; they are now "
                        "verified. No need to register again."}
        ins_note = None
        if insurance:
            match = next((i for i in config.ACCEPTED_INSURANCE if _norm(i) in _norm(insurance)
                          or _norm(insurance) in _norm(i)), None)
            if match is None:
                ins_note = (f"We are not in network with {insurance}. Tell the caller visits would be self-pay "
                            "($150 new patient) unless they have another plan, and confirm they want to continue.")
            insurance = match or insurance
        cur = self.conn.execute(
            "INSERT INTO patients(first_name, last_name, dob, phone, insurance, created_at) VALUES (?,?,?,?,?,?)",
            (first_name.strip().title(), last_name.strip().title(), dob.isoformat(), digits,
             insurance or None, self.now().isoformat(timespec="minutes")),
        )
        self.conn.commit()
        self.verified_patient_id = cur.lastrowid
        out = {"registered": True, "next_step": "Book a new_patient visit."}
        if ins_note:
            out["insurance_note"] = ins_note
        return out

    def tool_check_availability(self, visit_type: str, earliest_date: str = "", provider: str = "",
                                time_of_day: str = "any") -> dict:
        if visit_type not in config.VISIT_TYPES:
            return {"error": f"visit_type must be one of {list(config.VISIT_TYPES)}"}
        prov = self._provider_by_name(provider)
        if provider and prov is None:
            return {"error": f"No provider named {provider}. Providers: {config.CLINIC_INFO['providers']}"}
        if prov and visit_type not in prov["visit_types"]:
            return {"error": f"{prov['name']} does not do {config.VISIT_TYPES[visit_type][1]}s. "
                    "Offer another provider."}
        providers = [prov] if prov else [p for p in config.PROVIDERS if visit_type in p["visit_types"]]
        start_day = _parse_date(earliest_date) if earliest_date else None
        specific_day = start_day is not None
        start_day = max(start_day or self.now().date(), self.now().date())
        tod = (time_of_day or "any").lower()

        # Voice UX: never read out a list of twenty times. Offer at most three options, spread across
        # three different days, unless the caller asked for a specific day.
        options: list[dict] = []
        for offset in range(SEARCH_DAYS):
            day = start_day + timedelta(days=offset)
            if specific_day and offset == 1 and options:
                break  # the requested day had openings; don't pad with other days
            per_day_limit = 3 if (specific_day and offset == 0) else 1
            found_today = 0
            for slot in db.day_slots(day):
                if (tod == "morning" and slot.hour >= 12) or (tod == "afternoon" and slot.hour < 13):
                    continue
                p = next((p for p in providers if self._slot_problem(p, visit_type, slot) is None), None)
                if p is None:
                    continue
                options.append({"start": slot.isoformat(timespec="minutes"),
                                "when": human_time(slot), "provider": p["name"]})
                found_today += 1
                if found_today >= per_day_limit or len(options) >= 3:
                    break
            if len(options) >= 3:
                break
        if not options:
            return {"options": [], "message": "Nothing available in the next three weeks with those filters. "
                    "Offer a different provider or time of day, or offer the waitlist via transfer_to_staff."}
        out: dict = {"options": options, "visit_length_minutes": config.VISIT_TYPES[visit_type][0]}
        if specific_day and datetime.fromisoformat(options[0]["start"]).date() != start_day:
            out["note"] = f"Nothing open on {start_day:%A, %B} {start_day.day}; these are the next openings."
        return out

    def tool_book_appointment(self, visit_type: str, start: str, provider: str, reason: str = "") -> dict:
        if err := self._require_verified():
            return err
        if visit_type not in config.VISIT_TYPES:
            return {"error": f"visit_type must be one of {list(config.VISIT_TYPES)}"}
        prov = self._provider_by_name(provider)
        when = _parse_start(start)
        if prov is None or when is None:
            return {"error": "Need a valid provider and a start time like 2026-11-25T14:30 from check_availability."}
        if problem := self._slot_problem(prov, visit_type, when):
            return {"booked": False, "message": problem + " Run check_availability again and offer new times."}
        minutes = config.VISIT_TYPES[visit_type][0]
        if self._patient_conflict(when, minutes):
            return {"booked": False, "message": "The patient already has an appointment at that time."}
        cur = self.conn.execute(
            "INSERT INTO appointments(patient_id, provider_id, start, duration_min, visit_type, reason, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (self.verified_patient_id, prov["id"], when.isoformat(timespec="minutes"), minutes, visit_type,
             reason or None, self.now().isoformat(timespec="minutes")),
        )
        self.conn.commit()
        self.log("booking", appointment_id=cur.lastrowid)
        tip = config.CLINIC_INFO["what_to_bring"] if visit_type == "new_patient" else None
        return {"booked": True, "when": human_time(when), "provider": prov["name"],
                "visit_type": config.VISIT_TYPES[visit_type][1], "remind_caller": tip,
                "confirmation": "A text confirmation goes to the phone number on file."}

    def tool_get_my_appointments(self) -> dict:
        if err := self._require_verified():
            return err
        rows = self.conn.execute(
            "SELECT * FROM appointments WHERE patient_id=? AND status='booked' AND start >= ? ORDER BY start",
            (self.verified_patient_id, self.now().isoformat(timespec="minutes")),
        ).fetchall()
        return {"appointments": [self._appt_view(r) for r in rows]}

    def tool_cancel_appointment(self, appointment_id: int, caller_accepted_late_fee: bool = False) -> dict:
        if err := self._require_verified():
            return err
        appt = self._own_appointment(appointment_id)
        if appt is None:
            return {"error": "No upcoming appointment with that id for this patient. Use get_my_appointments."}
        start = datetime.fromisoformat(appt["start"])
        late = start - self.now() < timedelta(hours=config.LATE_CANCEL_HOURS)
        if late and not caller_accepted_late_fee:
            return {"cancelled": False, "requires_confirmation": True,
                    "message": f"This appointment is less than {config.LATE_CANCEL_HOURS} hours away, so there is a "
                    f"${config.LATE_CANCEL_FEE} late-cancellation fee. Tell the caller, offer to reschedule instead, "
                    "and only call again with caller_accepted_late_fee=true if they agree."}
        self.conn.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (appt["id"],))
        self.conn.commit()
        return {"cancelled": True, "was": human_time(start), "late_fee_applied": late}

    def tool_reschedule_appointment(self, appointment_id: int, new_start: str, provider: str = "") -> dict:
        if err := self._require_verified():
            return err
        appt = self._own_appointment(appointment_id)
        if appt is None:
            return {"error": "No upcoming appointment with that id for this patient. Use get_my_appointments."}
        prov = self._provider_by_name(provider) or self._provider(appt["provider_id"])
        when = _parse_start(new_start)
        if when is None:
            return {"error": "new_start must look like 2026-11-25T14:30."}
        if problem := self._slot_problem(prov, appt["visit_type"], when, ignore_appt=appt["id"]):
            return {"rescheduled": False, "message": problem + " Run check_availability and offer new times."}
        if self._patient_conflict(when, appt["duration_min"], ignore_appt=appt["id"]):
            return {"rescheduled": False, "message": "The patient already has an appointment at that time."}
        self.conn.execute("UPDATE appointments SET start=?, provider_id=? WHERE id=?",
                          (when.isoformat(timespec="minutes"), prov["id"], appt["id"]))
        self.conn.commit()
        return {"rescheduled": True, "from": human_time(datetime.fromisoformat(appt["start"])),
                "to": human_time(when), "provider": prov["name"]}

    def tool_request_prescription_refill(self, medication: str, pharmacy: str = "") -> dict:
        if err := self._require_verified():
            return err
        if any(c in medication.lower() for c in config.CONTROLLED_SUBSTANCES):
            return {"submitted": False, "reason": "controlled_substance",
                    "message": "This medication is a controlled substance. Refills need a visit with the provider; "
                    "they cannot be requested by phone. Offer to book a follow_up visit."}
        self.conn.execute(
            "INSERT INTO tasks(patient_id, kind, details, created_at) VALUES (?,?,?,?)",
            (self.verified_patient_id, "refill", f"{medication} -> {pharmacy or 'pharmacy on file'}",
             self.now().isoformat(timespec="minutes")),
        )
        self.conn.commit()
        return {"submitted": True, "message": "The nurse reviews refill requests within 2 business days and sends "
                "approved refills to the pharmacy. The patient gets a text when it's sent."}

    def tool_request_nurse_callback(self, question: str) -> dict:
        if err := self._require_verified():
            return err
        self.conn.execute(
            "INSERT INTO tasks(patient_id, kind, details, created_at) VALUES (?,?,?,?)",
            (self.verified_patient_id, "callback", question, self.now().isoformat(timespec="minutes")),
        )
        self.conn.commit()
        return {"submitted": True, "message": "A nurse will call back today if before 3 PM, otherwise next business day."}

    def tool_get_clinic_info(self, topic: str) -> dict:
        key = (topic or "").lower().replace(" ", "_")
        if key in config.CLINIC_INFO:
            return {"topic": key, "info": config.CLINIC_INFO[key]}
        return {"topics": list(config.CLINIC_INFO), "message": "Unknown topic; pick one of these."}

    def tool_escalate_emergency(self, symptoms: str) -> dict:
        self.flag("emergency_escalated", symptoms)
        self.conn.execute(
            "INSERT INTO tasks(patient_id, kind, details, priority, created_at) VALUES (?,?,?,?,?)",
            (self.verified_patient_id, "emergency", symptoms, "urgent", self.now().isoformat(timespec="minutes")),
        )
        self.conn.commit()
        self.outcome = "emergency_escalated"
        if safety.is_self_harm(symptoms):
            return {"say_to_caller": "I'm really glad you told me, and your safety matters. Please call or text 988, "
                    "the Suicide and Crisis Lifeline, right now. If you're in immediate danger, call 911.",
                    "agent_note": "Speak calmly and kindly, in your own words. Stay with them; do not continue "
                    "booking. Staff have been alerted."}
        return {"say_to_caller": "What you're describing could be serious. Please hang up and call 911 right now, or "
                "have someone take you to the nearest emergency room. Please don't wait.",
                "agent_note": "Say this calmly in your own words. Do not book an appointment. Staff have been alerted."}

    def tool_transfer_to_staff(self, reason: str) -> dict:
        self.conn.execute(
            "INSERT INTO tasks(patient_id, kind, details, priority, created_at) VALUES (?,?,?,?,?)",
            (self.verified_patient_id, "staff_transfer", reason, "high", self.now().isoformat(timespec="minutes")),
        )
        self.conn.commit()
        self.outcome = "transferred_to_staff"
        return {"transferring": True, "message": "Tell the caller you're connecting them with a team member now, and "
                "thank them for their patience."}

    def tool_end_call(self, outcome: str) -> dict:
        self.outcome = self.outcome or outcome
        return {"ok": True, "message": "Say a short goodbye."}


def _redact_result(result: dict) -> dict:
    return {k: ("[REDACTED]" if k in safety.SENSITIVE_ARG_KEYS else v) for k, v in result.items()}
