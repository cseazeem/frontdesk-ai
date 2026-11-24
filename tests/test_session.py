from datetime import datetime

import pytest

from frontdesk import db, safety
from frontdesk.session import CallSession

# A Tuesday morning. Seed data is anchored relative to this.
NOW = datetime(2026, 12, 1, 10, 15)


@pytest.fixture
def session():
    conn = db.connect(":memory:")
    db.init_db(conn, NOW)
    return CallSession(conn, channel="test", now=NOW)


def test_verification_tolerates_stt_spelling_but_needs_exact_dob(session):
    assert session.call_tool("verify_patient", {"first_name": "Mariah", "last_name": "Gonzales",
                                                "date_of_birth": "1985-03-14"})["verified"]
    s2 = CallSession(session.conn, now=NOW)
    r = s2.call_tool("verify_patient", {"first_name": "Maria", "last_name": "Gonzalez", "date_of_birth": "1985-03-15"})
    assert not r["verified"] and s2.verified_patient_id is None


def test_verification_locks_after_three_failures(session):
    bad = {"first_name": "Aisha", "last_name": "Rahman", "date_of_birth": "1991-01-25"}
    for _ in range(3):
        assert not session.call_tool("verify_patient", bad)["verified"]
    good = dict(bad, date_of_birth="1990-01-25")
    r = session.call_tool("verify_patient", good)
    assert r.get("locked") and not r["verified"]


def test_cannot_read_or_book_before_verification(session):
    assert session.call_tool("get_my_appointments", {})["error"] == "NOT_VERIFIED"
    opts = session.call_tool("check_availability", {"visit_type": "follow_up"})["options"]
    r = session.call_tool("book_appointment", {"visit_type": "follow_up", "start": opts[0]["start"],
                                               "provider": opts[0]["provider"]})
    assert r["error"] == "NOT_VERIFIED"


def test_availability_offers_three_different_days_and_books(session):
    session.call_tool("verify_patient", {"first_name": "Maria", "last_name": "Gonzalez", "date_of_birth": "1985-03-14"})
    opts = session.call_tool("check_availability", {"visit_type": "follow_up", "provider": "Dr. Chen",
                                                    "time_of_day": "afternoon"})["options"]
    assert len(opts) == 3
    days = {o["start"][:10] for o in opts}
    assert len(days) == 3
    assert all(int(o["start"][11:13]) >= 13 for o in opts)
    r = session.call_tool("book_appointment", {"visit_type": "follow_up", "start": opts[0]["start"],
                                               "provider": "Chen", "reason": "knee pain"})
    assert r["booked"]
    # The same slot is now gone.
    again = session.call_tool("book_appointment", {"visit_type": "follow_up", "start": opts[0]["start"],
                                                   "provider": "Chen"})
    assert not again["booked"]


def test_specific_day_returns_that_day(session):
    opts = session.call_tool("check_availability", {"visit_type": "sick_visit", "earliest_date": "2026-12-03"})
    assert opts["options"] and all(o["start"].startswith("2026-12-03") for o in opts["options"])


def test_provider_rules(session):
    r = session.call_tool("check_availability", {"visit_type": "annual_physical", "provider": "Priya Patel"})
    assert "error" in r
    # Dr. Johnson does not work Wednesdays (2026-12-02).
    session.call_tool("verify_patient", {"first_name": "Maria", "last_name": "Gonzalez", "date_of_birth": "1985-03-14"})
    r = session.call_tool("book_appointment", {"visit_type": "follow_up", "start": "2026-12-02T09:00",
                                               "provider": "Dr. Johnson"})
    assert not r["booked"] and "Wednesday" in r["message"]


def test_non_sick_visits_cannot_be_same_day(session):
    session.call_tool("verify_patient", {"first_name": "Maria", "last_name": "Gonzalez", "date_of_birth": "1985-03-14"})
    r = session.call_tool("book_appointment", {"visit_type": "follow_up", "start": "2026-12-01T16:30",
                                               "provider": "Chen"})
    assert not r["booked"]


def test_late_cancellation_needs_explicit_fee_acceptance(session):
    session.call_tool("verify_patient", {"first_name": "James", "last_name": "Carter", "date_of_birth": "1979-11-30"})
    appts = session.call_tool("get_my_appointments", {})["appointments"]
    assert len(appts) == 1
    aid = appts[0]["appointment_id"]
    r = session.call_tool("cancel_appointment", {"appointment_id": aid})
    assert r["requires_confirmation"] and not r["cancelled"]
    r = session.call_tool("cancel_appointment", {"appointment_id": aid, "caller_accepted_late_fee": True})
    assert r["cancelled"] and r["late_fee_applied"]


def test_reschedule_and_cannot_touch_other_patients_appointments(session):
    session.call_tool("verify_patient", {"first_name": "Robert", "last_name": "Miller", "date_of_birth": "1952-07-02"})
    aid = session.call_tool("get_my_appointments", {})["appointments"][0]["appointment_id"]
    opt = session.call_tool("check_availability", {"visit_type": "follow_up", "provider": "Chen",
                                                   "earliest_date": "2026-12-08"})["options"][0]
    r = session.call_tool("reschedule_appointment", {"appointment_id": aid, "new_start": opt["start"]})
    assert r["rescheduled"]
    # Switch to a different verified patient: Robert's appointment id is no longer theirs.
    session.call_tool("verify_patient", {"first_name": "Maria", "last_name": "Gonzalez", "date_of_birth": "1985-03-14"})
    assert "error" in session.call_tool("cancel_appointment", {"appointment_id": aid})


def test_controlled_substance_refill_is_refused(session):
    session.call_tool("verify_patient", {"first_name": "Kevin", "last_name": "Brooks", "date_of_birth": "1993-04-12"})
    assert not session.call_tool("request_prescription_refill", {"medication": "Adderall XR 20mg"})["submitted"]
    assert session.call_tool("request_prescription_refill", {"medication": "lisinopril"})["submitted"]


def test_register_new_patient_flags_out_of_network_insurance(session):
    r = session.call_tool("register_new_patient", {"first_name": "nina", "last_name": "patel",
                                                   "date_of_birth": "1999-02-02", "phone": "(512) 555-0100",
                                                   "insurance": "Oscar Health"})
    assert r["registered"] and "insurance_note" in r
    assert session.verified_patient_id is not None


def test_missed_emergency_is_flagged(session):
    session.record_user("I'd like to book a visit, I've had chest pain since this morning")
    record = session.to_record()
    kinds = {f["kind"] for f in record["flags"]}
    assert {"possible_emergency", "missed_emergency"} <= kinds


def test_escalated_emergency_is_not_flagged_as_missed(session):
    session.record_user("my chest hurts and my left arm is numb")
    session.call_tool("escalate_emergency", {"symptoms": "chest pain, arm numbness"})
    kinds = {f["kind"] for f in session.to_record()["flags"]}
    assert "missed_emergency" not in kinds and "emergency_escalated" in kinds


def test_phi_is_redacted_in_logs(session):
    session.record_user("my birthday is March 14th, 1985 and my number is 512-555-0142")
    session.call_tool("verify_patient", {"first_name": "Maria", "last_name": "Gonzalez", "date_of_birth": "1985-03-14"})
    text = str(session.to_record()["events"])
    assert "1985" not in text and "555-0142" not in text


@pytest.mark.parametrize("text", ["I can't breathe very well", "I think I'm having a stroke",
                                  "sometimes I want to kill myself", "tengo dolor de pecho"])
def test_emergency_detector(text):
    assert safety.detect_emergency(text)


@pytest.mark.parametrize("text", ["I need to reschedule my physical", "my kid has a cough",
                                  "what's your parking situation"])
def test_emergency_detector_no_false_positive(text):
    assert safety.detect_emergency(text) is None
