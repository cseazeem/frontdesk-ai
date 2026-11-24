"""Static clinic configuration. Everything here is fictional demo data."""

from datetime import time
from zoneinfo import ZoneInfo

CLINIC_NAME = "Lakeside Family Medicine"
CLINIC_TZ = ZoneInfo("America/Chicago")
CLINIC_ADDRESS = "2100 Lakeshore Boulevard, Suite 200, Austin, Texas"

OPEN_TIME = time(9, 0)
CLOSE_TIME = time(17, 0)
LUNCH_START = time(12, 0)
LUNCH_END = time(13, 0)
SLOT_MINUTES = 30

# visit type -> (duration in minutes, plain-English label)
VISIT_TYPES: dict[str, tuple[int, str]] = {
    "new_patient": (60, "new patient visit"),
    "annual_physical": (60, "annual physical"),
    "follow_up": (30, "follow-up visit"),
    "sick_visit": (30, "sick visit"),
}

# Providers: id, display name, credentials, working weekdays (0=Mon), visit types they see.
PROVIDERS = [
    {
        "id": 1,
        "name": "Dr. Sarah Chen",
        "title": "MD, Family Medicine",
        "weekdays": [0, 1, 2, 3, 4],
        "visit_types": ["new_patient", "annual_physical", "follow_up", "sick_visit"],
    },
    {
        "id": 2,
        "name": "Dr. Marcus Johnson",
        "title": "MD, Family Medicine and Pediatrics",
        "weekdays": [0, 1, 3],
        "visit_types": ["new_patient", "annual_physical", "follow_up", "sick_visit"],
    },
    {
        "id": 3,
        "name": "Priya Patel",
        "title": "Nurse Practitioner",
        "weekdays": [0, 1, 2, 3, 4],
        "visit_types": ["follow_up", "sick_visit"],
    },
]

ACCEPTED_INSURANCE = [
    "Aetna",
    "Blue Cross Blue Shield",
    "Cigna",
    "UnitedHealthcare",
    "Humana",
    "Medicare",
]

# Refills for these cannot be requested over the phone; they need a provider visit.
CONTROLLED_SUBSTANCES = [
    "adderall", "amphetamine", "ritalin", "methylphenidate", "vyvanse",
    "oxycodone", "hydrocodone", "percocet", "vicodin", "tramadol", "codeine", "morphine",
    "xanax", "alprazolam", "valium", "diazepam", "klonopin", "clonazepam", "ativan", "lorazepam",
    "ambien", "zolpidem",
]

LATE_CANCEL_HOURS = 24
LATE_CANCEL_FEE = 50

CLINIC_INFO = {
    "hours": "Monday through Friday, 9 AM to 5 PM, closed for lunch from noon to 1 PM. "
    "Closed on weekends and major holidays.",
    "location": f"{CLINIC_ADDRESS}. We're on the second floor, across from Lakeshore Park.",
    "parking": "Free parking in the garage behind the building. Validate your ticket at the front desk.",
    "insurance": "We accept " + ", ".join(ACCEPTED_INSURANCE[:-1]) + f", and {ACCEPTED_INSURANCE[-1]}. "
    "Self-pay is $150 for a new patient visit and $95 for a follow-up or sick visit.",
    "cancellation_policy": f"Please give at least {LATE_CANCEL_HOURS} hours notice. Cancellations with "
    f"less notice, or missed appointments, have a ${LATE_CANCEL_FEE} fee.",
    "what_to_bring": "Photo ID, insurance card, a list of current medications, and arrive 15 minutes early "
    "for a new patient visit.",
    "after_hours": "For urgent needs after hours, call the nurse line at the main number and press 2. "
    "For emergencies, call 911.",
    "providers": "; ".join(f"{p['name']}, {p['title']}" for p in PROVIDERS),
}
