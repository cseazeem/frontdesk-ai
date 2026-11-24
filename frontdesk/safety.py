"""Deterministic guardrails that run alongside the LLM.

The model is told how to handle emergencies and PHI, but a prompt is not a
control. These checks run on every caller utterance and every stored record,
so a missed emergency or leaked date of birth is caught even when the model slips.
"""

import re

# Phrases that should always trigger the emergency path. Kept deliberately broad:
# a false positive costs one extra question; a false negative can cost a life.
EMERGENCY_PATTERNS = [
    r"chest (pain|pressure|tightness|hurts)",
    r"(pain|pressure|tightness) in my chest",
    r"(can'?t|cannot|hard to|trouble|difficulty) breath",
    r"short(ness)? of breath",
    r"(arm|jaw) (is )?(numb|tingling)",
    r"numb(ness)? in my (arm|face|jaw)",
    r"face (is )?droop",
    r"slurr(ed|ing)",
    r"\bstroke\b",
    r"heart attack",
    r"(passed|passing) out|unconscious|fainted",
    r"(severe|heavy|won'?t stop) bleeding|bleeding (a lot|heavily)",
    r"(throat|tongue|lips?) (is |are )?(swelling|swollen|closing)",
    r"overdos(e|ed)",
    r"seizure",
    r"(kill|hurt|harm) (myself|me)",
    r"suicid",
    r"(want|going) to die",
    r"end (my|it all)",
    r"dolor (de|en el) pecho",
    r"no puedo respirar",
]
_EMERGENCY_RE = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)

SELF_HARM_RE = re.compile(r"(kill|hurt|harm) (myself|me)|suicid|(want|going) to die|end (my|it all)", re.IGNORECASE)


def detect_emergency(utterance: str) -> str | None:
    """Return the matched phrase if the utterance sounds like an emergency."""
    m = _EMERGENCY_RE.search(utterance or "")
    return m.group(0) if m else None


def is_self_harm(utterance: str) -> bool:
    return bool(SELF_HARM_RE.search(utterance or ""))


# --- PHI redaction -----------------------------------------------------------------

_MONTHS = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)"
_REDACTIONS = [
    # 1985-03-14, 03/14/1985, 3-14-85
    (re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"), "[DATE]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    # March 14th, 1985 / 14 March 1985 / March fourteenth nineteen eighty five
    (re.compile(rf"\b{_MONTHS}\.?\s+\w+(\s*,)?\s+(\d{{4}}|nineteen[\w\s-]+|two thousand[\w\s-]*)", re.IGNORECASE), "[DATE]"),
    (re.compile(rf"\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{_MONTHS}\s+\d{{4}}\b", re.IGNORECASE), "[DATE]"),
    # phone numbers
    (re.compile(r"(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "[PHONE]"),
    # long digit runs (member IDs, SSN-like)
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ID]"),
    (re.compile(r"\b\d{7,}\b"), "[ID]"),
]

SENSITIVE_ARG_KEYS = {"date_of_birth", "dob", "phone", "member_id"}


def redact_text(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def redact_args(args: dict) -> dict:
    return {k: ("[REDACTED]" if k in SENSITIVE_ARG_KEYS else v) for k, v in (args or {}).items()}
