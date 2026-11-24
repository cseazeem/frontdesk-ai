"""Tool definitions, written once and shared by the live voice bot and the caller simulator."""

from frontdesk import config

VISIT_TYPE_ENUM = list(config.VISIT_TYPES)
INFO_TOPICS = list(config.CLINIC_INFO)

# name -> (description, properties, required)
TOOL_SPECS: dict[str, tuple[str, dict, list[str]]] = {
    "verify_patient": (
        "Verify an existing patient's identity. Required before sharing or changing anything about their "
        "appointments, refills or records. Works for a parent calling about their child too: use the child's details.",
        {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
        },
        ["first_name", "last_name", "date_of_birth"],
    ),
    "register_new_patient": (
        "Register a new patient (someone who has never been seen here). Collect all fields first.",
        {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
            "phone": {"type": "string", "description": "10-digit US phone number"},
            "insurance": {"type": "string", "description": "Insurance company name, or empty for self-pay"},
        },
        ["first_name", "last_name", "date_of_birth", "phone"],
    ),
    "check_availability": (
        "Find up to three open appointment times. Use before offering any time to the caller.",
        {
            "visit_type": {"type": "string", "enum": VISIT_TYPE_ENUM},
            "earliest_date": {"type": "string", "description": "YYYY-MM-DD. Leave empty for the soonest openings."},
            "provider": {"type": "string", "description": "Provider name if the caller has a preference, else empty"},
            "time_of_day": {"type": "string", "enum": ["any", "morning", "afternoon"]},
        },
        ["visit_type"],
    ),
    "book_appointment": (
        "Book an appointment for the verified patient. Only after the caller has confirmed the time out loud.",
        {
            "visit_type": {"type": "string", "enum": VISIT_TYPE_ENUM},
            "start": {"type": "string", "description": "Exact start from check_availability, e.g. 2026-11-25T14:30"},
            "provider": {"type": "string"},
            "reason": {"type": "string", "description": "Short reason for visit in the caller's words"},
        },
        ["visit_type", "start", "provider"],
    ),
    "get_my_appointments": (
        "List the verified patient's upcoming appointments.",
        {},
        [],
    ),
    "cancel_appointment": (
        "Cancel one of the verified patient's appointments.",
        {
            "appointment_id": {"type": "integer"},
            "caller_accepted_late_fee": {
                "type": "boolean",
                "description": "Set true only after the caller explicitly agreed to the late-cancellation fee",
            },
        },
        ["appointment_id"],
    ),
    "reschedule_appointment": (
        "Move one of the verified patient's appointments to a new time from check_availability.",
        {
            "appointment_id": {"type": "integer"},
            "new_start": {"type": "string", "description": "e.g. 2026-11-25T14:30"},
            "provider": {"type": "string", "description": "Leave empty to keep the same provider"},
        },
        ["appointment_id", "new_start"],
    ),
    "request_prescription_refill": (
        "Send a refill request to the nurse for the verified patient.",
        {
            "medication": {"type": "string"},
            "pharmacy": {"type": "string", "description": "Pharmacy name and location, or empty for the one on file"},
        },
        ["medication"],
    ),
    "request_nurse_callback": (
        "Ask a nurse to call the verified patient back about a medical question you cannot answer.",
        {"question": {"type": "string", "description": "The caller's question, briefly"}},
        ["question"],
    ),
    "get_clinic_info": (
        "Look up clinic facts. Never guess hours, address, insurance or policies.",
        {"topic": {"type": "string", "enum": INFO_TOPICS}},
        ["topic"],
    ),
    "escalate_emergency": (
        "Call IMMEDIATELY if the caller describes possible emergency symptoms (chest pain, trouble breathing, "
        "stroke signs, severe bleeding, allergic swelling, thoughts of self-harm). Returns what to tell them.",
        {"symptoms": {"type": "string"}},
        ["symptoms"],
    ),
    "transfer_to_staff": (
        "Transfer to a human: the caller asks for a person, is very upset, has a billing or records request, "
        "or failed identity verification.",
        {"reason": {"type": "string"}},
        ["reason"],
    ),
    "end_call": (
        "End the call after the caller's needs are handled and you have said goodbye.",
        {
            "outcome": {
                "type": "string",
                "enum": ["booked", "rescheduled", "cancelled", "info_provided", "refill_requested",
                         "callback_requested", "no_action", "caller_hung_up"],
            }
        },
        ["outcome"],
    ),
}


def pipecat_function_schemas():
    """FunctionSchema objects (without handlers) for the Pipecat voice bot."""
    from pipecat.adapters.llm.schemas.function_schema import FunctionSchema

    return [
        FunctionSchema(name=name, description=desc, properties=props, required=req)
        for name, (desc, props, req) in TOOL_SPECS.items()
    ]


def genai_tool():
    """A google-genai Tool with every function declaration, for the text simulator."""
    from google.genai import types

    decls = []
    for name, (desc, props, req) in TOOL_SPECS.items():
        params = {"type": "object", "properties": props, "required": req} if props else None
        decls.append(types.FunctionDeclaration(name=name, description=desc, parameters_json_schema=params))
    return types.Tool(function_declarations=decls)
