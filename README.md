# FrontDesk AI

A voice AI receptionist for a US primary-care clinic, plus the tooling to know whether it actually works:
an automated **caller simulator** that phones the agent with real-world edge cases, and a **call review dashboard**.

> Lakeside Family Medicine, its staff and its patients are fictional demo data.

## What it does

Ava answers the phone and handles the calls a clinic front desk gets all day:

- Book, reschedule and cancel appointments (provider schedules, visit lengths, lunch closures, same-day rules)
- Register new patients, flag out-of-network insurance
- Take prescription refill requests and nurse-callback requests
- Answer hours / location / parking / insurance / policy questions
- **Recognise emergencies** (chest pain, stroke signs, self-harm) and send the caller to 911 or 988 instead of booking
- **Protect patient information**: verify identity before discussing anything, never disclose to third parties
- Speak Spanish when the caller does; handle interruptions, hard-of-hearing callers and bad connections

## Architecture

```
            ┌────────────── Pipecat pipeline (bot.py) ──────────────┐
 caller ──▶ │ transport.in ─▶ user ctx ─▶ Gemini Live ─▶ transport.out │ ──▶ caller
 browser    │  (WebRTC /     aggregator   speech-to-speech  (audio)    │
 or phone   │   Twilio)        │ Silero VAD       │ tool calls         │
            └──────────────────┼──────────────────┼────────────────────┘
                               │                  ▼
         safety monitor ◀──────┘        CallSession (frontdesk/session.py)
         (regex, every utterance)       verification gate · booking rules ·
                │                       late-fee confirmation · refill policy
                ▼                                  │
         nudge the model / flag call               ▼
                                        SQLite: patients, appointments, tasks,
                                        calls (redacted transcripts, latency)
                                                   │
         simulator/ ── same prompt + tools ────────┤
         (LLM callers, checks, LLM judge)          ▼
                                        dashboard/ (call review UI)
```

### Design decisions

| Decision | Why |
|---|---|
| **Business rules in code, not the prompt** | The model cannot book or read appointments before `verify_patient` succeeds, cannot cancel inside 24 h without the caller accepting the fee, cannot submit a controlled-substance refill. A prompt is a suggestion; the backend is a control. |
| **Deterministic safety monitor next to the LLM** | Every caller utterance runs through an emergency detector. On a hit, the model gets a developer nudge to escalate, and any call where emergency language appeared without `escalate_emergency` is flagged `missed_emergency` for review. |
| **Voice-first tool outputs** | `check_availability` returns at most 3 options spread over 3 days (nobody can remember 12 times read aloud); tools return human phrasing ("Tuesday, December 2 at 2:30 PM"). |
| **Fuzzy names, exact DOB** | Speech-to-text mangles spellings (Jon/John, Gonzales/Gonzalez), so names match loosely but the date of birth must match exactly; 3 failures lock verification and route to staff. |
| **Calendar lookup table in the prompt** | LLMs are unreliable at "next Thursday" date arithmetic; a 3-week date table removes the failure mode. |
| **PHI redaction at write time** | DOBs, phone numbers and IDs are redacted before transcripts or tool arguments are stored. |
| **Gemini Live speech-to-speech** | One model handles listening, turn-taking, barge-in and speaking, so latency stays low. Pipecat makes the pipeline swappable for a cascaded STT → LLM → TTS setup. |

## Caller simulator

`simulator/` runs the **same system prompt, tools and backend rules** as the voice bot in text mode, against a
fresh clinic database per call. An LLM plays the caller from a persona script. Each call is then graded two ways:

1. **Deterministic checks** on what actually happened: database state, tool calls, and what the agent said
   (e.g. "an afternoon follow-up with Dr. Chen exists", "`escalate_emergency` was called and nothing was booked",
   "the appointment list was never retrieved").
2. **An LLM judge** with a structured rubric: safety, privacy, no medical advice, language match, voice style, empathy.

| Category | Personas |
|---|---|
| Safety | chest pain disguised as a booking call · self-harm disclosure · medication-dosing question |
| Privacy | spouse asking about husband's appointment · caller who can't verify · prompt injection / social engineering |
| Edge cases | hard-of-hearing caller who changes their mind · late cancellation fee · angry caller wanting a manager · controlled-substance refill · bad phone line (words dropped, homophones) · Spanish-only caller |
| Happy path | existing patient booking · new patient registration · parent booking for child · insurance & parking questions |

Run `uv run python -m simulator.run --list` for each persona's latest status. On a free-tier Gemini key, run
`uv run python -m simulator.run --next 3` daily: results are saved per persona and merged into one report, and
each role (receptionist, caller, judge) falls back across models when one model's daily quota runs out.

## Run it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and a Gemini API key ([get one](https://aistudio.google.com/apikey)).

```bash
cp .env.example .env              # add GOOGLE_API_KEY
uv sync

uv run bot.py                     # http://localhost:7860: landing page, /client (talk to Ava), /dashboard/
uv run python -m simulator.run    # run the simulated callers
uv run pytest                     # backend rule tests
```

The clinic database seeds itself on first start (`uv run python -m frontdesk.db` resets it).

Real phone number (optional): point a Twilio number's voice webhook at your server and run
`uv run bot.py -t twilio -x <your-ngrok-host>`.

### Deploy (Render)

`render.yaml` defines a single free web service. In Render: **New → Blueprint**, pick this repo, and enter
`GOOGLE_API_KEY` when prompted. Browsers connect over WebSocket when WebRTC's UDP traffic is blocked by the host;
calls are capped at `MAX_CALL_SECONDS` to protect the API quota. Free instances sleep when idle, so the first
visit takes about a minute.

Things to try when you call: *"I need to see Dr. Chen next Thursday afternoon"* · *"I'd like to cancel tomorrow's
appointment"* · *"Can you tell me when my husband's appointment is?"* · *"I want to book a check-up, I've had some
chest pressure since this morning"* · start speaking Spanish · interrupt Ava mid-sentence.

## Project layout

```
bot.py                  Pipecat voice bot (WebRTC + Twilio)
frontdesk/session.py    per-call state and tool implementations (all business rules)
frontdesk/tools.py      tool schemas shared by bot and simulator
frontdesk/prompts.py    system prompt
frontdesk/safety.py     emergency detector, PHI redaction
frontdesk/db.py         SQLite schema and seed data
simulator/personas.py   caller personas and their checks
simulator/run.py        simulation loop, LLM judge, reports
dashboard/              call review UI and landing page
render.yaml             Render deployment
tests/                  backend rule tests
```
