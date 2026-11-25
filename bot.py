"""FrontDesk AI voice bot.

Run:  uv run bot.py            -> talk to Ava at http://localhost:7860, review calls at /dashboard/
      uv run bot.py -t twilio -x <your-ngrok-host>   -> answer a real phone number

Audio goes straight to Gemini Live (speech-to-speech), which handles listening, turn-taking,
interruptions and speaking. Pipecat wires it to the transport, runs our tools, and observes
every frame so we can log transcripts, latency and safety flags per call.
"""

import asyncio
import json
import os
import wave
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndWorkerFrame, LLMMessagesAppendFrame, LLMRunFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVISendTextFrame
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

from frontdesk import db, prompts
from frontdesk.session import CallSession
from frontdesk.tools import TOOL_SPECS

load_dotenv(override=True)

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL")  # unset = Pipecat's default Gemini Live model
VOICE = os.getenv("GEMINI_VOICE", "Aoede")

if not db.DEFAULT_DB_PATH.exists():
    from datetime import datetime

    from frontdesk import config

    db.DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.init_db(db.connect(), datetime.now(config.CLINIC_TZ).replace(tzinfo=None, second=0, microsecond=0))
CONN = db.connect()
RECORDINGS_DIR = db.DEFAULT_DB_PATH.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Tools after which the call should wrap up. The model still says its closing line:
# Gemini Live defers the end until the bot finishes its current turn.
ENDING_TOOLS = {"end_call", "transfer_to_staff"}

SAFETY_NUDGE = (
    "SAFETY MONITOR: the caller's last words may describe a medical emergency. If there is any chance of that, "
    "call escalate_emergency now and follow its script. Do not continue booking."
)

# Public demo guardrail: cap call length so one open tab can't burn the whole API quota.
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "300"))
MAX_PLAUSIBLE_LATENCY = 15.0  # seconds

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    # Browser over WebSocket: works on hosts that block the UDP traffic WebRTC needs (e.g. Render).
    "websocket": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True, audio_out_enabled=True, serializer=ProtobufFrameSerializer()
    ),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
}


class ChatTextObserver(BaseObserver):
    """Typed messages from the web client skip speech transcription; catch them so they're logged too."""

    def __init__(self):
        super().__init__()
        self.on_text = None
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        if isinstance(f, RTVISendTextFrame) and f.id not in self._seen and self.on_text:
            self._seen.add(f.id)
            await self.on_text(f.text)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    if isinstance(runner_args, WebSocketRunnerArguments):
        channel = "web" if runner_args.transport_type == "websocket" else "phone"
    else:
        channel = "webrtc"
    session = CallSession(CONN, channel=channel)
    logger.info(f"Call {session.call_id} started ({channel})")

    async def handle_tool(params: FunctionCallParams):
        result = session.call_tool(params.function_name, dict(params.arguments or {}))
        logger.info(f"[{session.call_id}] {params.function_name} -> {result}")
        await params.result_callback(result)
        if params.function_name in ENDING_TOOLS:
            await params.llm.push_frame(EndWorkerFrame())

    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(name=n, description=d, properties=p, required=r, handler=handle_tool)
            for n, (d, p, r) in TOOL_SPECS.items()
        ]
    )

    settings = GeminiLiveLLMService.Settings(system_instruction=prompts.system_prompt(session.now()), voice=VOICE)
    if LIVE_MODEL:
        settings.model = LIVE_MODEL
    llm = GeminiLiveLLMService(api_key=API_KEY, settings=settings, tools=tools)

    context = LLMContext([{"role": "developer", "content": prompts.GREETING_INSTRUCTION}])
    # Gemini Live does turn detection server-side. Local Silero VAD is supplemental: it emits
    # user-speech frames so the latency observer can time "caller stopped talking -> Ava started".
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    # Stereo call recording: caller on the left channel, Ava on the right. Reviewing real calls is how
    # you find what the simulator missed.
    recorder = AudioBufferProcessor(num_channels=2)

    pipeline = Pipeline(
        [transport.input(), user_aggregator, llm, transport.output(), recorder, assistant_aggregator]
    )

    latency = UserBotLatencyObserver()
    chat = ChatTextObserver()
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency, chat],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    nudged = False

    @latency.event_handler("on_latency_measured")
    async def on_latency(_observer, seconds: float):
        # A gap this long is not a reply: background noise tripped the local VAD, or the tab sat idle.
        if seconds > MAX_PLAUSIBLE_LATENCY:
            logger.warning(f"[{session.call_id}] ignoring implausible latency {seconds:.1f}s")
            return
        session.metrics["latencies"].append(round(seconds, 3))
        logger.info(f"[{session.call_id}] response latency {seconds:.2f}s")

    async def handle_caller_text(text: str):
        nonlocal nudged
        hit = session.record_user(text)
        if hit and not nudged:
            # Deterministic safety net: don't rely on the model alone to notice an emergency.
            nudged = True
            await worker.queue_frames(
                [LLMMessagesAppendFrame([{"role": "developer", "content": SAFETY_NUDGE}], run_llm=True)]
            )

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_speech(_agg, message: UserTurnMessageAddedMessage):
        await handle_caller_text(str(message.content))

    chat.on_text = handle_caller_text  # typed messages from the web client

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_agent_text(_agg, message: AssistantTurnStoppedMessage):
        if message.content:
            session.record_agent(str(message.content))

    @recorder.event_handler("on_audio_data")
    async def on_audio(_recorder, audio: bytes, sample_rate: int, num_channels: int):
        if not audio:
            return
        path = RECORDINGS_DIR / f"{session.call_id}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
        session.metrics["recording"] = path.name
        session.save()
        logger.info(f"[{session.call_id}] recording saved: {path}")

    async def enforce_time_limit():
        await asyncio.sleep(MAX_CALL_SECONDS)
        logger.info(f"[{session.call_id}] demo time limit reached")
        session.outcome = session.outcome or "time_limit"
        await worker.queue_frames([
            LLMMessagesAppendFrame([{"role": "developer", "content": "The demo call time is up. Apologize briefly, "
                                     "say goodbye, and call end_call."}], run_llm=True)
        ])
        await asyncio.sleep(20)
        await worker.queue_frames([EndWorkerFrame()])

    timer: asyncio.Task | None = None

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _client):
        nonlocal timer
        await recorder.start_recording()
        await worker.queue_frames([LLMRunFrame()])
        timer = asyncio.create_task(enforce_time_limit())

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        await runner.cancel()

    try:
        await runner.run()
    finally:
        if timer:
            timer.cancel()
        record = session.save()
        logger.info(f"Call {session.call_id} saved: outcome={record['outcome']} flags={record['flags']}")


async def bot(runner_args: RunnerArguments):
    if not API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY in .env")
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import app, main

    from dashboard.server import landing, router as dashboard_router

    # One server for everything: / (landing), /client (talk to Ava), /dashboard/ (call review),
    # plus the runner's WebRTC and WebSocket endpoints. Registered first, so our "/" wins.
    app.include_router(landing)
    app.include_router(dashboard_router, prefix="/dashboard")

    @app.middleware("http")
    async def public_ws_url(request: Request, call_next):
        """Behind a proxy (Render, Fly...), advertise the public wss:// URL instead of the bind address."""
        response = await call_next(request)
        if request.method != "POST" or request.url.path != "/start":
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            data = json.loads(body)
        except ValueError:
            return Response(body, status_code=response.status_code, media_type=response.media_type)
        if isinstance(data, dict) and data.get("wsUrl"):
            proto = request.headers.get("x-forwarded-proto", request.url.scheme)
            host = request.headers.get("x-forwarded-host") or request.headers.get("host")
            data["wsUrl"] = f"{'wss' if proto == 'https' else 'ws'}://{host}{urlparse(data['wsUrl']).path}"
        return JSONResponse(data, status_code=response.status_code)
    main()
