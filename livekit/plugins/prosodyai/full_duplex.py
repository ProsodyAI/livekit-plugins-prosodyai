"""Full-duplex bridge: room PCM up, backend speech down, typed events out.

The bridge owns everything ProsodyAI owns on this path: the room-clock
resampling, the uplink hold for the session handshake, and the ProsodySSM
readout plumbing (identity, transcripts, committed model events). The
speaking loop is a :class:`SpeechBackend`; PersonaPlex over the gateway
socket (``WS /v1/realtime``) is the default.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from .audio_resample import (
    float32_to_pcm16_le,
    pcm16_le_to_float32,
    resample_float32,
)
from .personaplex import (
    API_KEY_HEADER,
    DEFAULT_BASE_URL,
    GATEWAY_FRAME_SAMPLES,
    GATEWAY_PATH,
    GATEWAY_SAMPLE_RATE,
    PERSONAPLEX_CAPABILITIES,
    GatewayConnection,
    GatewayControlFrame,
    GatewayEnvError,
    PersonaPlexBackend,
    gateway_ws_url,
)
from .speech_backend import (
    BackendCapabilityError,
    SessionOpened,
    SpeechAudio,
    SpeechBackend,
    SpeechBackendCapabilities,
    SpeechItem,
    SpeechSessionConfig,
    SpeechText,
    require_capabilities,
)
from .wire import (
    KIND_AUDIO,
    KIND_EVENT,
    KIND_HANDSHAKE,
    KIND_IDENTITY,
    KIND_TEXT,
    KIND_TRANSCRIPT,
    ConversationBargeInEvent as BargeInEvent,
    ConversationStateDeltaEvent as StateDeltaEvent,
    ConversationTurnBoundaryEvent as TurnBoundaryEvent,
    ConversationWireEvent as ConversationEvent,
    GatewayAgentToolEvent as AgentToolEvent,
    GatewayAgentToolStatusEvent as AgentToolStatusEvent,
    GatewayIdentityResolvedEvent as IdentityResolvedEvent,
    GatewayModelEvent as ModelEvent,
    GatewayNewSpeakerEvent as NewSpeakerEvent,
    GatewaySpeakerChangeEvent as SpeakerChangeEvent,
    IdentityEvent,
    TextEvent,
    TranscriptDelta,
    TranscriptEvent,
    parse_conversation_event,
    parse_gateway_model_event,
    parse_identity_payload,
    parse_transcript_payload,
)

__all__ = [
    "API_KEY_HEADER",
    "DEFAULT_BASE_URL",
    "GATEWAY_FRAME_SAMPLES",
    "GATEWAY_PATH",
    "GATEWAY_SAMPLE_RATE",
    "KIND_AUDIO",
    "KIND_EVENT",
    "KIND_HANDSHAKE",
    "KIND_IDENTITY",
    "KIND_TEXT",
    "KIND_TRANSCRIPT",
    "PERSONAPLEX_CAPABILITIES",
    "AgentToolEvent",
    "AgentToolStatusEvent",
    "BackendCapabilityError",
    "BargeInEvent",
    "ConversationEvent",
    "FullDuplexBridge",
    "FullDuplexBridgeConfig",
    "GatewayConnection",
    "GatewayControlFrame",
    "GatewayEnvError",
    "GatewayEvent",
    "IdentityEvent",
    "IdentityResolvedEvent",
    "ModelEvent",
    "NewSpeakerEvent",
    "PersonaPlexBackend",
    "ReadyEvent",
    "SessionOpened",
    "SpeakerChangeEvent",
    "SpeechAudio",
    "SpeechBackend",
    "SpeechBackendCapabilities",
    "SpeechItem",
    "SpeechSessionConfig",
    "SpeechText",
    "StateDeltaEvent",
    "TextEvent",
    "TranscriptDelta",
    "TranscriptEvent",
    "TurnBoundaryEvent",
    "gateway_ws_url",
    "parse_control_event",
]

logger = logging.getLogger("livekit.plugins.prosodyai.full_duplex")

# Uplink wait for the backend to bind its session. Cold starts load several
# models, so the budget is generous; the wait ends in a raised error.
GATEWAY_READY_TIMEOUT = 120.0


@dataclass
class ReadyEvent:
    """The session handshake completed: the speaking loop is live."""


GatewayEvent = (
    ReadyEvent | TextEvent | IdentityEvent | TranscriptEvent | ModelEvent | ConversationEvent
)


def parse_control_event(kind: int, payload: bytes) -> GatewayEvent | None:
    """Parse one gateway control frame into a typed event; ``None`` when unparseable."""
    if kind == KIND_HANDSHAKE:
        return ReadyEvent()
    if kind == KIND_TEXT:
        return TextEvent(text=payload.decode("utf-8", errors="replace"))
    if kind not in (KIND_EVENT, KIND_TRANSCRIPT, KIND_IDENTITY):
        return None
    try:
        frame = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(frame, dict):
        return None
    if kind == KIND_EVENT:
        # The 0x06 channel carries two families: gateway tracker events
        # (``prosodyai.*``) and conversation events in the model wire's shape.
        model_event = parse_gateway_model_event(frame)
        if model_event is not None:
            return model_event
        return parse_conversation_event(frame)
    if kind == KIND_TRANSCRIPT:
        return parse_transcript_payload(frame)
    return parse_identity_payload(frame)


@dataclass(frozen=True)
class FullDuplexBridgeConfig:
    """Settings for one full-duplex session. The key is kept out of ``repr``.

    ``url`` and ``api_key`` locate the default PersonaPlex gateway; a bridge
    constructed with its own backend leaves them empty. ``voice_prompt`` and
    ``role_prompt`` reach the backend's session open when its capabilities
    declare those channels.
    """

    url: str = ""
    api_key: str = field(repr=False, default="")
    room_sample_rate: int = 16_000
    publish_sample_rate: int = GATEWAY_SAMPLE_RATE
    voice_prompt: str | None = None
    role_prompt: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        """The handshake credential, carried on the header so the URL stays loggable."""
        return {API_KEY_HEADER: self.api_key}


class FullDuplexBridge:
    """Mix-friendly duplex session: PCM16 uplink in, PCM16 downlink out + events."""

    def __init__(
        self, config: FullDuplexBridgeConfig, backend: SpeechBackend | None = None
    ) -> None:
        self._config = config
        if backend is None:
            if not config.url or not config.api_key:
                raise GatewayEnvError(
                    "the default PersonaPlex backend needs url and api_key on the "
                    "config; pass both, or pass a backend of your own"
                )
            backend = PersonaPlexBackend(url=config.url, api_key=config.api_key)
        self._backend = backend
        self._session_config = SpeechSessionConfig(
            voice_prompt=config.voice_prompt, role_prompt=config.role_prompt
        )
        require_capabilities(backend.capabilities, self._session_config)
        self._ready = asyncio.Event()
        self._closed = False

    @property
    def ready(self) -> asyncio.Event:
        return self._ready

    @property
    def backend(self) -> SpeechBackend:
        return self._backend

    async def _send_uplink(self, uplink_pcm16: AsyncIterator[bytes]) -> None:
        """Resample room PCM onto the backend's clock and stream it in.

        Holds audio until the backend binds its session; a handshake that
        never arrives raises so no call fails in silence.
        """
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=GATEWAY_READY_TIMEOUT)
        except asyncio.TimeoutError:
            raise RuntimeError(
                "the speech backend never completed its session handshake within "
                f"{GATEWAY_READY_TIMEOUT}s; no uplink audio was sent"
            ) from None
        backend_rate = self._backend.capabilities.sample_rate
        async for frame in uplink_pcm16:
            if self._closed or not frame:
                continue
            await self._backend.send_audio(
                resample_float32(
                    pcm16_le_to_float32(frame),
                    self._config.room_sample_rate,
                    backend_rate,
                )
            )

    def _as_gateway_event(self, item: SpeechItem | GatewayControlFrame) -> GatewayEvent | None:
        """One backend downlink item as the bridge's typed event vocabulary."""
        if isinstance(item, SessionOpened):
            return ReadyEvent()
        if isinstance(item, SpeechText):
            return TextEvent(text=item.text)
        if isinstance(item, GatewayControlFrame):
            return parse_control_event(item.kind, item.payload)
        return None

    async def _receive_downlink(
        self,
        *,
        on_downlink_pcm16: Callable[[bytes], Awaitable[None]],
        on_event: Callable[[GatewayEvent], Awaitable[None]] | None,
    ) -> None:
        """Split the backend's downlink into published PCM and typed events."""
        backend_rate = self._backend.capabilities.sample_rate
        async for item in self._backend.receive():
            if self._closed:
                break
            if isinstance(item, SpeechAudio):
                flat = item.samples
                if flat.size == 0:
                    continue
                if self._config.publish_sample_rate != backend_rate:
                    flat = resample_float32(flat, backend_rate, self._config.publish_sample_rate)
                await on_downlink_pcm16(float32_to_pcm16_le(flat))
                continue
            event = self._as_gateway_event(item)
            if event is None:
                continue
            if isinstance(event, ReadyEvent):
                self._ready.set()
            if on_event is not None:
                await on_event(event)

    async def run(
        self,
        uplink_pcm16: AsyncIterator[bytes],
        *,
        on_downlink_pcm16: Callable[[bytes], Awaitable[None]],
        on_event: Callable[[GatewayEvent], Awaitable[None]] | None = None,
    ) -> None:
        """Pump room PCM into the backend and publish its PCM + typed events.

        ``uplink_pcm16`` yields little-endian mono PCM16 at ``room_sample_rate``
        (typically 20 ms LiveKit frames). Downlink PCM is also LE mono PCM16 at
        ``publish_sample_rate``.
        """
        await self._backend.open(self._session_config)
        try:
            send_task = asyncio.create_task(self._send_uplink(uplink_pcm16), name="duplex-uplink")
            try:
                await self._receive_downlink(on_downlink_pcm16=on_downlink_pcm16, on_event=on_event)
            finally:
                self._closed = True
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
        finally:
            await self._backend.close()

    def close(self) -> None:
        self._closed = True
