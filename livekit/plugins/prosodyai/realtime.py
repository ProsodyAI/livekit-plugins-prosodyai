"""The ProsodyAI gateway as a LiveKit Agents realtime model.

The model is continuous full-duplex: the session emits exactly one generation
whose audio and text streams stay open for the life of the conversation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Literal

import numpy as np

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .full_duplex import (
    GATEWAY_SAMPLE_RATE,
    FullDuplexBridge,
    FullDuplexBridgeConfig,
    GatewayConnection,
    GatewayEvent,
    IdentityEvent,
    IdentityResolvedEvent,
    ModelEvent,
    NewSpeakerEvent,
    ReadyEvent,
    SpeakerChangeEvent,
    TextEvent,
    TranscriptEvent,
)
from .wire import WireEventType

# LiveKit RTP / RoomIO publish at 20 ms. The gateway emits ~80 ms Opus.
_PUBLISH_FRAME_BYTES = (GATEWAY_SAMPLE_RATE * 20 // 1000) * 2

__all__ = [
    "IdentityEvent",
    "IdentityResolvedEvent",
    "ModelEvent",
    "NewSpeakerEvent",
    "RealtimeModel",
    "RealtimeSession",
    "SessionEventType",
    "SpeakerChangeEvent",
    "TextEvent",
    "TranscriptEvent",
]

logger = logging.getLogger("livekit.plugins.prosodyai.realtime")


class SessionEventType(WireEventType):
    """Session event names, one per gateway readout. Members equal their string values."""

    IDENTITY = "prosody_identity"
    TEXT = "prosody_text"
    TRANSCRIPT = "prosody_transcript"
    EVENT = "prosody_event"


EventTypes = Literal[
    "prosody_identity",
    "prosody_text",
    "prosody_transcript",
    "prosody_event",
]


class RealtimeModel(llm.RealtimeModel):
    """Full-duplex speech with persistent speaker identity, as a realtime model."""

    def __init__(
        self,
        *,
        connection: GatewayConnection | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                # Full duplex: the model handles barge-in itself, so no server turn events.
                turn_detection=False,
                # Caller words come off the same socket, already attributed; no STT needed.
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )
        # Resolve at construction so a missing key fails the worker at startup.
        self._connection = connection or GatewayConnection.from_environment(
            base_url=base_url, api_key=api_key
        )
        self._sessions: list[RealtimeSession] = []

    @property
    def model(self) -> str:
        return "prosodyai-gateway"

    @property
    def provider(self) -> str:
        return "prosodyai"

    @property
    def sessions(self) -> list["RealtimeSession"]:
        """The model's open sessions, in the order they opened: how a worker
        reaches the identity events after handing the model to an
        ``AgentSession``. A session leaves this list when it closes."""
        return list(self._sessions)

    def session(self, *, turn_detection_disabled: bool = False) -> "RealtimeSession":
        # The runtime contains no turn detector; this compatibility parameter is inert.
        del turn_detection_disabled
        sess = RealtimeSession(self)
        self._sessions.append(sess)
        return sess

    def _forget(self, session: "RealtimeSession") -> None:
        """Drop one closed session.

        A model outlives its sessions (one worker process takes call after
        call), so a closed session left on this list holds its channels, its
        bridge, and whatever audio never drained for the life of the worker.
        """
        try:
            self._sessions.remove(session)
        except ValueError:
            pass

    async def aclose(self) -> None:
        sessions, self._sessions = self._sessions, []
        await asyncio.gather(*(sess.aclose() for sess in sessions), return_exceptions=True)


class RealtimeSession(llm.RealtimeSession[EventTypes]):
    """One conversation: one bridge, one open-ended generation."""

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._model = realtime_model
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext.empty()
        self._uplink = utils.aio.Chan[bytes]()
        self._audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._text_ch = utils.aio.Chan[str]()
        self._bridge: FullDuplexBridge | None = None
        self._bridge_task: asyncio.Task[None] | None = None
        self._generation_emitted = False
        self._closed = False

    # ------------------------------------------------------------- audio in

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        if self._bridge_task is None:
            self._start_bridge(room_sample_rate=frame.sample_rate)
        samples = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        try:
            self._uplink.send_nowait(samples.tobytes())
        except utils.aio.ChanClosed:
            pass

    def _start_bridge(self, *, room_sample_rate: int) -> None:
        self._bridge = FullDuplexBridge(
            FullDuplexBridgeConfig(
                url=self._model._connection.url,
                api_key=self._model._connection.api_key,
                room_sample_rate=room_sample_rate,
                publish_sample_rate=GATEWAY_SAMPLE_RATE,
            )
        )
        self._bridge_task = asyncio.create_task(self._run_bridge(), name="duplex-realtime-bridge")

    async def _run_bridge(self) -> None:
        assert self._bridge is not None
        try:
            await self._bridge.run(
                self._uplink,
                on_downlink_pcm16=self._on_downlink,
                on_event=self._on_event,
            )
        except Exception as exc:
            if not self._closed:
                self.emit(
                    "error",
                    llm.RealtimeModelError(
                        timestamp=time.time(),
                        label=self._model.label,
                        error=exc,
                        recoverable=False,
                    ),
                )

    # ------------------------------------------------------------ audio out

    async def _on_downlink(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        offset = 0
        while offset + 2 <= len(pcm):
            chunk = pcm[offset : offset + _PUBLISH_FRAME_BYTES]
            offset += len(chunk)
            if len(chunk) < 2:
                break
            self._audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=chunk,
                    sample_rate=GATEWAY_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
            )

    async def _on_event(self, event: GatewayEvent) -> None:
        if isinstance(event, ReadyEvent):
            self._emit_generation()
        elif isinstance(event, TextEvent):
            if event.text and not self._closed:
                self._text_ch.send_nowait(event.text)
                # Also a session event, so apps can skip the generation's text stream.
                self.emit(SessionEventType.TEXT.value, event)
        elif isinstance(event, TranscriptEvent):
            self._on_transcript(event)
        elif isinstance(event, (SpeakerChangeEvent, NewSpeakerEvent, IdentityResolvedEvent)):
            self.emit(SessionEventType.EVENT.value, event)
        elif isinstance(event, IdentityEvent):
            self.emit(SessionEventType.IDENTITY.value, event)

    def _on_transcript(self, event: TranscriptEvent) -> None:
        """Relay one committed span of caller words, twice: ``prosody_transcript``
        in the model's own shape, and the framework event in LiveKit's vocabulary
        so a stock ``AgentSession`` transcribes the user.
        """
        self.emit(SessionEventType.TRANSCRIPT.value, event)
        text = " ".join(delta.text for delta in event.deltas if delta.text).strip()
        if not text:
            return
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=utils.shortuuid("duplex_words_"),
                transcript=text,
                is_final=True,
            ),
        )

    def _emit_generation(self) -> None:
        """One conversation-length generation, opened when the gateway is ready."""
        if self._generation_emitted:
            return
        self._generation_emitted = True
        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        modalities: asyncio.Future[list[Literal["text", "audio"]]] = (
            asyncio.get_running_loop().create_future()
        )
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=utils.shortuuid("duplex_"),
                text_stream=self._text_ch,
                audio_stream=self._audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        function_ch.close()
        self.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=False,
                response_id=str(uuid.uuid4()),
            ),
        )

    # -------------------------------------------------------- llm interface

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    async def update_instructions(self, instructions: str) -> None:
        # Priming is the gateway's job; client instructions have no channel.
        logger.debug("update_instructions ignored (gateway owns priming)")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools) -> None:
        self._tools = llm.ToolContext(tools)

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        return None

    def push_video(self, frame: rtc.VideoFrame) -> None:
        return None

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        future: asyncio.Future[llm.GenerationCreatedEvent] = (
            asyncio.get_running_loop().create_future()
        )
        future.set_exception(
            llm.RealtimeError(
                "generate_reply() was called on a full-duplex session that streams "
                "replies continuously; consume the session's single open generation "
                "and drop the generate_reply() call"
            )
        )
        return future

    def commit_audio(self) -> None:
        return None

    def clear_audio(self) -> None:
        return None

    def interrupt(self) -> None:
        # Barge-in is the model's own behavior; there is no client-side generation to cancel.
        return None

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        return None

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bridge is not None:
            self._bridge.close()
        self._uplink.close()
        if self._bridge_task is not None:
            self._bridge_task.cancel()
            await asyncio.gather(self._bridge_task, return_exceptions=True)
        self._audio_ch.close()
        self._text_ch.close()
        # Nothing here outlives the call: the bridge holds the backend, and the
        # model holds this session until it hears the session is done.
        self._bridge = None
        self._bridge_task = None
        self._model._forget(self)
