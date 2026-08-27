"""The ProsodyAI gateway as a LiveKit Agents realtime model.

    from livekit.plugins import prosodyai

    session = AgentSession(llm=prosodyai.realtime.RealtimeModel())

The model is continuous full-duplex: the session emits exactly one generation
whose audio and text streams stay open for the life of the conversation.
Identity, transcript, and conversation events emit on both the session and
the model, so listeners can attach before ``AgentSession.start``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ..bridge import FullDuplexBridge
from ..events import (
    BargeInEvent,
    GatewayEvent,
    ReadyEvent,
    TextEvent,
    TranscriptEvent,
    TurnBoundaryEvent,
)
from ..gateway import SAMPLE_RATE, GatewayConnection
from ..log import logger
from ..wire import ConversationEventType, GatewayEventType, RoomEventType, WireEventType

NUM_CHANNELS = 1

__all__ = [
    "BargeInEvent",
    "ConversationEventType",
    "GatewayEventType",
    "RealtimeModel",
    "RealtimeSession",
    "RoomEventType",
    "TurnBoundaryEvent",
]


@dataclass
class _RealtimeOptions:
    api_key: str
    url: str


class RealtimeModel(llm.RealtimeModel, rtc.EventEmitter[WireEventType]):
    """Full-duplex speech with persistent speaker identity."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Connect a LiveKit AgentSession to the ProsodyAI speech model.

        Args:
            api_key: ProsodyAI API key. Defaults to ``PROSODYAI_API_KEY``.
            base_url: Gateway origin, for example ``https://api.prosodyai.app``
                or ``wss://api.prosodyai.app``. Defaults to the production origin.
        """
        llm.RealtimeModel.__init__(
            self,
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            ),
        )
        rtc.EventEmitter.__init__(self)
        connection = GatewayConnection.from_environment(
            api_key=api_key,
            base_url=base_url if is_given(base_url) else None,
        )
        self._opts = _RealtimeOptions(api_key=connection.api_key, url=connection.url)
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return "prosodyai-gateway"

    @property
    def provider(self) -> str:
        return "prosodyai"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        del turn_detection_disabled
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        await asyncio.gather(
            *(sess.aclose() for sess in list(self._sessions)),
            return_exceptions=True,
        )


class RealtimeSession(llm.RealtimeSession[WireEventType]):
    """One conversation: one bridge, one open-ended generation."""

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext.empty()
        self._uplink = utils.aio.Chan[bytes]()
        self._audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._text_ch = utils.aio.Chan[str]()
        self._bridge: FullDuplexBridge | None = None
        self._bridge_task: asyncio.Task[None] | None = None
        self._input_resampler: rtc.AudioResampler | None = None
        self._generation_emitted = False
        self._closed = False

    def _emit_plugin(self, name: WireEventType, event: object) -> None:
        self.emit(name, event)
        self._realtime_model.emit(name, event)

    # ------------------------------------------------------------- audio in

    @utils.log_exceptions(logger=logger)
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        if self._bridge_task is None:
            self._start_bridge()
        for resampled in self._resample_audio(frame):
            with contextlib.suppress(utils.aio.ChanClosed):
                self._uplink.send_nowait(bytes(resampled.data))

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler is not None:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler is not None:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _start_bridge(self) -> None:
        self._bridge = FullDuplexBridge(
            url=self._realtime_model._opts.url,
            api_key=self._realtime_model._opts.api_key,
            room_sample_rate=SAMPLE_RATE,
            publish_sample_rate=SAMPLE_RATE,
        )
        self._bridge_task = asyncio.create_task(self._run_bridge(), name="prosodyai-realtime")

    @utils.log_exceptions(logger=logger)
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
                        label=self._realtime_model.label,
                        error=exc,
                        recoverable=False,
                    ),
                )

    # ------------------------------------------------------------ audio out

    async def _on_downlink(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        pcm_view = memoryview(pcm)
        if len(pcm_view) < 2:
            return
        aligned = bytes(pcm_view[: len(pcm_view) - (len(pcm_view) % 2)])
        with contextlib.suppress(utils.aio.ChanClosed):
            self._audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=aligned,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=len(aligned) // 2,
                )
            )

    async def _on_event(self, event: GatewayEvent) -> None:
        if isinstance(event, ReadyEvent):
            self._emit_generation()
            return
        if isinstance(event, TextEvent):
            if event.text and not self._closed:
                with contextlib.suppress(utils.aio.ChanClosed):
                    self._text_ch.send_nowait(event.text)
                self._emit_plugin(event.TYPE, event)
            return
        if isinstance(event, TranscriptEvent):
            self._on_transcript(event)
            return
        if event.TYPE is None:
            return
        self._emit_plugin(event.TYPE, event)

    def _on_transcript(self, event: TranscriptEvent) -> None:
        self._emit_plugin(event.TYPE, event)
        text = " ".join(delta.text for delta in event.deltas if delta.text).strip()
        if not text:
            return
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=utils.shortuuid("prosodyai-words-"),
                transcript=text,
                is_final=True,
            ),
        )

    def _emit_generation(self) -> None:
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
                message_id=utils.shortuuid("prosodyai-"),
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
                response_id=utils.shortuuid("prosodyai-resp-"),
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
        logger.debug("update_instructions ignored (gateway owns priming)")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list) -> None:
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
            await utils.aio.cancel_and_wait(self._bridge_task)
        self._audio_ch.close()
        self._text_ch.close()
        self._bridge = None
        self._bridge_task = None
