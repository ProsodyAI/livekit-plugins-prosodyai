"""The ProsodyAI gateway as a LiveKit Agents duplex model.

    from livekit.plugins import prosodyai

    session = AgentSession(llm=prosodyai.DuplexModel())

``AgentSession`` wraps a :class:`livekit.agents.llm.DuplexModel` in its duplex
adapter. Audio out is ``audio_stream``. The model's own words are
``transcript_delta``. Caller words are
``input_audio_transcription_completed``. Identity and conversation events
still emit on the model, so a listener can attach before ``session.start``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import weakref
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from pydantic import BaseModel

from ..bridge import FullDuplexBridge
from ..events import (
    AgentToolStatusEvent,
    BargeInEvent,
    GatewayEvent,
    ReadyEvent,
    TextEvent,
    TranscriptEvent,
    TurnBoundaryEvent,
)
from ..gateway import SAMPLE_RATE, GatewayConnection
from ..thinking import (
    ClientControlType,
    RulesUpdate,
    ThoughtWrite,
    ToolDefinition,
    ToolResult,
    ToolsUpdate,
)
from ..tracks import Intent, Rule
from ..log import logger
from ..wire import (
    ConversationEventType,
    GatewayEventType,
    RoomEventType,
    ToolCallStatus,
    WireEventType,
)

NUM_CHANNELS = 1

__all__ = [
    "BargeInEvent",
    "ConversationEventType",
    "DuplexModel",
    "DuplexSession",
    "GatewayEventType",
    "RoomEventType",
    "TurnBoundaryEvent",
]


@dataclass
class _RealtimeOptions:
    api_key: str
    url: str


class DuplexModel(llm.DuplexModel, rtc.EventEmitter[WireEventType]):
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
        llm.DuplexModel.__init__(
            self,
            capabilities=llm.DuplexCapabilities(
                user_transcription=True,
                # A tool's result is a thought, and the model speaks from a
                # thought on its own.
                auto_tool_reply_generation=True,
                mutable_chat_context=True,
                mutable_instructions=True,
                mutable_tools=True,
            ),
        )
        rtc.EventEmitter.__init__(self)
        connection = GatewayConnection.from_environment(
            api_key=api_key,
            base_url=base_url if is_given(base_url) else None,
        )
        self._opts = _RealtimeOptions(api_key=connection.api_key, url=connection.url)
        self._sessions = weakref.WeakSet[DuplexSession]()

    @property
    def model(self) -> str:
        return "prosodyai-gateway"

    @property
    def provider(self) -> str:
        return "prosodyai"

    def session(self) -> DuplexSession:
        sess = DuplexSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        await asyncio.gather(
            *(sess.aclose() for sess in list(self._sessions)),
            return_exceptions=True,
        )


class DuplexSession(llm.DuplexSession[WireEventType]):
    """One conversation: one bridge, one continuous audio stream."""

    def __init__(self, duplex_model: DuplexModel) -> None:
        super().__init__(duplex_model)
        self._model = duplex_model
        self._tools = llm.ToolContext.empty()
        self._uplink = utils.aio.Chan[bytes]()
        self._audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._bridge: FullDuplexBridge | None = None
        self._bridge_task: asyncio.Task[None] | None = None
        self._input_resampler: rtc.AudioResampler | None = None
        self._closed = False
        #: Thinking-track messages, in order, sent once the call is open.
        self._controls = utils.aio.Chan[BaseModel]()
        self._controls_task: asyncio.Task[None] | None = None

    def _emit_plugin(self, name: WireEventType, event: object) -> None:
        self.emit(name, event)
        self._model.emit(name, event)

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
            url=self._model._opts.url,
            api_key=self._model._opts.api_key,
            room_sample_rate=SAMPLE_RATE,
            publish_sample_rate=SAMPLE_RATE,
        )
        self._bridge_task = asyncio.create_task(self._run_bridge(), name="prosodyai-duplex")
        self._controls_task = asyncio.create_task(
            self._send_controls(self._bridge), name="prosodyai-thinking"
        )

    @utils.log_exceptions(logger=logger)
    async def _send_controls(self, bridge: FullDuplexBridge) -> None:
        async for control in self._controls:
            await bridge.send_control(control)

    def _control(self, control: BaseModel) -> None:
        if not self._closed:
            with contextlib.suppress(utils.aio.ChanClosed):
                self._controls.send_nowait(control)

    # ------------------------------------------------------- thinking track

    def think(self, text: str, *, say: bool = False) -> None:
        """Tell the speech model something. It holds it as context, or with
        ``say`` speaks it now."""
        if text:
            self._control(
                ThoughtWrite(
                    type=ClientControlType.THOUGHT,
                    text=text,
                    intent=Intent.SAY if say else Intent.KNOW,
                )
            )

    def update_rules(self, rules: Sequence[Rule]) -> None:
        """This call's standing rules: when the caller says this, the model
        is told that. Replaces any earlier set."""
        self._control(RulesUpdate(type=ClientControlType.RULES_UPDATE, rules=list(rules)))

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
                        label=self._model.label,
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
            return
        if isinstance(event, TextEvent):
            if event.text and not self._closed:
                self.emit(
                    "transcript_delta",
                    llm.DuplexOutputTranscriptDelta(text=event.text),
                )
                self._emit_plugin(event.TYPE, event)
            return
        if isinstance(event, TranscriptEvent):
            self._on_transcript(event)
            return
        if isinstance(event, AgentToolStatusEvent) and event.status is ToolCallStatus.STARTED:
            # The call decided one of this agent's tools should run. LiveKit
            # runs it and hands the output back through ``_append_items``.
            self.emit(
                "function_call",
                llm.FunctionCall(
                    call_id=event.call_id,
                    name=event.name,
                    arguments=json.dumps(event.arguments),
                ),
            )
        if event.TYPE is None:
            return
        self._emit_plugin(event.TYPE, event)

    def _on_transcript(self, event: TranscriptEvent) -> None:
        # The room hears every delta, interim words included. The session's
        # transcription is the whole words: an interim is replaced, not said.
        self._emit_plugin(event.TYPE, event)
        text = " ".join(delta.text for delta in event.deltas if delta.final and delta.text).strip()
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

    @property
    def audio_stream(self) -> AsyncIterator[llm.DuplexAudioFrame]:
        return self._iter_audio()

    async def _iter_audio(self) -> AsyncIterator[llm.DuplexAudioFrame]:
        async for frame in self._audio_ch:
            yield llm.DuplexAudioFrame(frame=frame)

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    async def _update_instructions(self, instructions: str) -> None:
        self.think(instructions)

    async def _append_items(self, items: list[llm.ChatItem]) -> None:
        for item in items:
            if isinstance(item, llm.FunctionCallOutput):
                self._control(
                    ToolResult(
                        type=ClientControlType.TOOL_RESULT,
                        call_id=item.call_id,
                        output="" if item.is_error else item.output,
                        error=item.output if item.is_error else None,
                    )
                )
            elif isinstance(item, llm.ChatMessage) and item.text_content:
                self.think(item.text_content)

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools = llm.ToolContext(tools)
        self._control(
            ToolsUpdate(
                type=ClientControlType.TOOLS_UPDATE,
                tools=[
                    ToolDefinition.model_validate(
                        llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
                    )
                    for tool in self._tools.function_tools.values()
                ],
            )
        )

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        """Ask the model to speak now, from these instructions."""
        del tool_choice, tools
        if is_given(instructions):
            self.think(instructions, say=True)

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        del tool_choice

    async def aclose(self) -> None:
        await super().aclose()
        if self._closed:
            return
        self._closed = True
        if self._bridge is not None:
            self._bridge.close()
        self._uplink.close()
        self._controls.close()
        if self._controls_task is not None:
            await utils.aio.cancel_and_wait(self._controls_task)
        if self._bridge_task is not None:
            await utils.aio.cancel_and_wait(self._bridge_task)
        self._audio_ch.close()
        self._bridge = None
        self._bridge_task = None
