"""Gateway control-frame parse. Event shapes live in ``wire``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .frames import GatewayControlFrame
from .wire import (
    ConversationBargeInEvent as BargeInEvent,
    ConversationEntitySpanEvent as EntitySpanEvent,
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
    "AgentToolEvent",
    "AgentToolStatusEvent",
    "BargeInEvent",
    "ConversationEvent",
    "EntitySpanEvent",
    "GatewayEvent",
    "IdentityEvent",
    "IdentityResolvedEvent",
    "ModelEvent",
    "NewSpeakerEvent",
    "ReadyEvent",
    "SpeakerChangeEvent",
    "StateDeltaEvent",
    "TextEvent",
    "TranscriptDelta",
    "TranscriptEvent",
    "TurnBoundaryEvent",
    "parse_control_event",
]


@dataclass
class ReadyEvent:
    """The session handshake completed."""


GatewayEvent = (
    ReadyEvent | TextEvent | IdentityEvent | TranscriptEvent | ModelEvent | ConversationEvent
)


def _json_object(payload: bytes) -> Mapping[str, Any] | None:
    try:
        frame = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(frame, dict):
        return None
    return frame


def parse_control_event(frame: GatewayControlFrame) -> GatewayEvent | None:
    if frame.kind == GatewayControlFrame.HANDSHAKE:
        return ReadyEvent()
    if frame.kind == GatewayControlFrame.TEXT:
        return TextEvent(text=frame.payload.decode("utf-8", errors="replace"))
    body = _json_object(frame.payload)
    if body is None:
        return None
    if frame.kind == GatewayControlFrame.EVENT:
        return parse_gateway_model_event(body) or parse_conversation_event(body)
    if frame.kind == GatewayControlFrame.TRANSCRIPT:
        return parse_transcript_payload(body)
    if frame.kind == GatewayControlFrame.IDENTITY:
        return parse_identity_payload(body)
    return None
