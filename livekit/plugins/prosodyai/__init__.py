"""ProsodyAI plugin for LiveKit Agents.

Full-duplex speech with persistent speaker identity.

See https://prosodyai.app/docs and https://docs.livekit.io/agents/
"""

from . import realtime
from .bridge import FullDuplexBridge
from .events import (
    AgentToolEvent,
    AgentToolStatusEvent,
    BargeInEvent,
    ConversationEvent,
    EntitySpanEvent,
    GatewayEvent,
    IdentityEvent,
    IdentityResolvedEvent,
    ModelEvent,
    NewSpeakerEvent,
    ReadyEvent,
    SpeakerChangeEvent,
    StateDeltaEvent,
    TextEvent,
    TranscriptDelta,
    TranscriptEvent,
    TurnBoundaryEvent,
    parse_control_event,
)
from .gateway import GatewayConnection, GatewayEnvError, gateway_ws_url
from .realtime import RealtimeModel, RealtimeSession
from .version import __version__
from .wire import ConversationEventType, GatewayEventType, RoomEventType

__all__ = [
    "AgentToolEvent",
    "AgentToolStatusEvent",
    "BargeInEvent",
    "ConversationEvent",
    "ConversationEventType",
    "GatewayEventType",
    "RoomEventType",
    "EntitySpanEvent",
    "FullDuplexBridge",
    "GatewayConnection",
    "GatewayEnvError",
    "GatewayEvent",
    "IdentityEvent",
    "IdentityResolvedEvent",
    "ModelEvent",
    "NewSpeakerEvent",
    "ReadyEvent",
    "RealtimeModel",
    "RealtimeSession",
    "SpeakerChangeEvent",
    "StateDeltaEvent",
    "TextEvent",
    "TranscriptDelta",
    "TranscriptEvent",
    "TurnBoundaryEvent",
    "gateway_ws_url",
    "parse_control_event",
    "realtime",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class ProsodyAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(ProsodyAIPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
