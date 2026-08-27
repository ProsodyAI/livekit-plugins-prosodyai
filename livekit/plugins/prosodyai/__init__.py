"""ProsodyAI plugin for LiveKit Agents.

Inference runs server-side; this plugin is pure I/O. ``RealtimeModel`` and
``FullDuplexBridge`` carry full-duplex speech with persistent speaker identity.
"""

from .full_duplex import (
    AgentToolEvent,
    AgentToolStatusEvent,
    BackendCapabilityError,
    BargeInEvent,
    ConversationEvent,
    EntitySpanEvent,
    FullDuplexBridge,
    FullDuplexBridgeConfig,
    GatewayConnection,
    GatewayControlFrame,
    GatewayEnvError,
    GatewayEvent,
    IdentityEvent,
    IdentityResolvedEvent,
    ModelEvent,
    NewSpeakerEvent,
    PersonaPlexBackend,
    ReadyEvent,
    SessionOpened,
    SpeakerChangeEvent,
    SpeechAudio,
    SpeechBackend,
    SpeechBackendCapabilities,
    SpeechItem,
    SpeechSessionConfig,
    SpeechText,
    StateDeltaEvent,
    TextEvent,
    TranscriptDelta,
    TranscriptEvent,
    TurnBoundaryEvent,
    gateway_ws_url,
    parse_control_event,
)
from .realtime import (
    RealtimeModel,
    RealtimeSession,
)
from .version import __version__

__all__ = [
    "AgentToolEvent",
    "AgentToolStatusEvent",
    "BackendCapabilityError",
    "BargeInEvent",
    "ConversationEvent",
    "EntitySpanEvent",
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
    "RealtimeModel",
    "RealtimeSession",
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
