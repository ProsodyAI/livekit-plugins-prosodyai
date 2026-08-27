"""The ProsodyAI event wire, declared once.

This module is the canonical declaration of the model-to-consumer event
vocabulary: the event type strings, the field names, and the typed shapes
that every parse and serialize site derives from.

The module imports only the standard library. Two frozen wires are declared
here.

1. The model wire: committed tracker events on the prediction envelope's
   ``events`` list, sent by the deployment and parsed by the API. Fields
   speak the model's clock: ``frame_ms`` and the integer ``lane``.
2. The gateway wire: kind-tagged frames on the caller socket, serialized by
   the API gateway and parsed by consumers (the LiveKit plugin, the website,
   the CLI probes). Fields speak the caller's vocabulary: ``timestamp_ms``,
   ``session_id``, and ``speaker_id``, the session-local surface emitted by
   Sortformer. The integer model ``lane`` is a separate identity/state scope;
   its number does not derive or identify a gateway ``speaker_id``.

The bytes of both wires are frozen. Consumers align to the wire, and a
rename lands here before it lands anywhere else. Every field is a committed
fact; deliberation numbers never join these shapes.

Parsers here are strict. A recognized event missing a required field raises
``ValueError`` naming the event and the field, because a producer that
omits a required field broke the contract and a fabricated default would
turn the breakage into a fake committed fact. Optional fields are the ones
today's producers legitimately send as null; they parse to ``None``.
Unrecognized event types parse to ``None`` so the vocabulary
can grow model-side first without breaking a consumer.

Serializing mirrors parsing. ``parse_wire`` derives every coercion from the
declared fields, and ``to_wire`` walks the same declaration back out, so a
shape's payload round-trips through its own parser and a new field needs no
edit at either site.
"""

from __future__ import annotations

import collections.abc
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, Optional, Union, get_args, get_origin, get_type_hints

#: The discriminator key every typed event carries. Union dispatch reads it,
#: serialization writes it, and the 0x04/0x05 frame bodies drop it.
WIRE_TYPE_KEY = "type"


class WireEventType(str, Enum):
    """Base for the wire's event-type families.

    Members are the wire bytes: equality, hashing, and formatting all speak
    the wire string, so a member drops into JSON payloads and mapping keys
    unchanged.
    """

    def __str__(self) -> str:
        return self.value


def speaker_label(lane: int) -> str:
    """Serialize one integer identity/state lane as its stable internal key.

    This compatibility helper does not identify a session-local Sortformer
    speaker. A public ``speaker_id`` comes from diarization and must be bound
    to identity state explicitly.
    """
    return f"speaker_{lane}"


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _unwrap_optional(annotation: Any) -> Any:
    if not _is_optional(annotation):
        return annotation
    return next(arg for arg in get_args(annotation) if arg is not type(None))


def _coerce_item(annotation: Any, item: Any, key: str, owner: str) -> Any:
    """Coerce one tuple member, dispatching event unions on the ``type`` key.

    A union member whose type key is unrecognized, or a member that is not
    an object at all, parses to ``None`` and the caller's tuple filters it:
    the vocabulary grows model-side first without breaking a consumer.
    """
    if get_origin(annotation) is Union:
        if not isinstance(item, Mapping):
            return None
        by_type = {cls.TYPE.value: cls for cls in get_args(annotation)}
        shape = by_type.get(str(item.get(WIRE_TYPE_KEY)))
        return None if shape is None else parse_wire(shape, item, f"{owner} {key}")
    return _coerce(annotation, item, key, owner)


def _coerce(annotation: Any, value: Any, key: str, owner: str) -> Any:
    """Coerce one present wire value to its declared field type."""
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is bool:
        return bool(value)
    if annotation is str:
        return str(value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(str(value))
    if get_origin(annotation) is tuple:
        inner = get_args(annotation)[0]
        return tuple(
            parsed
            for parsed in (_coerce_item(inner, item, key, owner) for item in value)
            if parsed is not None
        )
    if get_origin(annotation) in (dict, collections.abc.Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{owner} field {key!r} must be an object")
        return dict(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return parse_wire(annotation, value, f"{owner} {key}")
    return value


def parse_wire(cls: type, entry: Mapping[str, Any], owner: str) -> Any:
    """Build one declared wire shape from its payload.

    The dataclass fields are the parse declaration: a missing or null
    required field raises ``ValueError`` naming the owner and the field, an
    ``Optional`` field parses null to ``None``, and a field with a declared
    default falls back to it. Every coercion derives from the field's
    annotation, so a new field on a shape parses itself.
    """
    if not isinstance(entry, Mapping):
        raise ValueError(f"{owner} must be an object")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for declared in fields(cls):
        annotation = hints[declared.name]
        value = entry.get(declared.name)
        if value is None:
            if declared.default is not MISSING or declared.default_factory is not MISSING:
                continue
            if _is_optional(annotation):
                kwargs[declared.name] = None
                continue
            raise ValueError(f"{owner} is missing required field {declared.name!r}")
        kwargs[declared.name] = _coerce(_unwrap_optional(annotation), value, declared.name, owner)
    return cls(**kwargs)


def _event_shape(family: type, union: Any, entry: Mapping[str, Any]) -> Any:
    """Dispatch one event payload to its declared shape on the ``type`` key.

    An unrecognized type parses to ``None`` so the vocabulary can grow
    model-side first without breaking a consumer.
    """
    try:
        event_type = family(str(entry.get(WIRE_TYPE_KEY)))
    except ValueError:
        return None
    shape = {cls.TYPE: cls for cls in get_args(union)}[event_type]
    return parse_wire(shape, entry, f"{event_type.value} event")


class WireShape:
    """One declared wire shape. Its fields are the whole serialize contract.

    A subclass that sets ``TYPE`` leads its payload with the discriminator;
    the 0x04 and 0x05 frame bodies drop it again through ``to_payload``.
    """

    TYPE: ClassVar[Optional[WireEventType]] = None

    def to_dict(self) -> dict[str, Any]:
        return to_wire(self)

    def to_payload(self) -> dict[str, Any]:
        """The same payload without the discriminator, for frames that tag
        their type in the frame kind byte instead of the body."""
        body = to_wire(self)
        body.pop(WIRE_TYPE_KEY, None)
        return body


def to_wire(value: Any) -> Any:
    """Serialize one declared shape, or one field of one, to its payload form.

    The inverse of ``parse_wire``, walking the same field declaration: the
    discriminator leads, enum members become their wire value, and nested
    shapes and sequences serialize through the same walk. Deriving it matters
    most for a nested union member, whose ``TYPE`` is a ``ClassVar`` and so
    never appears in ``dataclasses.asdict``; a payload missing it silently
    parses back to ``None``.
    """
    if isinstance(value, WireShape):
        payload: dict[str, Any] = {} if value.TYPE is None else {WIRE_TYPE_KEY: value.TYPE.value}
        for declared in fields(value):
            payload[declared.name] = to_wire(getattr(value, declared.name))
        return payload
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [to_wire(item) for item in value]
    if isinstance(value, Mapping):
        return {key: to_wire(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# The model wire: committed tracker events on the prediction envelope.


class TrackerEventType(WireEventType):
    """The identity-state commitments the deployment puts on the envelope."""

    SPEAKER_CHANGE = "speaker_change"
    NEW_SPEAKER = "new_speaker"
    IDENTITY_RESOLVED = "identity_resolved"


@dataclass(frozen=True)
class TrackerSpeakerChangeEvent(WireShape):
    """The outer recurrence committed identity state to a different lane.

    ``lane`` and ``previous_lane`` are identity/state scopes, not Sortformer
    speaker suffixes. ``frame_ms`` is retrodictive: it points at the segment's
    onset on the model's frame clock, while the commit itself lands when the
    segment closes (or when a held segment resolves late). The latency between
    the two is the model's business.
    """

    TYPE: ClassVar[TrackerEventType] = TrackerEventType.SPEAKER_CHANGE

    frame_ms: int
    lane: int
    previous_lane: Optional[int]
    known_id: Optional[str] = None
    previous_known_id: Optional[str] = None


@dataclass(frozen=True)
class TrackerNewSpeakerEvent(WireShape):
    """The outer recurrence opened a new identity/state lane."""

    TYPE: ClassVar[TrackerEventType] = TrackerEventType.NEW_SPEAKER

    frame_ms: int
    lane: int
    evidence_seconds: float


@dataclass(frozen=True)
class TrackerIdentityResolvedEvent(WireShape):
    """A lane matched a profile the organization enrolled, once per lane, at its first commit.

    ``verified`` is true on every committed decision: the decoder's absolute
    membership test is the only thing that writes a lane.
    """

    TYPE: ClassVar[TrackerEventType] = TrackerEventType.IDENTITY_RESOLVED

    frame_ms: int
    lane: int
    person_id: str
    verified: bool


TrackerEvent = Union[
    TrackerSpeakerChangeEvent,
    TrackerNewSpeakerEvent,
    TrackerIdentityResolvedEvent,
]
"""One committed identity-state event off the prediction envelope. The model
is the only author of these events."""


# ---------------------------------------------------------------------------
# The model wire: committed conversation events decoded by the learned event
# deciders. They ride the prediction envelope's ``events`` list beside the
# tracker events. Every field is a committed fact on
# the model's frame clock; the decision bands that produced the commit stay
# inside the model.


class ConversationEventType(WireEventType):
    """The learned event deciders' commitments, on the same envelope list."""

    STATE_DELTA = "state_delta"
    TURN_BOUNDARY = "turn_boundary"
    BARGE_IN = "barge_in"
    ENTITY_SPAN = "entity_span"


@dataclass(frozen=True)
class ConversationStateDeltaEvent(WireShape):
    """``state_delta``: the lane's state moved decisively against its own
    baseline.

    This is the significance signal: δ_t committed by the model that has been
    carrying the person's state. ``magnitude`` is the L2 norm of the mean
    per-frame departure from baseline over the excursion, in state units.
    Emitted at commit (``resolved`` false, duration so far) and once more when
    the return to baseline is observed (``resolved`` true, full duration).
    ``frame_ms`` is retrodictive to the excursion's onset.
    """

    TYPE: ClassVar[ConversationEventType] = ConversationEventType.STATE_DELTA

    frame_ms: int
    commit_ms: int
    duration_ms: int
    magnitude: float
    resolved: bool


@dataclass(frozen=True)
class ConversationTurnBoundaryEvent(WireShape):
    """``turn_boundary``: the model committed the floor passed between voices.

    An instantaneous committed fact. ``frame_ms`` is retrodictive: it points
    at where the boundary evidence began on the model's frame clock, while
    ``commit_ms`` is where the decision landed.
    """

    TYPE: ClassVar[ConversationEventType] = ConversationEventType.TURN_BOUNDARY

    frame_ms: int
    commit_ms: int


@dataclass(frozen=True)
class ConversationBargeInEvent(WireShape):
    """``barge_in``: the model committed a second voice entered against held
    speech.

    Emitted at commit (``resolved`` false, duration so far) and once more
    when the end of the overlap is observed (``resolved`` true, full
    duration). ``frame_ms`` is retrodictive to the overlap's onset.
    """

    TYPE: ClassVar[ConversationEventType] = ConversationEventType.BARGE_IN

    frame_ms: int
    commit_ms: int
    duration_ms: int
    resolved: bool


@dataclass(frozen=True)
class ConversationEntitySpanEvent(WireShape):
    """``entity_span``: the entity head committed a dictated entity's extent.

    ``kind`` is the entity family, spelled with the producing registry's
    enum value (``prosody_ssm.entities.EntityKind``); it stays a string on
    the wire so a kind added model-side parses before every consumer has
    heard of it. ``frame_ms`` is retrodictive to the span's onset,
    ``duration_ms`` is its extent on the frame clock, and ``commit_ms`` is
    where the decision landed. The words inside the extent belong to the
    transcript loop; the API pairs the two in its consumer and the entity
    graph's grounding gate decides what enters the record.
    """

    TYPE: ClassVar[ConversationEventType] = ConversationEventType.ENTITY_SPAN

    frame_ms: int
    commit_ms: int
    duration_ms: int
    kind: str


ConversationWireEvent = Union[
    ConversationStateDeltaEvent,
    ConversationTurnBoundaryEvent,
    ConversationBargeInEvent,
    ConversationEntitySpanEvent,
]
"""One committed conversation event off the prediction envelope, decoded by
the learned event deciders with carried state. The model is the only author."""


# ---------------------------------------------------------------------------
# The identity timeline: one model-owned history of identity/state decisions.
#
# Sortformer diarization and identity state are separate timelines. Sortformer
# supplies the session-local speaker surface. The identity tracker commits an
# audio span to an integer state lane, a new lane, or hold. These shapes are
# the canonical identity/state readout. A consumer may associate a lane with a
# Sortformer surface only through an explicit binding; matching suffixes are
# not a join.


IDENTITY_TIMELINE_SCHEMA_VERSION = 1


class IdentityDecision(WireEventType):
    """The tracker's verdict for one audio span."""

    COMMIT = "commit"  # the span belongs to ``lane``
    MINT = "mint"  # the span opened a fresh lane for a new voice
    HOLD = "hold"  # evidence stayed unresolved; the span remains unattributed


@dataclass(frozen=True)
class IdentitySpan(WireShape):
    """One committed lane decision over a span of audio-clock time.

    ``start_ms`` and ``end_ms`` bound the audio the decision covers, and are
    retrodictive when the segment closed (or a hold resolved) after the fact;
    ``commit_ms`` is where the verdict landed on the model's frame clock. A
    HOLD span carries ``lane`` for the candidate under test while keeping
    ``speaker_id`` null, because a hold attributes nothing. When present,
    ``speaker_id`` is an explicitly bound session-local Sortformer surface; it
    is never derived from ``lane``. ``late_resolved`` marks a span whose
    commitment arrived after its audio span closed, and ``unique_voice`` marks
    a minted lane the tracker will not merge into an existing one.
    """

    start_ms: int
    end_ms: int
    commit_ms: int
    lane: int
    speaker_id: Optional[str]
    decision: IdentityDecision
    late_resolved: bool = False
    unique_voice: bool = False


@dataclass(frozen=True)
class IdentityLane(WireShape):
    """One lane in the session's lane book.

    ``lane`` is the integer identity/state scope. ``speaker_id`` is the
    session-local Sortformer surface explicitly bound to that scope; its text
    is not derived from the lane number. ``person_id`` is the durable
    cross-session lineage identity, present only after a committed identity
    resolution; ``display_name`` labels it. ``resumed`` marks a lane that
    resumed a profile the organization enrolled, and ``is_agent`` marks the
    org's declared agent identity for self-recognition filtering.
    """

    lane: int
    speaker_id: str
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    resumed: bool = False
    is_agent: bool = False


@dataclass(frozen=True)
class IdentityTimeline(WireShape):
    """The session's whole identity history, in commit order.

    ``spans`` carries every identity/state assignment in the order the tracker
    committed them; ``lanes`` is the state-lane book those spans reference.
    ``events`` carries the committed tracker events verbatim so the durable
    record keeps the raw commit stream beside the derived spans. Sortformer
    diarization remains a separate session-local surface timeline.
    """

    schema_version: int
    model_provenance: Mapping[str, Any]
    spans: tuple[IdentitySpan, ...]
    lanes: tuple[IdentityLane, ...]
    events: tuple[TrackerEvent, ...] = ()


def parse_identity_span(entry: Mapping[str, Any]) -> IdentitySpan:
    """Parse one identity span. Strict: every required field must be present."""
    return parse_wire(IdentitySpan, entry, "identity span")


def parse_identity_lane(entry: Mapping[str, Any]) -> IdentityLane:
    """Parse one identity lane-book entry."""
    return parse_wire(IdentityLane, entry, "identity lane")


def parse_identity_timeline(entry: Mapping[str, Any]) -> IdentityTimeline:
    """Parse a canonical identity timeline payload."""
    return parse_wire(IdentityTimeline, entry, "identity timeline")


def parse_conversation_event(
    entry: Mapping[str, Any],
) -> Optional[ConversationWireEvent]:
    """Parse one committed conversation event off the prediction envelope.

    Strict on recognized types: a missing required field raises
    ``ValueError`` naming the event and the field. An unrecognized ``type``
    returns ``None`` because the vocabulary grows model-side first.
    """
    return _event_shape(ConversationEventType, ConversationWireEvent, entry)


def parse_tracker_event(entry: Mapping[str, Any]) -> Optional[TrackerEvent]:
    """Parse one committed tracker event off the prediction envelope.

    The model wire's single parse site. Strict on recognized types: a
    missing required field raises ``ValueError`` naming the event and the
    field. An unrecognized ``type`` returns ``None`` because the vocabulary
    grows model-side first.
    """
    return _event_shape(TrackerEventType, TrackerEvent, entry)


# ---------------------------------------------------------------------------
# The gateway wire: committed model events on the 0x06 caller frame.


class GatewayEventType(WireEventType):
    """The committed model events the gateway serializes onto 0x06 frames."""

    SPEAKER_CHANGE = "prosodyai.speaker_change"
    NEW_SPEAKER = "prosodyai.new_speaker"
    IDENTITY_RESOLVED = "prosodyai.identity_resolved"
    AGENT_TOOL = "prosodyai.agent_tool"
    AGENT_TOOL_STATUS = "prosodyai.agent_tool_status"


@dataclass(frozen=True)
class GatewaySpeakerChangeEvent(WireShape):
    """``prosodyai.speaker_change``: Sortformer moved the caller surface.

    ``speaker_id`` and ``previous_speaker_id`` are session-local Sortformer
    surfaces. They are not encodings of identity lane numbers.
    ``timestamp_ms`` points at the new surface's onset on the audio clock.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.SPEAKER_CHANGE

    timestamp_ms: int
    session_id: str
    speaker_id: str
    previous_speaker_id: Optional[str]
    person_id: Optional[str]
    display_name: Optional[str]
    is_agent: bool


@dataclass(frozen=True)
class GatewayNewSpeakerEvent(WireShape):
    """``prosodyai.new_speaker``: a new identity/state lane opened.

    ``speaker_id`` is the session-local Sortformer surface explicitly bound to
    the new state lane; it is not computed from the lane number. ``person_id``
    carries a durable identity when one is available and is otherwise null.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.NEW_SPEAKER

    timestamp_ms: int
    session_id: str
    speaker_id: str
    evidence_seconds: float
    person_id: Optional[str]
    display_name: Optional[str]


@dataclass(frozen=True)
class GatewayIdentityResolvedEvent(WireShape):
    """``prosodyai.identity_resolved``: identity state matched an enrolled profile.

    Fires once per identity lane, at its first committed segment.
    ``speaker_id`` is the explicitly bound session-local Sortformer surface,
    not a rendering of that lane. ``verified`` is true when the decision came
    from the decoder's absolute membership test.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.IDENTITY_RESOLVED

    timestamp_ms: int
    session_id: str
    speaker_id: str
    person_id: Optional[str]
    display_name: Optional[str]
    verified: bool


@dataclass(frozen=True)
class GatewayAgentToolEvent(WireShape):
    """``prosodyai.agent_tool``: one completed capability, shown to the caller.

    The speech model's monologue asked for the capability, the gateway ran
    it, and ``result`` is the clause Jarvis was given to say. The reading
    half of the deliberation is the monologue itself, which reaches the
    caller as the agent's own transcript.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.AGENT_TOOL

    session_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""


class ToolCallStatus(WireEventType):
    """The lifecycle stages of one capability invocation."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class GatewayAgentToolStatusEvent(WireShape):
    """``prosodyai.agent_tool_status``: one stage of a capability's lifecycle.

    ``started`` fires the moment the executor takes the monologue's intent;
    ``completed`` carries the result; ``failed`` carries the error. What the
    speech model receives is a clause to say, on the control channel, so
    these events are the operator's view of the same invocation and never
    join that channel. ``call_id`` ties the stages together.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.AGENT_TOOL_STATUS

    session_id: str
    call_id: str
    name: str
    status: ToolCallStatus
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None


GatewayModelEvent = Union[
    GatewaySpeakerChangeEvent,
    GatewayNewSpeakerEvent,
    GatewayIdentityResolvedEvent,
    GatewayAgentToolEvent,
    GatewayAgentToolStatusEvent,
]
"""A committed model decision off the gateway's 0x06 event channel."""


def parse_gateway_model_event(
    frame: Mapping[str, Any],
) -> Optional[GatewayModelEvent]:
    """Parse one 0x06 committed-model-event payload into its typed form.

    The gateway wire's single parse site for model events. Strict on
    recognized types: a missing required field raises ``ValueError`` naming
    the event and the field. An unrecognized ``type`` returns ``None`` so an
    unknown event never breaks the socket.
    """
    return _event_shape(GatewayEventType, GatewayModelEvent, frame)


# ---------------------------------------------------------------------------
# The gateway wire: identity (0x04) and transcript (0x05) caller frames.
# ``to_payload`` is the gateway serialize site (the frame body carries no
# ``type`` key). ``to_dict`` adds the room topic type for consumers that
# republish the event, and the type strings below name those republished
# events on the LiveKit data topic.


class RoomEventType(WireEventType):
    """The republished event names on the LiveKit data topic."""

    AGENT_STATE = "agent.state"
    TEXT = "prosodyai.text"
    IDENTITY = "prosodyai.identity"
    TRANSCRIPT = "prosodyai.transcript"
    MOMENT = "prosodyai.moment"


# Jarvis is the other party on the call, so his words carry this fixed label.
# A caller ``speaker_id`` names a session-local Sortformer surface. Identity
# state lanes are separate and may be associated only through explicit state.
AGENT_SPEAKER_ID = "agent"


@dataclass(frozen=True)
class AgentStateEvent(WireShape):
    """``agent.state``: the worker said where the call stands.

    The worker's own announcement about the socket it holds, published when
    the gateway binds. It reports the state of the connection and decides
    nothing about the conversation.
    """

    TYPE: ClassVar[RoomEventType] = RoomEventType.AGENT_STATE

    state: str


@dataclass(frozen=True)
class TextEvent(WireShape):
    """``prosodyai.text``: one token of the model's monologue, as it is spoken."""

    TYPE: ClassVar[RoomEventType] = RoomEventType.TEXT

    text: str


@dataclass(frozen=True)
class TranscriptDelta(WireShape):
    """One committed span of a speaker's words, times in their own audio."""

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class IdentityEvent(WireShape):
    """A committed identity resolution, announced mid-conversation.

    ``speaker_id`` is the session-local Sortformer surface explicitly bound to
    the resolved identity state. ``resolved_at_ms`` is the audio position (ms
    into the call) of the frame whose tracker assignment resolved the person.
    The model owns this clock, so it is the resolution time; consumers report
    it and derive nothing from it.
    """

    TYPE: ClassVar[RoomEventType] = RoomEventType.IDENTITY

    speaker_id: str
    person_id: str
    display_name: Optional[str]
    resumed: bool
    resolved_at_ms: int


@dataclass(frozen=True)
class TranscriptEvent(WireShape):
    """Words attributed to one session-local Sortformer speaker surface.

    Subtitles only. ``speaker_id`` is not an identity lane. Identity decisions
    arrive on the committed model-event channel and are never derived from
    this text.
    """

    TYPE: ClassVar[RoomEventType] = RoomEventType.TRANSCRIPT

    speaker_id: str
    deltas: tuple[TranscriptDelta, ...] = ()


@dataclass(frozen=True)
class MomentRecordEvent(WireShape):
    """One committed turn joined to the state movement over its span.

    The consumer-side moment record. Steps and words are separate loops on
    the frame path; this record is where they meet: the words of one
    committed turn beside the ``state_delta`` readout whose excursion
    overlapped the turn's span. ``delta`` carries the model's committed
    event verbatim. ``delta`` is ``None`` when the model committed no
    excursion over the span, so the absence of significance is an explicit
    fact on the record. ``speaker_id`` is the session-local Sortformer surface;
    ``person_id`` is the durable identity explicitly resolved to it at commit
    time.
    """

    TYPE: ClassVar[RoomEventType] = RoomEventType.MOMENT

    session_id: str
    speaker_id: str
    person_id: Optional[str]
    start_ms: int
    end_ms: int
    text: str
    delta: Optional[ConversationStateDeltaEvent] = None


def parse_identity_payload(frame: Mapping[str, Any]) -> IdentityEvent:
    """Parse one 0x04 committed-identity payload.

    Strict on required fields. Fields outside the declared shape are
    ignored so a grown resolution never breaks the socket.
    """
    return parse_wire(IdentityEvent, frame, f"{RoomEventType.IDENTITY.value} event")


def parse_transcript_payload(frame: Mapping[str, Any]) -> Optional[TranscriptEvent]:
    """Parse one 0x05 lane-attributed transcript payload.

    Strict on required fields, in every delta included. A payload without
    deltas parses to ``None``.
    """
    event = parse_wire(TranscriptEvent, frame, f"{RoomEventType.TRANSCRIPT.value} event")
    return event if event.deltas else None


# ---------------------------------------------------------------------------
# The realtime session event stream: one versioned envelope per event.

SESSION_EVENT_ENVELOPE_VERSION = 1
SESSION_ENVELOPE_KEYS = ("version", "session_id", "generation", "seq", "type")

#: The envelope's name in the vocabulary manifest. It is the one entry that is
#: not a discriminated shape, so it has no ``TYPE`` to name it.
SESSION_ENVELOPE_ENTRY = "session.envelope"

SESSION_STARTED = "session.started"
SESSION_RESET = "session.reset"
SESSION_FINALIZED = "session.finalized"
SESSION_PROCESSOR_ERROR = "session.processor_error"
PROSODY_DIRECTIVE = "prosody.directive"
ANALYSIS_FINAL = "analysis.final"


@dataclass(frozen=True)
class SessionEvent:
    """One versioned event off a realtime session's stream.

    The envelope keys are fixed and reserved; ``fields`` carries the event
    body and is flattened beside them at the serialize site.
    """

    version: int
    session_id: str
    generation: int
    seq: int
    type: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        envelope = {key: getattr(self, key) for key in SESSION_ENVELOPE_KEYS}
        return {**envelope, **dict(self.fields)}


# ---------------------------------------------------------------------------
# The room topic: what a joined client sees.

ROOM_TOPIC_EVENTS: tuple[type, ...] = (
    AgentStateEvent,
    TextEvent,
    IdentityEvent,
    TranscriptEvent,
    MomentRecordEvent,
    GatewaySpeakerChangeEvent,
    GatewayNewSpeakerEvent,
    GatewayIdentityResolvedEvent,
    GatewayAgentToolEvent,
    GatewayAgentToolStatusEvent,
    ConversationStateDeltaEvent,
    ConversationTurnBoundaryEvent,
    ConversationBargeInEvent,
    ConversationEntitySpanEvent,
)
"""Every shape republished on the LiveKit data topic, in the order a client
reads them: the worker's own announcement, the model's monologue, the
identity and transcript republications, the per-turn moment records, the
committed gateway events, and the
conversation events relayed verbatim. A client's event vocabulary is this
tuple. The tracker events stay off it: they are the model's own clock and the
gateway binds identity state to a session-local Sortformer surface before
anybody outside sees an identity event."""


def parse_room_event(payload: Mapping[str, Any]) -> Optional[Any]:
    """Parse one payload off the LiveKit data topic into its declared shape.

    The room topic's single parse site, and the reason a client needs no
    vocabulary of its own: the shapes come from :data:`ROOM_TOPIC_EVENTS`.
    Strict on a recognized type, and ``None`` for one nobody declared, so a
    grown vocabulary never breaks a joined client.
    """
    if not isinstance(payload, Mapping):
        return None
    shapes = {shape.TYPE.value: shape for shape in ROOM_TOPIC_EVENTS}
    shape = shapes.get(str(payload.get(WIRE_TYPE_KEY)))
    if shape is None:
        return None
    return parse_wire(shape, payload, f"{shape.TYPE.value} event")


# ---------------------------------------------------------------------------
# The vocabulary manifest: what the drift tests compare across trees.


def _field_names(event_class: type) -> tuple[str, ...]:
    return tuple(f.name for f in fields(event_class))


def vocabulary() -> dict[str, tuple[str, ...]]:
    """Every event type string mapped to its field names, in wire order."""
    entries: dict[str, tuple[str, ...]] = {
        event_class.TYPE.value: _field_names(event_class)
        for event_class in (
            TrackerSpeakerChangeEvent,
            TrackerNewSpeakerEvent,
            TrackerIdentityResolvedEvent,
            *ROOM_TOPIC_EVENTS,
        )
    }
    entries[SESSION_ENVELOPE_ENTRY] = SESSION_ENVELOPE_KEYS
    return entries
