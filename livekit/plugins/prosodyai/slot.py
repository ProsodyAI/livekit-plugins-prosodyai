from __future__ import annotations

import base64
import json
import time
from typing import Annotated, Any, Literal, Optional, Protocol, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.alias_generators import to_camel

from .tracks import StrEnum, Thought


class _Wire(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    def json_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True, exclude_none=True).encode()


class SlotControlType(StrEnum):
    CONFIGURE = "slot.configure"
    ADMITTED = "slot.admitted"
    REFUSED = "slot.refused"
    TOOL_CALL = "tool.call"
    TOOL_GATE = "tool.gate"
    FIRING = "directive.firing"
    USAGE = "slot.usage"
    BUNDLE = "slot.bundle"
    DROPPED = "slot.dropped"
    DEGRADED = "slot.degraded"
    AGENT_TEXT = "agent.text"
    THOUGHT_DONE = "thought.done"
    SAY_CANCEL = "say.cancel"
    READ = "directive.read"
    END = "slot.end"
    REASON = "reason.request"
    REASON_DELTA = "reason.delta"
    REASON_DONE = "reason.done"
    REASON_ERROR = "reason.error"
    REASON_CANCEL = "reason.cancel"


class Refusal(StrEnum):
    FULL = "REFUSAL_FULL"
    SHARE = "REFUSAL_SHARE"
    TOKEN = "REFUSAL_TOKEN"


class LimitKind(StrEnum):
    RATE = "LIMIT_KIND_RATE"
    CAPACITY = "LIMIT_KIND_CAPACITY"
    BUDGET = "LIMIT_KIND_BUDGET"


class TenancyLevel(StrEnum):
    PLATFORM = "TENANCY_LEVEL_PLATFORM"
    ORGANIZATION = "TENANCY_LEVEL_ORGANIZATION"
    ACCOUNT = "TENANCY_LEVEL_ACCOUNT"


class AdmissionDecision(_Wire):
    admitted: bool
    limit: Optional[LimitKind] = None
    limited_at: Optional[TenancyLevel] = None
    retry_after_ms: Optional[int] = None


class CallClaims(_Wire):
    model_config = ConfigDict(extra="forbid")

    call: str
    organization: str
    account: str
    agent: str
    version: int
    bundle: str = ""
    fetch: str = ""
    weight: int = 1
    seats: int = 0
    expires: int
    directed: bool = False

    @property
    def tenant(self) -> str:
        return tenant_key(self.organization, self.account)


def tenant_key(organization: str, account: str) -> str:
    """The one key a tenant is known by in the slot: an account exists inside its
    organization, so two organizations' accounts with the same name never share a
    bundle, a share of the rows, or a seat."""
    return f"{organization}/{account}"


class SlotConfigure(_Wire):
    type: Literal[SlotControlType.CONFIGURE] = SlotControlType.CONFIGURE
    token: str


class SlotAdmitted(_Wire):
    type: Literal[SlotControlType.ADMITTED] = SlotControlType.ADMITTED
    session: str
    bundle: str = ""
    degraded: bool = False
    transcribing: bool = False
    know: Optional[bool] = None
    directed: Optional[bool] = None


class SlotBundle(_Wire):
    type: Literal[SlotControlType.BUNDLE] = SlotControlType.BUNDLE
    bundle: str
    degraded: bool = False


class SlotDropped(_Wire):
    type: Literal[SlotControlType.DROPPED] = SlotControlType.DROPPED
    at_heard: int
    count: int


class SlotDegraded(_Wire):
    type: Literal[SlotControlType.DEGRADED] = SlotControlType.DEGRADED
    transcribing: bool
    directives: bool
    frame: int


class AgentText(_Wire):
    type: Literal[SlotControlType.AGENT_TEXT] = SlotControlType.AGENT_TEXT
    piece: str
    frame: int
    audio: int


class ThoughtOutcome(StrEnum):
    DELIVERED = "delivered"
    CUT = "cut"
    HELD = "held"
    LOST = "lost"


class ThoughtDone(_Wire):
    type: Literal[SlotControlType.THOUGHT_DONE] = SlotControlType.THOUGHT_DONE
    id: str
    frame: int
    spoken: int
    total: int
    outcome: ThoughtOutcome


class SayCancel(_Wire):
    type: Literal[SlotControlType.SAY_CANCEL] = SlotControlType.SAY_CANCEL
    id: str


class DirectiveRead(_Wire):
    type: Literal[SlotControlType.READ] = SlotControlType.READ
    frame: int
    labels: tuple[str, ...] = ()


class SlotEnd(_Wire):
    type: Literal[SlotControlType.END] = SlotControlType.END


class SlotRefused(_Wire):
    type: Literal[SlotControlType.REFUSED] = SlotControlType.REFUSED
    refusal: Refusal
    decision: Optional[AdmissionDecision] = None


class ToolCall(_Wire):
    type: Literal[SlotControlType.TOOL_CALL] = SlotControlType.TOOL_CALL
    call: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolGate(_Wire):
    type: Literal[SlotControlType.TOOL_GATE] = SlotControlType.TOOL_GATE
    call: str
    allow: bool
    directive: str = ""


class Firing(_Wire):
    type: Literal[SlotControlType.FIRING] = SlotControlType.FIRING
    directive: str
    version: int
    label: str
    score: float
    frame: int


class ReasonRequest(_Wire):
    type: Literal[SlotControlType.REASON] = SlotControlType.REASON
    id: str
    body: dict[str, Any]


class ReasonDelta(_Wire):
    type: Literal[SlotControlType.REASON_DELTA] = SlotControlType.REASON_DELTA
    id: str
    content: str


class ReasonDone(_Wire):
    type: Literal[SlotControlType.REASON_DONE] = SlotControlType.REASON_DONE
    id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    finish: str = ""


class ReasonError(_Wire):
    type: Literal[SlotControlType.REASON_ERROR] = SlotControlType.REASON_ERROR
    id: str
    code: str
    message: str = ""


class ReasonCancel(_Wire):
    type: Literal[SlotControlType.REASON_CANCEL] = SlotControlType.REASON_CANCEL
    id: str


class SlotUsage(_Wire):
    type: Literal[SlotControlType.USAGE] = SlotControlType.USAGE
    frames: int
    session_ms: int


CALLER = "caller"


class TranscriptDelta(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    text: str
    start_ms: int
    end_ms: int
    final: bool = True
    frame: Optional[int] = None


class Transcript(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    speaker_id: str
    deltas: tuple[TranscriptDelta, ...]

    def json_bytes(self) -> bytes:
        return self.model_dump_json(exclude_none=True).encode()


SlotControl = Union[
    SlotConfigure,
    SlotAdmitted,
    SlotRefused,
    ToolCall,
    ToolGate,
    Firing,
    SlotUsage,
    SlotBundle,
    SlotDropped,
    SlotDegraded,
    AgentText,
    ThoughtDone,
    SayCancel,
    DirectiveRead,
    SlotEnd,
    ReasonRequest,
    ReasonDelta,
    ReasonDone,
    ReasonError,
    ReasonCancel,
]
_SLOT_CONTROL: TypeAdapter[SlotControl] = TypeAdapter(
    Annotated[SlotControl, Field(discriminator="type")]
)


def parse_slot_control(body: bytes) -> Optional[SlotControl]:
    try:
        return _SLOT_CONTROL.validate_json(body)
    except ValidationError:
        return None


class TokenRejected(ValueError):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_call(claims: CallClaims, private_key: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    body = claims.json_bytes()
    signature = Ed25519PrivateKey.from_private_bytes(private_key).sign(body)
    return f"{_b64(body)}.{_b64(signature)}"


def verify_call(token: str, public_key: bytes, now: Optional[float] = None) -> CallClaims:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        encoded_body, encoded_signature = token.split(".")
        body, signature = _unb64(encoded_body), _unb64(encoded_signature)
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, body)
        claims = CallClaims.model_validate(json.loads(body))
    except (ValueError, InvalidSignature, ValidationError) as exc:
        raise TokenRejected(type(exc).__name__) from exc
    if claims.expires <= (time.time() if now is None else now):
        raise TokenRejected("expired")
    return claims


class TokenReplayed(TokenRejected):
    pass


class CallTokenLedger:
    """Every call token admits one call. A token seen again inside its lifetime is a
    replay and is refused; a token past its expiry is forgotten, since verify_call
    refuses it on its own."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def admit(self, token: str, claims: CallClaims, now: Optional[float] = None) -> None:
        moment = time.time() if now is None else now
        self._prune(moment)
        signature = token.rsplit(".", 1)[-1]
        if signature in self._seen:
            raise TokenReplayed("token already admitted a call")
        self._seen[signature] = claims.expires

    def _prune(self, now: float) -> None:
        for signature in [key for key, expires in self._seen.items() if expires <= now]:
            del self._seen[signature]


DirectiveAction = Union[Thought, ToolGate, Firing, DirectiveRead]


class DirectiveRuntime(Protocol):
    def seat(self, row: int, account: str, bundle: Optional[str]) -> None: ...

    def unseat(self, row: int) -> None: ...

    def caller_words(self, row: int, text: str, final: bool, frame: int) -> None: ...

    def agent_text(self, row: int, piece: str) -> None: ...

    def tool_call(self, row: int, call: ToolCall) -> ToolGate: ...

    def masked(self, row: int) -> Sequence[int]: ...

    def take(self, row: int) -> Sequence[DirectiveAction]: ...

    def alive(self) -> bool: ...
