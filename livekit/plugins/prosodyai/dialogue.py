from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import AwareDatetime, Field, PlainSerializer, model_validator

from .directives import Form, Int32, Lexicon, SemTerm, Situation, _exactly_one, _present, _Wire
from .tracks import StrEnum

Int64 = Annotated[
    int,
    Field(ge=-(2**63), le=2**63 - 1),
    PlainSerializer(str, return_type=str, when_used="json"),
]


class Participant(StrEnum):
    CALLER = "PARTICIPANT_CALLER"
    AGENT = "PARTICIPANT_AGENT"


class Force(StrEnum):
    ASSERT = "FORCE_ASSERT"
    ASK = "FORCE_ASK"
    DIRECT = "FORCE_DIRECT"
    CONFIRM = "FORCE_CONFIRM"
    REVERSE = "FORCE_REVERSE"
    CORRECT = "FORCE_CORRECT"
    ACKNOWLEDGE = "FORCE_ACKNOWLEDGE"
    GREET = "FORCE_GREET"
    CLOSE = "FORCE_CLOSE"


class Relation(StrEnum):
    ANSWER = "RELATION_ANSWER"
    ELABORATION = "RELATION_ELABORATION"
    EXPLANATION = "RELATION_EXPLANATION"
    CORRECTION = "RELATION_CORRECTION"
    SUBQUESTION = "RELATION_SUBQUESTION"
    ACKNOWLEDGEMENT = "RELATION_ACKNOWLEDGEMENT"
    CONTINUATION = "RELATION_CONTINUATION"
    BACKGROUND = "RELATION_BACKGROUND"


class Terminal(StrEnum):
    FALL = "TERMINAL_FALL"
    RISE = "TERMINAL_RISE"
    LEVEL = "TERMINAL_LEVEL"
    RISE_FALL_RISE = "TERMINAL_RISE_FALL_RISE"


class Floor(StrEnum):
    HOLD = "FLOOR_HOLD"
    YIELD = "FLOOR_YIELD"
    OVERLAP = "FLOOR_OVERLAP"


class Binding(StrEnum):
    BOUND = "BINDING_BOUND"
    ACCOMMODATED = "BINDING_ACCOMMODATED"
    UNBOUND = "BINDING_UNBOUND"
    REFUSED = "BINDING_REFUSED"


class Evidence(StrEnum):
    FRAGMENT = "EVIDENCE_FRAGMENT"
    CLASSIFIER = "EVIDENCE_CLASSIFIER"
    FRAGMENT_AND_CLASSIFIER = "EVIDENCE_FRAGMENT_AND_CLASSIFIER"
    ENTITY_SPAN = "EVIDENCE_ENTITY_SPAN"


class Grounding(StrEnum):
    PRESENTED = "GROUNDING_PRESENTED"
    ACCEPTED = "GROUNDING_ACCEPTED"
    DISPUTED = "GROUNDING_DISPUTED"


class IssueStatus(StrEnum):
    OPEN = "ISSUE_STATUS_OPEN"
    RESOLVED = "ISSUE_STATUS_RESOLVED"
    DROPPED = "ISSUE_STATUS_DROPPED"


class Answerhood(StrEnum):
    NONE = "ANSWERHOOD_NONE"
    PARTIAL = "ANSWERHOOD_PARTIAL"
    COMPLETE = "ANSWERHOOD_COMPLETE"


class TaskStatus(StrEnum):
    PENDING = "TASK_STATUS_PENDING"
    BLOCKED = "TASK_STATUS_BLOCKED"
    ACTIVE = "TASK_STATUS_ACTIVE"
    DONE = "TASK_STATUS_DONE"
    DECLINED = "TASK_STATUS_DECLINED"
    FAILED = "TASK_STATUS_FAILED"


class Source(StrEnum):
    TOOL = "SOURCE_TOOL"
    WEB = "SOURCE_WEB"
    VISION = "SOURCE_VISION"
    AUDIO = "SOURCE_AUDIO"
    CONNECTOR = "SOURCE_CONNECTOR"
    KNOWLEDGE = "SOURCE_KNOWLEDGE"


class Scope(StrEnum):
    MOVE = "SCOPE_MOVE"
    COMMON_GROUND = "SCOPE_COMMON_GROUND"
    CALLER_COMMITMENTS = "SCOPE_CALLER_COMMITMENTS"
    TOP_ISSUE = "SCOPE_TOP_ISSUE"
    ANY_ISSUE = "SCOPE_ANY_ISSUE"


class Delivery(StrEnum):
    SAY = "DELIVERY_SAY"
    KNOW = "DELIVERY_KNOW"


class Verdict(StrEnum):
    ALLOW = "VERDICT_ALLOW"
    HOLD = "VERDICT_HOLD"
    REJECT = "VERDICT_REJECT"


class Objection(StrEnum):
    INCONSISTENT = "OBJECTION_INCONSISTENT"
    IRRELEVANT = "OBJECTION_IRRELEVANT"
    UNBOUND_PRESUPPOSITION = "OBJECTION_UNBOUND_PRESUPPOSITION"
    FORBIDDEN = "OBJECTION_FORBIDDEN"
    UNGROUNDED_REFERENT = "OBJECTION_UNGROUNDED_REFERENT"
    FLOOR = "OBJECTION_FLOOR"
    UNSUPPORTED = "OBJECTION_UNSUPPORTED"
    ILL_TYPED = "OBJECTION_ILL_TYPED"


class DeliveryOutcome(StrEnum):
    DELIVERED = "DELIVERY_OUTCOME_DELIVERED"
    CUT = "DELIVERY_OUTCOME_CUT"
    HELD = "DELIVERY_OUTCOME_HELD"
    LOST = "DELIVERY_OUTCOME_LOST"
    PRIME = "DELIVERY_OUTCOME_PRIME"


class RequirementNeed(StrEnum):
    GROUNDED = "grounded"
    ESTABLISHED = "established"
    RESOLVED = "resolved"
    COMPLETED = "completed"


class ConditionTest(StrEnum):
    ENTAILS = "entails"
    ISSUE = "issue"
    TASK = "task"
    REFERENT = "referent"
    PROSODY = "prosody"
    FLOOR = "floor"
    FORCE = "force"
    ALL = "all"
    ANY = "any"
    NEGATED = "negated"


class HandoffTarget(StrEnum):
    SIP = "sip"
    PHONE = "phone"


class ActionAct(StrEnum):
    INJECT = "inject"
    ASK = "ask"
    GROUND = "ground"
    PERFORM = "perform"
    FORBID = "forbid"
    CEDE = "cede"
    RESOLVE = "resolve"
    HANDOFF = "handoff"
    PUSH = "push"
    DROP = "drop"


class TransitionCause(StrEnum):
    UTTERANCE = "utterance"
    OBSERVATION = "observation"
    SELECTION = "selection"
    TASK = "task"
    POLICY = "policy"
    DELIVERED = "delivered"
    SLOT_SESSION = "slot_session"


class Span(_Wire):
    start: Int32 = 0
    end: Int32 = 0


class Prominence(_Wire):
    first: Int32 = 0
    last: Int32 = 0


class Prosody(_Wire):
    terminal: Optional[Terminal] = None
    prominence: tuple[Prominence, ...] = ()
    floor: Optional[Floor] = None
    delta: float = 0.0


class Presupposition(_Wire):
    id: str = ""
    content: Optional[SemTerm] = None
    binding: Optional[Binding] = None
    antecedent: str = ""


class Move(_Wire):
    id: str = ""
    force: Optional[Force] = None
    content: Optional[SemTerm] = None
    referents: tuple[str, ...] = ()
    presuppositions: tuple[Presupposition, ...] = ()
    relation: Optional[Relation] = None
    target: str = ""
    evidence: Optional[Evidence] = None


class Utterance(_Wire):
    id: str = ""
    speaker: Optional[Participant] = None
    span: Optional[Span] = None
    prosody: Optional[Prosody] = None
    readings: tuple[Move, ...] = ()
    chosen: str = ""


class Commitment(_Wire):
    id: str = ""
    proposition: Optional[SemTerm] = None
    by: Optional[Participant] = None
    move: str = ""
    grounding: Optional[Grounding] = None
    session: str = ""


class Realization(_Wire):
    text: str = ""


class Answer(_Wire):
    move: str = ""
    proposition: Optional[SemTerm] = None
    answerhood: Optional[Answerhood] = None
    realization: Optional[Realization] = None


class Issue(_Wire):
    id: str = ""
    question: Optional[SemTerm] = None
    owner: Optional[Participant] = None
    raised: str = ""
    parent: str = ""
    status: Optional[IssueStatus] = None
    answers: tuple[Answer, ...] = ()


class Mention(_Wire):
    utterance: str = ""
    span: Optional[Span] = None


class Referent(_Wire):
    id: str = ""
    kind: str = ""
    value: Optional[SemTerm] = None
    introduced: str = ""
    mentions: tuple[Mention, ...] = ()
    rank: Int32 = 0
    grounding: Optional[Grounding] = None
    presented: Optional[Participant] = None
    dictated: bool = False
    session: str = ""


class Grounded(_Wire):
    referent: str = ""
    kind: str = ""
    grounding: Optional[Grounding] = None


class Established(_Wire):
    proposition: Optional[SemTerm] = None


class Resolved(_Wire):
    issue: str = ""


class Completed(_Wire):
    task: str = ""


class Requirement(_Wire):
    grounded: Optional[Grounded] = None
    established: Optional[Established] = None
    resolved: Optional[Resolved] = None
    completed: Optional[Completed] = None

    @model_validator(mode="after")
    def _one_need(self) -> Requirement:
        _exactly_one(self, RequirementNeed)
        return self

    @property
    def need(self) -> RequirementNeed:
        return _present(self, RequirementNeed)[0]


class Argument(_Wire):
    parameter: str = ""
    referent: str = ""


class Task(_Wire):
    id: str = ""
    owner: Optional[Participant] = None
    property: Optional[SemTerm] = None
    source: str = ""
    status: Optional[TaskStatus] = None
    requirements: tuple[Requirement, ...] = ()
    tool: str = ""
    arguments: tuple[Argument, ...] = ()


class Observation(_Wire):
    id: str = ""
    source: Optional[Source] = None
    origin: str = ""
    content: Optional[SemTerm] = None
    raw: Optional[dict[str, Any]] = None
    at: Optional[AwareDatetime] = None
    issue: str = ""


class State(_Wire):
    call: str = ""
    version: Int64 = 0
    commitments: tuple[Commitment, ...] = ()
    table: tuple[Issue, ...] = ()
    tasks: tuple[Task, ...] = ()
    referents: tuple[Referent, ...] = ()
    pending: tuple[Presupposition, ...] = ()
    utterances: tuple[Utterance, ...] = ()
    observations: tuple[Observation, ...] = ()
    floor: Optional[Participant] = None
    awaiting: tuple[str, ...] = ()
    updated: Optional[AwareDatetime] = None
    session: str = ""


class Entails(_Wire):
    situation: Optional[SemTerm] = None
    scope: Optional[Scope] = None


class IssueIs(_Wire):
    question: Optional[SemTerm] = None
    status: Optional[IssueStatus] = None


class TaskIs(_Wire):
    property: Optional[SemTerm] = None
    status: Optional[TaskStatus] = None


class ReferentIs(_Wire):
    kind: str = ""
    grounding: Optional[Grounding] = None


class ProsodyIs(_Wire):
    terminal: Optional[Terminal] = None
    floor: Optional[Floor] = None
    delta: float = 0.0


class FloorIs(_Wire):
    holder: Optional[Participant] = None


class Conditions(_Wire):
    conditions: tuple[Condition, ...] = ()


class Condition(_Wire):
    entails: Optional[Entails] = None
    issue: Optional[IssueIs] = None
    task: Optional[TaskIs] = None
    referent: Optional[ReferentIs] = None
    prosody: Optional[ProsodyIs] = None
    floor: Optional[FloorIs] = None
    force: Optional[Force] = None
    all: Optional[Conditions] = None
    any: Optional[Conditions] = None
    negated: Optional[Condition] = None

    @model_validator(mode="after")
    def _one_test(self) -> Condition:
        _exactly_one(self, ConditionTest)
        return self

    @property
    def test(self) -> ConditionTest:
        return _present(self, ConditionTest)[0]


class Inject(_Wire):
    delivery: Optional[Delivery] = None
    content: Optional[SemTerm] = None
    realization: Optional[Realization] = None


class Ask(_Wire):
    question: Optional[SemTerm] = None


class Ground(_Wire):
    kind: str = ""
    referent: str = ""


class Perform(_Wire):
    tool: str = ""
    arguments: tuple[Argument, ...] = ()
    requirements: tuple[Requirement, ...] = ()


class Forbid(_Wire):
    phrases: tuple[Realization, ...] = ()


class Cede(_Wire):
    pass


class Resolve(_Wire):
    question: Optional[SemTerm] = None
    sources: tuple[Source, ...] = ()


class SipTransfer(_Wire):
    uri: str = ""


class PhoneTransfer(_Wire):
    trunk_id: str = ""
    number: str = ""


class Handoff(_Wire):
    sip: Optional[SipTransfer] = None
    phone: Optional[PhoneTransfer] = None

    @model_validator(mode="after")
    def _one_target(self) -> Handoff:
        _exactly_one(self, HandoffTarget)
        return self

    @property
    def target(self) -> HandoffTarget:
        return _present(self, HandoffTarget)[0]


class Push(_Wire):
    question: Optional[SemTerm] = None
    parent: str = ""


class Drop(_Wire):
    issue: str = ""


class Action(_Wire):
    inject: Optional[Inject] = None
    ask: Optional[Ask] = None
    ground: Optional[Ground] = None
    perform: Optional[Perform] = None
    forbid: Optional[Forbid] = None
    cede: Optional[Cede] = None
    resolve: Optional[Resolve] = None
    handoff: Optional[Handoff] = None
    push: Optional[Push] = None
    drop: Optional[Drop] = None

    @model_validator(mode="after")
    def _one_act(self) -> Action:
        _exactly_one(self, ActionAct)
        return self

    @property
    def act(self) -> ActionAct:
        return _present(self, ActionAct)[0]


class Rule(_Wire):
    id: str = ""
    version: Int32 = 0
    directive: str = ""
    when: tuple[Condition, ...] = ()
    then: tuple[Action, ...] = ()
    priority: Int32 = 0
    once: bool = False


class Goal(_Wire):
    id: str = ""
    question: Optional[SemTerm] = None
    parent: str = ""


class DialoguePolicy(_Wire):
    id: str = ""
    account: str = ""
    version: Int32 = 0
    goals: tuple[Goal, ...] = ()
    rules: tuple[Rule, ...] = ()
    standing: tuple[Task, ...] = ()
    lexicon: str = ""
    updated: Optional[AwareDatetime] = None
    forms: tuple[Form, ...] = ()
    situations: tuple[Situation, ...] = ()


class Candidate(_Wire):
    id: str = ""
    action: Optional[Action] = None
    rule: str = ""


class Judgment(_Wire):
    candidate: str = ""
    verdict: Optional[Verdict] = None
    objections: tuple[Objection, ...] = ()
    answerhood: Optional[Answerhood] = None
    support: tuple[str, ...] = ()


class Selection(_Wire):
    candidates: tuple[Candidate, ...] = ()
    judgments: tuple[Judgment, ...] = ()
    chosen: str = ""


class TaskChange(_Wire):
    task: str = ""
    status: Optional[TaskStatus] = None


class Delivered(_Wire):
    candidate: str = ""
    frame: Int32 = 0
    spoken: Int32 = 0
    total: Int32 = 0
    outcome: Optional[DeliveryOutcome] = None


class SlotSession(_Wire):
    session: str = ""
    previous: str = ""


class DialogueStateDelta(_Wire):
    version: Int64 = 0
    changed: Optional[State] = None
    removed: tuple[str, ...] = ()
    table: tuple[str, ...] = ()
    floor_changed: bool = False
    awaiting_changed: bool = False


class Transition(_Wire):
    seq: Int64 = 0
    at: Optional[AwareDatetime] = None
    utterance: Optional[Utterance] = None
    observation: Optional[Observation] = None
    selection: Optional[Selection] = None
    task: Optional[TaskChange] = None
    policy: Optional[DialoguePolicy] = None
    delivered: Optional[Delivered] = None
    slot_session: Optional[SlotSession] = None
    rules: tuple[str, ...] = ()
    delta: Optional[DialogueStateDelta] = None

    @model_validator(mode="after")
    def _one_cause(self) -> Transition:
        _exactly_one(self, TransitionCause)
        return self

    @property
    def cause(self) -> TransitionCause:
        return _present(self, TransitionCause)[0]


class ResolveRequest(_Wire):
    call: str = ""
    issue: Optional[Issue] = None
    ground: tuple[Commitment, ...] = ()
    referents: tuple[Referent, ...] = ()
    observations: tuple[Observation, ...] = ()
    lexicon: Optional[Lexicon] = None


class ResolveResponse(_Wire):
    answers: tuple[Answer, ...] = ()
    gathered: tuple[Observation, ...] = ()
    asks: tuple[Ask, ...] = ()


class PutPolicyRequest(_Wire):
    account: str = ""
    policy: Optional[DialoguePolicy] = None


class PutPolicyResponse(_Wire):
    policy: Optional[DialoguePolicy] = None


class GetPolicyRequest(_Wire):
    account: str = ""
    version: Int32 = 0
    policy: str = ""


class GetPolicyResponse(_Wire):
    policy: Optional[DialoguePolicy] = None


class GetCallStateRequest(_Wire):
    call: str = ""


class GetCallStateResponse(_Wire):
    state: Optional[State] = None


class ListTransitionsRequest(_Wire):
    call: str = ""
    after: Int64 = 0
    limit: Int32 = 0


class ListTransitionsResponse(_Wire):
    transitions: tuple[Transition, ...] = ()
    next: Int64 = 0


for _recursive in (Conditions, Condition):
    _recursive.model_rebuild()
