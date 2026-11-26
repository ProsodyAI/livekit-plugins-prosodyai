from __future__ import annotations

from datetime import datetime
from typing import Annotated, Mapping, Optional, Protocol, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from .slot import TenancyLevel
from .tracks import StrEnum

Int32 = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]

_Case = TypeVar("_Case", bound=StrEnum)


class _Wire(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    def json_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True, exclude_none=True).encode()


def _present(model: BaseModel, cases: type[_Case]) -> list[_Case]:
    return [case for case in cases if getattr(model, case.value) is not None]


def _exactly_one(model: BaseModel, cases: type[_Case]) -> None:
    if len(_present(model, cases)) != 1:
        raise ValueError(f"exactly one of {', '.join(cases)} must be set")


class EffectKind(StrEnum):
    SET = "EFFECT_KIND_SET"
    READER = "EFFECT_KIND_READER"
    WRITER = "EFFECT_KIND_WRITER"
    CONT = "EFFECT_KIND_CONT"


class SemTypeKind(StrEnum):
    BASIC = "basic"
    FN = "fn"
    PROD = "prod"
    EFF = "eff"


class SemTermKind(StrEnum):
    VAR = "var"
    CONSTANT = "constant"
    LIT = "lit"
    LAM = "lam"
    APP = "app"
    SUM = "sum"
    PAIR = "pair"
    FST = "fst"
    SND = "snd"


class LitTermValue(StrEnum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    TEXT = "text"


class Effect(_Wire):
    kind: EffectKind
    index: tuple[SemType, ...] = ()


class BasicType(_Wire):
    name: str


class FnType(_Wire):
    dom: SemType
    cod: SemType


class ProdType(_Wire):
    left: SemType
    right: SemType


class EffType(_Wire):
    effect: Effect
    arg: SemType


class SemType(_Wire):
    basic: Optional[BasicType] = None
    fn: Optional[FnType] = None
    prod: Optional[ProdType] = None
    eff: Optional[EffType] = None

    @model_validator(mode="after")
    def _one_kind(self) -> SemType:
        _exactly_one(self, SemTypeKind)
        return self

    @property
    def kind(self) -> SemTypeKind:
        return _present(self, SemTypeKind)[0]


class VarTerm(_Wire):
    name: str
    ty: SemType


class ConstTerm(_Wire):
    name: str
    ty: SemType


class LitTerm(_Wire):
    boolean: Optional[bool] = None
    integer: Optional[Int32] = None
    text: Optional[str] = None
    ty: SemType

    @model_validator(mode="after")
    def _one_value(self) -> LitTerm:
        _exactly_one(self, LitTermValue)
        return self

    @property
    def value(self) -> LitTermValue:
        return _present(self, LitTermValue)[0]


class LamTerm(_Wire):
    param: str
    domain: SemType
    body: SemTerm


class AppTerm(_Wire):
    fun: SemTerm
    arg: SemTerm


class SumTerm(_Wire):
    parts: tuple[SemTerm, ...] = ()


class PairTerm(_Wire):
    left: SemTerm
    right: SemTerm


class ProjTerm(_Wire):
    of: SemTerm


class SemTerm(_Wire):
    var: Optional[VarTerm] = None
    constant: Optional[ConstTerm] = None
    lit: Optional[LitTerm] = None
    lam: Optional[LamTerm] = None
    app: Optional[AppTerm] = None
    sum: Optional[SumTerm] = None
    pair: Optional[PairTerm] = None
    fst: Optional[ProjTerm] = None
    snd: Optional[ProjTerm] = None

    @model_validator(mode="after")
    def _one_kind(self) -> SemTerm:
        _exactly_one(self, SemTermKind)
        return self

    @property
    def kind(self) -> SemTermKind:
        return _present(self, SemTermKind)[0]


class LexicalEntry(_Wire):
    word: str
    category: str
    meaning: SemTerm
    ty: SemType


class Lexicon(_Wire):
    entries: tuple[LexicalEntry, ...] = ()


class SemioticClass(StrEnum):
    CARDINAL = "SEMIOTIC_CLASS_CARDINAL"
    DECIMAL = "SEMIOTIC_CLASS_DECIMAL"
    MONEY = "SEMIOTIC_CLASS_MONEY"
    DATE = "SEMIOTIC_CLASS_DATE"
    TIME = "SEMIOTIC_CLASS_TIME"
    TELEPHONE = "SEMIOTIC_CLASS_TELEPHONE"


class Rendering(StrEnum):
    WORDS = "RENDERING_WORDS"
    DIGITS = "RENDERING_DIGITS"


class Entity(StrEnum):
    ADDRESS = "ENTITY_ADDRESS"
    PHONE = "ENTITY_PHONE"
    EMAIL = "ENTITY_EMAIL"
    AMOUNT = "ENTITY_AMOUNT"
    DATE = "ENTITY_DATE"
    NAME = "ENTITY_NAME"


class BundleStatus(StrEnum):
    COMPILING = "BUNDLE_STATUS_COMPILING"
    TRAINING = "BUNDLE_STATUS_TRAINING"
    READY = "BUNDLE_STATUS_READY"
    FAILED = "BUNDLE_STATUS_FAILED"


class DirectiveKind(StrEnum):
    SAY = "say"
    KNOW = "know"
    FORM = "form"
    BEFORE = "before"
    NEVER = "never"
    ALWAYS = "always"


class BeforeFirst(StrEnum):
    READBACK = "readback"
    CALLER = "caller"


class Situation(_Wire):
    label: str
    meaning: SemTerm


class Say(_Wire):
    when: Situation
    text: str


class Know(_Wire):
    when: Situation
    text: str


class Form(_Wire):
    semiotic: SemioticClass
    rendering: Rendering


class Readback(_Wire):
    entity: Entity


class Before(_Wire):
    readback: Optional[Readback] = None
    caller: Optional[Situation] = None
    tool: str

    @model_validator(mode="after")
    def _one_first(self) -> Before:
        _exactly_one(self, BeforeFirst)
        return self

    @property
    def first(self) -> BeforeFirst:
        return _present(self, BeforeFirst)[0]


class Never(_Wire):
    phrases: tuple[str, ...] = Field(min_length=1)


class Always(_Wire):
    text: str


class Directive(_Wire):
    id: str = ""
    version: Int32 = 0
    scope: TenancyLevel = TenancyLevel.ACCOUNT
    updated: Optional[datetime] = None
    say: Optional[Say] = None
    know: Optional[Know] = None
    form: Optional[Form] = None
    before: Optional[Before] = None
    never: Optional[Never] = None
    always: Optional[Always] = None

    @model_validator(mode="after")
    def _one_kind(self) -> Directive:
        _exactly_one(self, DirectiveKind)
        return self

    @property
    def kind(self) -> DirectiveKind:
        return _present(self, DirectiveKind)[0]

    def situations(self) -> tuple[Situation, ...]:
        if self.say is not None:
            return (self.say.when,)
        if self.know is not None:
            return (self.know.when,)
        if self.before is not None and self.before.caller is not None:
            return (self.before.caller,)
        return ()


class DirectiveRef(_Wire):
    id: str
    version: Int32 = 0


class LabelMetrics(_Wire):
    label: str
    precision: float = 0.0
    recall: float = 0.0
    positives: Int32 = 0
    negatives: Int32 = 0
    threshold: float = 0.0


class Bundle(_Wire):
    id: str
    organization: str = ""
    account: str
    status: BundleStatus
    directives: tuple[DirectiveRef, ...] = ()
    engine: str = ""
    encoder: str = ""
    job: str = ""
    metrics: tuple[LabelMetrics, ...] = ()
    failure: str = ""
    created: Optional[datetime] = None
    ready: Optional[datetime] = None


class BundleManifest(_Wire):
    bundle: str
    organization: str = ""
    account: str
    engine: str = ""
    encoder: str = ""
    tokenizer: str = ""
    directives: tuple[DirectiveRef, ...] = ()
    files: dict[str, str] = Field(default_factory=dict)
    lexicon: Int32 = 0


class PutDirectiveRequest(_Wire):
    account_id: str = ""
    directive: Directive
    expected_version: Optional[Int32] = None


class PutDirectiveResponse(_Wire):
    directive: Directive


class ListDirectivesRequest(_Wire):
    account_id: str = ""
    page_size: Int32 = 0
    page_token: str = ""


class ListDirectivesResponse(_Wire):
    directives: tuple[Directive, ...] = ()
    next_page_token: str = ""


class PutLexiconRequest(_Wire):
    account_id: str = ""
    lexicon: Lexicon


class PutLexiconResponse(_Wire):
    entries: Int32 = 0


class CreateBundleRequest(_Wire):
    account_id: str = ""
    directives: tuple[DirectiveRef, ...] = ()
    idempotency_key: str = ""


class CreateBundleResponse(_Wire):
    bundle: Bundle


class GetBundleRequest(_Wire):
    account_id: str = ""
    bundle_id: str = ""


class GetBundleResponse(_Wire):
    bundle: Bundle
    dataset_url: str = ""
    artifact_url: str = ""


class BundleObject(StrEnum):
    ARCHIVE = "bundle.tar"
    DATASET = "dataset.jsonl"
    RESULT = "bundle.json"


class BundleDispatch(Protocol):
    def __call__(
        self,
        bundle: Bundle,
        directives: Sequence[Directive],
        lexicon: Lexicon,
        uploads: Mapping[BundleObject, str],
        *,
        lexicon_version: int,
        attempt: int,
    ) -> str: ...


for _recursive in (
    Effect,
    FnType,
    ProdType,
    EffType,
    SemType,
    VarTerm,
    ConstTerm,
    LitTerm,
    LamTerm,
    AppTerm,
    SumTerm,
    PairTerm,
    ProjTerm,
    SemTerm,
):
    _recursive.model_rebuild()
