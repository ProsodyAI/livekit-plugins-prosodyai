"""Tracks: what a session is made of, and how models compose over them.

A full-duplex call is a set of streams on one frame clock. Two belong to the
speech model by construction: it *listens* to the caller and it *speaks*.
The third is *thinking*: text the speech model reads beside the audio, and
the only way anything else reaches it.

Every model on the card is a :class:`Stage` that declares the tracks it
reads and the tracks it writes. A :class:`Score` is the higher-order piece:
it takes stages, orders them by what each needs from the others, and is
itself one component, so the slot scheduler steps the whole composition for
every session of every tenant in one pass. Adding a capability is adding a
stage: a recognizer writes the caller's words to the thinking track, a
voicemail detector writes that nobody is there, a tool's result arrives from
outside the card; the speech model reads them all the same way and none of
them knows the others exist.

Thoughts also arrive between passes, from things too slow for a frame (a
deciding model, a customer's tool). A session's :class:`ThinkingTrack` holds
those until the next pass takes them.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, ConfigDict

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """``enum.StrEnum`` for Python 3.10, which the LiveKit plugin supports."""

        def __str__(self) -> str:
            return str(self.value)


class Track(StrEnum):
    """The streams of a session."""

    #: The caller's audio, one frame per pass.
    LISTENING = "listening"
    #: The agent's audio, one frame per pass.
    SPEAKING = "speaking"
    #: Thoughts for the speech model, any number per pass.
    THINKING = "thinking"


class Intent(StrEnum):
    """What the speech model does with a thought."""

    #: Hold it as context and speak from it when the conversation calls for it.
    KNOW = "know"
    #: Say it now.
    SAY = "say"


class Thought(BaseModel):
    """One thing the speech model is told by something that is not the caller."""

    model_config = ConfigDict(frozen=True)

    #: The stage or outside party that thought it.
    source: str
    text: str
    intent: Intent = Intent.KNOW


#: One slot's tracks for one pass.
Frame = Mapping[Track, Any]


class Stage(Protocol):
    """One model, as a function from the tracks it reads to the tracks it writes."""

    reads: frozenset[Track]
    writes: frozenset[Track]

    def start(self, slot: int) -> None:
        """A new session takes this slot."""

    def end(self, slot: int) -> None:
        """The slot's session is over."""

    def step(self, frames: Sequence[Optional[Frame]]) -> Sequence[Optional[Frame]]:
        """One pass. ``frames[slot]`` holds the tracks this stage reads, or
        None when the slot has nothing for it. Returns what it writes."""


class ScoreError(ValueError):
    """Stages that cannot be composed: something is read that nothing provides."""


class Score:
    """Stages composed by dataflow into one component.

    ``inputs`` are the tracks a session supplies itself. A stage runs once
    every track it reads has been supplied or written, so the order comes
    from the declarations and not from how the stages were listed. Thinking
    accumulates: every stage that writes it adds to what the next reader
    sees. Any other track has one writer.
    """

    def __init__(self, *stages: Stage, inputs: frozenset[Track]) -> None:
        self._inputs = inputs
        self._order = _ordered(stages, inputs)

    @property
    def order(self) -> tuple[Stage, ...]:
        return self._order

    def seat(self, slot: int, tenant: str) -> None:
        """Tell each stage that keeps tenants apart whose session this slot holds."""
        for stage in self._order:
            seat = getattr(stage, "seat", None)
            if seat is not None:
                seat(slot, tenant)

    def start(self, slot: int) -> None:
        for stage in self._order:
            stage.start(slot)

    def end(self, slot: int) -> None:
        for stage in self._order:
            stage.end(slot)

    def step(self, frames: Sequence[Optional[Frame]]) -> list[Optional[dict[Track, Any]]]:
        rows: list[Optional[dict[Track, Any]]] = [
            None if frame is None else dict(frame) for frame in frames
        ]
        for stage in self._order:
            given = [
                {track: row[track] for track in stage.reads}
                if row is not None and stage.reads <= row.keys()
                else None
                for row in rows
            ]
            if not any(frame is not None for frame in given):
                continue
            written = stage.step(given)
            for row, out in zip(rows, written, strict=True):
                if row is None or out is None:
                    continue
                for track, value in out.items():
                    if track is Track.THINKING:
                        row[track] = (*row.get(track, ()), *value)
                    else:
                        row[track] = value
        return rows


def _ordered(stages: Sequence[Stage], inputs: frozenset[Track]) -> tuple[Stage, ...]:
    """The stages in an order where each reads only what is already there.

    A stage that both reads and writes thinking reads what came before it;
    it does not wait on itself.
    """
    single_writer: dict[Track, Stage] = {}
    for stage in stages:
        for track in stage.writes - {Track.THINKING}:
            if track in inputs or track in single_writer:
                raise ScoreError(f"{track.value} has more than one source")
            single_writer[track] = stage
    available = set(inputs)
    pending = list(stages)
    ordered: list[Stage] = []
    while pending:
        # Thinking is ready for a reader once every other stage that writes
        # it has run, so a reader sees the whole pass's thoughts.
        ready = [
            stage
            for stage in pending
            if (stage.reads - {Track.THINKING}) <= available
            and (
                Track.THINKING not in stage.reads
                or not any(
                    Track.THINKING in other.writes
                    for other in pending
                    if other is not stage and Track.THINKING not in other.reads
                )
            )
        ]
        if not ready:
            missing = sorted(
                track.value
                for stage in pending
                for track in stage.reads - available - {Track.THINKING}
            )
            raise ScoreError(f"nothing provides: {', '.join(missing) or 'a thinking order'}")
        for stage in ready:
            ordered.append(stage)
            pending.remove(stage)
            available |= stage.writes
    return tuple(ordered)


class Rule(BaseModel):
    """One customer's fix: when the caller says this, the model is told that.

    ``hears`` are phrases; any one of them in the caller's committed words
    fires the rule. This is how a model is corrected for one customer
    without retraining it: the right sentence, in its context, at the moment
    it matters.
    """

    model_config = ConfigDict(frozen=True)

    hears: tuple[str, ...]
    thought: Thought
    #: Fire on the first match only; a fact told once stays in context.
    once: bool = True


class Rules:
    """One session's rules, read against the caller's words as they commit."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        self._rules = list(rules)
        self._fired: set[int] = set()

    def on_words(self, text: str) -> tuple[Thought, ...]:
        """The thoughts the caller's words call for."""
        heard = text.casefold()
        thoughts: list[Thought] = []
        for index, rule in enumerate(self._rules):
            if rule.once and index in self._fired:
                continue
            if any(phrase.casefold() in heard for phrase in rule.hears if phrase):
                self._fired.add(index)
                thoughts.append(rule.thought)
        return tuple(thoughts)


class ThinkingTrack:
    """A session's thoughts from outside the pass, waiting for the next one.

    Written from any thread, by anything that learned something late: a
    deciding model, a tool the customer's agent ran, the listening side.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting: list[Thought] = []

    def write(self, thought: Thought) -> None:
        with self._lock:
            self._waiting.append(thought)

    def take(self) -> tuple[Thought, ...]:
        """Every waiting thought, in the order written; the track is then empty."""
        with self._lock:
            taken, self._waiting = tuple(self._waiting), []
        return taken
