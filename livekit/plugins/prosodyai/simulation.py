"""Grade a duplex simulation on the words the gateway committed.

LiveKit's text simulation turns audio off, and a duplex model has nothing to
say on that path. The audio simulator still judges the spoken conversation.
This module grades the other half: caller words on the room topic either
carry a speaker or the run fails.

The reading is the one every room consumer makes (the SDK's
``RoomTranscript``, the website's room adapter): an interim is the word still
forming and its final replaces it; a ``speaker_change`` is an onset, and the
voice on the floor owns the unattributed words from its onset until the next
different onset, the first onset also owning the unattributed words before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .wire import AGENT_SPEAKER_ID, UNKNOWN_SPEAKER_ID, RoomEventType

_REQUIRE_SPEAKER = "require_speaker_on_words"
_SPEAKER_CHANGE = "prosodyai.speaker_change"
#: A span relabels a word it overlaps by a quarter of the word, or by 200 ms.
_OVERLAP_SHARE = 0.25
_OVERLAP_MS = 200


class SimulationTranscripts:
    """Room events recorded during one simulation job."""

    def __init__(self) -> None:
        self._by_job: dict[str, list[dict[str, Any]]] = {}

    def record(self, job_id: str, event: Mapping[str, Any]) -> None:
        self._by_job.setdefault(job_id, []).append(dict(event))

    def events(self, job_id: str) -> list[dict[str, Any]]:
        return list(self._by_job.get(job_id, ()))


def is_graded_event(payload: Mapping[str, Any]) -> bool:
    """The room events the grade reads: caller words and speaker onsets."""
    return payload.get("type") in (RoomEventType.TRANSCRIPT.value, _SPEAKER_CHANGE)


@dataclass
class _Word:
    text: str
    start_ms: int
    end_ms: int
    speaker_id: str
    committed: bool


def _known(speaker_id: Any) -> bool:
    return (
        isinstance(speaker_id, str)
        and bool(speaker_id.strip())
        and speaker_id != UNKNOWN_SPEAKER_ID
    )


def _relabel(words: list[_Word], start_ms: int, end_ms: int, speaker_id: str) -> None:
    span_end = max(start_ms + 1, end_ms)
    for word in words:
        if word.committed:
            continue
        word_end = max(word.start_ms + 1, word.end_ms)
        overlap = max(0, min(word_end, span_end) - max(word.start_ms, start_ms))
        duration = max(1, word_end - word.start_ms)
        if overlap < duration * _OVERLAP_SHARE and overlap < _OVERLAP_MS:
            continue
        word.speaker_id = speaker_id
        word.committed = True


def caller_words(events: list[Mapping[str, Any]]) -> list[_Word]:
    """The callers' whole words, each owned by the speaker a room reader sees."""
    words: list[_Word] = []
    floor: Optional[Mapping[str, Any]] = None
    spans: list[tuple[int, int, str]] = []
    for event in events:
        kind = event.get("type")
        if kind == _SPEAKER_CHANGE:
            onset = int(event.get("timestamp_ms") or 0)
            speaker = str(event.get("speaker_id") or "")
            if event.get("is_agent") is True or not speaker:
                continue
            if floor is None:
                floor = event
                spans.append((0, onset, speaker))
                _relabel(words, 0, onset, speaker)
            elif floor.get("speaker_id") == speaker or onset <= int(
                floor.get("timestamp_ms") or 0
            ):
                continue
            else:
                previous = (
                    int(floor.get("timestamp_ms") or 0),
                    onset,
                    str(floor.get("speaker_id")),
                )
                floor = event
                spans.append(previous)
                _relabel(words, *previous)
            continue
        if (
            kind != RoomEventType.TRANSCRIPT.value
            or event.get("speaker_id") == AGENT_SPEAKER_ID
        ):
            continue
        speaker_id = event.get("speaker_id")
        for delta in event.get("deltas") or ():
            if (
                not isinstance(delta, Mapping)
                or not delta.get("text")
                or delta.get("final") is False
            ):
                continue
            start_ms = int(delta.get("start_ms") or 0)
            end_ms = int(delta.get("end_ms") or start_ms)
            if _known(speaker_id):
                owner, committed = str(speaker_id), True
            elif floor is not None and start_ms >= int(floor.get("timestamp_ms") or 0):
                owner, committed = str(floor.get("speaker_id")), False
            else:
                owner, committed = UNKNOWN_SPEAKER_ID, False
            words.append(_Word(str(delta["text"]), start_ms, end_ms, owner, committed))
        for span in spans:
            _relabel(words, *span)
    return words


def attribution_failure(
    events: list[Mapping[str, Any]],
    *,
    require_speaker_on_words: bool,
) -> Optional[str]:
    """Why this duplex run failed the speaker contract, or None when it held.

    Only caller words are graded; the agent's own line is not a caller. A
    caller word no speaker owns once the committed onsets are applied fails
    the run. No caller words at all fails it too: the duplex path produced
    nothing to attribute.
    """
    if not require_speaker_on_words:
        return None
    words = caller_words(events)
    if not words:
        return "duplex simulation produced no caller words"
    unattributed = [word.text for word in words if not _known(word.speaker_id)]
    if unattributed:
        return f"{len(unattributed)} caller words arrived with no speaker: {unattributed!r}"
    return None


def require_speaker_on_words(userdata: Optional[Mapping[str, Any]]) -> bool:
    if not userdata:
        return False
    return bool(userdata.get(_REQUIRE_SPEAKER))
