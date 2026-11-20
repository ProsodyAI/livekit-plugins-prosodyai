"""Grade a duplex simulation on the words the gateway committed.

LiveKit's text simulation turns audio off, and a duplex model has nothing to
say on that path. The audio simulator still judges the spoken conversation.
This module grades the other half: caller words on the room topic either
carry a session-local speaker or the run fails.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .wire import RoomEventType

_REQUIRE_SPEAKER = "require_speaker_on_words"


class SimulationTranscripts:
    """Caller transcripts recorded during one simulation job."""

    def __init__(self) -> None:
        self._by_job: dict[str, list[dict[str, Any]]] = {}

    def record(self, job_id: str, event: Mapping[str, Any]) -> None:
        self._by_job.setdefault(job_id, []).append(dict(event))

    def events(self, job_id: str) -> list[dict[str, Any]]:
        return list(self._by_job.get(job_id, ()))


def attribution_failure(
    events: list[Mapping[str, Any]],
    *,
    require_speaker_on_words: bool,
) -> Optional[str]:
    """Why this duplex run failed the speaker contract, or None when it held.

    A transcript with no words is silence and is not a speaker failure. A
    transcript with words and no ``speaker_id`` is the failure this fork
    exists to catch. No caller transcript at all is the same failure: the
    duplex path produced nothing to attribute.
    """
    if not require_speaker_on_words:
        return None
    transcripts = [
        event
        for event in events
        if event.get("type") == RoomEventType.TRANSCRIPT.value
    ]
    if not transcripts:
        return "duplex simulation produced no caller transcript"
    for event in transcripts:
        deltas = event.get("deltas") or ()
        words = [
            delta.get("text")
            for delta in deltas
            if isinstance(delta, Mapping) and delta.get("text")
        ]
        if not words:
            continue
        speaker_id = event.get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            return f"caller words {words!r} arrived with no speaker_id"
    return None


def require_speaker_on_words(userdata: Optional[Mapping[str, Any]]) -> bool:
    if not userdata:
        return False
    return bool(userdata.get(_REQUIRE_SPEAKER))
