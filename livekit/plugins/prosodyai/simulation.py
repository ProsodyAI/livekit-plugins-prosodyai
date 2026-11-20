"""Grade a duplex simulation on the words the gateway committed.

LiveKit's text simulation turns audio off, and a duplex model has nothing to
say on that path. The audio simulator still judges the spoken conversation.
This module grades the other half: caller words on the room topic either
carry a speaker or the run fails.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .wire import AGENT_SPEAKER_ID, UNKNOWN_SPEAKER_ID, RoomEventType

_REQUIRE_SPEAKER = "require_speaker_on_words"


class SimulationTranscripts:
    """Room transcript events recorded during one simulation job."""

    def __init__(self) -> None:
        self._by_job: dict[str, list[dict[str, Any]]] = {}

    def record(self, job_id: str, event: Mapping[str, Any]) -> None:
        self._by_job.setdefault(job_id, []).append(dict(event))

    def events(self, job_id: str) -> list[dict[str, Any]]:
        return list(self._by_job.get(job_id, ()))


def _words(event: Mapping[str, Any]) -> list[str]:
    return [
        str(delta["text"])
        for delta in event.get("deltas") or ()
        if isinstance(delta, Mapping) and delta.get("text")
    ]


def attribution_failure(
    events: list[Mapping[str, Any]],
    *,
    require_speaker_on_words: bool,
) -> Optional[str]:
    """Why this duplex run failed the speaker contract, or None when it held.

    Only caller words are graded; the agent's own line is not a caller. A
    caller transcript with words and no speaker, or the unattributed label,
    fails the run. No caller words at all fails it too: the duplex path
    produced nothing to attribute.
    """
    if not require_speaker_on_words:
        return None
    caller = [
        event
        for event in events
        if event.get("type") == RoomEventType.TRANSCRIPT.value
        and event.get("speaker_id") != AGENT_SPEAKER_ID
        and _words(event)
    ]
    if not caller:
        return "duplex simulation produced no caller words"
    unattributed: list[str] = []
    for event in caller:
        speaker_id = event.get("speaker_id")
        if (
            not isinstance(speaker_id, str)
            or not speaker_id.strip()
            or speaker_id == UNKNOWN_SPEAKER_ID
        ):
            unattributed.extend(_words(event))
    if unattributed:
        return (
            f"{len(unattributed)} caller words arrived with no speaker: "
            f"{unattributed!r}"
        )
    return None


def require_speaker_on_words(userdata: Optional[Mapping[str, Any]]) -> bool:
    if not userdata:
        return False
    return bool(userdata.get(_REQUIRE_SPEAKER))
