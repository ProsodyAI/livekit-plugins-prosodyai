"""The duplex simulation fork grades speaker attribution without a cloud run."""

from livekit.plugins.prosodyai.simulation import attribution_failure
from livekit.plugins.prosodyai.wire import (
    AGENT_SPEAKER_ID,
    UNKNOWN_SPEAKER_ID,
    TranscriptDelta,
    TranscriptEvent,
)


def _transcript(speaker_id, *words: str) -> dict:
    deltas = tuple(
        TranscriptDelta(text=word, start_ms=index * 560, end_ms=(index + 1) * 560)
        for index, word in enumerate(words)
    )
    event = TranscriptEvent(speaker_id="speaker_0", deltas=deltas).to_dict()
    event["speaker_id"] = speaker_id
    return event


def test_attributed_caller_words_pass() -> None:
    heard = _transcript("speaker_0", "can you hear me")
    assert attribution_failure([heard], require_speaker_on_words=True) is None


def test_caller_words_without_a_speaker_fail() -> None:
    reason = attribution_failure(
        [_transcript(None, "Dude, so")], require_speaker_on_words=True
    )
    assert reason is not None
    assert "Dude, so" in reason


def test_unattributed_label_is_not_a_speaker() -> None:
    reason = attribution_failure(
        [
            _transcript(UNKNOWN_SPEAKER_ID, "We", "do"),
            _transcript("speaker_0", "so", "we"),
        ],
        require_speaker_on_words=True,
    )
    assert reason is not None
    assert reason.startswith("2 caller words")


def test_the_agent_line_is_not_a_caller() -> None:
    reason = attribution_failure(
        [_transcript(AGENT_SPEAKER_ID, "Hello, this is Jarvis.")],
        require_speaker_on_words=True,
    )
    assert reason == "duplex simulation produced no caller words"


def test_the_speaker_check_stays_off_unless_the_scenario_asks() -> None:
    assert attribution_failure([], require_speaker_on_words=False) is None
