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
    reason = attribution_failure([_transcript(None, "Dude, so")], require_speaker_on_words=True)
    assert reason is not None
    assert "Dude, so" in reason


def test_a_run_before_any_voice_belongs_to_the_first_voice() -> None:
    # Words the gateway emits before the tracker commits a lane arrive
    # unattributed; no other voice held the floor, so they are the first voice's.
    assert (
        attribution_failure(
            [
                _transcript(UNKNOWN_SPEAKER_ID, "We", "do"),
                _transcript("speaker_0", "so", "we"),
            ],
            require_speaker_on_words=True,
        )
        is None
    )


def test_a_run_between_two_voices_stays_unattributed() -> None:
    reason = attribution_failure(
        [
            _transcript("speaker_0", "We", "do"),
            _transcript(UNKNOWN_SPEAKER_ID, "so", "we"),
            _transcript("speaker_1", "are", "so"),
        ],
        require_speaker_on_words=True,
    )
    assert reason is not None
    assert "so" in reason


def test_a_word_still_forming_is_not_graded() -> None:
    """An interim delta is replaced by its final, which carries the speaker
    the interval settled on; grading the interim would fail a run on a word
    nobody was shown as said."""
    forming = TranscriptEvent(
        speaker_id="speaker_0",
        deltas=(TranscriptDelta(text="every", start_ms=0, end_ms=560, final=False),),
    ).to_dict()
    forming["speaker_id"] = UNKNOWN_SPEAKER_ID
    whole = _transcript("speaker_0", "everyone")
    assert attribution_failure([forming, whole], require_speaker_on_words=True) is None
    # Interim words alone are no caller words at all.
    assert attribution_failure([forming], require_speaker_on_words=True) is not None


def test_the_agent_line_is_not_a_caller() -> None:
    reason = attribution_failure(
        [_transcript(AGENT_SPEAKER_ID, "Hello, this is Prosody.")],
        require_speaker_on_words=True,
    )
    assert reason == "duplex simulation produced no caller words"


def test_the_speaker_check_stays_off_unless_the_scenario_asks() -> None:
    assert attribution_failure([], require_speaker_on_words=False) is None
