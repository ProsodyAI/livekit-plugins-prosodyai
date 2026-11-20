"""The duplex simulation fork grades speaker attribution without a cloud run."""

from livekit.plugins.prosodyai.simulation import attribution_failure
from livekit.plugins.prosodyai.wire import (
    RoomEventType,
    TranscriptDelta,
    TranscriptEvent,
)

HEARD = TranscriptEvent(
    speaker_id="speaker_0",
    deltas=(TranscriptDelta(text="can you hear me", start_ms=0, end_ms=560),),
).to_dict()


def test_words_without_a_speaker_fail_the_duplex_run() -> None:
    unattributed = {
        "type": RoomEventType.TRANSCRIPT.value,
        "speaker_id": None,
        "deltas": [{"text": "Dude, so", "start_ms": 560, "end_ms": 1120}],
    }
    reason = attribution_failure([unattributed], require_speaker_on_words=True)
    assert reason is not None
    assert "no speaker_id" in reason
    assert "Dude, so" in reason


def test_a_run_with_no_caller_transcript_fails() -> None:
    reason = attribution_failure([], require_speaker_on_words=True)
    assert reason == "duplex simulation produced no caller transcript"


def test_attributed_words_pass() -> None:
    assert attribution_failure([HEARD], require_speaker_on_words=True) is None


def test_the_speaker_check_stays_off_unless_the_scenario_asks() -> None:
    assert attribution_failure([], require_speaker_on_words=False) is None
