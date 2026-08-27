"""RealtimeModel against a fake gateway speaking the real wire protocol.

The fake server is the gateway contract in miniature; everything the session
promises AgentSession is asserted through the public interface.
"""

import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("sphn", reason="Opus bridging needs sphn")
import sphn
from websockets.asyncio.server import serve

from livekit import rtc
from livekit.plugins.prosodyai import EntitySpanEvent
from livekit.plugins.prosodyai.full_duplex import (
    GATEWAY_FRAME_SAMPLES,
    GATEWAY_SAMPLE_RATE,
    KIND_AUDIO,
    KIND_EVENT,
    KIND_HANDSHAKE,
    KIND_IDENTITY,
    KIND_TEXT,
    KIND_TRANSCRIPT,
)
from livekit.plugins.prosodyai.realtime import (
    IdentityEvent,
    RealtimeModel,
    SpeakerChangeEvent,
    TextEvent,
    TranscriptEvent,
)

ROOM_SAMPLE_RATE = 16_000
FRAME_SAMPLES = ROOM_SAMPLE_RATE * 20 // 1000  # 20 ms room frames

IDENTITY = {
    "speaker_id": "speaker_0",
    "person_id": "person:158a2b2e",
    "display_name": "Ada",
    "resumed": True,
    "resolved_at_ms": 3000,
}

# Exact 0x05 payload from the gateway's transcript sender.
TRANSCRIPT = {
    "speaker_id": "speaker_0",
    "deltas": [
        {"text": "hello", "start_ms": 0, "end_ms": 320},
        {"text": "there", "start_ms": 320, "end_ms": 640},
    ],
}

# Exact 0x06 payload from the gateway's event sender: a committed
# speaker_change with the model's retrodictive onset timestamp.
SPEAKER_CHANGE = {
    "session_id": "sess-test",
    "timestamp_ms": 4000,
    "type": "prosodyai.speaker_change",
    "speaker_id": "speaker_0",
    "previous_speaker_id": None,
    "person_id": "person:158a2b2e",
    "display_name": "Ada",
    "is_agent": False,
}


# The other family on the 0x06 channel: a committed entity span, the shape a
# dictated email arrives in. It carries no ``prosodyai.`` prefix, so the
# gateway-tracker parse declines it and the conversation parse claims it.
ENTITY_SPAN = {
    "type": "entity_span",
    "frame_ms": 5200,
    "commit_ms": 5600,
    "duration_ms": 1840,
    "kind": "email",
}


async def _fake_gateway(websocket) -> None:
    await websocket.send(bytes([KIND_HANDSHAKE]))
    responded = False
    writer = sphn.OpusStreamWriter(GATEWAY_SAMPLE_RATE)
    async for message in websocket:
        if isinstance(message, str) or not message:
            continue
        if message[0] != KIND_AUDIO or responded:
            continue
        responded = True
        await websocket.send(bytes([KIND_TEXT]) + b"hello there")
        await websocket.send(bytes([KIND_IDENTITY]) + json.dumps(IDENTITY).encode("utf-8"))
        await websocket.send(bytes([KIND_TRANSCRIPT]) + json.dumps(TRANSCRIPT).encode("utf-8"))
        await websocket.send(bytes([KIND_EVENT]) + json.dumps(SPEAKER_CHANGE).encode("utf-8"))
        await websocket.send(bytes([KIND_EVENT]) + json.dumps(ENTITY_SPAN).encode("utf-8"))
        clock = np.arange(4 * GATEWAY_FRAME_SAMPLES, dtype=np.float32) / GATEWAY_SAMPLE_RATE
        tone = 0.2 * np.sin(2.0 * np.pi * 440.0 * clock).astype(np.float32)
        for index in range(4):
            packet = writer.append_pcm(
                tone[index * GATEWAY_FRAME_SAMPLES : (index + 1) * GATEWAY_FRAME_SAMPLES]
            )
            if packet:
                await websocket.send(bytes([KIND_AUDIO]) + packet)


def _room_frame() -> rtc.AudioFrame:
    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    return rtc.AudioFrame(
        data=silence.tobytes(),
        sample_rate=ROOM_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=FRAME_SAMPLES,
    )


@pytest.mark.asyncio
async def test_full_duplex_session_against_the_wire_protocol(monkeypatch):
    monkeypatch.setenv("PROSODYAI_API_KEY", "psk_test")
    async with serve(_fake_gateway, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]

        model = RealtimeModel(base_url=f"ws://127.0.0.1:{port}")
        session = model.session()

        generations: asyncio.Queue = asyncio.Queue()
        identities: asyncio.Queue = asyncio.Queue()
        tokens: asyncio.Queue = asyncio.Queue()
        transcripts: asyncio.Queue = asyncio.Queue()
        model_events: asyncio.Queue = asyncio.Queue()
        conversation_events: asyncio.Queue = asyncio.Queue()
        session.on("generation_created", generations.put_nowait)
        session.on("prosody_identity", identities.put_nowait)
        session.on("prosody_text", tokens.put_nowait)
        session.on("prosody_transcript", transcripts.put_nowait)
        session.on("prosody_event", model_events.put_nowait)
        session.on("prosody_conversation", conversation_events.put_nowait)

        # ~400 ms of room audio: enough for the bridge to fill several 80 ms
        # model frames and the fake gateway to answer.
        for _ in range(20):
            session.push_audio(_room_frame())
            await asyncio.sleep(0.005)

        generation = await asyncio.wait_for(generations.get(), timeout=10.0)
        assert generation.user_initiated is False
        message = await asyncio.wait_for(generation.message_stream.__anext__(), timeout=5.0)
        assert await message.modalities == ["audio", "text"]

        text = await asyncio.wait_for(message.text_stream.__anext__(), timeout=10.0)
        assert text == "hello there"

        frame = await asyncio.wait_for(message.audio_stream.__anext__(), timeout=10.0)
        assert frame.sample_rate == GATEWAY_SAMPLE_RATE
        publish_samples = GATEWAY_SAMPLE_RATE * 20 // 1000
        assert frame.samples_per_channel == publish_samples

        second = await asyncio.wait_for(message.audio_stream.__anext__(), timeout=10.0)
        assert second.samples_per_channel == publish_samples

        identity: IdentityEvent = await asyncio.wait_for(identities.get(), timeout=10.0)
        assert identity.person_id == IDENTITY["person_id"]
        assert identity.display_name == "Ada"
        assert identity.resumed is True
        assert identity.resolved_at_ms == 3000
        assert identity.to_dict() == {
            "type": "prosodyai.identity",
            "speaker_id": identity.speaker_id,
            "person_id": IDENTITY["person_id"],
            "display_name": "Ada",
            "resumed": True,
            "resolved_at_ms": 3000,
        }

        token: TextEvent = await asyncio.wait_for(tokens.get(), timeout=10.0)
        assert token.text == "hello there"

        transcript: TranscriptEvent = await asyncio.wait_for(transcripts.get(), timeout=10.0)
        assert transcript.speaker_id == "speaker_0"
        assert [d.text for d in transcript.deltas] == ["hello", "there"]
        assert transcript.deltas[0].start_ms == 0
        assert transcript.deltas[-1].end_ms == 640

        change: SpeakerChangeEvent = await asyncio.wait_for(model_events.get(), timeout=10.0)
        assert isinstance(change, SpeakerChangeEvent)
        assert change.timestamp_ms == 4000
        assert change.speaker_id == "speaker_0"
        assert change.previous_speaker_id is None
        assert change.display_name == "Ada"
        assert change.is_agent is False

        # The conversation family rides the same 0x06 channel and reaches its
        # own name, so an app hears the dictated entity without filtering the
        # tracker's events out of the way.
        span: EntitySpanEvent = await asyncio.wait_for(conversation_events.get(), timeout=10.0)
        assert isinstance(span, EntitySpanEvent)
        assert span.kind == "email"
        assert span.frame_ms == 5200
        assert span.commit_ms == 5600
        assert span.duration_ms == 1840
        assert model_events.empty(), "a conversation event must not reach prosody_event"

        await session.aclose()
        await model.aclose()
