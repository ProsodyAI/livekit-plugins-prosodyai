"""RealtimeModel against a fake gateway speaking the real wire protocol."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("sphn", reason="Opus bridging needs sphn")
import sphn
from websockets.asyncio.server import serve

from livekit import rtc
from livekit.plugins.prosodyai import EntitySpanEvent
from livekit.plugins.prosodyai.frames import GatewayAudio, GatewayControlFrame
from livekit.plugins.prosodyai.gateway import FRAME_SAMPLES, SAMPLE_RATE
from livekit.plugins.prosodyai.realtime import RealtimeModel
from livekit.plugins.prosodyai.wire import (
    ConversationEventType,
    GatewayEventType,
    GatewaySpeakerChangeEvent,
    IdentityEvent,
    RoomEventType,
    TextEvent,
    TranscriptEvent,
)

ROOM_SAMPLE_RATE = 16_000
FRAME_ROOM_SAMPLES = ROOM_SAMPLE_RATE * 20 // 1000

IDENTITY = {
    "speaker_id": "speaker_0",
    "person_id": "person:158a2b2e",
    "display_name": "Ada",
    "resumed": True,
    "resolved_at_ms": 3000,
}

TRANSCRIPT = {
    "speaker_id": "speaker_0",
    "deltas": [
        {"text": "hello", "start_ms": 0, "end_ms": 320},
        {"text": "there", "start_ms": 320, "end_ms": 640},
    ],
}

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

ENTITY_SPAN = {
    "type": "entity_span",
    "frame_ms": 5200,
    "commit_ms": 5600,
    "duration_ms": 1840,
    "kind": "email",
}


async def _fake_gateway(websocket) -> None:
    await websocket.send(bytes([GatewayControlFrame.HANDSHAKE]))
    responded = False
    writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    async for message in websocket:
        if isinstance(message, str) or not message:
            continue
        if message[0] != GatewayAudio.KIND or responded:
            continue
        responded = True
        await websocket.send(bytes([GatewayControlFrame.TEXT]) + b"hello there")
        await websocket.send(
            bytes([GatewayControlFrame.IDENTITY]) + json.dumps(IDENTITY).encode("utf-8")
        )
        await websocket.send(
            bytes([GatewayControlFrame.TRANSCRIPT]) + json.dumps(TRANSCRIPT).encode("utf-8")
        )
        await websocket.send(
            bytes([GatewayControlFrame.EVENT]) + json.dumps(SPEAKER_CHANGE).encode("utf-8")
        )
        await websocket.send(
            bytes([GatewayControlFrame.EVENT]) + json.dumps(ENTITY_SPAN).encode("utf-8")
        )
        clock = np.arange(4 * FRAME_SAMPLES, dtype=np.float32) / SAMPLE_RATE
        tone = 0.2 * np.sin(2.0 * np.pi * 440.0 * clock).astype(np.float32)
        for index in range(4):
            packet = writer.append_pcm(
                tone[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES]
            )
            if packet:
                await websocket.send(bytes([GatewayAudio.KIND]) + packet)


def _room_frame() -> rtc.AudioFrame:
    silence = np.zeros(FRAME_ROOM_SAMPLES, dtype=np.int16)
    return rtc.AudioFrame(
        data=silence.tobytes(),
        sample_rate=ROOM_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=FRAME_ROOM_SAMPLES,
    )


@pytest.mark.asyncio
async def test_full_duplex_session_against_the_wire_protocol(monkeypatch):
    monkeypatch.setenv("PROSODYAI_API_KEY", "psk_test")
    async with serve(_fake_gateway, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]

        model = RealtimeModel(base_url=f"ws://127.0.0.1:{port}")
        generations: asyncio.Queue = asyncio.Queue()
        identities: asyncio.Queue = asyncio.Queue()
        tokens: asyncio.Queue = asyncio.Queue()
        transcripts: asyncio.Queue = asyncio.Queue()
        speaker_changes: asyncio.Queue = asyncio.Queue()
        conversation_events: asyncio.Queue = asyncio.Queue()
        model.on(RoomEventType.IDENTITY, identities.put_nowait)
        model.on(RoomEventType.TEXT, tokens.put_nowait)
        model.on(RoomEventType.TRANSCRIPT, transcripts.put_nowait)
        model.on(GatewayEventType.SPEAKER_CHANGE, speaker_changes.put_nowait)
        model.on(ConversationEventType.ENTITY_SPAN, conversation_events.put_nowait)

        session = model.session()
        session.on("generation_created", generations.put_nowait)

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
        assert frame.sample_rate == SAMPLE_RATE
        assert frame.num_channels == 1
        assert frame.samples_per_channel > 0

        identity: IdentityEvent = await asyncio.wait_for(identities.get(), timeout=10.0)
        assert identity.person_id == IDENTITY["person_id"]
        assert identity.display_name == "Ada"
        assert identity.resumed is True
        assert identity.resolved_at_ms == 3000

        token: TextEvent = await asyncio.wait_for(tokens.get(), timeout=10.0)
        assert token.text == "hello there"

        transcript: TranscriptEvent = await asyncio.wait_for(transcripts.get(), timeout=10.0)
        assert transcript.speaker_id == "speaker_0"
        assert [delta.text for delta in transcript.deltas] == ["hello", "there"]

        change: GatewaySpeakerChangeEvent = await asyncio.wait_for(
            speaker_changes.get(), timeout=10.0
        )
        assert change.timestamp_ms == 4000
        assert change.speaker_id == "speaker_0"
        assert change.display_name == "Ada"

        span: EntitySpanEvent = await asyncio.wait_for(conversation_events.get(), timeout=10.0)
        assert span.kind == "email"
        assert span.frame_ms == 5200

        await session.aclose()
        await model.aclose()
