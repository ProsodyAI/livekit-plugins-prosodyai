"""What a session leaves behind, and what it tells the caller on the way out.

A worker process takes call after call against one model, so anything a
closed session still holds it holds for the life of the worker. And a
session that never bound has to say so: the failure used to go out with the
teardown's cancel, which read as a clean hangup.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pytest.importorskip("sphn", reason="Opus bridging needs sphn")

from livekit.plugins.prosodyai import full_duplex, personaplex
from livekit.plugins.prosodyai.full_duplex import (
    GATEWAY_FRAME_SAMPLES,
    FullDuplexBridge,
    FullDuplexBridgeConfig,
)
from livekit.plugins.prosodyai.realtime import RealtimeModel
from livekit.plugins.prosodyai.speech_backend import SpeechSessionConfig
from livekit.plugins.prosodyai.wire import KIND_HANDSHAKE


class _FakeSocket:
    """A gateway that greets or stays silent, then holds the socket open."""

    def __init__(self, *, greet: bool) -> None:
        self._greet = greet
        self.sent: list[bytes] = []
        self._hangup = asyncio.Event()

    async def send(self, message: bytes) -> None:
        self.sent.append(message)

    async def __aenter__(self) -> "_FakeSocket":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def __aiter__(self):
        if self._greet:
            yield bytes([KIND_HANDSHAKE])
        await self._hangup.wait()

    def hangup(self) -> None:
        self._hangup.set()


def _install(monkeypatch: pytest.MonkeyPatch, socket: _FakeSocket) -> None:
    monkeypatch.setattr(personaplex, "ws_connect", lambda *a, **k: socket)


def _bridge() -> FullDuplexBridge:
    return FullDuplexBridge(
        FullDuplexBridgeConfig(url="ws://fake", api_key="psk_test", room_sample_rate=24_000)
    )


async def _frames(count: int):
    """``count`` gateway-sized frames of speech, then exhaust."""
    for index in range(count):
        yield (4_000 + index).to_bytes(2, "little", signed=True) * GATEWAY_FRAME_SAMPLES


async def _frames_then_hold(count: int):
    """The room's own shape: frames, then an open channel with nothing on it."""
    async for frame in _frames(count):
        yield frame
    await asyncio.sleep(3600)


# ------------------------------------------------- the handshake that never lands


@pytest.mark.asyncio
async def test_a_gateway_that_never_greets_raises_out_of_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The uplink's timeout is the caller's news, not the teardown's.

    The downlink waits on a socket that will never speak, so awaiting it
    alone meant the error died in the cancel and ``run`` returned as if the
    call had simply ended.
    """
    monkeypatch.setattr(full_duplex, "GATEWAY_READY_TIMEOUT", 0.1)
    socket = _FakeSocket(greet=False)
    _install(monkeypatch, socket)

    with pytest.raises(RuntimeError, match="never completed its session handshake"):
        await asyncio.wait_for(
            _bridge().run(_frames_then_hold(6), on_downlink_pcm16=lambda _pcm: asyncio.sleep(0)),
            timeout=5.0,
        )


@pytest.mark.asyncio
async def test_the_session_outlives_an_uplink_that_runs_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Room audio ending is not the call ending.

    Only a *failing* uplink ends the session; an exhausted one leaves the
    downlink to keep publishing whatever the model is still saying.
    """
    socket = _FakeSocket(greet=True)
    _install(monkeypatch, socket)

    runner = asyncio.create_task(
        _bridge().run(_frames(4), on_downlink_pcm16=lambda _pcm: asyncio.sleep(0))
    )
    await asyncio.sleep(0.2)
    assert not runner.done(), "the uplink running dry ended a session the gateway still held"

    socket.hangup()
    await asyncio.wait_for(runner, timeout=5.0)


# ------------------------------------------------------- what close() releases


@pytest.mark.asyncio
async def test_close_releases_the_opus_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Opus writer and reader are native allocations, and the uplink
    accumulator holds up to one frame of caller audio. A worker that takes
    call after call must not carry a set per call."""
    socket = _FakeSocket(greet=True)
    _install(monkeypatch, socket)
    backend = personaplex.PersonaPlexBackend(url="ws://fake", api_key="psk_test")
    await backend.open(SpeechSessionConfig())

    # A partial frame, so the accumulator is holding something at close.
    await backend.send_audio(np.zeros(GATEWAY_FRAME_SAMPLES + 500, dtype=np.float32))
    assert backend._pending_uplink.size == 500

    await backend.close()

    assert backend._writer is None
    assert backend._reader is None
    assert backend._pending_uplink.size == 0

    # A closed backend is inert rather than broken: the room can still push a
    # frame that raced the teardown.
    await backend.send_audio(np.zeros(GATEWAY_FRAME_SAMPLES, dtype=np.float32))
    assert [item async for item in backend.receive()] == []


# ------------------------------------------------- what the model keeps hold of


@pytest.mark.asyncio
async def test_the_model_forgets_sessions_as_they_close() -> None:
    """One model, many calls: a closed session left on the model's list holds
    its channels and its bridge for the life of the worker."""
    model = RealtimeModel(base_url="ws://fake", api_key="psk_test")

    first, second = model.session(), model.session()
    assert model.sessions == [first, second]

    await first.aclose()
    assert model.sessions == [second], "a closed session stayed on the model"

    await second.aclose()
    assert model.sessions == []

    # Closing twice is idempotent, and aclose() on the model is still safe
    # once every session has already gone.
    await second.aclose()
    await model.aclose()
    assert model.sessions == []


@pytest.mark.asyncio
async def test_model_aclose_closes_and_drops_its_open_sessions() -> None:
    model = RealtimeModel(base_url="ws://fake", api_key="psk_test")
    sessions = [model.session() for _ in range(3)]

    await model.aclose()

    assert model.sessions == []
    assert all(session._closed for session in sessions)
