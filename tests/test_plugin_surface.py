"""The public import surface AgentSession actually uses."""

from __future__ import annotations

import json

import pytest
from livekit.agents import Plugin
from livekit.plugins import prosodyai
from livekit.plugins.prosodyai.events import parse_control_event
from livekit.plugins.prosodyai.frames import GatewayControlFrame
from livekit.plugins.prosodyai.realtime import RealtimeModel, RealtimeSession


def test_registers_as_a_livekit_plugin() -> None:
    registered = [
        plugin
        for plugin in Plugin.registered_plugins
        if plugin.package == "livekit.plugins.prosodyai"
    ]
    assert len(registered) == 1
    assert registered[0].version == prosodyai.__version__


def test_realtime_model_is_on_the_package_and_the_submodule() -> None:
    assert prosodyai.RealtimeModel is RealtimeModel
    assert prosodyai.realtime.RealtimeModel is RealtimeModel
    assert prosodyai.RealtimeSession is RealtimeSession


def test_missing_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROSODYAI_API_KEY", raising=False)
    with pytest.raises(prosodyai.GatewayEnvError, match="PROSODYAI_API_KEY"):
        RealtimeModel()


@pytest.mark.asyncio
async def test_model_on_receives_identity_before_the_session_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROSODYAI_API_KEY", "psk_test")
    model = RealtimeModel()
    received: list[object] = []
    model.on("prosodyai.identity", received.append)

    session = model.session()
    event = prosodyai.IdentityEvent(
        speaker_id="speaker_0",
        person_id="person:1",
        display_name="Ada",
        resumed=True,
        resolved_at_ms=3000,
    )

    await session._on_event(event)
    assert received == [event]
    await session.aclose()
    await model.aclose()


@pytest.mark.asyncio
async def test_model_on_receives_turn_and_barge_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROSODYAI_API_KEY", "psk_test")
    model = RealtimeModel()
    turns: list[object] = []
    barges: list[object] = []
    model.on("turn_boundary", turns.append)
    model.on("barge_in", barges.append)

    session = model.session()
    turn = prosodyai.TurnBoundaryEvent(frame_ms=6400, commit_ms=6720)
    barge = prosodyai.BargeInEvent(
        frame_ms=9600, commit_ms=10080, duration_ms=480, resolved=True
    )
    await session._on_event(turn)
    await session._on_event(barge)
    assert turns == [turn]
    assert barges == [barge]
    await session.aclose()
    await model.aclose()


def _event_frame(payload: dict) -> GatewayControlFrame:
    return GatewayControlFrame(
        kind=GatewayControlFrame.EVENT,
        payload=json.dumps(payload).encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_example_string_listeners_fire_from_a_socket_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROSODYAI_API_KEY", "psk_test")
    model = RealtimeModel()
    turns: list[object] = []
    barges: list[object] = []
    model.on("turn_boundary", turns.append)
    model.on("barge_in", barges.append)
    session = model.session()

    turn = parse_control_event(
        _event_frame({"type": "turn_boundary", "frame_ms": 6400, "commit_ms": 6720})
    )
    barge = parse_control_event(
        _event_frame(
            {
                "type": "barge_in",
                "frame_ms": 9600,
                "commit_ms": 10080,
                "duration_ms": 480,
                "resolved": True,
            }
        )
    )
    assert isinstance(turn, prosodyai.TurnBoundaryEvent)
    assert isinstance(barge, prosodyai.BargeInEvent)
    await session._on_event(turn)
    await session._on_event(barge)
    assert turns == [turn]
    assert barges == [barge]
    await session.aclose()
    await model.aclose()
