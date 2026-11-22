"""The thinking track through the LiveKit duplex session, on the real wire.

A fake gateway speaks the call's binary protocol: it records every control
frame the session sends and, like the real one, asks for a tool once the
agent has declared it. The session is driven through LiveKit's own duplex
contract: the configuration the framework applies, the function-call event
it listens for, and the tool output it hands back.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("sphn", reason="Opus bridging needs sphn")
from websockets.asyncio.server import serve

from livekit import rtc
from livekit.agents import function_tool, llm
from livekit.plugins.prosodyai.frames import GatewayControlFrame
from livekit.plugins.prosodyai.realtime import DuplexModel
from livekit.plugins.prosodyai.thinking import (
    RulesUpdate,
    ThoughtWrite,
    ToolResult,
    ToolsUpdate,
    parse_client_control,
)
from livekit.plugins.prosodyai.tracks import Intent, Rule, Thought
from livekit.plugins.prosodyai.wire import (
    KIND_CONTROL,
    GatewayAgentToolStatusEvent,
    ToolCallStatus,
)

ROOM_SAMPLE_RATE = 16_000
FRAME_ROOM_SAMPLES = ROOM_SAMPLE_RATE * 20 // 1000


@function_tool
async def lookup_order(order_id: str) -> str:
    """Find an order by its number."""
    return "ships Friday"


def _room_frame() -> rtc.AudioFrame:
    silence = np.zeros(FRAME_ROOM_SAMPLES, dtype=np.int16)
    return rtc.AudioFrame(
        data=silence.tobytes(),
        sample_rate=ROOM_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=FRAME_ROOM_SAMPLES,
    )


@pytest.mark.asyncio
async def test_the_agents_configuration_tools_and_thoughts_reach_the_call(monkeypatch):
    monkeypatch.setenv("PROSODYAI_API_KEY", "psk_test")
    received: list = []
    tool_result = asyncio.Event()

    async def gateway(websocket) -> None:
        await websocket.send(bytes([GatewayControlFrame.HANDSHAKE]))
        async for message in websocket:
            if isinstance(message, str) or not message or message[0] != KIND_CONTROL:
                continue
            control = parse_client_control(message[1:])
            received.append(control)
            if isinstance(control, ToolsUpdate):
                # The call decides the declared tool should run.
                started = GatewayAgentToolStatusEvent(
                    session_id="sess-test",
                    call_id="sess-test:tool:1",
                    name=control.tools[0].name,
                    status=ToolCallStatus.STARTED,
                    arguments={"order_id": "42"},
                )
                await websocket.send(
                    bytes([GatewayControlFrame.EVENT])
                    + json.dumps(started.to_dict()).encode("utf-8")
                )
            if isinstance(control, ToolResult):
                tool_result.set()

    async with serve(gateway, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        model = DuplexModel(base_url=f"ws://127.0.0.1:{port}")
        assert model.capabilities.mutable_tools
        assert model.capabilities.mutable_instructions
        assert model.capabilities.auto_tool_reply_generation

        session = model.session()
        calls: asyncio.Queue = asyncio.Queue()
        session.on("function_call", calls.put_nowait)

        # What the framework applies right after it creates the session.
        await session._update_session(
            instructions="You are the support line for Acme.", tools=[lookup_order]
        )
        session.update_rules(
            [
                Rule(
                    hears=("refund",),
                    thought=Thought(source="policy", text="Refunds within 30 days only."),
                )
            ]
        )
        session.think("The caller is a premium customer.")
        session._generate_reply(instructions="Greet the caller by name.")
        for _ in range(10):
            session.push_audio(_room_frame())
            await asyncio.sleep(0.005)

        call = await asyncio.wait_for(calls.get(), timeout=10.0)
        assert isinstance(call, llm.FunctionCall)
        assert call.name == "lookup_order"
        assert json.loads(call.arguments) == {"order_id": "42"}

        # LiveKit ran the tool and hands its output back.
        await session._append_items(
            [
                llm.FunctionCallOutput(
                    call_id=call.call_id, name=call.name, output="ships Friday", is_error=False
                )
            ]
        )
        await asyncio.wait_for(tool_result.wait(), timeout=10.0)
        await session.aclose()
        await model.aclose()

    instructions, tools, rules, thought, say, result = received
    assert isinstance(instructions, ThoughtWrite) and instructions.intent is Intent.KNOW
    assert instructions.text == "You are the support line for Acme."
    assert isinstance(tools, ToolsUpdate)
    assert tools.tools[0].name == "lookup_order"
    assert "order_id" in tools.tools[0].parameters["properties"]
    assert isinstance(rules, RulesUpdate) and rules.rules[0].hears == ("refund",)
    assert isinstance(thought, ThoughtWrite) and thought.intent is Intent.KNOW
    assert isinstance(say, ThoughtWrite) and say.intent is Intent.SAY
    assert say.text == "Greet the caller by name."
    assert isinstance(result, ToolResult)
    assert result.call_id == "sess-test:tool:1" and result.output == "ships Friday"
