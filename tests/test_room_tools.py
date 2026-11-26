"""RoomTools skips SIP phone participants instead of timing out on RPC."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from livekit import rtc
from livekit.plugins.prosodyai.room_tools import RoomTools


class _LocalParticipant:
    def __init__(self) -> None:
        self.rpc_calls: list[tuple[str, str]] = []

    async def perform_rpc(self, *, destination_identity: str, method: str, **_kwargs: Any) -> str:
        self.rpc_calls.append((destination_identity, method))
        return '{"tools": []}'


@pytest.mark.asyncio
async def test_room_tools_skips_sip_participants() -> None:
    local = _LocalParticipant()
    room = SimpleNamespace(
        local_participant=local,
        remote_participants={
            "phone-1": SimpleNamespace(
                identity="phone-1",
                kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
            ),
            "browser-1": SimpleNamespace(
                identity="browser-1",
                kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
            ),
        },
    )
    sent: list[Any] = []

    async def _send(msg: Any) -> None:
        sent.append(msg)

    tools = RoomTools(room, _send)  # type: ignore[arg-type]
    await tools.collect()

    assert local.rpc_calls == [("browser-1", "prosodyai.tools")]
    assert len(sent) == 1
