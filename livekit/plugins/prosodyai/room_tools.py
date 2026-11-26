from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from livekit import rtc

from .events import AgentToolStatusEvent, GatewayEvent
from .thinking import ClientControlType, ToolDefinition, ToolResult, ToolsUpdate
from .wire import ToolCallStatus

logger = logging.getLogger(__name__)

ROOM_TOOLS_METHOD = "prosodyai.tools"
ROOM_TOOL_METHOD_PREFIX = "prosodyai.tool."
MANIFEST_TIMEOUT_S = 5.0
TOOL_CALL_TIMEOUT_S = 15.0

SendControl = Callable[[BaseModel], Awaitable[None]]


def room_tool_method(name: str) -> str:
    return f"{ROOM_TOOL_METHOD_PREFIX}{name}"


class RoomToolManifest(BaseModel):
    tools: list[ToolDefinition] = Field(default_factory=list)


class RoomToolReply(BaseModel):
    output: Optional[str] = None
    error: Optional[str] = None


class RoomTools:
    """The tools the room's participants offer the agent, and the loop that runs them.

    Each participant is asked ``prosodyai.tools`` and its manifest is forwarded to
    the gateway as the call's tool set. When the gateway's thinking track starts a
    call, the tool runs on the participant that declared it over
    ``prosodyai.tool.<name>``, and the reply goes back as the tool's result.
    """

    def __init__(
        self,
        room: rtc.Room,
        send_control: SendControl,
        *,
        manifest_timeout_s: float = MANIFEST_TIMEOUT_S,
        call_timeout_s: float = TOOL_CALL_TIMEOUT_S,
    ) -> None:
        self._room = room
        self._send = send_control
        self._manifest_timeout_s = manifest_timeout_s
        self._call_timeout_s = call_timeout_s
        self._owners: dict[str, str] = {}
        self._definitions: dict[str, ToolDefinition] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def tools(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def owner(self, name: str) -> Optional[str]:
        return self._owners.get(name)

    async def collect(self) -> None:
        for identity in list(self._room.remote_participants):
            await self.declare(identity)

    async def declare(self, identity: str) -> None:
        participant = self._room.remote_participants.get(identity)
        if (
            participant is not None
            and getattr(participant, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        ):
            return
        try:
            raw = await self._room.local_participant.perform_rpc(
                destination_identity=identity,
                method=ROOM_TOOLS_METHOD,
                payload="{}",
                response_timeout=self._manifest_timeout_s,
            )
        except rtc.RpcError as exc:
            if exc.code != rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD:
                logger.warning("tool manifest from %s failed: %s", identity, exc.message)
            return
        try:
            manifest = RoomToolManifest.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("tool manifest from %s unreadable: %s", identity, exc.errors()[0]["msg"])
            return
        self._drop(identity)
        for tool in manifest.tools:
            self._owners[tool.name] = identity
            self._definitions[tool.name] = tool
        logger.info("tools declared by %s: %s", identity, [tool.name for tool in manifest.tools])
        await self._announce()

    async def forget(self, identity: str) -> None:
        if self._drop(identity):
            await self._announce()

    def _drop(self, identity: str) -> bool:
        gone = [name for name, owner in self._owners.items() if owner == identity]
        for name in gone:
            del self._owners[name]
            del self._definitions[name]
        return bool(gone)

    async def _announce(self) -> None:
        await self._send(ToolsUpdate(type=ClientControlType.TOOLS_UPDATE, tools=self.tools))

    def observe(self, event: GatewayEvent) -> None:
        if not isinstance(event, AgentToolStatusEvent) or event.status != ToolCallStatus.STARTED:
            return
        task = asyncio.create_task(
            self.run(event.call_id, event.name, dict(event.arguments)),
            name=f"room-tool-{event.call_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        owner = self._owners.get(name)
        if owner is None:
            await self._result(call_id, error=f"no participant on the call offers {name}")
            return
        try:
            raw = await self._room.local_participant.perform_rpc(
                destination_identity=owner,
                method=room_tool_method(name),
                payload=json.dumps(arguments, separators=(",", ":")),
                response_timeout=self._call_timeout_s,
            )
        except rtc.RpcError as exc:
            await self._result(call_id, error=exc.message or f"{name} failed ({exc.code})")
            return
        try:
            reply = RoomToolReply.model_validate_json(raw)
        except ValidationError:
            reply = RoomToolReply(error=f"{name} answered with something that is not a tool reply")
        if reply.error is not None:
            await self._result(call_id, error=reply.error)
        else:
            await self._result(call_id, output=reply.output or "")

    async def _result(self, call_id: str, *, output: str = "", error: Optional[str] = None) -> None:
        await self._send(
            ToolResult(
                type=ClientControlType.TOOL_RESULT, call_id=call_id, output=output, error=error
            )
        )

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
