from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .tracks import Intent, StrEnum

logger = logging.getLogger(__name__)


class ClientControlType(StrEnum):
    """What a caller's agent sends on the control kind."""

    THOUGHT = "thought"
    TOOLS_UPDATE = "tools.update"
    TOOL_RESULT = "tool.result"


class ToolDefinition(BaseModel):
    """One function the agent can run, as it declares it."""

    name: str
    description: str = ""
    #: The arguments' JSON Schema, passed through to the deciding model.
    parameters: dict[str, Any] = Field(default_factory=dict)


class ThoughtWrite(BaseModel):
    """The agent tells the speech model something, now."""

    type: Literal[ClientControlType.THOUGHT]
    #: The slot drops a thought whose id it already saw this session.
    id: Optional[str] = None
    text: str
    intent: Intent = Intent.KNOW


class ToolsUpdate(BaseModel):
    """The tools the agent can run; replaces any earlier set."""

    type: Literal[ClientControlType.TOOLS_UPDATE]
    tools: list[ToolDefinition]


class ToolResult(BaseModel):
    """What a tool the gateway asked for returned."""

    type: Literal[ClientControlType.TOOL_RESULT]
    call_id: str
    output: str = ""
    error: Optional[str] = None


ClientControl = Union[ThoughtWrite, ToolsUpdate, ToolResult]
_CLIENT_CONTROL: TypeAdapter[ClientControl] = TypeAdapter(
    Annotated[ClientControl, Field(discriminator="type")]
)


def parse_client_control(body: bytes) -> Optional[ClientControl]:
    """The control message a frame body holds, or None for one that is not."""
    try:
        return _CLIENT_CONTROL.validate_json(body)
    except ValidationError as exc:
        logger.warning("unreadable control frame: %s", exc.errors()[0]["msg"])
        return None
