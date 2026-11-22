"""The thinking track on a call: what the speech model is told while it converses.

The speech model listens and speaks. Everything else reaches it as a
thought (``tracks.py``): text written into its context that it then
speaks from. On a call the thoughts come from the caller's agent, which
knows the customer: it writes a thought outright, declares rules ("when the
caller says this, tell the model that"), or declares tools and returns what
they produced. That is how a model is fixed for one customer in minutes.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .tracks import Intent, Rule

logger = logging.getLogger(__name__)


class ClientControlType(StrEnum):
    """What a caller's agent sends on the control kind."""

    THOUGHT = "thought"
    RULES_UPDATE = "rules.update"
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
    text: str
    intent: Intent = Intent.KNOW


class RulesUpdate(BaseModel):
    """The agent's standing rules for this call; replaces any earlier set."""

    type: Literal[ClientControlType.RULES_UPDATE]
    rules: list[Rule]


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


ClientControl = Union[ThoughtWrite, RulesUpdate, ToolsUpdate, ToolResult]
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
