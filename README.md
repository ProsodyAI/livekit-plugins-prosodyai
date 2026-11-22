# ProsodyAI plugin for LiveKit Agents

A full-duplex voice for LiveKit Agents: it listens and speaks at the same time, knows who is speaking, and can be steered while it talks.

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) and
[https://prosodyai.app/docs](https://prosodyai.app/docs) for more information.

## Installation

```bash
pip install livekit-plugins-prosodyai
```

Requires `livekit-agents` 1.8, which is where `DuplexModel` lives.

## Pre-requisites

You'll need an API key from ProsodyAI. It can be set as an environment variable: `PROSODYAI_API_KEY`

## Usage

`prosodyai.DuplexModel` is a LiveKit `DuplexModel`. Give it to an `AgentSession` as you would any speech model.

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import prosodyai

session = AgentSession(llm=prosodyai.DuplexModel())

await session.start(
    room=ctx.room,
    agent=Agent(instructions="You are the support line for Acme."),
)
```

The agent's instructions reach the model when the session starts and again whenever you change them.

## Tools

Tools are ordinary LiveKit function tools on the agent. The call decides when one should run, LiveKit runs it, and the model speaks from the result. There is nothing to wire.

```python
from livekit.agents import Agent, function_tool

class Support(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are the support line for Acme.")

    @function_tool
    async def lookup_order(self, order_id: str) -> str:
        """Find an order by its number."""
        return await orders.status(order_id)
```

## Steering the model

A full-duplex model does not wait for a turn, so you do not prompt it per turn. You tell it things while it talks. The session has three ways.

Tell it something it should know, or something it should say now:

```python
duplex = session.current_agent.duplex_session

duplex.think("The caller is a premium customer.")
duplex.think("Tell the caller their refund was approved.", say=True)
```

Give the call standing rules: when the caller says this, the model is told that.

```python
from livekit.plugins.prosodyai import Rule, Thought

duplex.update_rules([
    Rule(
        hears=("refund", "money back"),
        thought=Thought(source="policy", text="Refunds are available within 30 days only."),
    ),
])
```

`session.generate_reply(instructions=...)` also works, and is the same as `think(..., say=True)`.

## Who is speaking

The model commits a caller to a person when it recognises their voice. `speaker_id` is a label within one call; `person_id` is the person across calls.

```python
model = prosodyai.DuplexModel()

@model.on(prosodyai.RoomEventType.IDENTITY)
def on_identity(event: prosodyai.IdentityEvent) -> None:
    print(event.person_id, event.display_name)
```

## Events

Subscribe on the model or on one session.

| Enum | |
| --- | --- |
| `RoomEventType.TRANSCRIPT` | Words, the caller's and the agent's |
| `RoomEventType.IDENTITY` | A caller committed to a person |
| `GatewayEventType.SPEAKER_CHANGE` | The floor passed to another speaker |
| `GatewayEventType.AGENT_TOOL_STATUS` | A tool call started, completed or failed |
| `ConversationEventType.TURN_BOUNDARY` | The floor passed between voices |
| `ConversationEventType.BARGE_IN` | A second voice entered against held speech |
| `ConversationEventType.STATE_DELTA` | A speaker's state moved against their baseline |
| `ConversationEventType.ENTITY_SPAN` | A dictated entity's extent |

## License

MIT © [ProsodyAI](https://prosodyai.app)
