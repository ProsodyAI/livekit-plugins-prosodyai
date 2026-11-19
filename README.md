# ProsodyAI plugin for LiveKit Agents

ProsodyAI listens on each caller track. Any duplex model can speak.

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) and
[https://prosodyai.app/docs](https://prosodyai.app/docs) for more information.

## Installation

```bash
pip install livekit-plugins-prosodyai
```

Requires `livekit-agents` 1.8, which is where `DuplexModel` lives.

## Pre-requisites

You'll need an API key from ProsodyAI. It can be set as an environment variable: `PROSODYAI_API_KEY`

The speaking model is a `provider/model` string on the job metadata, or `DUPLEX_MODEL` when the dispatch does not set one. [`examples/agent.py`](examples/agent.py) passes that string to `AgentSession`, which resolves a duplex model itself.

## Usage

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import prosodyai

session = AgentSession(llm=duplex_model)
listener = prosodyai.Listener()

await session.start(
    room=ctx.room,
    agent=Agent(instructions="You are a helpful voice assistant."),
)
listener.start(room=ctx.room, session=session)
```

`Listener` opens one live session per caller audio track and raises what ProsodyAI commits. When a person is committed, that fact is given to the speaking model through its own context channel: `append_thinking` when the duplex session has it, otherwise a system item on the chat context when the model accepts one.

## Speaker identity

`speaker_id` is the label on one live session. `person_id` is the person. When a person is committed, `Listener` calls `tell_agent` on the session you passed to `start`.

```python
@listener.on(prosodyai.StreamEventType.IDENTITY_RESOLVED)
def on_identity(event: prosodyai.IdentityResolvedEvent, origin: prosodyai.TrackOrigin):
    print(origin.participant_identity, event.person_id, event.display_name)
```

## Events

Subscribe on the listener. The socket events are `StreamEventType`. Conversation events keep the names the model committed.

| Enum | |
| --- | --- |
| `StreamEventType.TRANSCRIPT_UPDATE` | Committed words |
| `StreamEventType.IDENTITY_RESOLVED` | A lane committed to a person |
| `StreamEventType.SPEAKER_UPDATE` | Attribution for one interval |
| `ConversationEventType.TURN_BOUNDARY` | The floor passed between voices |
| `ConversationEventType.BARGE_IN` | A second voice entered against held speech |
| `ConversationEventType.STATE_DELTA` | The lane's state moved against its baseline |
| `ConversationEventType.ENTITY_SPAN` | A dictated entity's extent |

## License

MIT © [ProsodyAI](https://prosodyai.app)
