# ProsodyAI plugin for LiveKit Agents

Full-duplex speech with persistent speaker identity.

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) and
[https://prosodyai.app/docs](https://prosodyai.app/docs) for more information.

## Installation

```bash
pip install livekit-plugins-prosodyai
```

## Pre-requisites

You'll need an API key from ProsodyAI. It can be set as an environment variable: `PROSODYAI_API_KEY`

## Usage

Use ProsodyAI within an `AgentSession`. A complete worker is in [`examples/agent.py`](examples/agent.py).

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import prosodyai

session = AgentSession(
    llm=prosodyai.realtime.RealtimeModel(),
)

await session.start(
    room=ctx.room,
    agent=Agent(instructions="You are a helpful voice assistant."),
)
```

`RealtimeModel` sends continuous room audio to ProsodyAI and returns generated
audio, streaming transcripts, identity updates, and conversation events. The
model handles barge-in; do not attach a turn detector.

## Speaker identity

Conversation-local diarization labels look like `speaker_0`. When the model
resolves an enrolled caller, it emits a durable `person_id` and display name.
Returning callers resume their saved speaker state.

Subscribe on the model before `session.start` so the listener is attached
when the first event arrives:

```python
model = prosodyai.realtime.RealtimeModel()


@model.on(prosodyai.RoomEventType.IDENTITY)
def on_identity(event):
    print(event.speaker_id, event.person_id, event.display_name)
```

## Events

Subscribe with the wire enums. Room and gateway events are `prosodyai.*`. Conversation events keep the names the model committed.

| Enum | |
| --- | --- |
| `RoomEventType.TRANSCRIPT` | Committed words with `speaker_id` and word-level timestamps |
| `RoomEventType.IDENTITY` | Returning person committed |
| `RoomEventType.TEXT` | Generated text stream |
| `GatewayEventType.SPEAKER_CHANGE` | The floor moved lanes |
| `GatewayEventType.NEW_SPEAKER` | A lane opened for a voice never heard here |
| `GatewayEventType.IDENTITY_RESOLVED` | A lane matched an enrolled profile |
| `ConversationEventType.TURN_BOUNDARY` | The floor passed between voices |
| `ConversationEventType.BARGE_IN` | A second voice entered against held speech |
| `ConversationEventType.STATE_DELTA` | The lane's state moved against its baseline |
| `ConversationEventType.ENTITY_SPAN` | A dictated entity's extent |

```python
@model.on(prosodyai.ConversationEventType.TURN_BOUNDARY)
def on_turn(event: prosodyai.TurnBoundaryEvent):
    print(event.frame_ms, event.commit_ms)


@model.on(prosodyai.ConversationEventType.BARGE_IN)
def on_barge_in(event: prosodyai.BargeInEvent):
    print(event.frame_ms, event.duration_ms, event.resolved)
```

User words also arrive as LiveKit `input_audio_transcription_completed` on
the realtime session, so a stock `AgentSession` transcribes the caller.

## Lower-level bridge

For workers that publish and subscribe to tracks directly:

```python
from livekit.plugins.prosodyai import FullDuplexBridge

bridge = FullDuplexBridge()

await bridge.run(
    uplink_pcm16(),
    on_downlink_pcm16=publish_pcm16,
    on_event=handle_gateway_event,
)
```

`uplink_pcm16()` yields little-endian mono PCM16. The bridge returns the same
format for publication.

## License

MIT © [ProsodyAI](https://prosodyai.app)
