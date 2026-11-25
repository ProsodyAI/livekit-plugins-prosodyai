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

### Tools a participant offers

A page or device on the call can offer tools of its own over LiveKit RPC, and the agent runs them where they live. The participant answers `prosodyai.tools` with a manifest (`{"tools": [{"name", "description", "parameters"}]}`) and serves each tool on `prosodyai.tool.<name>`, taking the arguments as a JSON payload and answering with `{"output": "..."}` or `{"error": "..."}`. `RoomTools` collects the manifests, forwards them to the call as its tool set, and when the call starts a tool, performs it on the participant that offered it and returns the reply. The TypeScript SDK's `registerRoomTools` is the participant side of the same contract.

```python
from livekit.plugins.prosodyai import RoomTools

tools = RoomTools(ctx.room, bridge.send_control)
await tools.collect()          # once the gateway session is open
tools.observe(event)           # for every gateway event the bridge yields
```

## Steering the model

A full-duplex model does not wait for a turn, so you do not prompt it per turn. You tell it things while it talks: something it should know, or something it should say now.

```python
duplex = session.current_agent.duplex_session

duplex.think("The caller is a premium customer.")
duplex.think("Tell the caller their refund was approved.", say=True)
```

`session.generate_reply(instructions=...)` also works, and is the same as `think(..., say=True)`.

## Policies

Standing behaviour is a policy on the account, and a policy is objects: rules over what the caller meant, never over the words they used. Its situations come from the account's directives. The classifier trained on them and a semantic parse of the caller's words recognise each one by its meaning, so "please cancel my account" and "can you cancel my account" are one situation, and "don't cancel my account" is not, though it holds the same words. Every action a rule proposes is judged against the call's state before it reaches the speech model.

```python
from livekit.plugins.prosodyai.dialogue import (
    Action,
    Condition,
    Delivery,
    DialoguePolicy,
    Entails,
    Established,
    Inject,
    Perform,
    Realization,
    Requirement,
    Rule,
    Scope,
)
from livekit.plugins.prosodyai.directives import ListDirectivesResponse


def support_policy(listing: ListDirectivesResponse) -> DialoguePolicy:
    meanings = {s.label: s.meaning for d in listing.directives for s in d.situations()}
    cancel = meanings["cancel_account"]
    asked_to_cancel = Condition(entails=Entails(situation=cancel, scope=Scope.MOVE))
    return DialoguePolicy(
        id="support",
        rules=(
            Rule(
                id="acct:cancel_terms",
                when=(asked_to_cancel,),
                then=(
                    Action(
                        inject=Inject(
                            delivery=Delivery.KNOW,
                            realization=Realization(
                                text="Cancellation takes effect at the end of the billing period."
                            ),
                        )
                    ),
                ),
                once=True,
            ),
            Rule(
                id="acct:cancel",
                when=(asked_to_cancel,),
                then=(
                    Action(
                        perform=Perform(
                            tool="cancel_account",
                            requirements=(
                                Requirement(established=Established(proposition=cancel)),
                            ),
                        )
                    ),
                ),
            ),
        ),
    )
```

`listing` is the body of `GET /v1/accounts/{account}/directives`, read with `ListDirectivesResponse.model_validate_json`. The first rule tells the model a fact the first time the caller asks to cancel. The second runs the agent's `cancel_account` tool, and only once the caller's request is established.

Put the policy with `PUT /v1/accounts/{account}/policies/support`, the body `support_policy(listing).json_bytes()`, and name `support` as the agent configuration's `dialogue_policy`. The gateway checks every term against the account's lexicon before it accepts a policy, and each call keeps the version it started with.

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
