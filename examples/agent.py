"""ProsodyAI on a LiveKit room.

pip install livekit-plugins-prosodyai
export PROSODYAI_API_KEY=psk_...
export LIVEKIT_URL=...
export LIVEKIT_API_KEY=...
export LIVEKIT_API_SECRET=...
python examples/agent.py dev
"""

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import prosodyai

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    model = prosodyai.realtime.RealtimeModel()

    @model.on("prosodyai.identity")
    def on_identity(event) -> None:
        print(event.speaker_id, event.person_id, event.display_name)

    @model.on("prosodyai.transcript")
    def on_transcript(event) -> None:
        print(" ".join(delta.text for delta in event.deltas if delta.text))

    @model.on("turn_boundary")
    def on_turn(event: prosodyai.TurnBoundaryEvent) -> None:
        print("turn", event.frame_ms, event.commit_ms)

    @model.on("barge_in")
    def on_barge_in(event: prosodyai.BargeInEvent) -> None:
        print("barge-in", event.frame_ms, event.duration_ms, event.resolved)

    session = AgentSession(llm=model)
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice assistant."),
    )


if __name__ == "__main__":
    cli.run_app(server)
