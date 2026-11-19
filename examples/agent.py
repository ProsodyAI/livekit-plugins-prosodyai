"""ProsodyAI as a LiveKit duplex model.

pip install livekit-plugins-prosodyai
export PROSODYAI_API_KEY=psk_...
export LIVEKIT_URL=...
export LIVEKIT_API_KEY=...
export LIVEKIT_API_SECRET=...
python examples/agent.py dev

``AgentSession`` accepts this ``DuplexModel`` the same way it accepts any other.
"""

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import prosodyai

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    model = prosodyai.DuplexModel()

    @model.on("prosodyai.identity")
    def on_identity(event) -> None:
        print(event.speaker_id, event.person_id, event.display_name)

    session = AgentSession(llm=model)
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice assistant."),
    )


if __name__ == "__main__":
    cli.run_app(server)
