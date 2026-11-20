"""ProsodyAI duplex model, and the audio simulation fork.

Text simulations turn audio off. This model speaks and hears only on the
audio path, so a text run is refused here the same way the framework refuses
it. Audio runs are graded twice: the simulator judges what was said, and
``on_simulation_end`` fails the run when caller words have no speaker.

    lk agent simulate audio examples/agent.py \\
        --scenarios examples/scenarios.yaml --yes
"""

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.simulation import SimulationContext, SimulationMode
from livekit.plugins import prosodyai
from livekit.plugins.prosodyai.simulation import (
    SimulationTranscripts,
    attribution_failure,
    require_speaker_on_words,
)

transcripts = SimulationTranscripts()
server = AgentServer()


def _refuse_text_simulation(ctx: JobContext) -> SimulationContext | None:
    simulation = ctx.simulation_context()
    if (
        simulation is not None
        and simulation.simulation_mode != SimulationMode.SIMULATION_MODE_AUDIO
    ):
        raise RuntimeError(
            "DuplexModel speaks only through audio; run `lk agent simulate audio`"
        )
    return simulation


async def on_simulation_end(ctx: SimulationContext) -> None:
    userdata = ctx.userdata()
    if not require_speaker_on_words(userdata):
        return
    reason = attribution_failure(
        transcripts.events(ctx.simulation_job_id),
        require_speaker_on_words=True,
    )
    if reason:
        ctx.fail(reason=reason)


@server.rtc_session(on_simulation_end=on_simulation_end)
async def entrypoint(ctx: JobContext) -> None:
    simulation = _refuse_text_simulation(ctx)
    model = prosodyai.DuplexModel()
    if simulation is not None:

        def on_transcript(event: prosodyai.TranscriptEvent) -> None:
            transcripts.record(simulation.simulation_job_id, event.to_dict())

        model.on(prosodyai.RoomEventType.TRANSCRIPT, on_transcript)

    session = AgentSession(llm=model)
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice assistant."),
    )


if __name__ == "__main__":
    cli.run_app(server)
