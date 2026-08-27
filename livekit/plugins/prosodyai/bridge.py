"""Room PCM up, gateway speech down, typed events beside it."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from .audio_resample import (
    float32_to_pcm16_le,
    pcm16_le_to_float32,
    resample_float32,
)
from .events import GatewayEvent, ReadyEvent, parse_control_event
from .frames import GatewayAudio
from .gateway import SAMPLE_RATE, GatewayConnection, GatewaySession

__all__ = ["FullDuplexBridge"]


async def _session_lifetime(
    *, send_task: asyncio.Task[None], receive_task: asyncio.Task[None]
) -> None:
    pending = {send_task, receive_task}
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
            if task is receive_task:
                return


class FullDuplexBridge:
    """Room PCM in, gateway speech out, typed events beside it."""

    def __init__(
        self,
        *,
        url: str = "",
        api_key: str = "",
        room_sample_rate: int = 16_000,
        publish_sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if not url or not api_key:
            resolved = GatewayConnection.from_environment()
            url, api_key = resolved.url, resolved.api_key
        self._room_sample_rate = room_sample_rate
        self._publish_sample_rate = publish_sample_rate
        self._session = GatewaySession(url=url, api_key=api_key)
        self._ready = asyncio.Event()
        self._closed = False

    @property
    def ready(self) -> asyncio.Event:
        return self._ready

    async def _send_uplink(self, uplink_pcm16: AsyncIterator[bytes]) -> None:
        await self._ready.wait()
        async for frame in uplink_pcm16:
            if self._closed or not frame:
                continue
            await self._session.send_audio(
                resample_float32(
                    pcm16_le_to_float32(frame),
                    self._room_sample_rate,
                    SAMPLE_RATE,
                )
            )

    async def _receive_downlink(
        self,
        *,
        on_downlink_pcm16: Callable[[bytes], Awaitable[None]],
        on_event: Callable[[GatewayEvent], Awaitable[None]] | None,
    ) -> None:
        async for item in self._session.receive():
            if self._closed:
                break
            if isinstance(item, GatewayAudio):
                flat = item.samples
                if flat.size == 0:
                    continue
                if self._publish_sample_rate != SAMPLE_RATE:
                    flat = resample_float32(flat, SAMPLE_RATE, self._publish_sample_rate)
                await on_downlink_pcm16(float32_to_pcm16_le(flat))
                continue
            event = parse_control_event(item)
            if event is None:
                continue
            if isinstance(event, ReadyEvent):
                self._ready.set()
            if on_event is not None:
                await on_event(event)

    async def run(
        self,
        uplink_pcm16: AsyncIterator[bytes],
        *,
        on_downlink_pcm16: Callable[[bytes], Awaitable[None]],
        on_event: Callable[[GatewayEvent], Awaitable[None]] | None = None,
    ) -> None:
        """``uplink_pcm16`` is little-endian mono PCM16 at ``room_sample_rate``.

        Downlink PCM is the same layout at ``publish_sample_rate``.
        """
        await self._session.open()
        send_task = asyncio.create_task(self._send_uplink(uplink_pcm16), name="duplex-uplink")
        receive_task = asyncio.create_task(
            self._receive_downlink(on_downlink_pcm16=on_downlink_pcm16, on_event=on_event),
            name="duplex-downlink",
        )
        try:
            await _session_lifetime(send_task=send_task, receive_task=receive_task)
        finally:
            self._closed = True
            send_task.cancel()
            receive_task.cancel()
            await asyncio.gather(send_task, receive_task, return_exceptions=True)
            await self._session.close()

    def close(self) -> None:
        self._closed = True
