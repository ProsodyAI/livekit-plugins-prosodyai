"""ProsodyAI gateway socket: connection, URL, protocol constants."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from websockets.asyncio.client import ClientConnection, connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from .frames import GatewayAudio, GatewayControlFrame

if TYPE_CHECKING:
    import sphn

SAMPLE_RATE = 24_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
REALTIME_PATH = "/v1/realtime"
API_KEY_HEADER = "x-api-key"
DEFAULT_ORIGIN = "https://api.prosodyai.app"

#: What a gateway restart looks like from a consumer: the live socket dropped
#: (1012 on a deploy or reload), or a redial failed while the service came
#: back. A caller holding a healthy room may re-dial on these; anything else
#: is a real defect and must keep failing loudly.
RECONNECTABLE_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionClosed,
    InvalidHandshake,
    OSError,
)

__all__ = [
    "API_KEY_HEADER",
    "DEFAULT_ORIGIN",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "REALTIME_PATH",
    "RECONNECTABLE_ERRORS",
    "SAMPLE_RATE",
    "GatewayConnection",
    "GatewayEnvError",
    "GatewaySession",
    "gateway_ws_url",
]


class GatewayEnvError(RuntimeError):
    """The gateway connection settings are missing or contradictory."""


def gateway_ws_url(*, base_url: str = DEFAULT_ORIGIN) -> str:
    origin = base_url.strip().rstrip("/")
    if origin.startswith("https://"):
        origin = "wss://" + origin[8:]
    elif origin.startswith("http://"):
        origin = "ws://" + origin[7:]
    if not origin.startswith(("ws://", "wss://")):
        raise GatewayEnvError(f"not an origin: {base_url!r}")
    return origin + REALTIME_PATH


@dataclass(frozen=True)
class GatewayConnection:
    """Resolved gateway endpoint. The key is kept out of ``repr``."""

    url: str
    api_key: str = field(repr=False)

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "GatewayConnection":
        environ = os.environ if env is None else env
        key = (api_key or environ.get("PROSODYAI_API_KEY") or "").strip()
        if not key:
            raise GatewayEnvError(
                "No gateway API key was found: PROSODYAI_API_KEY is unset and no "
                "api_key argument was passed; set the environment variable or pass api_key"
            )
        return cls(url=gateway_ws_url(base_url=base_url or DEFAULT_ORIGIN), api_key=key)

    @property
    def headers(self) -> dict[str, str]:
        return {API_KEY_HEADER: self.api_key}


class GatewaySession:
    """One ProsodyAI gateway socket."""

    def __init__(self, *, url: str, api_key: str) -> None:
        self._url = url
        self._api_key = api_key
        self._connect_ctx = None
        self._socket: ClientConnection | None = None
        self._writer: sphn.OpusStreamWriter | None = None
        self._reader: sphn.OpusStreamReader | None = None
        self._pending_uplink = np.zeros(0, dtype=np.float32)

    async def open(self) -> None:
        try:
            import sphn
        except ImportError as exc:
            raise RuntimeError(
                "sphn is required for Opus bridging "
                "(pip install 'livekit-plugins-prosodyai[duplex]')"
            ) from exc
        connect_ctx = ws_connect(
            self._url,
            additional_headers={API_KEY_HEADER: self._api_key},
            max_size=16 * 1024 * 1024,
            open_timeout=120.0,
            ping_interval=None,
        )
        socket = await connect_ctx.__aenter__()
        self._socket = socket
        self._connect_ctx = connect_ctx
        self._writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        self._reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self._pending_uplink = np.zeros(0, dtype=np.float32)

    async def send_audio(self, samples: np.ndarray) -> None:
        socket, writer = self._socket, self._writer
        if socket is None or writer is None:
            return
        self._pending_uplink = np.concatenate([self._pending_uplink, samples])
        while self._pending_uplink.shape[0] >= FRAME_SAMPLES:
            block = self._pending_uplink[:FRAME_SAMPLES]
            self._pending_uplink = self._pending_uplink[FRAME_SAMPLES:]
            packet = writer.append_pcm(block)
            if packet:
                await socket.send(bytes([GatewayAudio.KIND]) + packet)

    async def receive(self) -> AsyncIterator[GatewayAudio | GatewayControlFrame]:
        socket, reader = self._socket, self._reader
        if socket is None or reader is None:
            return
        async for message in socket:
            if isinstance(message, str) or not message:
                continue
            kind = message[0]
            payload = message[1:]
            if kind == GatewayAudio.KIND:
                pcm = reader.append_bytes(payload)
                if pcm is None or pcm.size == 0:
                    continue
                yield GatewayAudio(samples=np.asarray(pcm, dtype=np.float32).reshape(-1))
                continue
            yield GatewayControlFrame(
                kind=kind,
                payload=b"" if kind == GatewayControlFrame.HANDSHAKE else payload,
            )

    async def close(self) -> None:
        connect_ctx, self._connect_ctx = self._connect_ctx, None
        self._socket = None
        self._writer = None
        self._reader = None
        self._pending_uplink = np.zeros(0, dtype=np.float32)
        if connect_ctx is not None:
            await connect_ctx.__aexit__(None, None, None)
