"""PersonaPlex over the ProsodyAI gateway socket (``WS /v1/realtime``).

Caller PCM goes up as 24 kHz Opus in 80 ms frames on a kind-tagged binary
wire; speech comes back as protocol items and the ProsodySSM readout frames
as :class:`GatewayControlFrame` for the bridge to parse.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from websockets.asyncio.client import ClientConnection, connect as ws_connect

if TYPE_CHECKING:
    import sphn

from .speech_backend import (
    SessionOpened,
    SpeechAudio,
    SpeechBackendCapabilities,
    SpeechItem,
    SpeechSessionConfig,
    SpeechText,
)
from .wire import KIND_AUDIO, KIND_HANDSHAKE, KIND_TEXT

__all__ = [
    "API_KEY_HEADER",
    "DEFAULT_BASE_URL",
    "GATEWAY_FRAME_SAMPLES",
    "GATEWAY_PATH",
    "GATEWAY_SAMPLE_RATE",
    "PERSONAPLEX_CAPABILITIES",
    "GatewayConnection",
    "GatewayControlFrame",
    "GatewayEnvError",
    "PersonaPlexBackend",
    "gateway_ws_url",
]

# The gateway's audio wire format: Opus at 24 kHz in 80 ms frames.
GATEWAY_SAMPLE_RATE = 24_000
GATEWAY_FRAME_SAMPLES = 1_920

# The gateway origin, the socket it serves, and the header it authenticates on.
DEFAULT_BASE_URL = "https://api.prosodyai.app"
GATEWAY_PATH = "/v1/realtime"
API_KEY_HEADER = "x-api-key"

# What an origin may arrive as, and the transport scheme it connects on.
_WS_SCHEME = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}


class GatewayEnvError(RuntimeError):
    """The gateway connection settings are missing or contradictory."""


def gateway_ws_url(*, base_url: str = DEFAULT_BASE_URL) -> str:
    """The gateway socket URL for one origin; the key travels as ``x-api-key`` on the handshake, never in the URL."""
    parsed = urlsplit((base_url or "").strip().rstrip("/"))
    scheme = _WS_SCHEME.get(parsed.scheme.lower())
    if scheme is None or not parsed.netloc:
        raise GatewayEnvError(
            "gateway base_url must be an http(s) or ws(s) origin such as "
            f"{DEFAULT_BASE_URL}, got {base_url!r}"
        )
    if parsed.query or parsed.fragment:
        raise GatewayEnvError(
            f"gateway base_url is an origin and carries no query or fragment, got {base_url!r}"
        )
    return urlunsplit((scheme, parsed.netloc, f"{parsed.path}{GATEWAY_PATH}", "", ""))


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
        return cls(url=gateway_ws_url(base_url=base_url or DEFAULT_BASE_URL), api_key=key)


@dataclass(frozen=True)
class GatewayControlFrame:
    """One non-speech gateway frame, handed to the bridge verbatim:
    identity (0x04), transcripts (0x05), committed model events (0x06)."""

    kind: int
    payload: bytes


PERSONAPLEX_CAPABILITIES = SpeechBackendCapabilities(
    full_duplex=True,
    accepts_voice_prompt=False,
    accepts_role_prompt=False,
    sample_rate=GATEWAY_SAMPLE_RATE,
)
"""Continuous full-duplex at 24 kHz; the gateway primes voice and role
server-side, so neither prompt channel exists on this session."""


class PersonaPlexBackend:
    """PersonaPlex speaking loop on one gateway socket."""

    def __init__(self, *, url: str, api_key: str) -> None:
        self._url = url
        self._api_key = api_key
        self._connect_ctx = None
        self._socket: ClientConnection | None = None
        self._writer: sphn.OpusStreamWriter | None = None
        self._reader: sphn.OpusStreamReader | None = None
        # Uplink accumulator: caller float32 held until a full 80 ms frame lands.
        self._pending_uplink = np.zeros(0, dtype=np.float32)

    @property
    def capabilities(self) -> SpeechBackendCapabilities:
        return PERSONAPLEX_CAPABILITIES

    async def open(self, config: SpeechSessionConfig) -> None:
        """Connect the gateway socket; prompts have no channel here."""
        del config
        try:
            import sphn
        except ImportError as exc:
            raise RuntimeError(
                "sphn is required for Opus bridging "
                "(pip install 'livekit-plugins-prosodyai[duplex]')"
            ) from exc
        self._writer = sphn.OpusStreamWriter(GATEWAY_SAMPLE_RATE)
        self._reader = sphn.OpusStreamReader(GATEWAY_SAMPLE_RATE)
        connect_ctx = ws_connect(
            self._url,
            additional_headers={API_KEY_HEADER: self._api_key},
            max_size=16 * 1024 * 1024,
            open_timeout=120.0,
            # Client pings stay disabled: uplink audio is the liveness signal.
            ping_interval=None,
        )
        self._socket = await connect_ctx.__aenter__()
        self._connect_ctx = connect_ctx

    async def send_audio(self, samples: np.ndarray) -> None:
        """Pack caller float32 onto the gateway's 80 ms grid and stream it as Opus."""
        if self._socket is None or self._writer is None:
            return
        self._pending_uplink = np.concatenate([self._pending_uplink, samples])
        while self._pending_uplink.shape[0] >= GATEWAY_FRAME_SAMPLES:
            block = self._pending_uplink[:GATEWAY_FRAME_SAMPLES]
            self._pending_uplink = self._pending_uplink[GATEWAY_FRAME_SAMPLES:]
            packet = self._writer.append_pcm(block)
            if packet:
                await self._socket.send(bytes([KIND_AUDIO]) + packet)

    async def receive(self) -> AsyncIterator[SpeechItem | GatewayControlFrame]:
        """Yield the socket's downlink in arrival order; frames beyond
        handshake, audio, and text arrive as :class:`GatewayControlFrame`."""
        if self._socket is None or self._reader is None:
            return
        async for message in self._socket:
            if isinstance(message, str) or not message:
                continue
            kind = message[0]
            payload = message[1:]
            if kind == KIND_HANDSHAKE:
                yield SessionOpened()
                continue
            if kind == KIND_TEXT:
                yield SpeechText(text=payload.decode("utf-8", errors="replace"))
                continue
            if kind == KIND_AUDIO:
                pcm = self._reader.append_bytes(payload)
                if pcm is None or pcm.size == 0:
                    continue
                yield SpeechAudio(samples=np.asarray(pcm, dtype=np.float32).reshape(-1))
                continue
            yield GatewayControlFrame(kind=kind, payload=payload)

    async def close(self) -> None:
        connect_ctx, self._connect_ctx = self._connect_ctx, None
        self._socket = None
        if connect_ctx is not None:
            await connect_ctx.__aexit__(None, None, None)
