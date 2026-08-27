"""Binary frames on the gateway socket."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GatewayAudio",
    "GatewayControlFrame",
]


@dataclass(frozen=True)
class GatewayControlFrame:
    """One non-speech frame: handshake, text, identity, transcript, or event."""

    HANDSHAKE = 0x00
    TEXT = 0x02
    CONTROL = 0x03
    IDENTITY = 0x04
    TRANSCRIPT = 0x05
    EVENT = 0x06

    kind: int
    payload: bytes


@dataclass(frozen=True)
class GatewayAudio:
    """Spoken PCM from the gateway: mono float32 at 24 kHz."""

    KIND = 0x01

    samples: np.ndarray
