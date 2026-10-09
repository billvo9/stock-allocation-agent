"""
Deterministic seed derivation.

Every random choice in a run is derived from one recorded base seed and a
purpose path, by SHA-256 of their text, so a choice never depends on how
many other draws were made, on the frame's length, or on PYTHONHASHSEED.

    derive_seed(0, "random_noise", "fold", 3) -> int in [0, 2**32)
    unit_uniform(0, "block_permutation", 7, -2, "NVDA") -> float in [0, 1)
"""

from __future__ import annotations

import hashlib

SEED_RULE = "sha256('<base_seed>:<part>:<part>...') first 8 bytes, big-endian"


def _digest(base_seed: int, parts: tuple[object, ...]) -> bytes:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer.")
    text = ":".join(str(part) for part in (base_seed, *parts))
    return hashlib.sha256(text.encode("utf-8")).digest()


def derive_seed(base_seed: int, *parts: object) -> int:
    """A RandomState-compatible seed in [0, 2**32) for one purpose."""

    return int.from_bytes(_digest(base_seed, parts)[:8], "big") % (2**32)


def unit_uniform(base_seed: int, *parts: object) -> float:
    """A uniform number in [0, 1) for one purpose (53-bit resolution)."""

    return (int.from_bytes(_digest(base_seed, parts)[:8], "big") >> 11) / float(1 << 53)
