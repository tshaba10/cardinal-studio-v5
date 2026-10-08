"""
Addressable (keyed) random numbers - guide Step 6.

Every random number is a pure function of *what it is for*:

    u = F(seed, stream, entity, step, sub)

* ``stream``  - the node (or node + purpose) that consumes the draw,
                e.g. ``"account_cycle.payment_amount#0"``.
* ``entity``  - ordinal of the root entity the row belongs to (e.g. account #17).
* ``step``    - absolute time step (e.g. months since 1970-01), 0 for static rows.
* ``sub``     - sub-index, e.g. the sequence number of a transaction within a cycle.

The guide illustrates this with one ``numpy.random.Generator(Philox(key))`` per
draw. That is correct but costs ~10 microseconds per generator, which is far too
slow for tens of millions of draws. Here we use the same *idea* with a
vectorised counter-based hash: the key tuple is mixed through the SplitMix64
finaliser (a well-tested 64-bit bijective mixer) in numpy, so a million keyed
draws cost a few milliseconds and the guarantees are identical:

1. Same contract + seed -> identical numbers on any machine.
2. Execution order / chunking / number of workers is irrelevant.
3. Adding a node never changes another node's draws (new stream = new keys).

The per-stream base key is derived with BLAKE2b (never Python's ``hash()``,
which is randomised per process).
"""
from __future__ import annotations

from hashlib import blake2b

import numpy as np
from scipy.special import ndtri

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)
_TWO_POW_M53 = 2.0 ** -53


def mix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser on a uint64 array (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
        return z ^ (z >> _S31)


def _as_u64(a) -> np.ndarray:
    """Cast any integer-like (including negative ints) to uint64 bit patterns."""
    arr = np.asarray(a)
    if arr.dtype == np.uint64:
        return arr
    return arr.astype(np.int64).astype(np.uint64)


def stream_key(seed: int, stream: str) -> int:
    """Stable 64-bit key for (seed, stream)."""
    digest = blake2b(f"{seed}|{stream}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class KeyedRNG:
    """Vectorised addressable random number source."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._keys: dict[str, np.uint64] = {}

    def _key(self, stream: str) -> np.uint64:
        k = self._keys.get(stream)
        if k is None:
            k = np.uint64(stream_key(self.seed, stream))
            self._keys[stream] = k
        return k

    # ------------------------------------------------------------------ #
    def bits(self, stream: str, entity, step=0, sub=0) -> np.ndarray:
        """Raw 64-bit hash for every (entity, step, sub) address."""
        e, s, b = np.broadcast_arrays(_as_u64(entity), _as_u64(step), _as_u64(sub))
        h = mix64(self._key(stream) ^ e)
        h = mix64(h ^ s)
        h = mix64(h ^ b)
        return np.atleast_1d(h)

    def uniform(self, stream: str, entity, step=0, sub=0) -> np.ndarray:
        """Uniform draws strictly inside (0, 1) - safe for inverse CDFs."""
        h = self.bits(stream, entity, step, sub)
        return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * _TWO_POW_M53

    def normal(self, stream: str, entity, step=0, sub=0) -> np.ndarray:
        """Standard normal draws via the inverse CDF of keyed uniforms."""
        return ndtri(self.uniform(stream, entity, step, sub))
