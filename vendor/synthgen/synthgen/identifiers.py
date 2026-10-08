"""
Identifier and pattern-formatted string generation.

A ``format.pattern`` in the contract is a tiny regex subset:

    ACC-[A-Z0-9]{10}      literal text, character classes, fixed repetition
    [456][0-9]{3}         explicit sets and ranges
    MER-\\d{6}            \\d = [0-9]

Two generation modes are provided:

* ``unique``  - a *bijection* from a unique integer ordinal to a pattern string.
                The ordinal is first scrambled with a keyed Feistel permutation
                (so IDs look random rather than sequential) and then written in
                the mixed radix defined by the pattern. Because both steps are
                bijective, uniqueness is guaranteed by construction - no
                collision checks or retries, regardless of volume.
* ``random``  - an arbitrary keyed draw (used for non-unique strings).

Both are fully deterministic and vectorised.
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass

import numpy as np

from .rng import KeyedRNG, mix64, stream_key

_CLASS_ESCAPES = {
    "d": string.digits,
    "w": string.ascii_letters + string.digits + "_",
}


def _expand_set(body: str) -> str:
    """Expand the inside of ``[...]`` (supports ranges like A-Z)."""
    out: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            esc = body[i + 1]
            out.extend(_CLASS_ESCAPES.get(esc, esc))
            i += 2
            continue
        if i + 2 < len(body) and body[i + 1] == "-":
            lo, hi = ord(c), ord(body[i + 2])
            if hi < lo:
                raise ValueError(f"bad range {c}-{body[i + 2]} in pattern")
            out.extend(chr(x) for x in range(lo, hi + 1))
            i += 3
            continue
        out.append(c)
        i += 1
    # keep order but drop duplicates
    seen: dict[str, None] = {}
    for ch in out:
        seen.setdefault(ch, None)
    return "".join(seen)


def parse_pattern(pattern: str) -> list[bytes]:
    """Return one alphabet (bytes) per output character position."""
    positions: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "[":
            j = pattern.find("]", i + 1)
            if j < 0:
                raise ValueError(f"unclosed '[' in pattern {pattern!r}")
            alphabet = _expand_set(pattern[i + 1:j])
            i = j + 1
        elif c == "\\" and i + 1 < len(pattern):
            alphabet = _CLASS_ESCAPES.get(pattern[i + 1], pattern[i + 1])
            i += 2
        elif c in "(){}*+?|^$.":
            raise ValueError(f"unsupported regex syntax {c!r} in pattern {pattern!r}")
        else:
            alphabet = c
            i += 1
        # optional {n}
        reps = 1
        if i < len(pattern) and pattern[i] == "{":
            j = pattern.find("}", i)
            if j < 0:
                raise ValueError(f"unclosed '{{' in pattern {pattern!r}")
            reps = int(pattern[i + 1:j])
            i = j + 1
        if not alphabet:
            raise ValueError(f"empty character class in pattern {pattern!r}")
        positions.extend([alphabet] * reps)
    if not positions:
        raise ValueError("empty pattern")
    for a in positions:
        if any(ord(ch) > 127 for ch in a):
            raise ValueError("only ASCII patterns are supported")
    return [a.encode("ascii") for a in positions]


@dataclass
class PatternCodec:
    """Encodes integers into strings matching a pattern."""

    pattern: str
    seed: int
    stream: str

    def __post_init__(self):
        self.alphabets = parse_pattern(self.pattern)
        self.bases = np.array([len(a) for a in self.alphabets], dtype=np.int64)
        # capacity may exceed 2**63; cap the integer domain we use at 2**62
        cap = 1
        for b in self.bases:
            cap *= int(b)
        self.capacity = cap
        self.domain = min(cap, 2 ** 62)
        # Feistel network works on an even number of bits covering the domain
        nbits = max(2, math.ceil(math.log2(self.domain))) if self.domain > 1 else 2
        self.half_bits = (nbits + 1) // 2
        self._half_mask = np.uint64((1 << self.half_bits) - 1)
        self._key = np.uint64(stream_key(self.seed, "pattern|" + self.stream))
        # lookup tables: one 256-entry row per position (only first `base` used)
        # Positions not needed to hold the domain (leading ones, when the
        # pattern's capacity exceeds 2**62) are filled with keyed noise so IDs
        # do not all start with the same characters. Uniqueness still comes
        # from the trailing positions, which encode the ordinal bijectively.
        prod, self._free_positions = 1, 0
        for p in range(len(self.bases) - 1, -1, -1):
            if prod >= self.domain:
                self._free_positions = p + 1
                break
            prod *= int(self.bases[p])
        self._lut = np.zeros((len(self.alphabets), 256), dtype=np.uint8)
        for p, a in enumerate(self.alphabets):
            self._lut[p, :len(a)] = np.frombuffer(a, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    def _feistel(self, x: np.ndarray) -> np.ndarray:
        """Keyed 4-round balanced Feistel permutation on 2*half_bits bits."""
        hb = np.uint64(self.half_bits)
        left = (x >> hb) & self._half_mask
        right = x & self._half_mask
        for r in range(4):
            f = mix64(self._key ^ right ^ np.uint64(0x1000193 * (r + 1))) & self._half_mask
            left, right = right, left ^ f
        return (left << hb) | right

    def permute(self, ordinals: np.ndarray) -> np.ndarray:
        """Bijective scramble of [0, domain) via cycle-walking."""
        x = np.asarray(ordinals, dtype=np.int64)
        if x.size and (x.min() < 0 or x.max() >= self.domain):
            raise ValueError(
                f"pattern {self.pattern!r} cannot hold ordinal {int(x.max())} "
                f"(capacity {self.domain}); widen the pattern")
        y = self._feistel(x.astype(np.uint64))
        dom = np.uint64(self.domain)
        bad = y >= dom
        while bad.any():                      # expected < 2 extra rounds
            y[bad] = self._feistel(y[bad])
            bad = y >= dom
        return y.astype(np.int64)

    def encode(self, numbers: np.ndarray) -> np.ndarray:
        """Write integers (< domain) in the pattern's mixed radix -> 'S' array."""
        n = np.asarray(numbers, dtype=np.int64).copy()
        width = len(self.alphabets)
        out = np.empty((n.size, width), dtype=np.uint8)
        for p in range(width - 1, -1, -1):
            base = self.bases[p]
            if base == 1:
                out[:, p] = self._lut[p, 0]
                continue
            digit = n % base
            n //= base
            out[:, p] = self._lut[p, digit]
        return out.view(f"S{width}").reshape(-1)

    def unique(self, ordinals: np.ndarray) -> np.ndarray:
        """Guaranteed-unique strings for distinct ordinals."""
        perm = self.permute(ordinals)
        out = self.encode(perm)
        if self._free_positions:
            width = len(self.alphabets)
            mat = out.view(np.uint8).reshape(-1, width).copy()
            noise = mix64(perm.astype(np.uint64) ^ self._key)
            for p in range(self._free_positions):
                base = int(self.bases[p])
                if base > 1:
                    digit = ((noise >> np.uint64(5 * p)) % np.uint64(base)).astype(np.int64)
                    mat[:, p] = self._lut[p, digit]
            out = mat.view(f"S{width}").reshape(-1)
        return out

    def random(self, rng: KeyedRNG, entity, step=0, sub=0) -> np.ndarray:
        """Keyed (non-unique) pattern strings."""
        bits = rng.bits(self.stream + "#pattern", entity, step, sub)
        nums = (bits % np.uint64(self.domain)).astype(np.int64)
        return self.encode(nums)


def luhn_check_digit(digits: np.ndarray) -> np.ndarray:
    """Luhn check digit for an (n, k) uint8 matrix of payload digits."""
    d = digits.astype(np.int64)[:, ::-1]            # rightmost payload digit first
    doubled = d.copy()
    doubled[:, 0::2] *= 2                           # double every 2nd from the right
    doubled = np.where(doubled > 9, doubled - 9, doubled)
    total = doubled.sum(axis=1)
    return (10 - total % 10) % 10
