"""
Inverse-CDF samplers.

Every sampler turns *keyed uniforms* (see ``rng.py``) into draws, so a draw
is a pure function of its address. Parameters may be scalars or per-row arrays
(the engine resolves conditioning such as "mean depends on segment" into
per-row arrays before calling these functions).

Bounds are honoured *inside* the draw (guide Step 2: constraints belong inside
the draw): a normal with ``min``/``max`` is sampled as a truncated normal by
mapping the uniform into ``[F(min), F(max)]`` - no clipping (which would pile
mass on the bounds) and no rejection loops (which would break addressability).
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats

_TINY = 1e-300


def _arr(x, n=None):
    a = np.asarray(x, dtype=np.float64)
    if n is not None and a.ndim == 0:
        a = np.full(n, float(a))
    return a


# --------------------------------------------------------------------------- #
# Discrete
# --------------------------------------------------------------------------- #

def categorical(u: np.ndarray, weights) -> np.ndarray:
    """Index into ``weights`` (normalised internally) for each uniform."""
    w = np.asarray(weights, dtype=np.float64)
    cdf = np.cumsum(w / w.sum())
    cdf[-1] = 1.0                              # guard against rounding
    return np.searchsorted(cdf, u, side="right").astype(np.int64)


def bernoulli(u: np.ndarray, p) -> np.ndarray:
    return u < np.asarray(p, dtype=np.float64)


def uniform_int(u: np.ndarray, low, high) -> np.ndarray:
    """Integer uniform on the *inclusive* range [low, high]."""
    lo = np.asarray(low, dtype=np.int64)
    hi = np.asarray(high, dtype=np.int64)
    span = np.maximum(hi - lo + 1, 1)
    return lo + np.minimum(np.floor(u * span).astype(np.int64), span - 1)


def poisson(u: np.ndarray, lam, max_value=None) -> np.ndarray:
    lam = np.maximum(np.asarray(lam, dtype=np.float64), 0.0)
    k = stats.poisson.ppf(u, lam)
    k = np.where(np.isfinite(k), k, 0).astype(np.int64)
    if max_value is not None:
        k = np.minimum(k, int(max_value))
    return k


# --------------------------------------------------------------------------- #
# Continuous (with optional truncation)
# --------------------------------------------------------------------------- #

def _truncated_std_normal(u, a, b):
    """Map u into the standard-normal quantiles between a and b (may be +-inf)."""
    fa = special.ndtr(a)
    fb = special.ndtr(b)
    p = fa + u * (fb - fa)
    p = np.clip(p, _TINY, 1 - 1e-16)
    return special.ndtri(p)


def normal(u, mean, std, low=None, high=None) -> np.ndarray:
    mean = _arr(mean)
    std = np.maximum(_arr(std), 1e-12)
    if low is None and high is None:
        return mean + std * special.ndtri(u)
    a = (_arr(-np.inf if low is None else low) - mean) / std
    b = (_arr(np.inf if high is None else high) - mean) / std
    return mean + std * _truncated_std_normal(u, a, b)


def lognormal(u, mu, sigma, low=None, high=None) -> np.ndarray:
    """``exp(N(mu, sigma))``; bounds are in value space (low must be > 0)."""
    lo = None if low is None or float(np.min(low)) <= 0 else np.log(_arr(low))
    hi = None if high is None else np.log(_arr(high))
    return np.exp(normal(u, mu, sigma, lo, hi))


def uniform(u, low, high) -> np.ndarray:
    lo, hi = _arr(low), _arr(high)
    return lo + u * (hi - lo)


def beta(u, a, b, low=0.0, high=1.0) -> np.ndarray:
    x = special.betaincinv(_arr(a), _arr(b), u)
    return _arr(low) + x * (_arr(high) - _arr(low))


def exponential(u, scale) -> np.ndarray:
    return -_arr(scale) * np.log1p(-u)


# --------------------------------------------------------------------------- #
# Link functions for linear predictors
# --------------------------------------------------------------------------- #

def apply_link(eta: np.ndarray, link: str) -> np.ndarray:
    if link == "identity":
        return eta
    if link == "logit":
        return special.expit(eta)
    if link == "log":
        return np.exp(eta)
    if link == "probit":
        return special.ndtr(eta)
    raise ValueError(f"unknown link {link!r}")


TERM_TRANSFORMS = {
    "identity": lambda x: x,
    "gt0": lambda x: (x > 0).astype(np.float64),
    "log": lambda x: np.log(np.maximum(x, 1e-12)),
    "log1p": lambda x: np.log1p(np.maximum(x, 0)),
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0)),
    "square": lambda x: x * x,
}

#: names of distributions the engine knows how to draw (used by the validator)
CONTINUOUS = {"normal", "lognormal", "uniform", "beta", "exponential"}
DISCRETE = {"categorical", "bernoulli", "uniform_int", "poisson", "constant", "date_uniform"}
ALL_DISTS = CONTINUOUS | DISCRETE
