"""
The rules library - guide Step 7.

All domain knowledge lives here as **named, versioned, pure functions**. The
engine itself knows nothing about credit cards; a contract node says
``"rule": "payment.minimum_due"`` and the engine looks the rule up, converts
the node's inputs to the types the rule declares, calls it and converts the
result into the node's column type.

Three kinds of rule exist:

``rule``      deterministic: typed inputs + params -> one output (+ optional
              NULL mask). No randomness, no global state.
``sampler``   used by ``drawn`` nodes whose draw needs domain logic (e.g. a
              payment amount that depends on what is owed). It receives keyed
              uniforms from the engine (``u``), so it is still a pure function
              of its address - just not of its inputs alone.
``group``     a rule over child rows that needs to know which parent row each
              child belongs to (e.g. credit-limit declines are sequential
              within a statement cycle). Receives ``group`` (parent row index)
              and ``order`` (sort key) via ``ctx``.

Input / output / param type vocabulary
--------------------------------------
``money``      int64 minor units at scale 2 (cents) - alias of ``fixed2``
``fixedN``     int64 minor units at scale N (``fixed6`` for rates, etc.)
``float``      float64 in natural units
``int``        int64
``bool``       bool
``date``       int64 days since 1970-01-01
``timestamp``  int64 seconds since epoch
``category``   object array of labels
``any``        whatever the column holds, in natural units
``list``       (params only) a JSON list
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

REQUIRED = object()      # sentinel for params without a default

_FIXED_RE = re.compile(r"^fixed(\d{1,2})$")
SCALAR_TYPES = {"money", "float", "int", "bool", "date", "timestamp", "category", "any"}


def type_scale(t: str) -> Optional[int]:
    """Scale of a fixed-point rule type (``money`` -> 2), else None."""
    if t == "money":
        return 2
    m = _FIXED_RE.match(t)
    return int(m.group(1)) if m else None


def valid_rule_type(t: str) -> bool:
    return t in SCALAR_TYPES or type_scale(t) is not None


@dataclass
class RuleSpec:
    name: str
    version: str
    kind: str                                  # rule | sampler | group
    func: Callable
    inputs: dict[str, str]
    output: str
    params: dict[str, tuple[str, Any]] = field(default_factory=dict)
    uniforms: int = 0
    description: str = ""
    source: str = "library"
    tests: list = field(default_factory=list)
    #: True for rules that receive ``(values, null_mask)`` pairs instead of
    #: NULL-filled arrays (custom expression rules have SQL NULL semantics)
    null_aware: bool = False
    #: inputs where NULL is meaningful and must NOT make the output NULL
    #: (they are filled with a neutral value, e.g. a payment day when no payment)
    nullable_inputs: tuple = ()

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def signature(self) -> str:
        ins = ", ".join(f"{k}: {v}" for k, v in self.inputs.items())
        ps = ", ".join(f"{k}: {t}" + ("" if d is REQUIRED else f" = {d!r}")
                       for k, (t, d) in self.params.items())
        return f"{self.name}@{self.version}({ins}; params: {ps}) -> {self.output}"


def _version_tuple(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in str(v).split("."))


class Registry:
    """Name -> version -> RuleSpec."""

    def __init__(self):
        self._rules: dict[str, dict[str, RuleSpec]] = {}

    def register(self, spec: RuleSpec, override: bool = False) -> None:
        for t in list(spec.inputs.values()) + [spec.output]:
            if not valid_rule_type(t):
                raise ValueError(f"rule {spec.key}: unknown type {t!r}")
        versions = self._rules.setdefault(spec.name, {})
        if spec.version in versions and not override:
            raise ValueError(f"rule {spec.key} is already registered "
                             f"(set \"override\": true to replace it)")
        versions[spec.version] = spec

    def resolve(self, name: str, version: Optional[str] = None) -> RuleSpec:
        if name not in self._rules:
            raise KeyError(f"rule {name!r} is not in the rules library")
        versions = self._rules[name]
        if version is None:
            version = max(versions, key=_version_tuple)
        if version not in versions:
            raise KeyError(f"rule {name!r} has no version {version!r} "
                           f"(available: {sorted(versions)})")
        return versions[version]

    def copy(self) -> "Registry":
        r = Registry()
        r._rules = {k: dict(v) for k, v in self._rules.items()}
        return r

    def all(self) -> list[RuleSpec]:
        return [s for vs in self._rules.values() for s in vs.values()]


#: package name -> specs (filled by the decorators when a package is imported)
_PACKAGES: dict[str, list[RuleSpec]] = {}


def registrar(package: str):
    """Return (rule, sampler, group) decorators that file specs under ``package``."""

    def _make(kind):
        def deco_factory(name, version, inputs, output, params=None, uniforms=0, description="",
                         nullable_inputs=()):
            def deco(fn):
                spec = RuleSpec(
                    name=name, version=str(version), kind=kind, func=fn,
                    inputs=dict(inputs), output=output,
                    params={k: (v if isinstance(v, tuple) else (v, REQUIRED))
                            for k, v in (params or {}).items()},
                    uniforms=uniforms,
                    description=description or (fn.__doc__ or "").strip().split("\n")[0],
                    source=f"library:{package}",
                    nullable_inputs=tuple(nullable_inputs),
                )
                _PACKAGES.setdefault(package, []).append(spec)
                return fn
            return deco
        return deco_factory

    return _make("rule"), _make("sampler"), _make("group")


def load_packages(packages: list[str]) -> Registry:
    """Import the requested rule packages and return a registry with them."""
    import importlib

    reg = Registry()
    for pkg in packages:
        if pkg not in _PACKAGES:
            try:
                importlib.import_module(f"{__name__}.{pkg}")
            except ModuleNotFoundError:
                raise ValueError(f"unknown rule package {pkg!r}") from None
        for spec in _PACKAGES.get(pkg, []):
            reg.register(spec)
    return reg
