"""
User-defined rules declared in the contract's ``"rules"`` block.

The guide warns against free-text formulas because "two implementations
disagree and nobody tests them". Custom rules address both concerns:

* they are expressed in the restricted expression language of
  ``synthgen.expressions`` - one interpreter, one meaning;
* they are **named and versioned** like library rules and referenced from nodes
  by name, never inlined;
* they **must ship test cases** (inputs -> expected output). The tests run when
  the contract is validated, so a broken rule stops the run before any data is
  generated.

Example::

    "rules": {
      "custom.payment_ratio": {
        "version": "1.0",
        "description": "Payment as a share of the statement it pays",
        "inputs":  {"payment": "float", "previous_balance": "float"},
        "params":  {"cap": {"type": "float", "default": 5.0}},
        "output":  "float",
        "expression": "where(previous_balance > 0, min(payment / previous_balance, cap), NULL)",
        "tests": [
          {"inputs": {"payment": 50, "previous_balance": 200}, "expected": 0.25},
          {"inputs": {"payment": 0,  "previous_balance": 0},   "expected": null}
        ]
      }
    }
"""
from __future__ import annotations

import numpy as np

from ..expressions import Expression, Val
from . import REQUIRED, Registry, RuleSpec

#: types a custom rule may declare (fixed-point types are library-only)
CUSTOM_TYPES = {"float", "int", "bool", "date", "timestamp", "category", "any"}


class CustomRuleError(ValueError):
    pass


def build_custom_rule(name: str, spec: dict) -> RuleSpec:
    """Compile one contract rule definition into a RuleSpec."""
    if "expression" not in spec:
        raise CustomRuleError(f"custom rule {name!r}: 'expression' is required")
    inputs = dict(spec.get("inputs", {}))
    output = spec.get("output", "float")
    for t in list(inputs.values()) + [output]:
        if t not in CUSTOM_TYPES:
            raise CustomRuleError(f"custom rule {name!r}: type {t!r} not allowed "
                                  f"(use one of {sorted(CUSTOM_TYPES)})")
    params: dict[str, tuple] = {}
    for pname, pdef in spec.get("params", {}).items():
        if isinstance(pdef, dict):
            params[pname] = (pdef.get("type", "any"), pdef.get("default", REQUIRED))
        else:                                   # shorthand: "cap": 5.0
            params[pname] = ("any", pdef)

    expr = Expression(spec["expression"])
    unknown = set(expr.names) - set(inputs) - set(params)
    if unknown:
        raise CustomRuleError(f"custom rule {name!r} reads undeclared names {sorted(unknown)}")

    def func(**kwargs):
        # every argument arrives as (values, mask) because null_aware=True
        n = 1
        for k in inputs:
            n = max(n, np.size(kwargs[k][0]))
        env = {}
        for k, (vals, mask) in kwargs.items():
            env[k] = Val(np.asarray(vals), mask)
        out = expr.evaluate(env, n)
        if out.labels is not None:
            out = out.materialize()
        return out.v, out.m

    rs = RuleSpec(
        name=name, version=str(spec.get("version", "1.0")), kind="rule", func=func,
        inputs=inputs, output=output, params=params,
        description=spec.get("description", ""), source="contract",
        tests=list(spec.get("tests", [])), null_aware=True,
    )
    return rs


def run_rule_tests(spec: RuleSpec) -> list[str]:
    """Execute a custom rule's embedded tests. Returns a list of failure messages."""
    failures: list[str] = []
    if not spec.tests:
        return [f"custom rule {spec.key} has no tests (at least one is required)"]
    for i, case in enumerate(spec.tests):
        kwargs = {}
        try:
            for k in spec.inputs:
                if k not in case.get("inputs", {}):
                    raise CustomRuleError(f"test #{i} is missing input {k!r}")
                v = case["inputs"][k]
                if v is None:
                    kwargs[k] = (np.zeros(1), np.ones(1, dtype=bool))
                else:
                    kwargs[k] = (np.asarray([v], dtype=object if isinstance(v, str) else None), None)
            for p, (_, default) in spec.params.items():
                v = case.get("params", {}).get(p, default)
                if v is REQUIRED:
                    raise CustomRuleError(f"test #{i} is missing required param {p!r}")
                kwargs[p] = (np.asarray(v), None)
            vals, mask = spec.func(**kwargs)
        except Exception as e:  # noqa: BLE001 - report any failure as a test failure
            failures.append(f"{spec.key} test #{i}: raised {type(e).__name__}: {e}")
            continue
        expected = case.get("expected")
        is_null = bool(mask is not None and np.asarray(mask).reshape(-1)[0])
        got = np.asarray(vals).reshape(-1)[0]
        if expected is None:
            if not is_null:
                failures.append(f"{spec.key} test #{i}: expected NULL, got {got!r}")
            continue
        if is_null:
            failures.append(f"{spec.key} test #{i}: expected {expected!r}, got NULL")
            continue
        tol = float(case.get("tolerance", 1e-9))
        if isinstance(expected, bool) or isinstance(expected, str):
            ok = (bool(got) == expected) if isinstance(expected, bool) else (str(got) == expected)
        else:
            ok = abs(float(got) - float(expected)) <= tol
        if not ok:
            failures.append(f"{spec.key} test #{i}: expected {expected!r}, got {got!r}")
    return failures


def register_custom_rules(registry: Registry, rules: dict) -> list[str]:
    """Compile, test and register all contract rules. Returns failures."""
    failures: list[str] = []
    for name, spec in (rules or {}).items():
        try:
            rs = build_custom_rule(name, spec)
            registry.register(rs, override=bool(spec.get("override", False)))
        except (ValueError, KeyError) as e:
            failures.append(str(e))
            continue
        failures.extend(run_rule_tests(rs))
    return failures
