"""
Calibration - guide Step 9: hit portfolio targets by tuning *behaviour
parameters*, never by editing output values.

Each target with a ``calibrate`` block names one numeric parameter (a JSON
pointer into the contract, e.g. the intercept shift of the missed-payment
model) and a search range. The compiler has already verified that this
parameter has a *mechanism*: its node feeds the target column.

Algorithm: coordinate-wise bisection on a sample of roots. For each target in
turn, bisect its parameter until the sampled metric is within a quarter of the
tolerance (or iterations run out); repeat the whole pass a few times because
targets interact (more spending raises utilisation, which raises delinquency).
The tuned values are written back into the contract, so the full run is
reproducible from the contract alone.
"""
from __future__ import annotations

import copy
import json
from typing import Callable

import numpy as np

from .contract.compiler import compile_contract
from .contract.loader import resolve_refs
from .engine.simulate import make_stepped_setup, run_static, simulate_chunk
from .expressions import Expression


def _set_pointer(doc: dict, pointer: str, value) -> None:
    keys = [k.replace("~1", "/").replace("~0", "~") for k in pointer.strip("/").split("/")]
    node = doc
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    node[keys[-1]] = value


def _get_pointer(doc: dict, pointer: str):
    node = doc
    for k in pointer.strip("/").split("/"):
        node = node[int(k)] if isinstance(node, list) else node[k]
    return node


def measure(raw: dict, sample_roots: int) -> dict[str, float]:
    """Simulate a sample in memory and evaluate every target metric."""
    doc = copy.deepcopy(raw)
    doc.setdefault("volume", {})
    plan = compile_contract(resolve_refs(doc))
    # shrink every stepped entity's root volume to the sample size
    for ent in plan.entities.values():
        if ent.role == "stepped":
            root = plan.entities[ent.parent]
            root.count = min(root.count, sample_roots)
    static = run_static(plan)
    tables: dict = {}
    for entity in plan.stepped:
        setup = make_stepped_setup(plan, entity, static)
        tables.update(simulate_chunk(plan, static, setup, 0, setup.n_roots, keep_all_columns=True))
    tables.update(static)
    results = {}
    for t in plan.targets:
        cols = tables.get(t["table"], {})
        text = t["where"] if t.get("metric", "share") == "share" else t["of"]
        expr = Expression(text)
        if not cols:
            results[t["id"]] = float("nan")
            continue
        n = len(next(iter(cols.values())))
        r = expr.evaluate({c: cols[c].to_val() for c in expr.names}, n)
        v = np.asarray(r.v, dtype=float)
        valid = np.ones(n, bool) if r.m is None else ~r.m
        results[t["id"]] = float(v[valid].mean()) if valid.any() else float("nan")
    return results


def calibrate(raw: dict, log: Callable = print) -> tuple[dict, dict]:
    """Return (calibrated contract document, report)."""
    doc = copy.deepcopy(raw)
    settings = doc.get("calibration", {})
    rounds = int(settings.get("rounds", 2))
    sample = int(settings.get("sample_roots", 4000))
    max_iter = int(settings.get("max_iterations", 14))
    targets = [t for t in doc.get("targets", []) if t.get("calibrate")]
    history = []
    if not targets:
        log("no targets have a 'calibrate' block - nothing to do")
    for rnd in range(rounds):
        for t in targets:
            cal = t["calibrate"]
            ptr = cal["parameter"]
            lo, hi = float(cal["low"]), float(cal["high"])
            if rnd:
                # later rounds only correct for interactions between targets:
                # search a narrow window around the current value
                cur = float(_get_pointer(doc, ptr))
                span = (hi - lo) / 16
                lo, hi = max(lo, cur - span), min(hi, cur + span)
            increasing = cal.get("direction", "increasing") == "increasing"
            tol = float(t.get("tolerance", 0.0))
            best = (float("inf"), _get_pointer(doc, ptr))
            for it in range(max_iter):
                x = (lo + hi) / 2 if it else float(_get_pointer(doc, ptr))
                _set_pointer(doc, ptr, round(x, 6))
                value = measure(doc, sample)[t["id"]]
                err = value - t["target"]
                history.append({"round": rnd + 1, "target": t["id"], "parameter": ptr,
                                "value": round(x, 6), "measured": value})
                log(f"  round {rnd + 1} {t['id']}: {ptr.split('/')[-3]}...={x:+.4f} -> {value:.4f} "
                    f"(target {t['target']})")
                if abs(err) < best[0]:
                    best = (abs(err), round(x, 6))
                if abs(err) <= tol / 4:
                    break
                if (err > 0) == increasing:
                    hi = x
                else:
                    lo = x
            _set_pointer(doc, ptr, best[1])
    final = measure(doc, sample) if targets else {}
    report = {
        "sample_roots": sample,
        "final": {t["id"]: {"measured": final.get(t["id"]), "target": t["target"],
                            "tolerance": t.get("tolerance")} for t in doc.get("targets", [])},
        "parameters": {t["calibrate"]["parameter"]: _get_pointer(doc, t["calibrate"]["parameter"])
                       for t in targets},
        "history": history,
    }
    doc.setdefault("calibration", {})["last_result"] = {
        "parameters": report["parameters"],
        "final": report["final"],
    }
    return doc, report


def write_contract(doc: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
