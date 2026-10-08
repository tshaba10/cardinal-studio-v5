"""
Generation phases - guide Step 8.

Phase 0/1  static entities (quotas, joint draws, references)    ``run_static``
Phase 2    the time loop for each stepped entity, with event
           children generated inside each step                   ``simulate_chunk``
Phase 3    emit canonical long tables (one CSV per grain)         ``generate``
Phase 4    verification (``synthgen.verify``)                    ``generate(verify=True)``

Scaling
-------
Roots (e.g. accounts) are split into fixed-size chunks. Each chunk is
simulated independently - legal because entities only interact through
shared series - and written to a part file; parts are concatenated in chunk
order. Rows are sorted canonically within a chunk (root ordinal, step,
event time, sequence) and chunks are contiguous ranges of root ordinals, so
the final files are **byte-identical for any chunk size and worker count**.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..contract.compiler import Plan
from ..output import CodecCache, emitted_columns, format_table, header
from ..rng import KeyedRNG
from .executor import Executor, Frame
from .time import MonthlyCalendar, build_shared
from .values import Col, concat_cols


# --------------------------------------------------------------------------- #
# Phase 1 - static entities
# --------------------------------------------------------------------------- #

def run_static(plan: Plan) -> dict[str, dict[str, Col]]:
    """Generate every static entity in full. Returns entity -> column -> Col."""
    rng = KeyedRNG(plan.seed)
    frames: dict[str, Frame] = {}
    static: dict[str, dict[str, Col]] = {}
    for ent in plan.entities.values():
        if ent.role == "static":
            ords = np.arange(ent.count, dtype=np.int64)
            frames[ent.name] = Frame(ent.name, ent.count, ords, 0, 0)
            static[ent.name] = frames[ent.name].cols          # shared dict, filled in place
    Executor(plan, rng, static).run_units(plan.static_order, frames)
    return static


# --------------------------------------------------------------------------- #
# Phase 2 - time loop
# --------------------------------------------------------------------------- #

@dataclass
class SteppedSetup:
    entity: str
    calendar: MonthlyCalendar
    shared: dict
    n_roots: int


def make_stepped_setup(plan: Plan, entity: str, static: dict) -> SteppedSetup:
    ent = plan.entities[entity]
    anchors = plan.time.get("anchors", {}).get(entity, {})
    parent = static[ent.parent]
    start = parent[anchors["start_date"].split(".")[1]].values
    cday = None
    if plan.time.get("calendar", "monthly_cycle_day") == "monthly_cycle_day":
        cday = parent[anchors["cycle_day"].split(".")[1]].values
    cal = MonthlyCalendar(plan.time, start, cday)
    shared = build_shared(plan.shared, cal.sim_start, cal.window_end, plan.seed)
    return SteppedSetup(entity, cal, shared, len(start))


def _empty_state(plan: Plan, entity: str, size: int) -> dict[str, Col]:
    state = {}
    for col in plan.lagged.get(entity, []):
        node = plan.entities[entity].nodes[col]
        dtype = {"decimal": np.int64, "int": np.int64, "float": np.float64, "bool": bool,
                 "date": np.int64, "timestamp": np.int64}.get(node.ctype.kind, np.int64)
        init = node.init if node.init is not None else 0
        mask = np.ones(size, bool) if node.init is None and node.spec.get("init") is None else None
        state[col] = Col(np.full(size, init, dtype=dtype), node.ctype, mask, node.labels)
    return state


def simulate_chunk(plan: Plan, static: dict, setup: SteppedSetup, lo: int, hi: int,
                   keep_all_columns: bool = False) -> dict[str, dict[str, Col]]:
    """Simulate roots [lo, hi) of one stepped entity; return emitted tables.

    Returned tables are sorted canonically and contain only output-window rows.
    With ``keep_all_columns`` internal (``emit: false``) columns are kept too
    (used by calibration and tests).
    """
    ent = plan.entities[setup.entity]
    cal = setup.calendar
    roots = np.arange(lo, hi, dtype=np.int64)
    first = np.maximum(cal.first_step[roots], cal.sim_start)
    terminated = np.zeros(roots.size, dtype=bool)
    state = {ent.name: _empty_state(plan, ent.name, roots.size)}
    n_steps_total = cal.window_end - cal.sim_start + 1
    ex = Executor(plan, KeyedRNG(plan.seed), static, shared=setup.shared,
                  shared_origin=cal.sim_start, state=state, state_offset=lo,
                  g_base=cal.sim_start, n_steps_total=n_steps_total)
    units = plan.stepped[ent.name]
    family = [ent.name] + ent.children
    pieces: dict[str, list] = {e: [] for e in family}

    g0 = int(first.min()) if roots.size else cal.window_end + 1
    for g in range(g0, cal.window_end + 1):
        active = (first <= g) & ~terminated
        if not active.any():
            continue
        act = roots[active]
        frame = Frame(ent.name, act.size, act, g, 0)
        frame.ctx = cal.context(g, act, first[active] == g)
        frames = {ent.name: frame}
        ex.run_units(units, frames)

        # carry state: this step's values become @t-1 for the next step
        for col, st in state[ent.name].items():
            cur = frame.cols[col]
            st.values[act - lo] = cur.values
            if cur.mask is not None or st.mask is not None:
                if st.mask is None:
                    st.mask = np.zeros(st.values.size, bool)
                st.mask[act - lo] = False if cur.mask is None else cur.mask
        if ent.terminal is not None:
            term = frame.cols[ent.terminal.column].values.astype(bool)
            terminated[act[term] - lo] = True

        if g < cal.window_start:                      # warm-up: simulate, do not emit
            continue
        for e in family:
            f = frames.get(e)
            if f is None:
                if e == ent.name:
                    continue
                f = ex.expand(e, frames)                # zero-count steps still need a frame
            keep = {c: col for c, col in f.cols.items()
                    if keep_all_columns or plan.entities[e].nodes[c].emit}
            sort_keys = {"root": f.root, "step": np.full(f.n, g, np.int64),
                         "seq": np.asarray(f.sub, dtype=np.int64) * np.ones(f.n, np.int64)}
            for ob in plan.entities[e].order_by:
                sort_keys["ob_" + ob] = f.cols[ob].values
            pieces[e].append((keep, sort_keys))

    out: dict[str, dict[str, Col]] = {}
    for e, plist in pieces.items():
        if not plist:
            out[e] = {}
            continue
        cols = {c: concat_cols([p[0][c] for p in plist]) for c in plist[0][0]}
        keys = {k: np.concatenate([p[1][k] for p in plist]) for k in plist[0][1]}
        ob = [keys["ob_" + c] for c in plan.entities[e].order_by]
        # lexsort: last key is primary -> (root, step, order_by..., seq)
        order = np.lexsort([keys["seq"]] + ob[::-1] + [keys["step"], keys["root"]])
        out[e] = {c: col.take(order) for c, col in cols.items()}
    return out


# --------------------------------------------------------------------------- #
# Phase 3 - writing (parallel over chunks)
# --------------------------------------------------------------------------- #

_WORKER: dict = {}       # populated before forking; inherited by worker processes


def _chunk_worker(args) -> dict:
    """Simulate one chunk and write one part file per table in its family."""
    lo, hi, idx = args
    plan, static, setup, parts_dir = (_WORKER["plan"], _WORKER["static"],
                                      _WORKER["setup"], _WORKER["parts_dir"])
    tables = simulate_chunk(plan, static, setup, lo, hi)
    codecs = CodecCache(plan)
    written = {}
    for e, cols in tables.items():
        if not plan.entities[e].emit:
            continue
        path = Path(parts_dir) / f"{e}.part{idx:06d}.csv"
        rows = 0
        with open(path, "wb") as fh:
            if cols:
                rows = len(next(iter(cols.values())))
                for blob in format_table(plan, e, cols, codecs):
                    fh.write(blob)
        written[e] = (str(path), rows)
    return written


def _write_static(plan: Plan, static: dict, out_dir: Path, log) -> dict:
    codecs = CodecCache(plan)
    counts = {}
    for name, cols in static.items():
        ent = plan.entities[name]
        if not ent.emit or not emitted_columns(plan, name):
            continue
        path = out_dir / f"{name}.csv"
        with open(path, "wb") as fh:
            fh.write(header(plan, name))
            for blob in format_table(plan, name, cols, codecs):
                fh.write(blob)
        counts[name] = ent.count
        log(f"  wrote {path.name}: {ent.count:,} rows")
    return counts


def generate(plan: Plan, out_dir: Optional[str] = None, workers: Optional[int] = None,
             chunk_size: Optional[int] = None, log: Callable = print) -> dict:
    """Run every phase and write one CSV per emitted entity. Returns a run report."""
    t0 = time.time()
    out = Path(out_dir or plan.output["directory"])
    out.mkdir(parents=True, exist_ok=True)
    workers = int(plan.output.get("workers", 1) if workers is None else workers)
    if workers <= 0:                                   # 0 = use every core
        workers = os.cpu_count() or 1
    chunk_size = int(chunk_size or plan.output.get("chunk_size", 10000))

    log("phase 1: static entities")
    static = run_static(plan)
    counts = _write_static(plan, static, out, log)

    for entity in plan.stepped:
        setup = make_stepped_setup(plan, entity, static)
        ent = plan.entities[entity]
        _check_event_capacity(plan, ent, setup)
        cal = setup.calendar
        log(f"phase 2: time loop for {entity}: {setup.n_roots:,} roots, steps "
            f"{cal.sim_start - cal.window_start + 1}..{cal.window_end - cal.window_start + 1} "
            f"(warm-up included), chunk={chunk_size}, workers={workers}")
        parts_dir = out / "_parts"
        parts_dir.mkdir(exist_ok=True)
        jobs = [(lo, min(lo + chunk_size, setup.n_roots), i)
                for i, lo in enumerate(range(0, setup.n_roots, chunk_size))]
        _WORKER.update(plan=plan, static=static, setup=setup, parts_dir=str(parts_dir))
        results: list[dict] = []
        if workers > 1 and len(jobs) > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
                for k, r in enumerate(pool.imap(_chunk_worker, jobs)):
                    results.append(r)
                    log(f"  chunk {k + 1}/{len(jobs)} done ({time.time() - t0:.1f}s)")
        else:
            for k, job in enumerate(jobs):
                results.append(_chunk_worker(job))
                log(f"  chunk {k + 1}/{len(jobs)} done ({time.time() - t0:.1f}s)")
        # Phase 3: concatenate parts in chunk order -> canonical file
        for e in [entity] + ent.children:
            if not plan.entities[e].emit:
                continue
            path = out / f"{e}.csv"
            total = 0
            with open(path, "wb") as fh:
                fh.write(header(plan, e))
                for r in results:
                    part, rows = r[e]
                    total += rows
                    with open(part, "rb") as src:
                        shutil.copyfileobj(src, fh, 16 * 1024 * 1024)
                    os.remove(part)
            counts[e] = total
            log(f"  wrote {path.name}: {total:,} rows")
        shutil.rmtree(parts_dir, ignore_errors=True)

    report = {
        "contract": plan.raw.get("contract", {}),
        "seed": plan.seed,
        "contract_sha256": hashlib.sha256(
            json.dumps(plan.raw, sort_keys=True).encode()).hexdigest(),
        "rows": counts,
        "seconds": round(time.time() - t0, 2),
        "rules_used": sorted({n.rule.key for e in plan.entities.values()
                              for n in e.nodes.values() if n.rule is not None}),
    }
    with open(out / "run_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"done in {report['seconds']}s")
    return report


def _check_event_capacity(plan: Plan, ent, setup: SteppedSetup) -> None:
    """Event identifiers pack (root, step, seq); make sure the pattern can hold them."""
    n_steps = setup.calendar.window_end - setup.calendar.sim_start + 1
    for child in ent.children:
        ce = plan.entities[child]
        need = setup.n_roots * n_steps * ce.max_per_parent
        for node in ce.nodes.values():
            if node.kind == "identifier" and node.codec is not None and need > node.codec.domain:
                raise ValueError(f"{node.full}: pattern holds {node.codec.domain} ids but up to "
                                 f"{need} are addressable; widen the pattern")


__all__ = ["run_static", "simulate_chunk", "make_stepped_setup", "generate"]
