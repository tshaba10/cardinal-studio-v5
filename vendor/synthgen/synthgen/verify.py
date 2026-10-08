"""
Post-run verification - guide Step 11.

Reads the CSV files back **from disk** (so it checks the artifact users get,
not the generator's memory) and checks:

Schema (automatic, from the contract)
  * header matches the emitted columns
  * non-nullable columns contain no NULLs
  * categorical values are within their declared label set
  * bounds (drawn columns must never violate; derived columns are checked
    according to their ``on_violation``: error / warn)
  * decimal precision and scale, integer width
  * primary keys and ``unique`` columns are unique
  * foreign keys resolve (parent keys, references, declared composite FKs)

Contract invariants (``invariants`` block)
  * ``row``          an expression that must be true on every row
  * ``continuity``   opening value of step t == closing value of step t-1
  * ``monotonic``    a cumulative column never decreases within a partition
  * ``aggregate``    a parent column equals the roll-up of its child rows

Targets (``targets`` block) are measured and reported; a target outside its
tolerance is a *warning* (targets are calibration goals, not invariants).

Performance: every table is read exactly once, in chunks, with typed columns
(numbers as float64, categorical strings as pandas categories). Everything
that needs a table - FK checks, aggregate roll-ups, targets, sequence checks -
is accumulated during that single pass. Keys are hashed to uint64, so
uniqueness and referential checks on tens of millions of rows fit in memory.
Tables are processed in contract order (parents before children).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .contract.compiler import Plan
from .expressions import Expression, Val
from .output import emitted_columns
from .rng import mix64

CHUNK_ROWS = 2_000_000


def _hash_keys(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Order-sensitive uint64 hash of key columns, consistent across tables.

    Strings hash by value whatever their pandas dtype (category / object) and
    numbers hash as float64, so a foreign key and its primary key always agree.
    """
    h = np.zeros(len(df), dtype=np.uint64)
    for c in cols:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = pd.util.hash_array(np.asarray(s.cat.categories.astype(str), dtype=object))
            codes = s.cat.codes.to_numpy()
            hc = np.where(codes >= 0, cats[np.maximum(codes, 0)], np.uint64(0))
        elif pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
            hc = pd.util.hash_array(s.to_numpy(dtype=float, na_value=np.nan))
        else:
            hc = pd.util.hash_array(s.astype(object).to_numpy(dtype=object))
        h = mix64(h ^ hc.astype(np.uint64))
    return h


def _sorted_unique_count_dups(h: np.ndarray) -> tuple[np.ndarray, int]:
    h = np.sort(h)
    dups = int((h[1:] == h[:-1]).sum()) if h.size else 0
    return h, dups


class Verifier:
    def __init__(self, plan: Plan, out_dir: str | Path, log: Callable = print):
        self.plan = plan
        self.dir = Path(out_dir)
        self.log = log
        self.checks: list[dict] = []
        self.key_sets: dict[str, np.ndarray] = {}     # "entity:a+b" -> sorted hashes

    # ------------------------------------------------------------------ #
    def add(self, name, table, violations, severity="error", detail="", rows=None):
        self.checks.append({"check": name, "table": table, "violations": int(violations),
                            "severity": severity if violations else "ok",
                            "rows_checked": rows, "detail": detail})

    def _path(self, entity: str) -> Path:
        return self.dir / f"{entity}.csv"

    def _dtypes(self, entity: str) -> dict:
        """Numbers float64; label sets, booleans and repeated keys as category
        (parsed in C); unique identifiers and free text as Python strings."""
        out = {}
        for c in emitted_columns(self.plan, entity):
            node = self.plan.entities[entity].nodes[c]
            k = node.ctype.kind
            if k in ("int", "decimal", "float"):
                out[c] = "float64"
            elif k == "bool" or node.labels is not None or node.kind in ("parent_key", "reference"):
                out[c] = "category"
            else:
                out[c] = object
        return out

    def _read(self, entity: str, usecols=None):
        dtypes = self._dtypes(entity)
        if usecols is not None:
            dtypes = {c: dtypes[c] for c in usecols}
        return pd.read_csv(self._path(entity), chunksize=CHUNK_ROWS, dtype=dtypes,
                           keep_default_na=False, na_values=[""], usecols=usecols)

    def _typed(self, entity: str, df: pd.DataFrame) -> dict[str, Val]:
        """Columns -> expression values (natural units; categoricals as codes+labels)."""
        env = {}
        ent = self.plan.entities[entity]
        for c in df.columns:
            node = ent.nodes[c]
            s = df[c]
            mask = s.isna().to_numpy()
            m = mask if mask.any() else None
            k = node.ctype.kind
            if k == "bool":
                v = (s == "true").to_numpy(dtype=bool, na_value=False) if isinstance(
                    s.dtype, pd.CategoricalDtype) else s.to_numpy(dtype=bool, na_value=False)
            elif isinstance(s.dtype, pd.CategoricalDtype):
                labels = tuple(str(x) for x in s.cat.categories)
                env[c] = Val(s.cat.codes.to_numpy().astype(np.int64), m, labels)
                continue
            elif k in ("int", "decimal", "float"):
                v = s.to_numpy(dtype=float, na_value=0.0)
            elif k == "date":
                d = pd.to_datetime(s, format="%Y-%m-%d")
                v = ((d - pd.Timestamp("1970-01-01")).dt.days).fillna(0).to_numpy(dtype=np.int64)
            elif k == "timestamp":
                d = pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S")
                v = ((d - pd.Timestamp("1970-01-01")).dt.total_seconds()).fillna(0).to_numpy(dtype=np.int64)
            else:
                v = s.fillna("").to_numpy(dtype=object)
            env[c] = Val(v, m)
        return env

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        t0 = time.time()
        tables = [e for e, ent in self.plan.entities.items()
                  if ent.emit and emitted_columns(self.plan, e) and self._path(e).exists()]
        self.agg_child: dict[str, list] = defaultdict(list)    # invariant -> partial sums
        self.agg_parent: dict[str, tuple] = {}
        self.targets_acc = {t["id"]: [0.0, 0.0] for t in self.plan.targets}
        for e in tables:
            self.log(f"verify {e}")
            self._table(e, tables)
        self._finish_aggregates(tables)
        self._finish_targets(tables)
        errors = [c for c in self.checks if c["severity"] == "error"]
        warnings = [c for c in self.checks if c["severity"] == "warning"]
        result = {"ok": not errors, "errors": len(errors), "warnings": len(warnings),
                  "seconds": round(time.time() - t0, 2), "checks": self.checks}
        with open(self.dir / "verification_report.json", "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    # ------------------------------------------------------------------ #
    def _table(self, entity: str, tables: list[str]) -> None:
        plan, ent = self.plan, self.plan.entities[entity]
        expected = emitted_columns(plan, entity)
        with open(self._path(entity), "r") as fh:
            header = fh.readline().strip().split(",")
        self.add("header", entity, int(header != expected), detail=",".join(header))
        if header != expected:
            return

        pk = ent.spec.get("primary_key") or [c for c, n in ent.nodes.items() if n.kind == "identifier"]
        unique_cols = [c for c in expected if ent.nodes[c].spec.get("unique") and [c] != pk]
        row_invs = [(inv, Expression(inv["expression"])) for inv in plan.invariants
                    if inv["type"] == "row" and inv.get("table") == entity]
        seq_invs = [inv for inv in plan.invariants
                    if inv["type"] in ("continuity", "monotonic") and inv.get("table") == entity]
        agg_as_child = [(inv, Expression(inv["where"]) if inv.get("where") else None)
                        for inv in plan.invariants if inv["type"] == "aggregate" and inv["child"] == entity]
        agg_as_parent = [inv for inv in plan.invariants
                         if inv["type"] == "aggregate" and inv["parent"] == entity]
        my_targets = [(t, Expression(t["where"] if t.get("metric", "share") == "share" else t["of"]))
                      for t in plan.targets if t["table"] == entity]
        fks = self._fk_specs(entity)
        fk_missing = defaultdict(int)

        counts: dict[str, int] = defaultdict(int)
        pk_hashes, uniq_hashes = [], {c: [] for c in unique_cols}
        seq_cols = sorted({c for inv in seq_invs for c in
                           inv.get("partition_by", []) + [inv["order_by"]] +
                           ([inv["current"], inv["previous"]] if inv["type"] == "continuity"
                            else [inv["column"]])})
        seq_parts = []
        parent_parts = {inv["name"]: ([], []) for inv in agg_as_parent}
        total = 0

        for df in self._read(entity):
            total += len(df)
            self._column_checks(entity, df, expected, counts)
            pk_hashes.append(_hash_keys(df, pk))
            for c in unique_cols:
                uniq_hashes[c].append(_hash_keys(df, [c]))
            # row invariants + targets share one typed environment
            names = sorted({n for _, e in row_invs for n in e.names}
                           | {n for _, e in my_targets for n in e.names}
                           | {n for _, e in agg_as_child if e for n in e.names})
            env = self._typed(entity, df[names]) if names else {}
            for inv, expr in row_invs:
                r = expr.evaluate(env, len(df))
                ok = np.asarray(r.v, dtype=bool)
                if r.m is not None:                    # a NULL result is a violation
                    ok = ok & ~r.m
                counts[f"invariant:{inv['name']}"] += int((~ok).sum())
            for t, expr in my_targets:
                r = expr.evaluate(env, len(df))
                v = np.asarray(r.v, dtype=float)
                valid = np.ones(len(df), bool) if r.m is None else ~r.m
                self.targets_acc[t["id"]][0] += float(v[valid].sum())
                self.targets_acc[t["id"]][1] += float(valid.sum())
            for cols, tent, tcols in fks:
                ref = self.key_sets.get(f"{tent}:{'+'.join(tcols)}")
                if ref is None:
                    continue
                sub = df[cols].dropna()
                h = _hash_keys(sub, cols)
                if ref.size == 0:
                    fk_missing[(tuple(cols), tent)] += len(h)
                    continue
                pos = np.minimum(np.searchsorted(ref, h), ref.size - 1)
                fk_missing[(tuple(cols), tent)] += int((ref[pos] != h).sum())
            for inv, expr in agg_as_child:
                mask = np.ones(len(df), bool)
                if expr is not None:
                    r = expr.evaluate(env, len(df))
                    mask = np.asarray(r.v, bool) & (~r.m if r.m is not None else True)
                if inv.get("fn", "sum") == "count":
                    val = mask.astype(float)
                else:
                    val = np.where(mask, df[inv["child_column"]].to_numpy(dtype=float, na_value=0.0), 0.0)
                self.agg_child[inv["name"]].append(
                    pd.Series(val).groupby(_hash_keys(df, inv["keys"])).sum())
            for inv in agg_as_parent:
                hs, vs = parent_parts[inv["name"]]
                hs.append(_hash_keys(df, inv["keys"]))
                vs.append(df[inv["parent_column"]].to_numpy(dtype=float, na_value=0.0))
            if seq_cols:
                seq_parts.append(df[seq_cols].copy())

        for key, n in sorted(counts.items()):
            self.add(key, entity, n, "warning" if "[warn]" in key else "error", rows=total)
        pk_sorted, dups = _sorted_unique_count_dups(np.concatenate(pk_hashes) if pk_hashes
                                                    else np.array([], np.uint64))
        self.add(f"primary_key:{'+'.join(pk)}", entity, dups, rows=total)
        self.key_sets[f"{entity}:{'+'.join(pk)}"] = pk_sorted
        for c in unique_cols:
            hs, d = _sorted_unique_count_dups(np.concatenate(uniq_hashes[c]))
            self.add(f"unique:{c}", entity, d, rows=total)
            self.key_sets[f"{entity}:{c}"] = hs
        for cols, tent, tcols in fks:
            if f"{tent}:{'+'.join(tcols)}" in self.key_sets or tent in tables:
                self.add(f"foreign_key:{'+'.join(cols)}->{tent}", entity,
                         fk_missing[(tuple(cols), tent)], rows=total)
        for inv in agg_as_parent:
            hs, vs = parent_parts[inv["name"]]
            self.agg_parent[inv["name"]] = (np.concatenate(hs) if hs else np.array([], np.uint64),
                                            np.concatenate(vs) if vs else np.array([]))
        if seq_invs and seq_parts:
            df_all = pd.concat(seq_parts, ignore_index=True)
            for inv in seq_invs:
                self._sequence_invariant(inv, df_all)
        self.log(f"  {total:,} rows")

    def _column_checks(self, entity: str, df: pd.DataFrame, expected: list[str], counts) -> None:
        ent = self.plan.entities[entity]
        for c in expected:
            node = ent.nodes[c]
            s = df[c]
            na = s.isna().to_numpy()
            if not node.nullable and na.any():
                counts[f"not_null:{c}"] += int(na.sum())
            k = node.ctype.kind
            if isinstance(s.dtype, pd.CategoricalDtype):
                allowed = node.labels if node.labels is not None else (
                    ("true", "false") if k == "bool" else None)
                if allowed is not None:
                    bad = [x for x in s.cat.categories if str(x) not in allowed]
                    if bad:
                        counts[f"allowed_values:{c}"] += int(s.isin(bad).sum())
            if k not in ("int", "decimal", "float"):
                continue
            x = s.to_numpy(dtype=float, na_value=np.nan)[~na]
            if not x.size:
                continue
            if k == "decimal":
                sc, p = node.ctype.scale, node.ctype.precision
                scaled = np.abs(x) * 10.0 ** sc
                counts[f"decimal_scale:{c}"] += int((np.abs(scaled - np.rint(scaled)) > 1e-4).sum())
                counts[f"decimal_precision:{c}"] += int((np.abs(x) >= 10.0 ** (p - sc)).sum())
            if k == "int":
                lo, hi = node.ctype.int_range
                counts[f"int_width:{c}"] += int(((x < lo) | (x > hi) | (x != np.rint(x))).sum())
            blo, bhi, mode = node.bounds
            if blo is not None or bhi is not None:
                scale = 10 ** node.ctype.scale if k == "decimal" else 1
                bad = np.zeros(x.size, bool)
                if blo is not None:
                    bad |= x < blo / scale - 1e-9
                if bhi is not None:
                    bad |= x > bhi / scale + 1e-9
                counts[f"bounds[{'warn' if mode == 'warn' else 'error'}]:{c}"] += int(bad.sum())

    def _fk_specs(self, entity: str) -> list[tuple]:
        ent = self.plan.entities[entity]
        fks = []
        for c, node in ent.nodes.items():
            if node.emit and node.kind in ("parent_key", "reference"):
                tent, tcol = node.spec["references"].split(".")
                fks.append(([c], tent, [tcol]))
        for fk in ent.spec.get("foreign_keys", []):
            fks.append((fk["columns"], fk["references"], fk["ref_columns"]))
        return fks

    def _sequence_invariant(self, inv: dict, df: pd.DataFrame) -> None:
        part, order = inv.get("partition_by", []), inv["order_by"]
        # factorize partitions (hash based, C speed) and sort by (partition, order)
        codes = [pd.factorize(df[p])[0] for p in part]
        o = df[order].to_numpy(dtype=float, na_value=np.nan)
        idx = np.lexsort([o] + codes[::-1])
        o = o[idx]
        same = np.ones(len(df), bool)
        for c in codes:
            v = c[idx]
            same[1:] &= v[1:] == v[:-1]
        if len(df):
            same[0] = False
        if inv["type"] == "continuity":
            consecutive = same[1:] & (o[1:] - o[:-1] == 1)
            cur = df[inv["current"]].to_numpy(dtype=float, na_value=np.nan)[idx]
            prev = df[inv["previous"]].to_numpy(dtype=float, na_value=np.nan)[idx]
            bad = consecutive & (np.abs(cur[1:] - prev[:-1]) > inv.get("tolerance", 0.005))
        else:
            x = df[inv["column"]].to_numpy(dtype=float, na_value=np.nan)[idx]
            d = x[1:] - x[:-1]
            down = inv.get("direction", "non_decreasing") == "non_decreasing"
            bad = same[1:] & ((d < -1e-9) if down else (d > 1e-9))
        self.add(f"invariant:{inv['name']}", inv["table"], int(bad.sum()), rows=len(df))

    def _finish_aggregates(self, tables: list[str]) -> None:
        for inv in self.plan.invariants:
            if inv["type"] != "aggregate" or inv["parent"] not in tables or inv["child"] not in tables:
                continue
            parts = self.agg_child.get(inv["name"], [])
            agg = pd.concat(parts).groupby(level=0).sum() if parts else pd.Series(dtype=float)
            ph, pv = self.agg_parent.get(inv["name"], (np.array([], np.uint64), np.array([])))
            cv = agg.reindex(ph).fillna(0.0).to_numpy()
            bad = int((np.abs(pv - cv) > inv.get("tolerance", 0.005)).sum())
            orphan = int((~np.isin(agg.index.to_numpy(), ph)).sum())
            self.add(f"invariant:{inv['name']}", inv["parent"], bad + orphan, rows=len(ph),
                     detail=f"{orphan} child groups without a parent row" if orphan else "")

    def _finish_targets(self, tables: list[str]) -> None:
        for t in self.plan.targets:
            if t["table"] not in tables:
                continue
            num, den = self.targets_acc[t["id"]]
            value = num / den if den else float("nan")
            off = not (abs(value - t["target"]) <= t.get("tolerance", 0))
            self.add(f"target:{t['id']}", t["table"], int(off), "warning",
                     detail=f"measured {value:.4f}, target {t['target']} +/- {t.get('tolerance', 0)}")


def verify(plan: Plan, out_dir: str | Path, log: Callable = print) -> dict:
    return Verifier(plan, out_dir, log).run()


def summarize(result: dict) -> str:
    lines = [f"verification: {'PASSED' if result['ok'] else 'FAILED'} "
             f"({result['errors']} errors, {result['warnings']} warnings, "
             f"{len(result['checks'])} checks, {result['seconds']}s)"]
    for c in result["checks"]:
        if c["severity"] != "ok" or c["check"].startswith("target:"):
            lines.append(f"  [{c['severity']:7}] {c['table']}.{c['check']}: "
                         f"{c['violations']} {c['detail']}")
    return "\n".join(lines)
