"""
Node execution - one strategy per node kind (guide Step 2).

The executor runs one *execution unit* (a node, or a joint group) for all the
rows of a ``Frame`` at once. A frame is the set of rows of one entity being
generated together:

* static phase - all rows of a static entity (e.g. every account);
* time loop    - every active row of a stepped entity at one absolute step g,
                 plus the event rows (transactions) expanded under them.

Vectorising across entities (instead of looping entity by entity as in the
guide's pseudo-code) is what makes 100k accounts x 36 cycles run in minutes:
the only sequential dimension left is time, which is inherent.

Every random draw is keyed by ``(seed, node stream, root ordinal, step, seq)``
so results are identical however rows are batched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import ndtr

from .. import distributions as D
from ..contract.compiler import STEP_CONTEXT, NodePlan, Plan, Ref
from ..expressions import Expression
from ..rng import KeyedRNG
from ..rules import type_scale
from ..types import ColumnType, coerce_scalar_to_storage, float_to_fixed, parse_date
from .values import (DATE, FLOAT, INT, Col, bound_at_creation, null_fill,
                     to_rule_type, to_storage)

_REF_RE = re.compile(r"^([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)(?:@t-(\d+))?$")
_CTX_TYPES = {"date": DATE, "int": INT}


class GenerationError(RuntimeError):
    pass


@dataclass
class Frame:
    entity: str
    n: int
    root: np.ndarray                    # root static ordinal per row (RNG entity key)
    step: int = 0                       # absolute step g (0 for static rows)
    sub: object = 0                     # sequence within parent (events) for RNG keys
    parent_row: Optional[np.ndarray] = None
    ctx: dict = field(default_factory=dict)
    cols: dict = field(default_factory=dict)
    ordinal: Optional[np.ndarray] = None  # unique ordinal (identifiers)


def runtime_ref(text: str) -> Ref:
    m = _REF_RE.match(text.strip())
    ent, col, lag = m.group(1), m.group(2), int(m.group(3) or 0)
    ns = ent if ent in ("step", "shared", "row") else "node"
    return Ref(text, ns, ent, col, lag)


class Executor:
    def __init__(self, plan: Plan, rng: KeyedRNG, static: dict[str, dict[str, Col]],
                 shared: Optional[dict] = None, shared_origin: int = 0,
                 state: Optional[dict] = None, state_offset: int = 0,
                 g_base: int = 0, n_steps_total: int = 1):
        self.plan = plan
        self.rng = rng
        self.static = static                  # entity -> column -> Col (full tables)
        self.shared = shared or {}
        self.shared_origin = shared_origin
        self.state = state or {}              # stepped entity -> column -> Col (chunk-local)
        self.state_offset = state_offset
        self.g_base = g_base
        self.n_steps_total = n_steps_total
        self.rounding = plan.rounding
        self._exprs: dict[str, Expression] = {}
        self._refs: dict[str, Ref] = {}

    # ------------------------------------------------------------------ #
    # input resolution
    # ------------------------------------------------------------------ #
    def ref(self, text: str) -> Ref:
        r = self._refs.get(text)
        if r is None:
            r = self._refs[text] = runtime_ref(text)
        return r

    def resolve(self, ref: Ref, frame: Frame, frames: dict[str, Frame]) -> Col:
        """Return the referenced column aligned to ``frame``'s rows."""
        if ref.ns == "step":
            src = frame if frame.ctx else frames[self.plan.entities[frame.entity].parent]
            vals = src.ctx[ref.column]
            if src is not frame:
                vals = vals[frame.parent_row]
            return Col(vals, _CTX_TYPES[STEP_CONTEXT[ref.column]])
        if ref.ns == "shared":
            x = self.shared[ref.column][frame.step - self.shared_origin]
            return Col(np.full(frame.n, x), FLOAT)
        if ref.ns == "row":
            if ref.column == "seq":
                return Col(np.asarray(frame.sub, dtype=np.int64), INT)
            return Col(frame.ordinal if frame.ordinal is not None else frame.root, INT)
        if ref.lag:
            st = self.state[ref.entity][ref.column]
            return st.take(frame.root - self.state_offset)
        if ref.entity == frame.entity:
            if ref.column not in frame.cols:
                raise GenerationError(f"{ref.raw} read before it was computed")
            return frame.cols[ref.column]
        if ref.entity in frames and frames[ref.entity] is not frame:
            pf = frames[ref.entity]
            if frame.parent_row is not None and self.plan.entities[frame.entity].parent == ref.entity:
                return pf.cols[ref.column].take(frame.parent_row)
        if ref.entity in self.static:
            # static ancestor: rows map through the root ordinal
            return self.static[ref.entity][ref.column].take(frame.root)
        raise GenerationError(f"cannot resolve {ref.raw} from {frame.entity}")

    def param(self, value, frame: Frame, frames) -> object:
        """Resolve a parameter: literal, {"input": ref} or {"by": ref, "values": {...}}."""
        if isinstance(value, dict) and "input" in value and len(value) <= 2:
            col = self.resolve(self.ref(value["input"]), frame, frames)
            return to_rule_type(col, "any", self.rounding)
        if isinstance(value, dict) and "by" in value and "values" in value:
            col = self.resolve(self.ref(value["by"]), frame, frames)
            table = [value["values"].get(lab, value.get("default")) for lab in col.labels]
            if any(t is None for t in table):
                raise GenerationError(f"missing parameter value for one of {col.labels}")
            if any(isinstance(t, (list, dict)) for t in table):
                arr = np.empty(len(table), dtype=object)
                arr[:] = table
            else:
                arr = np.asarray(table)
            return arr[col.values]
        return value

    def conditioned_params(self, node: NodePlan, frame: Frame, frames) -> dict:
        """Turn {"conditioned_on": c, "params": {label: {...}}} into per-row params."""
        spec = node.spec
        params = spec.get("params", {})
        if not spec.get("conditioned_on"):
            return {k: self.param(v, frame, frames) for k, v in params.items()}
        col = self.resolve(self.ref(spec["conditioned_on"]), frame, frames)
        keys: list = []
        for lab in col.labels:
            for k in params.get(lab, params.get("default", {})):
                if k not in keys:
                    keys.append(k)
        out = {}
        for k in keys:
            table = [params.get(lab, params.get("default", {})).get(k) for lab in col.labels]
            out[k] = np.asarray(table, dtype=float)[col.values]
        return out

    # ------------------------------------------------------------------ #
    # drivers
    # ------------------------------------------------------------------ #
    def run_units(self, units: list[str], frames: dict[str, Frame]) -> None:
        for unit in units:
            if unit.startswith("joint:"):
                _, ent, group = unit.split(":")
                self.run_joint(ent, group, frames[ent], frames)
                continue
            node = self.plan.node(unit)
            frame = frames.get(node.entity)
            if frame is None:
                frame = self.expand(node.entity, frames)
            frame.cols[node.column] = self.run_node(node, frame, frames)

    def expand(self, entity: str, frames: dict[str, Frame]) -> Frame:
        """Create the event frame (e.g. transactions) under its stepped parent."""
        ent = self.plan.entities[entity]
        parent = frames[ent.parent]
        counts = parent.cols[ent.count_ref.column].values.astype(np.int64)
        if counts.size and (counts.min() < 0 or counts.max() > ent.max_per_parent):
            raise GenerationError(
                f"{ent.count_ref.raw} produced {counts.max()} rows for one parent, above "
                f"max_per_parent={ent.max_per_parent}; bound the count distribution")
        total = int(counts.sum())
        parent_row = np.repeat(np.arange(parent.n, dtype=np.int64), counts)
        starts = np.cumsum(counts) - counts
        seq = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
        root = parent.root[parent_row]
        ordinal = ((root * self.n_steps_total) + (parent.step - self.g_base)) * ent.max_per_parent + seq
        frame = Frame(entity, total, root, parent.step, seq, parent_row, ordinal=ordinal)
        frames[entity] = frame
        return frame

    # ------------------------------------------------------------------ #
    def keys(self, frame: Frame):
        return frame.root, frame.step, frame.sub

    def uniform(self, stream: str, frame: Frame) -> np.ndarray:
        u = self.rng.uniform(stream, *self.keys(frame))
        if u.size != frame.n:
            u = np.broadcast_to(u, (frame.n,)).copy()
        return u

    def run_node(self, node: NodePlan, frame: Frame, frames) -> Col:
        kind = node.kind
        if kind == "identifier":
            ords = frame.ordinal if frame.ordinal is not None else frame.root
            return Col(ords.astype(np.int64), node.ctype, ref=node.full)
        if kind == "parent_key":
            target = node.spec["references"]
            tent = target.split(".")[0]
            vals = frame.root if tent == self.plan.entities[frame.entity].root else None
            if vals is None:
                raise GenerationError(f"{node.full}: parent_key must reference the root entity")
            return Col(vals.astype(np.int64), node.ctype, ref=target)
        if kind == "reference":
            return self.run_reference(node, frame, frames)
        if kind == "lookup":
            via = self.resolve(self.ref(node.spec["via"]), frame, frames)
            src = self.ref(node.spec["from"])
            col = self.static[src.entity][src.column].take(via.values)
            if via.mask is not None:
                col.mask = via.mask if col.mask is None else (col.mask | via.mask)
            return col
        if kind == "context":
            src = self.resolve(self.ref(node.spec["source"]), frame, frames)
            rtype = {"date": "date", "timestamp": "timestamp", "int": "int",
                     "float": "float", "bool": "bool"}.get(src.ctype.kind, "any")
            return Col(to_storage(src.values, rtype, node, self.rounding), node.ctype, src.mask)
        if kind == "aggregate":
            return self.run_aggregate(node, frame, frames)
        if kind in ("derived", "state") or (kind == "drawn" and node.rule is not None):
            return self.run_rule(node, frame, frames)
        if kind == "drawn":
            u = self.uniform(f"{node.full}#0", frame)
            return self.finish_draw(node, frame, frames, self.draw(node, frame, frames, u))
        raise GenerationError(f"{node.full}: cannot execute kind {kind!r}")

    # ------------------------------------------------------------------ #
    # drawn values
    # ------------------------------------------------------------------ #
    def draw(self, node: NodePlan, frame: Frame, frames, u: np.ndarray):
        """Return (values, representation) for a dist-based draw using uniforms u."""
        spec = node.spec
        dist = spec.get("dist", "categorical" if spec.get("allowed_values") else None)
        lo_s, hi_s, mode = node.bounds
        lo = hi = None
        if mode == "truncate":                  # bounds in value units for truncation
            lo = None if lo_s is None else self._value_units(lo_s, node.ctype)
            hi = None if hi_s is None else self._value_units(hi_s, node.ctype)

        if dist == "categorical":
            return self.draw_categorical(node, frame, frames, u)
        if dist == "constant":
            v = spec["params"]["value"]
            return np.full(frame.n, coerce_scalar_to_storage(v, node.ctype, self.rounding)), "storage"
        if dist == "date_uniform":
            p = spec["params"]
            start = parse_date(p["start"]) if isinstance(p["start"], str) else p["start"]
            end = parse_date(p["end"]) if isinstance(p["end"], str) else p["end"]
            return D.uniform_int(u, start, end), "date"

        p = self.conditioned_params(node, frame, frames)
        lp = spec.get("linear_predictor")
        eta = self.linear_predictor(node, frame, frames, lp) if lp else None
        if dist == "normal":
            mean = eta if eta is not None else p["mean"]
            return D.normal(u, mean, p["std"], lo, hi), "float"
        if dist == "lognormal":
            return D.lognormal(u, p["mu"], p["sigma"], lo, hi), "float"
        if dist == "uniform":
            return D.uniform(u, p["low"], p["high"]), "float"
        if dist == "beta":
            return D.beta(u, p["a"], p["b"], p.get("low", 0.0), p.get("high", 1.0)), "float"
        if dist == "exponential":
            return D.exponential(u, p["scale"]), "float"
        if dist == "uniform_int":
            return D.uniform_int(u, p["low"], p["high"]), "int"
        if dist == "bernoulli":
            prob = D.apply_link(eta, lp.get("link", "logit")) if eta is not None else p["p"]
            return D.bernoulli(u, prob), "bool"
        if dist == "poisson":
            lam = D.apply_link(eta, lp.get("link", "log")) if eta is not None else p["lam"]
            return D.poisson(u, lam, p.get("max")), "int"
        raise GenerationError(f"{node.full}: unsupported dist {dist!r}")

    @staticmethod
    def _value_units(v, ctype: ColumnType):
        return v / 10 ** ctype.scale if ctype.kind == "decimal" else v

    def draw_categorical(self, node: NodePlan, frame: Frame, frames, u):
        spec = node.spec
        params = spec.get("params") or spec.get("allowed_values")

        def encode(values: list) -> np.ndarray:
            if node.labels is not None:
                return np.array([node.labels.index(v) for v in values], dtype=np.int64)
            return np.array([coerce_scalar_to_storage(v, node.ctype, self.rounding) for v in values])

        if spec.get("allocation") == "exact":
            # Phase 0 quota: exact shares via largest remainder, rows assigned
            # by a keyed random permutation (deterministic, order independent).
            w = np.asarray(params.get("weights") or [1] * len(params["values"]), dtype=float)
            w = w / w.sum()
            raw = w * frame.n
            counts = np.floor(raw).astype(np.int64)
            short = frame.n - counts.sum()
            if short:
                counts[np.argsort(-(raw - counts), kind="stable")[:short]] += 1
            order = np.argsort(u, kind="stable")
            idx = np.empty(frame.n, dtype=np.int64)
            idx[order] = np.repeat(np.arange(len(w)), counts)
            return encode(params["values"])[idx], "storage"

        cond = spec.get("conditioned_on")
        if not cond:
            idx = D.categorical(u, params.get("weights") or [1] * len(params["values"]))
            return encode(params["values"])[idx], "storage"
        col = self.resolve(self.ref(cond), frame, frames)
        out = None
        for code, lab in enumerate(col.labels):
            rows = col.values == code
            if not rows.any():
                continue
            p = params.get(lab, params.get("default"))
            idx = D.categorical(u[rows], p.get("weights") or [1] * len(p["values"]))
            vals = encode(p["values"])[idx]
            if out is None:
                out = np.zeros(frame.n, dtype=vals.dtype)
            out[rows] = vals
        if out is None:
            out = np.zeros(frame.n, dtype=np.int64)
        return out, "storage"

    def linear_predictor(self, node: NodePlan, frame: Frame, frames, lp: dict) -> np.ndarray:
        """eta = intercept + shift + sum(coef * (transform(x) - center) / scale)."""
        eta = np.asarray(self.param(lp.get("intercept", 0.0), frame, frames), dtype=float)
        eta = eta + float(lp.get("intercept_shift", 0.0))
        eta = np.broadcast_to(eta, (frame.n,)).astype(float)
        for term in lp.get("terms", []):
            col = self.resolve(self.ref(term["input"]), frame, frames)
            x = to_rule_type(col, "float", self.rounding)
            x = D.TERM_TRANSFORMS[term.get("transform", "identity")](x)
            x = (x - float(term.get("center", 0.0))) / float(term.get("scale", 1.0))
            if col.mask is not None:                 # NULL contributes null_as (default 0)
                x = np.where(col.mask, float(term.get("null_as", 0.0)), x)
            eta = eta + float(term["coef"]) * x
        return eta

    def finish_draw(self, node: NodePlan, frame: Frame, frames, drawn) -> Col:
        """Convert, bound (at creation), zero-inflate and NULL-inject a draw."""
        values, rep = drawn
        if rep != "storage":
            values = to_storage(values, rep, node, self.rounding)
        values = bound_at_creation(values, node)
        zm = node.spec.get("zero_mass")
        if zm is not None:
            prob = self.param(zm, frame, frames)
            z = self.uniform(f"{node.full}#zero", frame) < np.asarray(prob, dtype=float)
            values = np.where(z, np.zeros(1, dtype=values.dtype), values)
        mask = None
        if node.null_rate:
            mask = self.uniform(f"{node.full}#null", frame) < node.null_rate
        return Col(values, node.ctype, mask, node.labels)

    def run_joint(self, entity: str, group: str, frame: Frame, frames) -> None:
        """Gaussian copula: correlated normals -> uniforms -> each member's marginal."""
        g = self.plan.entities[entity].joint_groups[group]
        members = g["members"]
        z = np.vstack([self.rng.normal(f"{entity}.@joint.{group}#{i}", *self.keys(frame))
                       for i in range(len(members))])
        u = ndtr(g["cholesky"] @ z)
        u = np.clip(u, 1e-16, 1 - 1e-16)
        for i, col in enumerate(members):
            node = self.plan.entities[entity].nodes[col]
            frame.cols[col] = self.finish_draw(node, frame, frames, self.draw(node, frame, frames, u[i]))

    def run_reference(self, node: NodePlan, frame: Frame, frames) -> Col:
        target = node.spec["references"]
        tent = target.split(".")[0]
        n_target = len(next(iter(self.static[tent].values())))
        u = self.uniform(f"{node.full}#0", frame)
        if node.spec.get("weights"):
            wref = self.ref(node.spec["weights"])
            w = to_rule_type(self.static[tent][wref.column], "float", self.rounding)
            cdf = np.cumsum(np.maximum(w, 0))
            ords = np.searchsorted(cdf, u * cdf[-1], side="right")
            ords = np.minimum(ords, n_target - 1)
        else:
            ords = np.minimum((u * n_target).astype(np.int64), n_target - 1)
        mask = None
        if node.null_rate:
            mask = self.uniform(f"{node.full}#null", frame) < node.null_rate
        return Col(ords.astype(np.int64), node.ctype, mask, ref=target)

    # ------------------------------------------------------------------ #
    # rules
    # ------------------------------------------------------------------ #
    def run_rule(self, node: NodePlan, frame: Frame, frames) -> Col:
        rule = node.rule
        args: dict = {}
        out_mask = None
        for name, rtype in rule.inputs.items():
            col = self.resolve(node.rule_inputs[name], frame, frames)
            v = to_rule_type(col, rtype, self.rounding)
            if rule.null_aware:
                args[name] = (v, col.mask)
                continue
            args[name] = null_fill(v, col.mask)
            if col.mask is not None and name not in rule.nullable_inputs:
                out_mask = col.mask.copy() if out_mask is None else (out_mask | col.mask)
        for pname, (ptype, default) in rule.params.items():
            raw = node.spec.get("params", {}).get(pname, default)
            val = self._convert_param(self.param(raw, frame, frames), ptype)
            args[pname] = (np.asarray(val), None) if rule.null_aware else val

        if rule.kind == "sampler":
            res = rule.func(lambda i: self.uniform(f"{node.full}#{i}", frame), frame.n, **args)
        elif rule.kind == "group":
            parent = frames[self.plan.entities[frame.entity].parent]
            ctx = {"group": frame.parent_row, "seq": np.asarray(frame.sub), "n_groups": parent.n}
            res = rule.func(ctx, **args)
        else:
            res = rule.func(**args)
        if isinstance(res, tuple):
            vals, m = res
        else:
            vals, m = res, None
        vals = np.asarray(vals)
        if vals.ndim == 0 or vals.shape[0] != frame.n:
            vals = np.broadcast_to(vals, (frame.n,)).copy()
        if m is not None:
            m = np.broadcast_to(np.asarray(m, dtype=bool), (frame.n,))
            out_mask = m.copy() if out_mask is None else (out_mask | m)
        if node.labels is not None and out_mask is not None and out_mask.any():
            # NULL slots carry placeholder values; blank them before label encoding
            vals = vals.astype(object)
            vals[out_mask] = None
        storage = to_storage(vals, rule.output, node, self.rounding)
        if node.kind == "drawn":
            storage = bound_at_creation(storage, node)
            if node.null_rate:
                nm = self.uniform(f"{node.full}#null", frame) < node.null_rate
                out_mask = nm if out_mask is None else (out_mask | nm)
        if out_mask is not None and not out_mask.any():
            out_mask = None
        return Col(storage, node.ctype, out_mask, node.labels)

    def _convert_param(self, val, ptype: str):
        scale = type_scale(ptype)
        if scale is not None:
            return float_to_fixed(np.asarray(val, dtype=float), scale, self.rounding)
        if ptype == "int":
            return np.asarray(val).astype(np.int64)
        if ptype == "float":
            return np.asarray(val, dtype=float)
        if ptype == "bool":
            return np.asarray(val).astype(bool)
        return val                                  # list / any: passed through

    # ------------------------------------------------------------------ #
    # aggregates
    # ------------------------------------------------------------------ #
    def run_aggregate(self, node: NodePlan, frame: Frame, frames) -> Col:
        spec = node.spec
        child = frames.get(spec["over"])
        if child is None:
            child = self.expand(spec["over"], frames)
        fn = spec.get("fn", "sum")
        n_groups = frame.n
        valid = np.ones(child.n, dtype=bool)
        if spec.get("where"):
            expr = self._exprs.get(node.full)
            if expr is None:
                expr = self._exprs[node.full] = Expression(spec["where"])
            env = {name: child.cols[name].to_val() for name in expr.names}
            res = expr.evaluate(env, child.n)
            valid = np.asarray(res.v, dtype=bool)
            if res.m is not None:
                valid = valid & ~res.m
        src = None
        if spec.get("of"):
            src = child.cols[spec["of"]]
            if src.mask is not None:
                valid = valid & ~src.mask
        # child rows are contiguous per parent (expansion order) -> prefix sums
        bounds = np.searchsorted(child.parent_row, np.arange(n_groups + 1), side="left")
        starts, ends = bounds[:-1], bounds[1:]
        count = np.concatenate([[0], np.cumsum(valid, dtype=np.int64)])
        cnt = count[ends] - count[starts]
        if fn == "count":
            return Col(to_storage(cnt, "int", node, self.rounding), node.ctype)
        vals = src.values
        rep = self._rep_of(src)
        if fn in ("sum", "mean"):
            acc = np.concatenate([[0], np.cumsum(np.where(valid, vals, 0))])
            sums = acc[ends] - acc[starts]
            if fn == "sum":
                return Col(to_storage(sums, rep, node, self.rounding), node.ctype)
            fl = to_rule_type(Col(sums, src.ctype), "float", self.rounding)
            mean = np.where(cnt > 0, fl / np.maximum(cnt, 1), 0.0)
            mask = cnt == 0
            return Col(to_storage(mean, "float", node, self.rounding), node.ctype,
                       mask if mask.any() else None)
        # min / max over non-empty groups
        fill = np.iinfo(np.int64).max if fn == "min" else np.iinfo(np.int64).min
        if vals.dtype.kind == "f":
            fill = np.inf if fn == "min" else -np.inf
        arr = np.where(valid, vals, fill)
        out = np.full(n_groups, fill, dtype=arr.dtype)
        nonempty = ends > starts
        if nonempty.any() and arr.size:
            red = (np.minimum if fn == "min" else np.maximum).reduceat(arr, starts[nonempty])
            out[nonempty] = red
        mask = cnt == 0
        out = np.where(mask, 0, out)
        return Col(to_storage(out, rep, node, self.rounding), node.ctype, mask if mask.any() else None)

    @staticmethod
    def _rep_of(col: Col) -> str:
        if col.ctype.kind == "decimal":
            return f"fixed{col.ctype.scale}"
        if col.ctype.kind == "float":
            return "float"
        return "int"
