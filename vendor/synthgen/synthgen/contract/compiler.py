"""
Contract compiler and validator - guide Steps 1-4.

``compile_contract(raw_contract) -> Plan`` turns the JSON contract into an
executable plan and **rejects the contract before any data is generated** if
anything is wrong. All problems are collected and reported together.

Checks performed
----------------
* structure: entity roles, parents, volumes, time block
* every column has a valid type, kind and kind-specific fields
* every input exists (node / step context / shared series / row context)
* lags: only ``@t-1``, only on time-stepped entities, and every node read at a
  lag declares an ``init`` value
* every rule exists in the library (respecting version pins) and the node's
  inputs and params match the rule signature exactly
* drawn nodes are not also derived; NULL rates only on drawn/decorative nodes
  (derived NULLs must come from the rule - "derive nulls from the event")
* categorical weights are non-negative, match the values and sum to 1;
  conditioned parameters cover every category of the conditioning column
* bounds: min < max; quantization steps representable at the column scale
* joint groups: correlation matrices symmetric, unit-diagonal, positive definite
* identifier patterns have the capacity for the requested volume
* no same-step cycles (topological sort over non-lagged edges)
* custom rules compile and their embedded tests pass
* calibration targets are feasible and have a *mechanism* - a path in the
  graph from the tuned parameter's node to the target column
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Any, Optional

import numpy as np

from ..distributions import ALL_DISTS, CONTINUOUS
from ..expressions import Expression, ExpressionError
from ..identifiers import PatternCodec
from ..rules import REQUIRED, Registry, RuleSpec, load_packages, type_scale
from ..rules.custom import register_custom_rules
from ..types import (ROUNDING_MODES, ColumnType, coerce_scalar_to_storage, parse_date,
                     parse_type, step_to_units)
from .loader import ContractError

NODE_KINDS = {"identifier", "parent_key", "reference", "context", "lookup", "drawn",
              "decorative", "joint", "derived", "state", "aggregate"}
ROLES = {"static", "stepped", "event"}
STEP_CONTEXT = {
    "statement_date": "date", "cycle_start_date": "date", "days_in_cycle": "int",
    "cycle_number": "int", "months_on_book": "int", "step_index": "int",
    "month_of_year": "int",
}
ROW_CONTEXT = {"ordinal": "int", "seq": "int"}
AGG_FNS = {"sum", "count", "mean", "min", "max"}
_REF_RE = re.compile(r"^([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)(?:@t-(\d+))?$")


# --------------------------------------------------------------------------- #
# Plan data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Ref:
    """A reference to a value: ``entity.column``, ``entity.column@t-1``,
    ``step.<name>``, ``shared.<name>`` or ``row.<name>``."""

    raw: str
    ns: str            # node | step | shared | row
    entity: str        # entity name, or the namespace for step/shared/row
    column: str
    lag: int = 0

    @property
    def full(self) -> str:
        return f"{self.entity}.{self.column}"


@dataclass
class NodePlan:
    full: str
    entity: str
    column: str
    kind: str
    spec: dict
    ctype: ColumnType
    emit: bool = True
    nullable: bool = False
    null_rate: float = 0.0
    bounds: tuple = (None, None, None)       # (min, max, on_violation) in storage units
    quant: Optional[tuple] = None             # (step_units, mode)
    labels: Optional[tuple] = None            # categorical string labels
    numeric_values: Optional[np.ndarray] = None  # categorical over numeric values
    refs: list = field(default_factory=list)  # every input this node reads
    rule: Optional[RuleSpec] = None
    rule_inputs: dict = field(default_factory=dict)   # rule input name -> Ref
    init: Any = None                          # storage value for @t-1 at first step
    has_init: bool = False
    group: Optional[str] = None               # joint group name
    codec: Optional[PatternCodec] = None      # identifiers

    @property
    def is_categorical(self) -> bool:
        return self.labels is not None


@dataclass
class EntityPlan:
    name: str
    role: str
    spec: dict
    count: int = 0
    parent: Optional[str] = None
    root: Optional[str] = None                # static ancestor (for stepped / event)
    nodes: dict = field(default_factory=dict)  # column -> NodePlan
    count_ref: Optional[Ref] = None           # event entities: per-parent count node
    max_per_parent: int = 0
    terminal: Optional[Ref] = None
    joint_groups: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    emit: bool = True
    order_by: list = field(default_factory=list)


@dataclass
class Plan:
    raw: dict
    seed: int
    registry: Registry
    rounding: str
    entities: dict                 # name -> EntityPlan
    static_order: list             # execution units for phase 1
    stepped: dict                  # stepped entity -> execution units (incl. event children)
    time: dict
    shared: dict
    invariants: list
    targets: list
    output: dict
    lagged: dict                   # stepped entity -> list of columns carried as state
    dependents: dict               # node full -> set of nodes it (transitively) feeds

    def node(self, full: str) -> NodePlan:
        e, c = full.split(".", 1)
        return self.entities[e].nodes[c]


# --------------------------------------------------------------------------- #
# Compiler
# --------------------------------------------------------------------------- #

class _Compiler:
    def __init__(self, raw: dict):
        self.raw = raw
        self.problems: list[str] = []

    # ---- helpers ---------------------------------------------------------- #
    def err(self, msg: str) -> None:
        self.problems.append(msg)

    def parse_ref(self, text: Any, where: str) -> Optional[Ref]:
        if not isinstance(text, str):
            self.err(f"{where}: reference must be a string, got {text!r}")
            return None
        m = _REF_RE.match(text.strip())
        if not m:
            self.err(f"{where}: cannot parse reference {text!r} (expected entity.column[@t-1])")
            return None
        ent, col, lag = m.group(1), m.group(2), int(m.group(3) or 0)
        if ent == "step":
            ns = "step"
        elif ent == "shared":
            ns = "shared"
        elif ent == "row":
            ns = "row"
        else:
            ns = "node"
        if lag and ns != "node":
            self.err(f"{where}: lag is only allowed on node references ({text})")
        if lag > 1:
            self.err(f"{where}: only @t-1 lags are supported ({text})")
        return Ref(text, ns, ent, col, lag)

    # ---- entry ------------------------------------------------------------ #
    def compile(self) -> Plan:
        raw = self.raw
        seed = raw.get("seed")
        if not isinstance(seed, int):
            self.err("'seed' must be an integer")
            seed = 0
        conventions = raw.get("conventions", {})
        rounding = conventions.get("rounding", {}).get("mode", "half_up") \
            if isinstance(conventions.get("rounding"), dict) else conventions.get("rounding", "half_up")
        if rounding not in ROUNDING_MODES:
            self.err(f"conventions.rounding {rounding!r} must be one of {ROUNDING_MODES}")
            rounding = "half_up"
        self.rounding = rounding

        # rules library + custom rules (custom rule tests run here)
        lib = raw.get("rule_library", {})
        try:
            registry = load_packages(lib.get("packages", ["common"]))
        except ValueError as e:
            self.err(str(e))
            registry = Registry()
        for failure in register_custom_rules(registry, raw.get("rules", {})):
            self.err(f"custom rule: {failure}")
        self.registry = registry
        self.pins = lib.get("pins", {})

        entities = self._entities()
        self.entities = entities
        for ent in entities.values():
            for col, nspec in ent.spec.get("nodes", {}).items():
                node = self._node(ent, col, nspec)
                if node is not None:
                    ent.nodes[col] = node
        self._infer_labels()
        self._entity_links()
        for ent in entities.values():
            for node in ent.nodes.values():
                self._check_node(ent, node)
        self._check_joint_groups()

        time = self._time()
        shared = self._shared(time)
        lagged = self._lags()
        static_order = self._order([e for e in entities.values() if e.role == "static"], "static")
        stepped = {}
        for e in entities.values():
            if e.role == "stepped":
                stepped[e.name] = self._order([e] + [entities[c] for c in e.children], e.name)
        dependents = self._dependents()
        invariants = self._invariants()
        targets = self._targets(dependents)
        output = dict(raw.get("output", {}))
        output.setdefault("directory", "output")
        output.setdefault("chunk_size", 10000)
        output.setdefault("workers", 1)

        if self.problems:
            raise ContractError(self.problems)
        return Plan(raw=raw, seed=seed, registry=registry, rounding=rounding,
                    entities=entities, static_order=static_order, stepped=stepped,
                    time=time, shared=shared, invariants=invariants, targets=targets,
                    output=output, lagged=lagged, dependents=dependents)

    # ---- entities --------------------------------------------------------- #
    def _entities(self) -> dict:
        raw_entities = self.raw.get("entities")
        if not isinstance(raw_entities, dict) or not raw_entities:
            self.err("'entities' must be a non-empty object")
            return {}
        volume = self.raw.get("volume", {})
        out: dict[str, EntityPlan] = {}
        for name, spec in raw_entities.items():
            if name in ("step", "shared", "row"):
                self.err(f"entity name {name!r} is reserved")
                continue
            role = spec.get("role")
            if role not in ROLES:
                self.err(f"entity {name}: role must be one of {sorted(ROLES)}")
                continue
            ent = EntityPlan(name=name, role=role, spec=spec, parent=spec.get("parent"),
                             emit=spec.get("emit", True), order_by=spec.get("order_by", []))
            if role == "static":
                count = spec.get("count", volume.get(name))
                if not isinstance(count, int) or count < 1:
                    self.err(f"entity {name}: volume must be a positive integer "
                             f"(set volume.{name} or entities.{name}.count)")
                    count = 1
                ent.count = count
            out[name] = ent
        return out

    def _entity_links(self) -> None:
        for ent in self.entities.values():
            if ent.role == "static":
                if ent.parent:
                    self.err(f"entity {ent.name}: static entities cannot have a parent "
                             f"(use a 'reference' column for many-to-one links)")
                ent.root = ent.name
                continue
            parent = self.entities.get(ent.parent or "")
            if parent is None:
                self.err(f"entity {ent.name}: parent {ent.parent!r} does not exist")
                continue
            if ent.role == "stepped" and parent.role != "static":
                self.err(f"entity {ent.name}: a stepped entity's parent must be static")
            if ent.role == "event" and parent.role != "stepped":
                self.err(f"entity {ent.name}: an event entity's parent must be stepped")
            parent.children.append(ent.name)
            ent.root = parent.name if parent.role == "static" else parent.parent
            if ent.role == "event":
                ent.count_ref = self.parse_ref(ent.spec.get("count"), f"entity {ent.name}.count")
                if ent.count_ref is not None:
                    if ent.count_ref.entity != parent.name or ent.count_ref.lag:
                        self.err(f"entity {ent.name}: count must be a same-step column of "
                                 f"its parent {parent.name}")
                    elif ent.count_ref.column not in parent.nodes:
                        self.err(f"entity {ent.name}: count column {ent.count_ref.raw} missing")
                    elif parent.nodes[ent.count_ref.column].ctype.kind != "int":
                        self.err(f"entity {ent.name}: count column must be an integer")
                mpp = ent.spec.get("max_per_parent", 500)
                if not isinstance(mpp, int) or mpp < 1:
                    self.err(f"entity {ent.name}: max_per_parent must be a positive integer")
                    mpp = 500
                ent.max_per_parent = mpp
            if ent.role == "stepped" and ent.spec.get("terminal_when"):
                ent.terminal = self.parse_ref(ent.spec["terminal_when"], f"entity {ent.name}.terminal_when")
                if ent.terminal and (ent.terminal.entity != ent.name
                                     or ent.terminal.column not in ent.nodes):
                    self.err(f"entity {ent.name}: terminal_when must name one of its own columns")

    # ---- nodes ------------------------------------------------------------ #
    def _node(self, ent: EntityPlan, col: str, spec: dict) -> Optional[NodePlan]:
        full = f"{ent.name}.{col}"
        if not isinstance(spec, dict):
            self.err(f"{full}: node spec must be an object")
            return None
        kind = spec.get("kind")
        if kind not in NODE_KINDS:
            self.err(f"{full}: kind {kind!r} must be one of {sorted(NODE_KINDS)}")
            return None
        if kind == "decorative":                  # decorative == drawn without relationships
            kind = "drawn"
        try:
            ctype = parse_type(spec.get("type", "string"))
        except ValueError as e:
            self.err(f"{full}: {e}")
            return None
        node = NodePlan(full=full, entity=ent.name, column=col, kind=kind, spec=spec,
                        ctype=ctype, emit=spec.get("emit", True),
                        nullable=bool(spec.get("nullable", False)),
                        null_rate=float(spec.get("null_rate", 0) or 0))

        # bounds -> storage units
        b = spec.get("bounds")
        if b:
            lo = coerce_scalar_to_storage(b.get("min"), ctype, self.rounding) if b.get("min") is not None else None
            hi = coerce_scalar_to_storage(b.get("max"), ctype, self.rounding) if b.get("max") is not None else None
            mode = b.get("on_violation", "truncate" if kind in ("drawn", "joint") else "error")
            if mode not in ("truncate", "clip", "error", "warn", "reject"):
                self.err(f"{full}: bounds.on_violation {mode!r} is not supported")
            if mode == "reject":           # reject == redraw; inverse-CDF truncation is exact redraw
                mode = "truncate"
            if lo is not None and hi is not None and lo > hi:
                self.err(f"{full}: bounds.min must be <= bounds.max")
            node.bounds = (lo, hi, mode)

        q = spec.get("quantization")
        if q:
            step = q.get("round_to")
            qmode = q.get("mode", self.rounding)
            if qmode not in ROUNDING_MODES:
                self.err(f"{full}: quantization.mode must be one of {ROUNDING_MODES}")
            if ctype.kind not in ("decimal", "int"):
                self.err(f"{full}: quantization only applies to decimal/int columns")
            elif not isinstance(step, (int, float)) or step <= 0:
                self.err(f"{full}: quantization.round_to must be a positive number")
            else:
                try:
                    node.quant = (step_to_units(float(step), ctype.scale), qmode)
                except ValueError as e:
                    self.err(f"{full}: {e} (column type {ctype})")

        if "init" in spec:
            node.has_init = True
            try:
                node.init = coerce_scalar_to_storage(spec["init"], ctype, self.rounding)
            except (ValueError, TypeError):
                self.err(f"{full}: init value {spec['init']!r} is not valid for type {ctype}")
        if node.null_rate and kind not in ("drawn",):
            self.err(f"{full}: null_rate is only allowed on drawn/decorative columns; "
                     f"derived NULLs must come from the rule that computes them")
        if not 0 <= node.null_rate < 1:
            self.err(f"{full}: null_rate must be in [0, 1)")
        if node.null_rate and not node.nullable:
            self.err(f"{full}: null_rate > 0 requires nullable: true")
        return node

    def _infer_labels(self) -> None:
        """Determine categorical label sets (needed to store strings as codes)."""
        pending_lookups = []
        for ent in self.entities.values():
            for node in ent.nodes.values():
                spec = node.spec
                av = spec.get("allowed_values", {})
                values = None
                if node.kind in ("drawn", "joint") and spec.get("dist", "categorical" if av else None) == "categorical" \
                        and not spec.get("rule"):
                    values = self._categorical_values(node)
                elif av.get("values") is not None:
                    values = list(av["values"])
                if node.kind == "lookup":
                    pending_lookups.append(node)
                    continue
                if values is None:
                    continue
                if node.ctype.kind == "string":
                    if not all(isinstance(v, str) for v in values):
                        self.err(f"{node.full}: string categories must be strings")
                        continue
                    node.labels = tuple(dict.fromkeys(values))
                else:
                    try:
                        node.numeric_values = np.array(
                            [coerce_scalar_to_storage(v, node.ctype, self.rounding) for v in values])
                    except (TypeError, ValueError):
                        self.err(f"{node.full}: allowed values do not match type {node.ctype}")
        # lookups inherit labels from their source (may chain)
        for _ in range(len(pending_lookups) + 1):
            for node in pending_lookups:
                src = self._lookup_source(node)
                if src is not None:
                    node.labels = src.labels
                    node.numeric_values = src.numeric_values

    def _categorical_values(self, node: NodePlan) -> Optional[list]:
        spec = node.spec
        params = spec.get("params")
        av = spec.get("allowed_values", {})
        if spec.get("conditioned_on") and isinstance(params, dict):
            vals: list = []
            for k, p in params.items():
                if isinstance(p, dict) and "values" in p:
                    vals.extend(p["values"])
            return vals or av.get("values")
        if isinstance(params, dict) and "values" in params:
            return list(params["values"])
        return av.get("values")

    def _lookup_source(self, node: NodePlan) -> Optional[NodePlan]:
        src = node.spec.get("from")
        if not isinstance(src, str) or "." not in src:
            return None
        e, c = src.split(".", 1)
        return self.entities.get(e, EntityPlan("", "static", {})).nodes.get(c)

    # ---- per-kind validation --------------------------------------------- #
    def _add_ref(self, node: NodePlan, text, where: str) -> Optional[Ref]:
        ref = self.parse_ref(text, f"{node.full}.{where}")
        if ref is not None:
            node.refs.append(ref)
        return ref

    def _check_node(self, ent: EntityPlan, node: NodePlan) -> None:
        spec, kind, full = node.spec, node.kind, node.full
        for g in spec.get("given", []):          # documented extra dependencies
            self._add_ref(node, g, "given")

        if kind == "identifier":
            if node.ctype.kind != "string":
                self.err(f"{full}: identifiers must be strings")
            pattern = spec.get("format", {}).get("pattern")
            if not pattern:
                self.err(f"{full}: identifier needs format.pattern (e.g. 'ACC-[A-Z0-9]{{10}}')")
            else:
                try:
                    node.codec = PatternCodec(pattern, 0, full)
                except ValueError as e:
                    self.err(f"{full}: {e}")
            if ent.role == "stepped":
                self.err(f"{full}: stepped entities use composite keys (parent key + step)")
        elif kind in ("parent_key", "reference"):
            target = self.parse_ref(spec.get("references"), f"{full}.references")
            if target is not None:
                tent = self.entities.get(target.entity)
                tnode = tent.nodes.get(target.column) if tent else None
                if tnode is None or tnode.kind != "identifier":
                    self.err(f"{full}: references {target.raw} which is not an identifier column")
                elif kind == "parent_key" and target.entity not in self._ancestors(ent):
                    self.err(f"{full}: parent_key must reference an ancestor entity's identifier")
                elif kind == "reference" and tent.role != "static":
                    self.err(f"{full}: references may only target static entities")
                elif kind == "reference":
                    node.refs.append(target)       # dependency on the target table
            if kind == "reference" and spec.get("weights"):
                w = self._add_ref(node, spec["weights"], "weights")
                if w is not None and target is not None and w.entity != target.entity:
                    self.err(f"{full}: weights must be a column of {target.entity}")
        elif kind == "context":
            self._add_ref(node, spec.get("source"), "source")
        elif kind == "lookup":
            via = self._add_ref(node, spec.get("via"), "via")
            src = self._add_ref(node, spec.get("from"), "from")
            if via is not None and src is not None:
                vnode = self._resolve_node(via)
                if vnode is None or vnode.kind != "reference":
                    self.err(f"{full}: 'via' must be a reference column of the same entity")
                else:
                    tgt = vnode.spec.get("references", "").split(".")[0]
                    if src.entity != tgt:
                        self.err(f"{full}: 'from' must be a column of {tgt}")
        elif kind in ("drawn", "joint"):
            self._check_drawn(ent, node)
        elif kind in ("derived", "state"):
            self._check_rule_node(ent, node, expected_kinds=("rule", "group"))
            if kind == "state" and not node.has_init:
                self.err(f"{full}: state nodes need an 'init' value")
        elif kind == "aggregate":
            self._check_aggregate(ent, node)

        # validate each ref resolves to something that exists & is reachable
        for ref in node.refs:
            self._check_ref(ent, node, ref)

    def _resolve_node(self, ref: Ref) -> Optional[NodePlan]:
        ent = self.entities.get(ref.entity)
        return ent.nodes.get(ref.column) if ent else None

    def _ancestors(self, ent: EntityPlan) -> list[str]:
        out, cur = [], ent
        while cur is not None and cur.parent:
            out.append(cur.parent)
            cur = self.entities.get(cur.parent)
        return out

    def _check_ref(self, ent: EntityPlan, node: NodePlan, ref: Ref) -> None:
        where = f"{node.full}: input {ref.raw}"
        if ref.ns == "step":
            if ent.role == "static":
                self.err(f"{where}: static entities have no time step")
            elif ref.column not in STEP_CONTEXT:
                self.err(f"{where}: unknown step context (known: {sorted(STEP_CONTEXT)})")
            return
        if ref.ns == "shared":
            if ent.role == "static":
                self.err(f"{where}: shared series need a time step; static entities cannot read them")
            elif ref.column not in self.raw.get("shared", {}):
                self.err(f"{where}: shared series {ref.column!r} is not declared")
            return
        if ref.ns == "row":
            if ref.column not in ROW_CONTEXT:
                self.err(f"{where}: unknown row context (known: {sorted(ROW_CONTEXT)})")
            elif ref.column == "seq" and ent.role != "event":
                self.err(f"{where}: row.seq only exists on event entities")
            return
        target = self._resolve_node(ref)
        if target is None:
            self.err(f"{where}: no such column")
            return
        tent = self.entities[ref.entity]
        if ref.lag:
            if tent.role != "stepped":
                self.err(f"{where}: lags are only meaningful on stepped entities")
            if not target.has_init:
                self.err(f"{where}: {ref.full} is read at @t-1 so it must declare 'init'")
            if ent.role == "static":
                self.err(f"{where}: static entities cannot read time-lagged values")
        # visibility: own entity, ancestors, referenced static tables (via lookup)
        if ref.entity == ent.name:
            return
        if ref.entity in self._ancestors(ent):
            return
        if node.kind in ("lookup", "reference"):
            return
        if node.kind == "aggregate" and ref.entity in ent.children:
            return
        self.err(f"{where}: {ref.entity} is not visible from {ent.name} "
                 f"(use a reference + lookup column for non-ancestor entities)")

    def _check_drawn(self, ent: EntityPlan, node: NodePlan) -> None:
        spec, full = node.spec, node.full
        if node.kind == "joint":
            node.group = spec.get("group")
            if node.group not in ent.spec.get("joint_groups", {}):
                self.err(f"{full}: joint group {node.group!r} is not declared on {ent.name}")
        if spec.get("rule"):
            if node.kind == "joint":
                self.err(f"{full}: joint members must use a 'dist', not a rule")
            if spec.get("dist"):
                self.err(f"{full}: a drawn node has either a 'dist' or a sampler 'rule', not both")
            self._check_rule_node(ent, node, expected_kinds=("sampler",))
            return
        dist = spec.get("dist", "categorical" if spec.get("allowed_values") else None)
        if dist not in ALL_DISTS:
            self.err(f"{full}: dist {dist!r} must be one of {sorted(ALL_DISTS)}")
            return
        if node.kind == "joint" and dist not in CONTINUOUS | {"poisson", "categorical", "uniform_int"}:
            self.err(f"{full}: dist {dist!r} cannot be a joint (copula) member")
        cond = spec.get("conditioned_on")
        cond_ref = self._add_ref(node, cond, "conditioned_on") if cond else None
        params = spec.get("params", {})
        if dist == "categorical" and not params:
            params = spec.get("allowed_values", {})
        if cond_ref is not None:
            cnode = self._resolve_node(cond_ref) if cond_ref.ns == "node" else None
            if cnode is None or cnode.labels is None:
                self.err(f"{full}: conditioned_on must be a categorical (string) column")
            else:
                missing = [lab for lab in cnode.labels if lab not in params and "default" not in params]
                if missing:
                    self.err(f"{full}: params missing for categories {missing} of {cond_ref.raw}")
                for lab, p in params.items():
                    self._check_dist_params(node, dist, p, f"params[{lab!r}]")
        elif dist != "categorical" or spec.get("linear_predictor") is None:
            self._check_dist_params(node, dist, params, "params")
        # per-param dynamic references ({"input": ...} / {"by": ...})
        self._collect_param_refs(node, params)
        lp = spec.get("linear_predictor")
        if lp is not None:
            if dist not in ("bernoulli", "poisson", "normal"):
                self.err(f"{full}: linear_predictor is only supported for bernoulli/poisson/normal")
            self._collect_param_refs(node, {"intercept": lp.get("intercept", 0)})
            for i, term in enumerate(lp.get("terms", [])):
                self._add_ref(node, term.get("input"), f"linear_predictor.terms[{i}]")
                if term.get("transform", "identity") not in (
                        "identity", "gt0", "log", "log1p", "sqrt", "square"):
                    self.err(f"{full}: unknown term transform {term.get('transform')!r}")
        zm = spec.get("zero_mass")
        if zm is not None:
            self._collect_param_refs(node, {"zero_mass": zm})
        alloc = spec.get("allocation")
        if alloc not in (None, "random", "exact"):
            self.err(f"{full}: allocation must be 'random' or 'exact'")
        if alloc == "exact" and (ent.role != "static" or dist != "categorical" or cond):
            self.err(f"{full}: exact allocation needs an unconditioned categorical on a static entity")
        if dist in ("normal", "lognormal") and node.bounds[2] == "truncate":
            lo = node.bounds[0]
            if dist == "lognormal" and lo is not None and lo <= 0:
                self.err(f"{full}: lognormal lower bound must be > 0")

    def _collect_param_refs(self, node: NodePlan, params) -> None:
        def walk(v, path):
            if isinstance(v, dict):
                if "input" in v and len(v) <= 2:
                    self._add_ref(node, v["input"], path)
                    return
                if "by" in v and "values" in v:
                    r = self._add_ref(node, v["by"], path)
                    rn = self._resolve_node(r) if r is not None and r.ns == "node" else None
                    if r is not None and r.ns == "node" and (rn is None or rn.labels is None):
                        self.err(f"{node.full}.{path}: 'by' must be a categorical column")
                    elif rn is not None:
                        miss = [lab for lab in rn.labels if lab not in v["values"] and "default" not in v]
                        if miss:
                            self.err(f"{node.full}.{path}: missing values for {miss}")
                    return
                for k, x in v.items():
                    walk(x, f"{path}.{k}")
        walk(params, "params")

    def _check_dist_params(self, node, dist, p, where) -> None:
        full = node.full
        if not isinstance(p, dict):
            self.err(f"{full}.{where}: must be an object")
            return

        def need(*keys):
            for k in keys:
                if k not in p:
                    self.err(f"{full}.{where}: '{dist}' needs param {k!r}")

        if dist == "categorical":
            vals, w = p.get("values"), p.get("weights")
            if not isinstance(vals, list) or not vals:
                self.err(f"{full}.{where}: categorical needs a non-empty 'values' list")
                return
            if w is not None:
                if len(w) != len(vals):
                    self.err(f"{full}.{where}: weights and values differ in length")
                elif any((not isinstance(x, (int, float))) or x < 0 for x in w):
                    self.err(f"{full}.{where}: weights must be non-negative numbers")
                elif abs(sum(w) - 1.0) > 1e-6:
                    self.err(f"{full}.{where}: weights sum to {sum(w):.6f}, not 1")
        elif dist == "bernoulli":
            if "p" not in p and node.spec.get("linear_predictor") is None:
                self.err(f"{full}.{where}: bernoulli needs 'p' or a linear_predictor")
            elif isinstance(p.get("p"), (int, float)) and not 0 <= p["p"] <= 1:
                self.err(f"{full}.{where}: p must be in [0, 1]")
        elif dist == "normal":
            if node.spec.get("linear_predictor") is None:
                need("mean")
            need("std")
        elif dist == "lognormal":
            need("mu", "sigma")
        elif dist in ("uniform", "uniform_int"):
            need("low", "high")
        elif dist == "beta":
            need("a", "b")
        elif dist == "exponential":
            need("scale")
        elif dist == "poisson":
            if "lam" not in p and node.spec.get("linear_predictor") is None:
                self.err(f"{full}.{where}: poisson needs 'lam' or a linear_predictor")
        elif dist == "constant":
            need("value")
        elif dist == "date_uniform":
            need("start", "end")
            for k in ("start", "end"):
                if isinstance(p.get(k), str):
                    try:
                        parse_date(p[k])
                    except ValueError:
                        self.err(f"{full}.{where}: {k} is not a valid date")

    def _check_rule_node(self, ent: EntityPlan, node: NodePlan, expected_kinds) -> None:
        spec, full = node.spec, node.full
        name = spec.get("rule")
        if not name:
            self.err(f"{full}: {node.kind} nodes must name a library rule (free-text formulas "
                     f"are not allowed; add a custom rule in the 'rules' block instead)")
            return
        version = spec.get("version") or self.pins.get(name)
        try:
            rule = self.registry.resolve(name, version)
        except KeyError as e:
            self.err(f"{full}: {e.args[0]}")
            return
        if rule.kind not in expected_kinds:
            self.err(f"{full}: rule {rule.key} is a {rule.kind}; {node.kind} nodes need a "
                     f"{' or '.join(expected_kinds)}")
            return
        if rule.kind == "group" and ent.role != "event":
            self.err(f"{full}: group rule {rule.key} can only run on event entities")
        node.rule = rule
        inputs = spec.get("inputs", {})
        if isinstance(inputs, list):
            self.err(f"{full}: 'inputs' must map rule argument names to columns, "
                     f"e.g. {{\"balance\": \"account_cycle.balance\"}}; rule signature: "
                     f"{rule.signature()}")
            return
        missing = set(rule.inputs) - set(inputs)
        extra = set(inputs) - set(rule.inputs)
        if missing or extra:
            self.err(f"{full}: inputs do not match {rule.signature()} "
                     f"(missing {sorted(missing)}, unexpected {sorted(extra)})")
        for arg, text in inputs.items():
            ref = self._add_ref(node, text, f"inputs.{arg}")
            if ref is not None:
                node.rule_inputs[arg] = ref
        params = spec.get("params", {})
        unknown = set(params) - set(rule.params)
        if unknown:
            self.err(f"{full}: unknown params {sorted(unknown)} for {rule.signature()}")
        for p, (_, default) in rule.params.items():
            if default is REQUIRED and p not in params:
                self.err(f"{full}: required param {p!r} missing for {rule.key}")
        self._collect_param_refs(node, params)
        out_scale = type_scale(rule.output)
        if node.ctype.kind == "bool" and rule.output not in ("bool", "any"):
            self.err(f"{full}: rule {rule.key} returns {rule.output}, column is boolean")
        if out_scale is not None and node.ctype.kind not in ("decimal", "int", "float"):
            self.err(f"{full}: rule {rule.key} returns a number, column is {node.ctype}")

    def _check_aggregate(self, ent: EntityPlan, node: NodePlan) -> None:
        spec, full = node.spec, node.full
        over = spec.get("over")
        if over not in ent.children:
            self.err(f"{full}: 'over' must name a child event entity of {ent.name}")
            return
        child = self.entities[over]
        fn = spec.get("fn", "sum")
        if fn not in AGG_FNS:
            self.err(f"{full}: fn must be one of {sorted(AGG_FNS)}")
        of = spec.get("of")
        if fn != "count" and not of:
            self.err(f"{full}: '{fn}' needs 'of'")
        if of:
            self._add_ref(node, f"{over}.{of}", "of")
        where = spec.get("where")
        if where:
            try:
                expr = Expression(where)
            except ExpressionError as e:
                self.err(f"{full}: where: {e}")
                return
            for name in expr.names:
                if name not in child.nodes:
                    self.err(f"{full}: where refers to unknown {over} column {name!r}")
                else:
                    node.refs.append(Ref(f"{over}.{name}", "node", over, name))

    def _check_joint_groups(self) -> None:
        for ent in self.entities.values():
            for gname, g in ent.spec.get("joint_groups", {}).items():
                members = g.get("members", [])
                if g.get("method", "gaussian_copula") != "gaussian_copula":
                    self.err(f"{ent.name}.{gname}: only method 'gaussian_copula' is supported")
                for m in members:
                    n = ent.nodes.get(m)
                    if n is None or n.kind != "joint" or n.group != gname:
                        self.err(f"{ent.name}.{gname}: member {m!r} must be a joint node in this group")
                declared = [c for c, n in ent.nodes.items() if n.kind == "joint" and n.group == gname]
                if sorted(declared) != sorted(members):
                    self.err(f"{ent.name}.{gname}: members list must equal the joint nodes "
                             f"declaring this group ({declared})")
                corr = np.asarray(g.get("correlation", np.eye(len(members))), dtype=float)
                if corr.shape != (len(members), len(members)):
                    self.err(f"{ent.name}.{gname}: correlation must be {len(members)}x{len(members)}")
                    continue
                if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
                    self.err(f"{ent.name}.{gname}: correlation must be symmetric with unit diagonal")
                    continue
                try:
                    chol = np.linalg.cholesky(corr)
                except np.linalg.LinAlgError:
                    self.err(f"{ent.name}.{gname}: correlation matrix is not positive definite")
                    continue
                ent.joint_groups[gname] = {"members": members, "cholesky": chol}

    # ---- time & shared ---------------------------------------------------- #
    def _time(self) -> dict:
        stepped = [e for e in self.entities.values() if e.role == "stepped"]
        t = dict(self.raw.get("time", {}))
        if not stepped:
            return t
        if not t:
            self.err("a stepped entity requires a 'time' block")
            return t
        if t.get("calendar", "monthly_cycle_day") not in ("monthly_cycle_day", "monthly"):
            self.err("time.calendar must be 'monthly_cycle_day' or 'monthly'")
        try:
            parse_date(t.get("output_start", ""))
        except ValueError:
            self.err("time.output_start must be a YYYY-MM-DD date")
        steps = t.get("output_steps", self.raw.get("volume", {}).get("steps"))
        if not isinstance(steps, int) or steps < 1:
            self.err("time.output_steps (or volume.steps) must be a positive integer")
            steps = 1
        t["output_steps"] = steps
        wu = t.get("warmup_max_steps")
        if wu is not None and (not isinstance(wu, int) or wu < 0):
            self.err("time.warmup_max_steps must be a non-negative integer or null")
        for ent in stepped:
            anchors = t.get("anchors", {}).get(ent.name, t.get("anchor", {}))
            parent = self.entities.get(ent.parent or "")
            for key in ("start_date",) + (("cycle_day",) if t.get("calendar", "monthly_cycle_day") == "monthly_cycle_day" else ()):
                ref = self.parse_ref(anchors.get(key), f"time.anchor.{key}")
                if ref is None:
                    continue
                if parent is None or ref.entity != parent.name or ref.column not in parent.nodes:
                    self.err(f"time.anchor.{key} must be a column of {ent.parent}")
                    continue
                kind = parent.nodes[ref.column].ctype.kind
                if key == "start_date" and kind != "date":
                    self.err("time.anchor.start_date must be a date column")
                if key == "cycle_day":
                    if kind != "int":
                        self.err("time.anchor.cycle_day must be an integer column")
                    hi = parent.nodes[ref.column].bounds[1]
                    if hi is None or hi > 28:
                        self.err("time.anchor.cycle_day needs bounds.max <= 28 (avoids month-end gaps)")
            t.setdefault("anchors", {})[ent.name] = anchors
        return t

    def _shared(self, time: dict) -> dict:
        shared = self.raw.get("shared", {})
        for name, s in shared.items():
            proc = s.get("process")
            if proc not in ("ar1", "seasonal", "constant", "explicit"):
                self.err(f"shared.{name}: process must be ar1|seasonal|constant|explicit")
            p = s.get("params", {})
            if proc == "ar1":
                for k in ("mean", "phi", "sigma"):
                    if k not in p:
                        self.err(f"shared.{name}: ar1 needs {k!r}")
                if abs(p.get("phi", 0)) >= 1:
                    self.err(f"shared.{name}: |phi| must be < 1 for a stationary AR(1)")
            if proc == "seasonal" and len(p.get("by_month", [])) != 12:
                self.err(f"shared.{name}: seasonal needs 12 'by_month' factors")
            if proc == "explicit" and not isinstance(p.get("values"), dict):
                self.err(f"shared.{name}: explicit needs 'values' {{'YYYY-MM': x}}")
            for sh in s.get("shocks", []):
                try:
                    parse_date(sh["from"])
                    parse_date(sh["to"])
                except (KeyError, ValueError):
                    self.err(f"shared.{name}: shocks need valid 'from'/'to' dates")
        return shared

    def _lags(self) -> dict:
        out: dict[str, list] = {}
        for ent in self.entities.values():
            for node in ent.nodes.values():
                for ref in node.refs:
                    if ref.lag and ref.ns == "node":
                        cols = out.setdefault(ref.entity, [])
                        if ref.column not in cols:
                            cols.append(ref.column)
        return out

    # ---- ordering --------------------------------------------------------- #
    def _order(self, ents: list[EntityPlan], label: str) -> list[str]:
        """Topologically sort the nodes of these entities (same-step edges only)."""
        names = {n.full for e in ents for n in e.nodes.values()}
        graph: dict[str, set] = {}
        unit_of: dict[str, str] = {}
        for e in ents:
            for g in e.joint_groups:
                unit = f"joint:{e.name}:{g}"
                for m in e.joint_groups[g]["members"]:
                    unit_of[f"{e.name}.{m}"] = unit
        for e in ents:
            for n in e.nodes.values():
                u = unit_of.get(n.full, n.full)
                deps = graph.setdefault(u, set())
                for r in n.refs:
                    if r.ns == "node" and not r.lag and r.full in names:
                        du = unit_of.get(r.full, r.full)
                        if du != u:
                            deps.add(du)
                if e.role == "event" and e.count_ref is not None:
                    deps.add(e.count_ref.full)
        try:
            return list(TopologicalSorter(graph).static_order())
        except CycleError as err:
            self.err(f"same-step cycle in {label}: {' -> '.join(map(str, err.args[1]))}")
            return []

    def _dependents(self) -> dict:
        """Transitive 'feeds' relation over all edges (lagged included)."""
        feeds: dict[str, set] = {}
        for ent in self.entities.values():
            for n in ent.nodes.values():
                for r in n.refs:
                    if r.ns == "node":
                        feeds.setdefault(r.full, set()).add(n.full)
                if ent.role == "event" and ent.count_ref is not None:
                    feeds.setdefault(ent.count_ref.full, set()).add(n.full)
        closure: dict[str, set] = {}

        def reach(x):
            if x in closure:
                return closure[x]
            closure[x] = set()
            stack, seen = list(feeds.get(x, ())), set()
            while stack:
                y = stack.pop()
                if y in seen:
                    continue
                seen.add(y)
                stack.extend(feeds.get(y, ()))
            closure[x] = seen
            return seen

        for ent in self.entities.values():
            for n in ent.nodes.values():
                reach(n.full)
        return closure

    # ---- invariants & targets -------------------------------------------- #
    def _invariants(self) -> list:
        out = []
        for i, inv in enumerate(self.raw.get("invariants", [])):
            name = inv.get("name", f"invariant_{i}")
            typ = inv.get("type", "row")
            table = inv.get("table") or inv.get("parent")
            if table not in self.entities:
                self.err(f"invariant {name}: unknown table {table!r}")
                continue
            if typ == "row":
                try:
                    expr = Expression(inv["expression"])
                except (KeyError, ExpressionError) as e:
                    self.err(f"invariant {name}: {e}")
                    continue
                bad = [x for x in expr.names if x not in self.entities[table].nodes]
                if bad:
                    self.err(f"invariant {name}: unknown columns {bad}")
            elif typ in ("continuity", "monotonic"):
                for k in (["current", "previous"] if typ == "continuity" else ["column"]):
                    if inv.get(k) not in self.entities[table].nodes:
                        self.err(f"invariant {name}: {k} column missing")
            elif typ == "aggregate":
                child = inv.get("child")
                if child not in self.entities:
                    self.err(f"invariant {name}: unknown child {child!r}")
            else:
                self.err(f"invariant {name}: unknown type {typ!r}")
            out.append(dict(inv, name=name, type=typ))
        return out

    def _targets(self, dependents: dict) -> list:
        targets = self.raw.get("targets", [])
        shares: dict[tuple, list] = {}
        for t in targets:
            tid = t.get("id", "?")
            table = t.get("table")
            if table not in self.entities:
                self.err(f"target {tid}: unknown table {table!r}")
                continue
            metric = t.get("metric", "share")
            expr_text = t.get("where") if metric == "share" else t.get("of")
            try:
                expr = Expression(expr_text)
            except (ExpressionError, AttributeError, TypeError) as e:
                self.err(f"target {tid}: bad expression ({e})")
                continue
            cols = [f"{table}.{c}" for c in expr.names]
            for c in expr.names:
                if c not in self.entities[table].nodes:
                    self.err(f"target {tid}: unknown column {c!r}")
            if metric == "share":
                tv = t.get("target")
                if not isinstance(tv, (int, float)) or not 0 <= tv <= 1:
                    self.err(f"target {tid}: a share target must be in [0, 1]")
                m = re.match(r"^\s*(\w+)\s*>=\s*([-\d.]+)\s*$", expr_text or "")
                if m and isinstance(tv, (int, float)):
                    shares.setdefault((table, m.group(1)), []).append((float(m.group(2)), tv, tid))
            cal = t.get("calibrate")
            if cal:
                path = cal.get("parameter", "")
                pm = re.match(r"^/entities/([^/]+)/nodes/([^/]+)/", path)
                if not pm:
                    self.err(f"target {tid}: calibrate.parameter must point inside a node "
                             f"(/entities/<e>/nodes/<col>/...)")
                    continue
                pnode = f"{pm.group(1)}.{pm.group(2)}"
                reach = dependents.get(pnode, set()) | {pnode}
                if not any(c in reach for c in cols):
                    self.err(f"target {tid}: no mechanism - {pnode} does not feed "
                             f"{', '.join(cols)} so tuning it cannot move this target")
                lo, hi = cal.get("low"), cal.get("high")
                if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo < hi):
                    self.err(f"target {tid}: calibrate needs numeric low < high")
        # nested threshold targets must be ordered: share(x>=60) <= share(x>=30)
        for (table, col), items in shares.items():
            items.sort()
            for (t1, v1, id1), (t2, v2, id2) in zip(items, items[1:]):
                if v2 > v1 + 1e-12:
                    self.err(f"targets {id1} and {id2} are infeasible: share of {col} >= {t2} "
                             f"({v2}) exceeds share of {col} >= {t1} ({v1})")
        return list(targets)


def compile_contract(raw: dict) -> Plan:
    """Validate and compile a (``$ref``-resolved) contract dict."""
    plan = _Compiler(raw).compile()
    # identifier capacity checks need volumes, done after compile
    problems = []
    for ent in plan.entities.values():
        for node in ent.nodes.values():
            if node.kind == "identifier" and node.codec is not None and ent.role == "static":
                if ent.count > node.codec.domain:
                    problems.append(f"{node.full}: pattern holds {node.codec.domain} ids, "
                                    f"volume is {ent.count}")
    if problems:
        raise ContractError(problems)
    return plan


def describe_rules(registry: Registry) -> str:
    """Human readable listing of the rules library (``synthgen rules``)."""
    lines = []
    for spec in sorted(registry.all(), key=lambda s: (s.name, s.version)):
        lines.append(f"{spec.signature()}\n    [{spec.kind}, {spec.source}] {spec.description}")
    return "\n".join(lines)


__all__ = ["compile_contract", "Plan", "NodePlan", "EntityPlan", "Ref", "ContractError",
           "describe_rules", "STEP_CONTEXT"]
