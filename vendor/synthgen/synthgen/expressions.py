"""
A small, safe, vectorised expression language.

Used for three things:
1. **Custom rules** declared in the contract (``"rules"`` block). These let a
   user add domain rules without touching Python, while still being *named,
   versioned and tested* (each custom rule carries test cases that are run
   at validation time - guide Step 7).
2. ``where`` filters of aggregate nodes (``"transaction_type in ['purchase']"``).
3. Row-level invariants and calibration metrics.

The source is parsed with Python's ``ast`` module and only a whitelist of node
types and functions is accepted - there is no ``eval``/``exec``, no attribute
access, no subscripts, no imports and no lambdas, so a contract cannot run
arbitrary code.

Semantics
---------
* Every operation is element-wise over numpy arrays.
* SQL-like NULLs: each value carries an optional null mask. Arithmetic and
  comparisons with a NULL operand yield NULL. ``x / 0`` yields NULL.
  ``coalesce``, ``isnull``, ``notnull`` and ``where`` handle NULLs explicitly.
* ``and`` / ``or`` / ``not`` are element-wise boolean operators.
* Categorical values compare against string literals by label.
* Multi-statement bodies are allowed: ``"r = a / b; where(r > 1, 1, r)"`` -
  the value of the final expression is the result.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .types import day_of_month, float_to_fixed, month_index, month_of_year


@dataclass
class Val:
    """A column of values plus an optional NULL mask and categorical labels."""

    v: np.ndarray
    m: Optional[np.ndarray] = None
    labels: Optional[tuple] = None   # if set, ``v`` holds integer codes

    def materialize(self) -> "Val":
        """Replace categorical codes with label strings (object array)."""
        if self.labels is None:
            return self
        lab = np.asarray(list(self.labels) + [None], dtype=object)
        codes = np.where(self.v < 0, len(self.labels), self.v)
        return Val(lab[codes], self.m)


def _or_masks(*masks):
    out = None
    for m in masks:
        if m is None:
            continue
        out = m.copy() if out is None else (out | m)
    return out


class ExpressionError(ValueError):
    pass


_BIN_OPS: dict[type, Callable] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}
_CMP_OPS: dict[type, Callable] = {
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
    ast.Lt: np.less, ast.LtE: np.less_equal,
    ast.Gt: np.greater, ast.GtE: np.greater_equal,
}
_ALLOWED_NODES = (
    ast.Module, ast.Expression, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.List, ast.Tuple, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.In, ast.NotIn, *_BIN_OPS.keys(), ast.Div, ast.FloorDiv, *_CMP_OPS.keys(),
)


class Expression:
    """A compiled expression. ``names`` lists the free variables it reads."""

    def __init__(self, source: str):
        self.source = source
        try:
            text = "\n".join(line.strip() for line in source.replace(";", "\n").splitlines())
            tree = ast.parse(text.strip(), mode="exec")
        except SyntaxError as e:
            raise ExpressionError(f"syntax error in expression {source!r}: {e.msg}") from None
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ExpressionError(
                    f"{type(node).__name__} is not allowed in expressions ({source!r})")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                    fname = getattr(node.func, "id", "?")
                    raise ExpressionError(f"unknown function {fname!r} in {source!r}")
                if node.keywords:
                    raise ExpressionError("keyword arguments are not supported")
        body = tree.body
        if not body or not isinstance(body[-1], ast.Expr):
            raise ExpressionError(f"expression {source!r} must end with an expression")
        for stmt in body[:-1]:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                raise ExpressionError("only simple 'name = expr' assignments are allowed")
        self._body = body
        assigned = {s.targets[0].id for s in body[:-1]}
        reads = {n.id for n in ast.walk(tree)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and n.id not in FUNCTIONS and n.id not in _CONSTANT_NAMES}
        self.names = sorted(reads - assigned)

    # ------------------------------------------------------------------ #
    def evaluate(self, env: dict[str, Val], n: int) -> Val:
        scope = dict(env)
        for stmt in self._body[:-1]:
            scope[stmt.targets[0].id] = self._eval(stmt.value, scope, n)
        out = self._eval(self._body[-1].value, scope, n)
        return _broadcast(out, n)

    # ------------------------------------------------------------------ #
    def _eval(self, node, scope, n) -> Val:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return Val(np.zeros(n), np.ones(n, dtype=bool))
            return Val(np.asarray(node.value))
        if isinstance(node, ast.Name):
            if node.id in _CONSTANT_NAMES:
                return _CONSTANT_NAMES[node.id](n)
            if node.id not in scope:
                raise ExpressionError(f"unknown name {node.id!r} in {self.source!r}")
            return scope[node.id]
        if isinstance(node, ast.UnaryOp):
            x = self._eval(node.operand, scope, n)
            if isinstance(node.op, ast.Not):
                return Val(np.logical_not(x.v), x.m)
            if isinstance(node.op, ast.USub):
                return Val(np.negative(x.v), x.m)
            return x
        if isinstance(node, ast.BinOp):
            a = self._eval(node.left, scope, n)
            b = self._eval(node.right, scope, n)
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                av = np.asarray(a.v, dtype=np.float64)
                bv = np.asarray(b.v, dtype=np.float64)
                zero = bv == 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    q = av / np.where(zero, 1.0, bv)
                    if isinstance(node.op, ast.FloorDiv):
                        q = np.floor(q)
                q = np.where(zero, 0.0, q)
                m = _or_masks(a.m, b.m, np.broadcast_to(zero, np.shape(q)) if np.any(zero) else None)
                return Val(q, m)
            return Val(_BIN_OPS[type(node.op)](a.v, b.v), _or_masks(a.m, b.m))
        if isinstance(node, ast.BoolOp):
            vals = [self._eval(v, scope, n) for v in node.values]
            fn = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            out = vals[0].v.astype(bool)
            for v in vals[1:]:
                out = fn(out, np.asarray(v.v).astype(bool))
            return Val(out, _or_masks(*[v.m for v in vals]))
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, scope, n)
            result, mask = None, None
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    if not isinstance(comp, (ast.List, ast.Tuple)):
                        raise ExpressionError("'in' requires a literal list, e.g. x in ['a','b']")
                    items = [c.value for c in comp.elts if isinstance(c, ast.Constant)]
                    r = _isin(left, items)
                    if isinstance(op, ast.NotIn):
                        r = ~r
                    right = None
                else:
                    right = self._eval(comp, scope, n)
                    r = _compare(left, right, _CMP_OPS[type(op)])
                    mask = _or_masks(mask, right.m)
                mask = _or_masks(mask, left.m)
                result = r if result is None else (result & r)
                if right is not None:
                    left = right
            return Val(result, mask)
        if isinstance(node, ast.IfExp):
            return FUNCTIONS["where"](self, n, self._eval(node.test, scope, n),
                                      self._eval(node.body, scope, n),
                                      self._eval(node.orelse, scope, n))
        if isinstance(node, ast.Call):
            args = [self._eval(a, scope, n) for a in node.args]
            return FUNCTIONS[node.func.id](self, n, *args)
        if isinstance(node, (ast.List, ast.Tuple)):
            raise ExpressionError("lists are only allowed on the right of 'in'")
        raise ExpressionError(f"unsupported syntax {type(node).__name__}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _broadcast(x: Val, n: int) -> Val:
    v = np.asarray(x.v)
    if v.ndim == 0 or v.shape[0] != n:
        v = np.broadcast_to(v, (n,)).copy()
    m = x.m
    if m is not None and (np.ndim(m) == 0 or np.shape(m)[0] != n):
        m = np.broadcast_to(m, (n,)).copy()
    return Val(v, m, x.labels)


def _isin(x: Val, items: list) -> np.ndarray:
    if x.labels is not None:
        codes = [x.labels.index(i) for i in items if i in x.labels]
        return np.isin(x.v, codes)
    return np.isin(x.v, items)


def _compare(a: Val, b: Val, fn) -> np.ndarray:
    # categorical vs string literal -> compare codes
    if a.labels is not None and b.labels is None and np.ndim(b.v) == 0 and isinstance(b.v.item(), str):
        lit = b.v.item()
        code = a.labels.index(lit) if lit in a.labels else -2
        return fn(a.v, code)
    if b.labels is not None and a.labels is None and np.ndim(a.v) == 0 and isinstance(a.v.item(), str):
        lit = a.v.item()
        code = b.labels.index(lit) if lit in b.labels else -2
        return fn(code, b.v)
    if a.labels is not None or b.labels is not None:
        a, b = a.materialize(), b.materialize()
    return fn(a.v, b.v)


def _num(x: Val) -> np.ndarray:
    return np.asarray(x.v, dtype=np.float64)


def _f_max(e, n, *args):
    out = _num(args[0])
    for a in args[1:]:
        out = np.maximum(out, _num(a))
    return Val(out, _or_masks(*[a.m for a in args]))


def _f_min(e, n, *args):
    out = _num(args[0])
    for a in args[1:]:
        out = np.minimum(out, _num(a))
    return Val(out, _or_masks(*[a.m for a in args]))


def _f_where(e, n, c, a, b):
    cond = np.asarray(c.v).astype(bool)
    if a.labels is not None or b.labels is not None:
        a, b = a.materialize(), b.materialize()
    v = np.where(cond, a.v, b.v)
    am = np.zeros(1, bool) if a.m is None else a.m
    bm = np.zeros(1, bool) if b.m is None else b.m
    m = np.where(cond, am, bm)
    if c.m is not None:
        m = m | c.m
    return Val(v, m if np.any(m) else None)


def _f_coalesce(e, n, *args):
    out = _broadcast(args[0], n)
    v, m = out.v.copy(), (np.zeros(n, bool) if out.m is None else out.m.copy())
    for a in args[1:]:
        if not m.any():
            break
        a = _broadcast(a, n)
        v = np.where(m, a.v, v)
        m = m & (np.zeros(n, bool) if a.m is None else a.m)
    return Val(v, m if m.any() else None)


def _f_round(mode):
    def fn(e, n, x, digits=None):
        d = 0 if digits is None else int(np.asarray(digits.v).item())
        q = float_to_fixed(_num(x), d, mode).astype(np.float64) / (10.0 ** d)
        return Val(q, x.m)
    return fn


def _unary(fn):
    return lambda e, n, x: Val(fn(_num(x)), x.m)


def _f_isnull(e, n, x):
    return Val(np.zeros(n, bool) if x.m is None else np.broadcast_to(x.m, (n,)).copy())


def _f_clip(e, n, x, lo, hi):
    return Val(np.clip(_num(x), _num(lo), _num(hi)), _or_masks(x.m, lo.m, hi.m))


FUNCTIONS: dict[str, Callable] = {
    "max": _f_max,
    "min": _f_min,
    "abs": _unary(np.abs),
    "floor": _unary(np.floor),
    "ceil": _unary(np.ceil),
    "sqrt": _unary(lambda x: np.sqrt(np.maximum(x, 0))),
    "log": _unary(lambda x: np.log(np.maximum(x, 1e-300))),
    "exp": _unary(np.exp),
    "round_half_up": _f_round("half_up"),
    "round_half_even": _f_round("half_even"),
    "clip": _f_clip,
    "where": _f_where,
    "if_else": _f_where,
    "coalesce": _f_coalesce,
    "isnull": _f_isnull,
    "notnull": lambda e, n, x: Val(~_f_isnull(e, n, x).v),
    "to_int": _unary(np.trunc),
    "to_float": _unary(lambda x: x),
    "month_of": lambda e, n, d: Val(month_of_year(month_index(d.v)), d.m),
    "day_of": lambda e, n, d: Val(day_of_month(d.v), d.m),
    "year_of": lambda e, n, d: Val(1970 + month_index(d.v) // 12, d.m),
}

_CONSTANT_NAMES: dict[str, Callable[[int], Val]] = {
    "NULL": lambda n: Val(np.zeros(n), np.ones(n, dtype=bool)),
    "TRUE": lambda n: Val(np.asarray(True)),
    "FALSE": lambda n: Val(np.asarray(False)),
}
