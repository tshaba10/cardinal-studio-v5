"""
Loading a contract: JSON parsing, ``$ref`` resolution and CLI overrides.

``$ref`` values are JSON pointers into the same document, e.g.::

    "params": {"$ref": "#/parameters/segment_profiles/fico"}

They let a contract define a value once (a segment profile table, a fee
schedule) and reference it from many nodes. References are resolved before
compilation, recursively, with cycle detection.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when a contract is invalid. ``problems`` lists every issue found."""

    def __init__(self, problems):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        super().__init__("contract is invalid:\n  - " + "\n  - ".join(self.problems))


def _pointer_get(doc: Any, pointer: str) -> Any:
    if not pointer.startswith("#/"):
        raise ContractError(f"$ref {pointer!r} must be a local JSON pointer like '#/parameters/x'")
    node = doc
    for raw in pointer[2:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(key)]
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            raise ContractError(f"$ref {pointer!r} does not resolve (missing {key!r})")
    return node


def resolve_refs(doc: Any) -> Any:
    """Return a copy of ``doc`` with every ``{"$ref": "#/..."}`` replaced."""
    root = doc

    def walk(node, stack):
        if isinstance(node, dict):
            if set(node) == {"$ref"}:
                ptr = node["$ref"]
                if ptr in stack:
                    raise ContractError(f"circular $ref: {' -> '.join(stack + [ptr])}")
                return walk(copy.deepcopy(_pointer_get(root, ptr)), stack + [ptr])
            return {k: walk(v, stack) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, stack) for v in node]
        return node

    return walk(doc, [])


def _parse_override_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(doc: dict, overrides: list[str]) -> dict:
    """Apply ``dotted.path=value`` overrides (value parsed as JSON when possible).

    Example: ``volume.account=1000``, ``time.output_steps=5``, ``seed=7``.
    """
    doc = copy.deepcopy(doc)
    for item in overrides or []:
        if "=" not in item:
            raise ContractError(f"override {item!r} must look like path.to.key=value")
        path, value = item.split("=", 1)
        keys = path.strip().split(".")
        node = doc
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = _parse_override_value(value.strip())
    return doc


def load_contract(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Read a contract file, apply overrides, and resolve references."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as e:
            raise ContractError(f"{path}: invalid JSON ({e})") from None
    raw = apply_overrides(raw, overrides or [])
    return resolve_refs(raw)
