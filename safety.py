"""Preflight limits around the bundled compiler, not a public-service sandbox."""

import ast
import json
import math
import re

MAX_BYTES = 1_000_000


def parse_json(data):
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > MAX_BYTES:
        raise ValueError("Keep the contract under 1 MB.")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are not allowed: " + key)
            result[key] = value
        return result

    def bad_number(text):
        raise ValueError("Use finite JSON numbers, not " + text)

    try:
        value = json.loads(
            raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=bad_number
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(
            "Use a valid UTF-8 JSON object. Check quotes, commas and nesting."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("The contract must be one JSON object, not a list or placeholder text.")
    return value


def preflight(doc):
    # Count expansion BEFORE the generator resolves references, bounding shared-subtree expansion.
    visits = 0

    def pointer(ref):
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError("Only local $ref pointers such as #/parameters/x are allowed.")
        obj = doc
        for raw in ref[2:].split("/"):
            key = raw.replace("~1", "/").replace("~0", "~")
            try:
                if isinstance(obj, list):
                    if not key.isdigit():
                        raise ValueError("List references need a nonnegative index.")
                    obj = obj[int(key)]
                else:
                    obj = obj[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("A local reference does not exist: " + ref) from exc
        return obj

    def walk(value, depth=0, stack=()):
        nonlocal visits
        visits += 1
        if visits > 50_000 or depth > 40:
            raise ValueError(
                "This contract is too deeply nested or expands beyond the learning-app limit."
            )
        if isinstance(value, dict):
            if "$ref" in value:
                if set(value) != {"$ref"}:
                    raise ValueError("$ref objects cannot include extra keys.")
                ref = value["$ref"]
                if ref in stack:
                    raise ValueError("Circular $ref: remove the loop between referenced values.")
                return walk(pointer(ref), depth + 1, stack + (ref,))
            return {k: walk(v, depth + 1, stack) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) > 1000:
                raise ValueError("A contract list exceeds 1000 items; use a smaller definition.")
            return [walk(v, depth + 1, stack) for v in value]
        if isinstance(value, str) and len(value) > 6000:
            raise ValueError("A contract text field is too long; use a smaller definition.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or abs(value) > 1e15:
                raise ValueError("A number is non-finite or outside the learning-app size limit.")
        return value

    expanded = walk(doc)
    allowed = {
        "contract",
        "seed",
        "volume",
        "rule_library",
        "conventions",
        "parameters",
        "time",
        "shared",
        "entities",
        "rules",
        "invariants",
        "targets",
        "calibration",
        "output",
    }
    if set(expanded) - allowed:
        raise ValueError(
            "Unknown top-level fields. Upload a filled-in generator contract, not contract_structure.json or a Cardinal review-notes file."
        )
    if not isinstance(expanded.get("contract"), dict) or not isinstance(
        expanded["contract"].get("name"), str
    ):
        raise ValueError("contract.name must contain the name of a filled-in contract.")
    lib = expanded.get("rule_library", {})
    if not isinstance(lib, dict) or not isinstance(lib.get("packages", ["common"]), list):
        raise ValueError("rule_library.packages must be a list.")
    if any(pkg not in ("common", "credit_card") for pkg in lib.get("packages", ["common"])):
        raise ValueError("Only the bundled common and credit_card rule packages are allowed.")
    custom = expanded.get("rules", {})
    if not isinstance(custom, dict) or len(custom) > 30:
        raise ValueError("Use at most 30 custom rule definitions.")
    for name, rule in custom.items():
        if (
            not name.startswith("custom.")
            or not isinstance(rule, dict)
            or rule.get("override", False)
        ):
            raise ValueError(
                "Custom rules must use custom.* names and cannot override library rules."
            )
        tests = rule.get("tests", [])
        if not isinstance(tests, list) or not 1 <= len(tests) <= 30:
            raise ValueError("Every custom rule needs 1–30 test cases.")

    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("expression", "where", "of") and isinstance(item, str):
                    if len(item) > 2000:
                        raise ValueError("Keep each expression under 2000 characters.")
                    try:
                        nodes = list(ast.walk(ast.parse(item)))
                    except SyntaxError as exc:
                        raise ValueError(
                            "An expression is not valid syntax; check its formula or filter."
                        ) from exc
                    if len(nodes) > 250:
                        raise ValueError("An expression is too complex for this POC.")
                    # Synthgen applies its own exact language whitelist as well.
                    if any(
                        isinstance(
                            n,
                            (
                                ast.Import,
                                ast.ImportFrom,
                                ast.Attribute,
                                ast.Subscript,
                                ast.Lambda,
                                ast.ListComp,
                                ast.DictComp,
                                ast.GeneratorExp,
                                ast.For,
                                ast.While,
                                ast.Pow,
                            ),
                        )
                        for n in nodes
                    ):
                        raise ValueError(
                            "Expressions cannot use Python access, imports, loops or powers. Use the generator’s small expression language."
                        )
                if key == "pattern" and isinstance(item, str):
                    if len(item) > 120 or any(int(n) > 64 for n in re.findall(r"\{(\d+)\}", item)):
                        raise ValueError(
                            "Identifier patterns must be short, with repeats no larger than 64."
                        )
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(expanded)
    entities = expanded.get("entities", {})
    if not isinstance(entities, dict) or not 1 <= len(entities) <= 10:
        raise ValueError("A contract needs between 1 and 10 entities.")
    for name, entity in entities.items():
        if (
            not isinstance(entity, dict)
            or not isinstance(entity.get("nodes"), dict)
            or not entity["nodes"]
        ):
            raise ValueError(f"{name}: provide a non-empty nodes object.")
        if any(not isinstance(node, dict) for node in entity["nodes"].values()):
            raise ValueError(f"{name}: each column must be a node object.")
    if sum(len(e.get("nodes", {})) for e in entities.values() if isinstance(e, dict)) > 200:
        raise ValueError("Use at most 200 columns across this learning contract.")
    return expanded
