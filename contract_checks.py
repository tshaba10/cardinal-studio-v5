"""Deterministic intent anchors, coverage pointers, and execution boundary checks."""

import json
import re
from safety import preflight
from models import Brief


def resolve_pointer(doc, pointer):
    if not pointer.startswith("/"):
        raise ValueError(
            "Coverage needs a JSON Pointer such as /entities/account/nodes/account_id."
        )
    value = doc
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list) and (
            not part.isdigit() or (len(part) > 1 and part.startswith("0"))
        ):
            raise ValueError(
                "JSON Pointer array indexes must be nonnegative integers without leading zeros."
            )
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def brief_errors(brief):
    b = Brief.model_validate(brief)
    errors = list(b.questions)
    if not b.name.strip() or not b.purpose.strip():
        errors.append("Confirm a dataset name and purpose in the chat.")
    if not b.volumes:
        errors.append(
            "Confirm the number of records for each static entity, for example 500 accounts."
        )
    if b.seed is None:
        errors.append("Confirm a random seed, for example 42.")
    if not any(r.status == "confirmed" for r in b.requirements):
        errors.append("Confirm the dataset requirements before designing.")
    if any(r.status == "proposed" for r in b.requirements):
        errors.append("Accept, revise or explicitly defer every proposed requirement in the chat.")
    return errors


def execution_check(contract):
    doc = preflight(contract)

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    for name, entity in doc["entities"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name.lower() in {
            "con",
            "prn",
            "aux",
            "nul",
            *("com" + str(i) for i in range(1, 10)),
            *("lpt" + str(i) for i in range(1, 10)),
        }:
            raise ValueError(
                "Use a safe snake_case entity name, not a path or Windows device name: " + name
            )
        for col in entity["nodes"]:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", col):
                raise ValueError("Use snake_case column names: " + str(col))
            node = entity["nodes"][col]
            refs = [
                node.get(key)
                for key in (
                    "inputs",
                    "conditioned_on",
                    "references",
                    "via",
                    "from",
                    "given",
                    "linear_predictor",
                )
            ]
            refs += (
                [v for v in node.get("params", {}).values() if isinstance(v, dict)]
                if isinstance(node.get("params", {}), dict)
                else []
            )
            if name + "." + col in set(strings(refs)):
                raise ValueError(
                    f"{name}.{col}: a column cannot depend on itself in the same step. Use a different input, or a supported @t-1 state reference with init."
                )
    if doc.get("calibration"):
        raise ValueError(
            "Automatic calibration is not executed by this application. Explicitly defer it or extend the generator integration."
        )
    if doc.get("output", {}).get("format", "csv") != "csv":
        raise ValueError("This release exports CSV tables in a ZIP. Ask for CSV output.")
    return doc


def intent_errors(contract, brief, design):
    errors = list(design.get("unresolved", []))
    b = Brief.model_validate(brief)
    try:
        doc = execution_check(contract)
        if doc.get("seed") != b.seed:
            errors.append("The contract seed differs from the confirmed seed.")
        expected = {v.entity: v.count for v in b.volumes}
        actual = {
            name: doc.get("volume", {}).get(name)
            for name, ent in doc["entities"].items()
            if ent.get("role", "static") == "static"
        }
        if actual != expected:
            errors.append(
                "Static entity volumes must exactly match the approved brief: "
                + json.dumps(expected)
            )
        required = {r.id for r in b.requirements if r.status == "confirmed"}
        mappings = design.get("coverage", [])
        ids = [m["requirement_id"] for m in mappings]
        if set(ids) != required or len(ids) != len(set(ids)):
            errors.append("Provide exactly one coverage entry for every confirmed requirement ID.")
        for entry in mappings:
            if not entry["pointers"] or not entry["explanation"].strip():
                errors.append(
                    entry["requirement_id"] + ": explain the implementation and give JSON Pointers."
                )
            for pointer in entry["pointers"]:
                try:
                    resolve_pointer(contract, pointer)
                except (KeyError, IndexError, ValueError, TypeError):
                    errors.append(
                        entry["requirement_id"] + ": missing contract location " + pointer
                    )
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        errors.append(str(exc))
    return errors
