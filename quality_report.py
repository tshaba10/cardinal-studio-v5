"""Read every emitted CSV row. Trusted checks supplement, never replace, synthgen.verify.

No LLM writes or executes this policy. Formulas cover emitted same-table common.add,
common.subtract and common.ratio nodes without bounds/quantization; coverage gaps are explicit.
SQLite holds exact keys on disk so relationship checking is independent of table order.
"""

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory

POLICY_VERSION = "cardinal-quality-v1"
ROUNDING = {
    "half_up": "ROUND_HALF_UP",
    "half_even": "ROUND_HALF_EVEN",
    "floor": "ROUND_FLOOR",
    "ceil": "ROUND_CEILING",
    "down": "ROUND_DOWN",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(contract):
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def download_allowed(result):
    # AI advice has intentionally no role in this decision.
    return (
        result.get("status") == "complete"
        and result.get("verification", {}).get("ok") is True
        and result.get("quality", {}).get("gate_passed") is True
    )


def numeric(text):
    value = Decimal(text)
    if not value.is_finite():
        raise ValueError("Non-finite numeric value")
    return value


def profile_add(profile, raw, kind, categorical=False):
    if raw == "":
        profile["missing"] += 1
        return
    profile["non_null"] += 1
    if kind in ("int", "decimal", "float"):
        value = numeric(raw)
        if kind == "float" and not math.isfinite(float(value)):
            raise ValueError("Non-finite float")
        profile["numeric_values"] = profile.get("numeric_values", 0) + 1
        profile["_sum"] = profile.get("_sum", Decimal(0)) + value
        profile["_min"] = min(profile.get("_min", value), value)
        profile["_max"] = max(profile.get("_max", value), value)
    elif kind in ("date", "timestamp"):
        value = date.fromisoformat(raw) if kind == "date" else datetime.fromisoformat(raw)
        text = value.isoformat()
        if text != raw:
            raise ValueError("Non-canonical ISO date/time")
        profile["min"] = min(profile.get("min", text), text)
        profile["max"] = max(profile.get("max", text), text)
    if categorical:
        counts = profile.setdefault("category_counts", {})
        if raw in counts or len(counts) < 50:
            counts[raw] = counts.get(raw, 0) + 1
        else:
            profile["other_category_rows"] = profile.get("other_category_rows", 0) + 1


def finish_profile(profile):
    if "_sum" in profile:
        profile["min"], profile["max"] = str(profile.pop("_min")), str(profile.pop("_max"))
        profile["mean"] = str(profile.pop("_sum") / profile["numeric_values"])
    return profile


def formula_spec(entity, col, nodes, emitted):
    node = nodes[col]
    args = {
        "common.add": ("a", "b"),
        "common.subtract": ("a", "b"),
        "common.ratio": ("numerator", "denominator"),
    }.get(node.spec.get("rule"))
    if not args or node.kind != "derived":
        return None
    if (
        node.spec.get("bounds")
        or node.spec.get("quantization")
        or node.ctype.kind not in ("int", "decimal", "float")
    ):
        return False
    refs = [node.spec.get("inputs", {}).get(arg, "") for arg in args]
    cols = [ref.removeprefix(entity + ".") for ref in refs]
    if any(
        ref != entity + "." + col
        or col not in emitted
        or nodes[col].ctype.kind not in ("int", "decimal", "float")
        for ref, col in zip(refs, cols)
    ):
        return False
    return cols


def formula_ok(node, inputs, raw, row, rounding):
    values = [row[c] for c in inputs]
    if "" in values:
        return raw == ""
    a, b = map(numeric, values)
    rule = node.spec["rule"]
    if rule == "common.ratio":
        if b == 0:
            return raw == ""
        if node.spec.get("params", {}).get("floor_numerator_at_zero", False):
            a = max(a, Decimal(0))
        expected = a / b
    else:
        # The bundled common arithmetic signatures use fixed6 inputs.
        unit = Decimal("0.000001")
        a, b = (v.quantize(unit, rounding=rounding) for v in (a, b))
        expected = a + b if rule == "common.add" else a - b
    if raw == "":
        return False
    actual = numeric(raw)
    if node.ctype.kind == "float":
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9)
    expected = expected.quantize(Decimal(1).scaleb(-node.ctype.scale), rounding=rounding)
    return actual == expected


def audit_dataset(plan, output, contract, run, verification, log=lambda message: None):
    """Return JSON-safe observations, explicit check coverage, and a fail-closed gate."""
    output = Path(output)
    checks, tables, manifest = [], {}, {}
    rounding = ROUNDING.get(plan.rounding, "ROUND_HALF_UP")
    entities = {
        name: ent
        for name, ent in plan.entities.items()
        if ent.emit and any(n.emit for n in ent.nodes.values())
    }
    columns = {name: [c for c, n in ent.nodes.items() if n.emit] for name, ent in entities.items()}
    key_specs, relationships = {}, []

    def add(check, table, violations=0, rows=None, detail="", examples=None, warning=False):
        checks.append(
            dict(
                check=check,
                table=table,
                violations=int(violations),
                rows_checked=rows,
                severity=("warning" if warning else "error") if violations else "ok",
                detail=detail,
                row_numbers=(examples or [])[:3],
                source=POLICY_VERSION,
            )
        )

    def key_spec(table, cols, unique=False):
        identity = (table, tuple(cols))
        key_specs[identity] = key_specs.get(identity, False) or unique
        return json.dumps(identity)

    for name, ent in entities.items():
        pk = ent.spec.get("primary_key") or [
            c for c, n in ent.nodes.items() if n.kind == "identifier"
        ]
        if pk and all(c in columns[name] for c in pk):
            key_spec(name, pk, True)
        for col in columns[name]:
            node = ent.nodes[col]
            if node.spec.get("unique"):
                key_spec(name, [col], True)
            if node.kind in ("parent_key", "reference"):
                target, target_col = node.spec["references"].split(".")
                relationships.append((name, [col], target, [target_col]))
        for fk in ent.spec.get("foreign_keys", []):
            relationships.append((name, fk["columns"], fk["references"], fk["ref_columns"]))

    active_fks = []
    for source, cols, target, target_cols in relationships:
        label = "foreign_key:" + "+".join(cols) + "->" + target
        if (
            target not in entities
            or not all(c in columns[source] for c in cols)
            or not all(c in columns[target] for c in target_cols)
        ):
            add(
                "coverage:" + label,
                source,
                1,
                detail="Not independently checked: required columns/table are not emitted.",
                warning=True,
            )
            continue
        key_spec(target, target_cols)
        active_fks.append((source, cols, target, target_cols))

    with (
        TemporaryDirectory(prefix="quality-", dir=output.parent) as temporary,
        closing(sqlite3.connect(str(Path(temporary) / "keys.sqlite"))) as db,
        localcontext() as context,
    ):
        context.prec = 60
        db.execute("CREATE TABLE keys (spec TEXT, value TEXT, PRIMARY KEY(spec,value))")
        db.execute("CREATE TABLE refs (relation INTEGER, value TEXT, row_no INTEGER)")

        def encode_key(row, cols, nodes):
            if any(row[c] == "" for c in cols):
                return None
            result = []
            for col in cols:
                text = row[col]
                if nodes[col].ctype.kind in ("int", "decimal", "float"):
                    value = numeric(text)
                    text = "0" if value == 0 else str(value.normalize())
                result.append(text)
            return json.dumps(result)

        for name, ent in entities.items():
            log("quality audit: " + name)
            path = output / (name + ".csv")
            if not path.is_file():
                add(
                    "file_present",
                    name,
                    1,
                    detail="Expected CSV is missing. Regenerate the dataset.",
                )
                continue
            manifest[name + ".csv"] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            profiles = {
                c: dict(type=str(ent.nodes[c].ctype), missing=0, non_null=0) for c in columns[name]
            }
            violations, examples = Counter(), {}
            registered = set()
            formulas = {}
            primary = ent.spec.get("primary_key") or [
                c for c, n in ent.nodes.items() if n.kind == "identifier"
            ]
            if primary and all(c in columns[name] for c in primary):
                registered.add("primary_key_not_null")
            for col in columns[name]:
                node = ent.nodes[col]
                registered.add("valid_type:" + col)
                if not node.nullable:
                    registered.add("not_null:" + col)
                if node.codec is not None:
                    registered.add("identifier_format:" + col)
                inputs = formula_spec(name, col, ent.nodes, columns[name])
                if inputs:
                    formulas[col] = inputs
                    registered.add("trusted_formula:" + col)
                elif node.kind in ("derived", "state", "aggregate", "lookup"):
                    add(
                        "coverage:formula:" + col,
                        name,
                        1,
                        detail="No additional independent formula check for this node; see configured generator invariants.",
                        warning=True,
                    )
            my_keys = [
                (cols, unique, json.dumps((table, cols)))
                for (table, cols), unique in key_specs.items()
                if table == name
            ]
            my_fks = [
                (i, cols) for i, (source, cols, _, _) in enumerate(active_fks) if source == name
            ]
            total = 0
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                add(
                    "header",
                    name,
                    int(reader.fieldnames != columns[name]),
                    detail="Expected: " + ", ".join(columns[name]),
                )
                if reader.fieldnames != columns[name]:
                    continue
                for total, row in enumerate(reader, 1):
                    if total % 100_000 == 0:
                        log(f"quality audit: {name}, {total:,} rows scanned")

                    def fail(key):
                        registered.add(key)
                        violations[key] += 1
                        examples.setdefault(key, [])
                        if len(examples[key]) < 3:
                            examples[key].append(total)

                    if None in row or any(v is None for v in row.values()):
                        fail("row_shape")
                        continue
                    if (
                        primary
                        and all(c in row for c in primary)
                        and any(row[c] == "" for c in primary)
                    ):
                        fail("primary_key_not_null")
                    for col in columns[name]:
                        node, raw = ent.nodes[col], row[col]
                        if raw == "" and not node.nullable:
                            fail("not_null:" + col)
                        try:
                            profile_add(
                                profiles[col],
                                raw,
                                node.ctype.kind,
                                node.labels is not None or node.spec.get("dist") == "categorical",
                            )
                            if raw and node.ctype.kind == "bool" and raw not in ("true", "false"):
                                raise ValueError("Invalid boolean")
                            if (
                                raw
                                and node.ctype.kind == "int"
                                and numeric(raw) != numeric(raw).to_integral_value()
                            ):
                                raise ValueError("Invalid integer")
                            if raw and node.ctype.kind == "int":
                                low, high = node.ctype.int_range
                                if not low <= numeric(raw) <= high:
                                    raise ValueError("Integer outside its declared width")
                            if raw and node.ctype.kind == "decimal":
                                value = numeric(raw)
                                unit = Decimal(1).scaleb(-node.ctype.scale)
                                if value != value.quantize(unit) or abs(value) >= Decimal(10) ** (
                                    node.ctype.precision - node.ctype.scale
                                ):
                                    raise ValueError(
                                        "Decimal outside its declared precision or scale"
                                    )
                        except (ValueError, InvalidOperation, OverflowError):
                            fail("valid_type:" + col)
                        if raw and node.codec is not None:
                            alphabets = node.codec.alphabets
                            if len(raw) != len(alphabets) or any(
                                ord(char) not in alphabet for char, alphabet in zip(raw, alphabets)
                            ):
                                fail("identifier_format:" + col)
                        if col in formulas:
                            try:
                                if not formula_ok(node, formulas[col], raw, row, rounding):
                                    fail("trusted_formula:" + col)
                            except (ValueError, InvalidOperation, OverflowError):
                                fail("trusted_formula:" + col)
                        if node.spec.get("dist") in ("categorical", "date_uniform") and raw:
                            params = node.spec.get("params") or node.spec.get("allowed_values", {})
                            by = node.spec.get("conditioned_on")
                            if by:
                                by_col = by.removeprefix(name + ".")
                                if by != name + "." + by_col or by_col not in row:
                                    registered.add("coverage:distribution:" + col)
                                    continue
                                params = params.get(row[by_col], params.get("default", {}))
                            label = "distribution_support:" + col
                            registered.add(label)
                            try:
                                if node.spec["dist"] == "date_uniform":
                                    valid = (
                                        date.fromisoformat(params["start"])
                                        <= date.fromisoformat(raw)
                                        <= date.fromisoformat(params["end"])
                                    )
                                elif node.ctype.kind in ("int", "decimal", "float"):
                                    valid = numeric(raw) in [
                                        numeric(str(v)) for v in params["values"]
                                    ]
                                else:
                                    valid = raw in [
                                        str(v).lower() if isinstance(v, bool) else str(v)
                                        for v in params["values"]
                                    ]
                                if not valid:
                                    fail(label)
                            except (KeyError, TypeError, ValueError, InvalidOperation):
                                fail(label)
                    for cols, unique, spec_id in my_keys:
                        label = "unique:" + "+".join(cols)
                        if unique:
                            registered.add(label)
                        try:
                            value = encode_key(row, cols, ent.nodes)
                        except (ValueError, InvalidOperation):
                            continue  # Already blocked by valid_type checks.
                        if value is None:
                            continue
                        cursor = db.execute(
                            "INSERT OR IGNORE INTO keys VALUES (?,?)", (spec_id, value)
                        )
                        if unique and cursor.rowcount == 0:
                            fail(label)
                    for relation, cols in my_fks:
                        try:
                            value = encode_key(row, cols, ent.nodes)
                        except (ValueError, InvalidOperation):
                            continue
                        if value is not None:
                            db.execute("INSERT INTO refs VALUES (?,?,?)", (relation, value, total))
            tables[name] = dict(
                rows=total,
                role=ent.role,
                columns={c: finish_profile(p) for c, p in profiles.items()},
            )
            log(f"quality audit: {name}, finished {total:,} rows")
            add(
                "recorded_row_count",
                name,
                int(total != run.get("rows", {}).get(name)),
                total,
                "CSV count compared with the generator run report.",
            )
            if ent.role == "static":
                add(
                    "requested_row_count",
                    name,
                    int(total != ent.count),
                    total,
                    f"Requested {ent.count}; read {total}.",
                )
            for label in sorted(registered):
                if label.startswith("coverage:"):
                    add(
                        label,
                        name,
                        1,
                        total,
                        "Cross-table distribution support not independently checked.",
                        warning=True,
                    )
                else:
                    add(label, name, violations[label], total, examples=examples.get(label))

        db.execute("CREATE INDEX refs_relation ON refs(relation)")
        for i, (source, cols, target, target_cols) in enumerate(active_fks):
            spec_id = json.dumps((target, tuple(target_cols)))
            query = "FROM refs r WHERE r.relation=? AND NOT EXISTS (SELECT 1 FROM keys k WHERE k.spec=? AND k.value=r.value)"
            count = db.execute("SELECT COUNT(*) " + query, (i, spec_id)).fetchone()[0]
            example = [
                r[0] for r in db.execute("SELECT row_no " + query + " LIMIT 3", (i, spec_id))
            ]
            add(
                "foreign_key:" + "+".join(cols) + "->" + target,
                source,
                count,
                tables.get(source, {}).get("rows"),
                examples=example,
            )

    checks = [dict(c, source="bundled_generator") for c in verification.get("checks", [])] + checks
    errors = sum(c["severity"] == "error" for c in checks)
    warnings = sum(c["severity"] == "warning" for c in checks)
    return dict(
        policy_version=POLICY_VERSION,
        contract_sha256=config_hash(contract),
        gate_passed=verification.get("ok") is True and errors == 0,
        errors=errors,
        warnings=warnings,
        tables=tables,
        checks=checks,
        files=manifest,
        limitations=[
            "Checks cover emitted CSVs and implemented/configured rules, not every natural-language requirement.",
            "Independent formulas cover supported same-table common.add/subtract/ratio nodes only; gaps are warnings.",
            "These independently implemented checks test config execution, not whether the designer chose the right business rule.",
            "Random proportions are observations, not exact quotas. Only explicit target tolerances are evaluated.",
            "Monthly business completeness requires approved invariants/targets; date ranges alone do not establish it.",
            "Passing these checks does not establish industry realism, representativeness, or privacy.",
            "Category summaries retain at most 50 labels per column; remaining rows are counted separately.",
        ],
    )
