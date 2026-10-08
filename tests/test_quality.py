import csv
import json
from pathlib import Path
from threading import Event
import pytest
from generation_worker import run
from quality_report import audit_dataset, download_allowed
from quality_review import evidence_for as build_evidence, quality_graph, QualityReviewRun

ROOT = Path(__file__).resolve().parents[1]


def review_brief(contract):
    return dict(
        name="demo",
        purpose="Fictional demo",
        seed=contract["seed"],
        volumes=[dict(entity="account", count=contract["volume"]["account"])],
        questions=[],
        assumptions=[],
        requirements=[dict(id="R1", text="Create 140 fictional accounts.", status="confirmed")],
    )


def evidence_for(quality):
    contract = json.loads((ROOT / "references/dependent_accounts.json").read_text())
    contract["volume"]["account"] = 140
    return build_evidence(quality, review_brief(contract), contract)


def answer(verdict="supported", refs=None):
    return dict(
        summary="Compared measured evidence with the approved request.",
        requirements=[
            dict(
                requirement_id="R1",
                verdict=verdict,
                evidence_ids=["T1"] if refs is None else refs,
                explanation="Measured table count is 140.",
                suggested_action="Inspect the report.",
            )
        ],
        limitations=["No raw rows reviewed; aggregate evidence only."],
    )


@pytest.fixture
def generated(tmp_path):
    # No API calls. Use the actual generator, then independently corrupt real CSVs below.
    contract = json.loads((ROOT / "references/dependent_accounts.json").read_text())
    contract["volume"]["account"] = 140
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    result = run(tmp_path)
    assert result["status"] == "complete", result
    (tmp_path / "result.json").write_text(json.dumps(result))
    from synthgen.contract.compiler import compile_contract

    return contract, compile_contract(contract), tmp_path, result


def mutate_csv(folder, mutate):
    path = folder / "data/account.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(generated):
    contract, plan, folder, result = generated
    # Deliberately give the audit a passing upstream report. It must detect corruption itself.
    return audit_dataset(plan, folder / "data", contract, result["run"], {"ok": True, "checks": []})


def test_quality_in_archive_and_report(generated):
    import zipfile

    _, _, folder, result = generated
    assert download_allowed(result)
    quality = result["quality"]
    assert quality["tables"]["account"]["rows"] == 140
    assert quality["tables"]["account"]["columns"]["opening_balance"]["numeric_values"] == 140
    assert any(c["check"] == "trusted_formula:available_credit" for c in quality["checks"])
    with zipfile.ZipFile(folder / "dataset.zip") as archive:
        assert json.loads(archive.read("quality_report.json")) == quality


def test_corruption_after_preview_rows_is_blocked(generated):
    mutate_csv(generated[2], lambda rows: rows[125].update(available_credit="999999.00"))
    quality = audit(generated)
    assert not quality["gate_passed"]
    failure = next(c for c in quality["checks"] if c["check"] == "trusted_formula:available_credit")
    assert failure["violations"] == 1 and failure["row_numbers"] == [126]


def test_actual_row_count_not_trusted_generator_counter(generated):
    mutate_csv(generated[2], lambda rows: rows.pop())
    quality = audit(generated)
    assert not quality["gate_passed"]
    assert quality["tables"]["account"]["rows"] == 139
    assert any(
        c["check"] == "requested_row_count" and c["severity"] == "error" for c in quality["checks"]
    )


def test_missing_duplicate_and_nonfinite(generated):
    def corrupt(rows):
        rows[1]["account_id"] = rows[0]["account_id"]
        rows[2]["opening_balance"] = ""
        rows[3]["credit_limit"] = "NaN"

    mutate_csv(generated[2], corrupt)
    errors = {c["check"] for c in audit(generated)["checks"] if c["severity"] == "error"}
    assert {"unique:account_id", "not_null:opening_balance", "valid_type:credit_limit"} <= errors


def test_conditional_support_not_just_union_of_values(generated):
    def corrupt(rows):
        row = rows[120]
        row["credit_limit"] = "5000.00" if row["segment"] == "standard" else "1000.00"

    mutate_csv(generated[2], corrupt)
    quality = audit(generated)
    assert any(
        c["check"] == "distribution_support:credit_limit" and c["violations"]
        for c in quality["checks"]
    )


def test_identifier_format_and_decimal_precision(generated):
    def corrupt(rows):
        rows[125]["account_id"] = "NOT-AN-ACCOUNT"
        rows[126]["opening_balance"] = "0.001"

    mutate_csv(generated[2], corrupt)
    errors = {c["check"] for c in audit(generated)["checks"] if c["severity"] == "error"}
    assert {"identifier_format:account_id", "valid_type:opening_balance"} <= errors


def test_relationship_checks_independent_of_table_order(tmp_path):
    from synthgen.contract.compiler import compile_contract

    contract = dict(
        contract={"name": "relationships"},
        seed=42,
        volume={"child": 2, "parent": 1},
        entities={
            "child": dict(
                role="static",
                primary_key=["id"],
                nodes={
                    "id": {
                        "kind": "identifier",
                        "type": "string",
                        "format": {"pattern": "C[0-9]{4}"},
                    },
                    "parent_id": {"kind": "reference", "type": "string", "references": "parent.id"},
                },
            ),
            "parent": dict(
                role="static",
                primary_key=["id"],
                nodes={
                    "id": {
                        "kind": "identifier",
                        "type": "string",
                        "format": {"pattern": "P[0-9]{4}"},
                    }
                },
            ),
        },
    )
    plan = compile_contract(contract)
    output = tmp_path / "data"
    output.mkdir()
    (output / "child.csv").write_text("id,parent_id\nC0001,P0001\nC0002,ORPHAN\n")
    (output / "parent.csv").write_text("id\nP0001\n")
    quality = audit_dataset(
        plan, output, contract, {"rows": {"child": 2, "parent": 1}}, {"ok": True, "checks": []}
    )
    check = next(c for c in quality["checks"] if c["check"] == "foreign_key:parent_id->parent")
    assert check["violations"] == 1 and check["row_numbers"] == [2]
    assert not quality["gate_passed"]


def test_optional_evidence_has_no_raw_rows_or_category_labels(generated):
    quality = generated[3]["quality"]
    quality["tables"]["account"]["columns"]["segment"]["category_counts"][
        "PRIVATE_CATEGORY_SENTINEL"
    ] = 1
    evidence = evidence_for(quality)
    serialized = json.dumps(evidence)
    assert "category_counts" not in serialized and "PRIVATE_CATEGORY_SENTINEL" not in serialized
    assert str(generated[2]) not in serialized
    assert evidence["approved_brief"]["requirements"][0]["id"] == "R1"
    assert evidence["scope"]["raw_rows_shared"] is False
    assert evidence["gate_passed"] is True


def test_ai_cannot_unlock_failed_gate(monkeypatch, generated):
    import quality_review

    quality = dict(generated[3]["quality"], gate_passed=False)
    evidence = evidence_for(quality)
    monkeypatch.setattr(quality_review, "ask", lambda *a, **k: (answer(), {}))
    result = quality_graph().invoke({"evidence": evidence})
    assert result["review"]["summary"]
    assert not download_allowed(
        dict(
            status="complete",
            verification={"ok": True},
            quality=quality,
            ai_review=result["review"],
        )
    )


def test_invalid_check_citation_is_rejected(monkeypatch, generated):
    import quality_review

    monkeypatch.setattr(quality_review, "ask", lambda *a, **k: (answer(refs=["C999999"]), {}))
    with pytest.raises(ValueError, match="not present"):
        quality_graph().invoke({"evidence": evidence_for(generated[3]["quality"])})


def test_ai_timeout_no_retry_and_python_gate_unchanged(monkeypatch, generated):
    import quality_review

    calls = []

    def failed(*a, **k):
        calls.append(1)
        raise ValueError("Claude timed out.")

    monkeypatch.setattr(quality_review, "ask", failed)
    run = QualityReviewRun(evidence_for(generated[3]["quality"]))
    run.thread.join(timeout=5)
    assert run.snapshot()["status"] == "failed" and len(calls) == 1
    assert download_allowed(generated[3])


def test_ai_stop_discards_late_answer(monkeypatch, generated):
    import quality_review

    entered, release = Event(), Event()

    def fake(*a, **k):
        entered.set()
        release.wait(timeout=5)
        return answer(), {}

    monkeypatch.setattr(quality_review, "ask", fake)
    run = QualityReviewRun(evidence_for(generated[3]["quality"]))
    assert entered.wait(timeout=5)
    run.stop()
    release.set()
    run.thread.join(timeout=5)
    assert run.snapshot()["status"] == "stopped"
    assert run.snapshot()["result"] == {}


def test_evidence_is_bounded_and_selection_disclosed(generated):
    quality = generated[3]["quality"]
    quality["checks"] *= 10
    evidence = evidence_for(quality)
    assert len(evidence["checks"]) == 120
    assert evidence["scope"]["total_checks"] > 120


def quality_app(generated):
    from streamlit.testing.v1 import AppTest
    from jobs import fingerprint

    class Finished:
        def poll(self):
            return 0

    contract, _, folder, _ = generated
    a = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
    a.session_state.result = {"contract": contract, "brief": review_brief(contract)}
    a.session_state.approved = fingerprint(contract)
    a.session_state.page = 2
    a.session_state.job = dict(folder=str(folder), process=Finished(), cancelled=False)
    return a.run(timeout=20)


def test_optional_ai_is_click_and_consent_only(monkeypatch, generated):
    import quality_review

    calls = []

    def fake(system, payload, model, role="quality_reviewer", **kw):
        calls.append(payload)
        assert payload["approved_brief"]["requirements"]
        return answer(), {"model": "mock", "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(quality_review, "ask", fake)
    a = quality_app(generated)
    assert not a.exception and not calls

    def action():
        return next(b for b in a.button if b.label == "Review dataset against my requirements")

    assert action().disabled
    a.checkbox(key="_consent_widget").check().run()
    assert action().disabled  # Separate permission for the aggregate evidence.
    next(c for c in a.checkbox if c.label.startswith("Share the approved")).check().run()
    assert not action().disabled and not calls
    action().click().run()
    a.session_state.quality_run.thread.join(timeout=5)
    a.run()
    assert not a.exception and len(calls) == 1
    assert a.session_state.quality_answer["status"] == "complete"
    assert any(c.label.startswith("I reviewed the preview") for c in a.checkbox)
    a.session_state.page = 3
    a.run()
    assert not a.exception
    assert "Optional dataset requirements review" in a.selectbox[0].options


def test_failed_quality_never_offers_dataset_acceptance(monkeypatch, generated):
    import quality_review

    result = generated[3]
    result.update(status="failed", error="Quality gate failed.")
    result["quality"]["gate_passed"] = False
    (generated[2] / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(quality_review, "ask", lambda *a, **kw: (answer(), {}))
    a = quality_app(generated)
    assert not a.exception
    assert not any(c.label.startswith("I reviewed the preview") for c in a.checkbox)
    assert any("blocked" in e.value for e in a.error)
    a.checkbox(key="_consent_widget").check().run()
    next(c for c in a.checkbox if c.label.startswith("Share the approved")).check().run()
    next(b for b in a.button if b.label == "Review dataset against my requirements").click().run()
    a.session_state.quality_run.thread.join(timeout=5)
    a.run()
    assert not a.exception
    assert not any(c.label.startswith("I reviewed the preview") for c in a.checkbox)


def test_every_requirement_must_be_reviewed(monkeypatch, generated):
    import quality_review

    incomplete = dict(summary="All fine", requirements=[], limitations=[])
    monkeypatch.setattr(quality_review, "ask", lambda *a, **k: (incomplete, {}))
    with pytest.raises(ValueError, match="omitted or invented"):
        quality_graph().invoke({"evidence": evidence_for(generated[3]["quality"])})


def test_insufficient_evidence_is_not_supported(monkeypatch, generated):
    import quality_review

    monkeypatch.setattr(
        quality_review, "ask", lambda *a, **k: (answer("insufficient_evidence", []), {})
    )
    value = quality_graph().invoke({"evidence": evidence_for(generated[3]["quality"])})
    assert value["review"]["requirements"][0]["verdict"] == "insufficient_evidence"


def test_wrong_contract_report_cannot_be_reviewed(generated):
    contract = dict(generated[0], seed=999)
    with pytest.raises(ValueError, match="different config"):
        build_evidence(generated[3]["quality"], review_brief(contract), contract)


def test_warning_cannot_be_cited_as_support(monkeypatch, generated):
    import quality_review

    evidence = evidence_for(generated[3]["quality"])
    evidence["checks"][0]["severity"] = "warning"
    monkeypatch.setattr(
        quality_review, "ask", lambda *a, **k: (answer(refs=[evidence["checks"][0]["id"]]), {})
    )
    with pytest.raises(ValueError, match="failed or incomplete"):
        quality_graph().invoke({"evidence": evidence})


def test_ai_cannot_defer_confirmed_requirement(monkeypatch, generated):
    import quality_review

    monkeypatch.setattr(quality_review, "ask", lambda *a, **k: (answer("deferred", []), {}))
    with pytest.raises(ValueError, match="deferral"):
        quality_graph().invoke({"evidence": evidence_for(generated[3]["quality"])})
