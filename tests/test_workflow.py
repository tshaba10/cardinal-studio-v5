import workflow
from models import Design, Review
from graphs import svg, markdown


def test_design_graph_order(monkeypatch, brief, design):
    calls = []

    def fake(system, payload, model, role="interviewer"):
        calls.append(role)
        answer = (
            design
            if model is Design
            else dict(summary="Reviewed", missing_requirements=[], concerns=[])
        )
        return answer, {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    result = workflow.design_graph().invoke(dict(brief=brief, attempt=0))
    assert result["trace"] == ["designer", "validator", "reviewer"]
    assert result["report"]["ok"]
    assert calls == ["designer", "reviewer"]


def test_repair_bounded(monkeypatch, brief, design):
    monkeypatch.setattr(workflow, "ask", lambda *a, **kw: (design, {}))
    monkeypatch.setattr(
        workflow, "validate_contract", lambda c, **kw: {"ok": False, "errors": ["bad node"]}
    )
    result = workflow.design_graph().invoke(dict(brief=brief, attempt=0))
    assert result["attempt"] == 2
    assert result["trace"] == ["designer", "validator"] * 2
    assert not result.get("review")


def test_unresolved_decision_does_not_loop(monkeypatch, brief, design):
    monkeypatch.setattr(
        workflow,
        "ask",
        lambda *a, **kw: (dict(design, unresolved=["Choose the amount units."]), {}),
    )
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    result = workflow.design_graph().invoke(dict(brief=brief))
    assert result["attempt"] == 1
    assert not result["report"]["ok"]


def test_review_can_use_only_remaining_repair(monkeypatch, brief, design):
    def fake(system, payload, model, role="interviewer"):
        if model is Design:
            return design, {}
        return dict(
            summary="Wrong units", missing_requirements=[], concerns=["Units mismatch"], warnings=[]
        ), {}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    result = workflow.design_graph().invoke(dict(brief=brief))
    assert result["attempt"] == 2
    assert result["trace"] == ["designer", "validator", "reviewer"] * 2
    assert result["review"]["concerns"]


def test_advice_does_not_trigger_repair(monkeypatch, brief, design):
    def fake(system, payload, model, role="interviewer"):
        return (
            design
            if model is Design
            else dict(
                summary="OK",
                missing_requirements=[],
                concerns=[],
                warnings=["Illustrative assumptions, not industry data"],
            )
        ), {}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    result = workflow.design_graph().invoke(dict(brief=brief))
    assert result["attempt"] == 1


def test_invalid_contract_can_repair(monkeypatch, brief, design):
    count = 0

    def fake(system, payload, model, role="interviewer"):
        nonlocal count
        count += 1
        if model is Review:
            return dict(summary="OK", missing_requirements=[], concerns=[]), {}
        if count == 1:
            return dict(design, contract_json="not JSON"), {}
        assert payload["errors"]
        return design, {}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    assert workflow.design_graph().invoke(dict(brief=brief))["attempt"] == 2


def test_interview_receives_history_and_review_feedback(monkeypatch, brief):
    def fake(system, payload, model, role="interviewer"):
        assert payload["messages"][0]["content"] == "hello"
        assert payload["feedback"]["concerns"] == ["wrong dates"]
        return dict(reply="Let us correct the dates", brief=brief), {}

    monkeypatch.setattr(workflow, "ask", fake)
    result = workflow.interview_graph().invoke(
        dict(
            brief=brief,
            messages=[{"role": "user", "content": "hello"}],
            feedback={"concerns": ["wrong dates"]},
        )
    )
    assert result["reply"]


def test_offline_graphs_have_real_nodes():
    graph = workflow.design_graph()
    diagram = svg(graph)
    assert "designer" in diagram and "reviewer" in diagram and "repair" in diagram
    assert "validator" in markdown({"Design": graph})


def test_designer_runtime_notes_do_not_block_but_reach_reviewer(monkeypatch, brief, design):
    notes = [
        "Realised sample proportions may differ from the approved probabilities.",
        "The hidden helper is not emitted; verify calculated output after generation.",
    ]

    def fake(system, payload, model, role="interviewer"):
        if model is Design:
            return dict(design, warnings=notes, suggested_next_steps=[]), {}
        assert payload["designer_warnings"] == notes
        return dict(
            summary="Approved meaning retained",
            missing_requirements=[],
            concerns=[],
            warnings=notes,
        ), {}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    result = workflow.design_graph().invoke(dict(brief=brief))
    assert result["report"]["ok"] and result["attempt"] == 1
    assert result["trace"] == ["designer", "validator", "reviewer"]


def test_real_sequential_id_choice_remains_blocking(monkeypatch, brief, design):
    question = "R1 requires sequential IDs. May I use unique random IDs instead, or keep strict sequencing blocked?"
    monkeypatch.setattr(
        workflow,
        "ask",
        lambda *a, **kw: (
            dict(
                design,
                unresolved=[question],
                warnings=["Check output at runtime"],
                suggested_next_steps=["Confirm whether ordering is essential."],
            ),
            {},
        ),
    )
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    result = workflow.design_graph().invoke(dict(brief=brief))
    assert not result["report"]["ok"] and question in result["report"]["errors"]
    assert result["trace"] == ["designer", "validator"]


def test_warning_cannot_hide_requirement_mismatch(monkeypatch, brief, design):
    def fake(system, payload, model, role="interviewer"):
        if model is Design:
            return dict(design, warnings=["Sequential IDs will be checked later."]), {}
        return dict(
            summary="Not implemented",
            missing_requirements=[],
            concerns=["R1: ID pattern does not implement the approved sequential order."],
            warnings=[],
        ), {}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: {"ok": True, "errors": []})
    result = workflow.design_graph().invoke(dict(brief=brief))
    assert result["attempt"] == 2 and result["review"]["concerns"]


def test_design_feedback_schema_defaults_and_explicit_advisories(design):
    minimal = Design.model_validate(design)
    assert minimal.warnings == [] and minimal.suggested_next_steps == []
    updated = Design.model_validate(
        dict(
            design,
            warnings=["Sampling variation"],
            suggested_next_steps=["Please choose the currency."],
        )
    )
    assert updated.warnings and updated.suggested_next_steps
