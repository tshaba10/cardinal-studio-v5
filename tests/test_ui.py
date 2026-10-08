from pathlib import Path
import streamlit.testing.v1 as testing

ROOT = Path(__file__).resolve().parents[1]


def app():
    return testing.AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)


def button(a, label):
    return next(b for b in a.button if b.label == label)


def test_start_is_chat_first():
    a = app()
    assert not a.exception
    assert len(a.chat_input) == 1
    assert a.radio[0].value == "quick"
    assert not any("ready-made" in b.label or "existing generator" in b.label for b in a.button)
    assert any("Diagrams & session notes" in b.label for b in a.button)


def test_chat_requires_consent():
    a = app()
    assert a.chat_input[0].disabled


def test_generation_activity_and_stop_follow_user_between_pages(monkeypatch):
    import jobs

    job = dict(folder="mock-generation", cancelled=False)
    monkeypatch.setattr(
        jobs,
        "job_status",
        lambda j: (
            dict(status="cancelled", error="Stopped for test")
            if j["cancelled"]
            else dict(status="running")
        ),
    )
    monkeypatch.setattr(jobs, "progress", lambda j: "Checking 250 account rows")
    monkeypatch.setattr(jobs, "cancel_job", lambda j: j.update(cancelled=True))
    a = app()
    a.session_state.job = job
    for page in (0, 1, 2, 3):
        a.session_state.page = page
        a.run()
        assert not a.exception
        assert len([b for b in a.button if b.label == "Stop generation"]) == 1
        assert any("Generating and checking" in i.value for i in a.info)
        assert any("position:fixed" in m.value for m in a.markdown)
    button(a, "Stop generation").click().run()
    assert not a.exception
    assert a.session_state.job["cancelled"]
    assert not any(b.label == "Stop generation" for b in a.button)
    assert not any("position:fixed" in m.value for m in a.markdown)


def test_quality_stop_available_outside_export_page():
    class Explaining:
        from design_runner import Control

        control = Control()
        evidence = {"report_sha256": "demo"}
        stopped = False

        def alive(self):
            return True

        def snapshot(self):
            return dict(
                status="stopped" if self.stopped else "running",
                error="Stopped" if self.stopped else None,
                result={},
                **self.control.snapshot(),
            )

        def stop(self):
            self.stopped = True

    a = app()
    a.session_state.quality_run = Explaining()
    a.session_state.page = 3
    a.run()
    assert not a.exception
    button(a, "Stop dataset review").click().run()
    assert not a.exception
    assert a.session_state.quality_run.stopped
    assert button(a, "Stop dataset review").disabled


def test_prepare_action_follows_latest_message_in_long_chat(brief):
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.session_state.brief = brief
    a.session_state.messages = [
        dict(role="user" if i % 2 == 0 else "assistant", content=f"Demo message {i}")
        for i in range(40)
    ]
    a.run()
    assert not a.exception
    children = list(a.main.children.values())
    latest_message = max(i for i, element in enumerate(children) if element.type == "chat_message")
    requirements = next(
        i
        for i, element in enumerate(children)
        if element.type == "expander" and element.label == "Your captured requirements"
    )
    action = next(
        i
        for i, element in enumerate(children)
        if element.type == "button" and element.label == "Prepare my design"
    )
    assert latest_message < requirements < action
    assert not button(a, "Prepare my design").disabled
    assert len([b for b in a.button if b.label == "Prepare my design"]) == 1
    assert any("This uses paid AI calls" in c.value for c in a.caption)


def test_bottom_prepare_action_keeps_readiness_and_consent_checks(brief):
    a = app()
    a.session_state.brief = brief
    a.run()
    assert button(a, "Prepare my design").disabled
    a.checkbox(key="_consent_widget").check().run()
    assert not button(a, "Prepare my design").disabled
    a.session_state.brief = dict(brief, questions=["Which currency should amounts use?"])
    a.run()
    assert button(a, "Prepare my design").disabled
    assert any("what is missing" in c.value for c in a.caption)
    assert not a.chat_input[0].disabled


def test_export_requires_approval():
    a = app()
    a.session_state.page = 2
    a.run()
    assert not a.exception
    assert any("approve" in info.value for info in a.info)


def test_consent_survives_navigation():
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.session_state.page = 3
    a.run()
    a.session_state.page = 0
    a.run()
    assert a.session_state.consent and not a.chat_input[0].disabled


def test_chat_and_design_end_to_end(monkeypatch, brief, design):
    import workflow
    from models import Interview, Design

    def fake(system, payload, model, role="interviewer", **kw):
        if model is Interview:
            answer = dict(reply="Your requirements are ready for review.", brief=brief)
        elif model is Design:
            answer = design
        else:
            answer = dict(
                summary="Reviewed against all requirements.",
                missing_requirements=[],
                concerns=[],
                warnings=["Illustrative data only"],
            )
        return answer, dict(model="mock", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(workflow, "ask", fake)
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.chat_input[0].set_value("Create the fictional demo").run(timeout=20)
    a.session_state.interview_run.thread.join(timeout=5)
    a.run()
    assert not a.exception
    assert a.session_state.brief == brief and len(a.session_state.messages) == 2
    button(a, "Prepare my design").click().run(timeout=30)
    run = a.session_state.design_run
    run.thread.join(timeout=30)
    assert not run.alive()
    a.run()
    assert not a.exception
    assert a.session_state.page == 1 and a.session_state.result["report"]["ok"]
    next(c for c in a.checkbox if c.label.startswith("I reviewed the design")).check().run()
    button(a, "Approve and continue").click().run()
    assert not a.exception
    assert a.session_state.page == 2 and a.session_state.approved
    a.session_state.page = 0
    a.run()
    a.chat_input[0].set_value("Review this same design again").run(timeout=20)
    a.session_state.interview_run.thread.join(timeout=5)
    a.run()
    assert a.session_state.approved is None and a.session_state.result is None


def test_reviewer_concern_blocks_approval(brief, design, contract):
    a = app()
    a.session_state.result = dict(
        design=design,
        contract=contract,
        report={"ok": True, "errors": []},
        review={
            "summary": "Check units",
            "missing_requirements": [],
            "concerns": ["Utilization units do not match."],
        },
    )
    a.session_state.page = 1
    a.run()
    next(c for c in a.checkbox if c.label.startswith("I reviewed the design")).check().run()
    assert button(a, "Approve and continue").disabled


def test_notes_work_with_failed_partial_run():
    a = app()
    a.session_state.result = {"attempt": 0, "run_status": "failed", "run_error": "Claude HTTP 429"}
    a.session_state.page = 1
    a.run()
    assert not a.exception
    assert button(a, "Approve and continue").disabled
    a.session_state.page = 3
    a.run()
    assert not a.exception


def test_failed_build_preserves_chat_and_diagnostics(monkeypatch, brief, design):
    import workflow

    monkeypatch.setattr(workflow, "ask", lambda *args, **kw: (design, {}))
    monkeypatch.setattr(
        workflow,
        "validate_contract",
        lambda c, **kw: {"ok": False, "errors": ["account.x: unknown rule"]},
    )
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.session_state.brief = brief
    a.run()
    button(a, "Prepare my design").click().run()
    a.session_state.design_run.thread.join(timeout=20)
    a.run()
    assert not a.exception
    assert a.session_state.page == 0
    assert a.session_state.brief == brief
    assert a.session_state.result["attempt"] == 2
    assert a.session_state.run_history[0]["attempts"][-1]["errors"]
    assert not a.chat_input[0].disabled


def test_failed_design_feedback_follows_history_and_precedes_prepare(brief, design):
    a = app()
    a.session_state.brief = brief
    a.session_state.messages = [
        dict(role="assistant", content=f"Historical reply {i}") for i in range(30)
    ]
    a.session_state.result = dict(
        run_status="complete",
        report=dict(ok=False, errors=["R1 needs a decision"]),
        design=dict(
            design,
            unresolved=["R1: May I use unique random IDs rather than sequential IDs?"],
            warnings=["Sampling varies"],
            suggested_next_steps=["Reply: Keep strict sequential IDs."],
        ),
    )
    a.run()
    assert not a.exception
    children = list(a.main.children.values())
    latest = max(i for i, element in enumerate(children) if element.type == "chat_message")
    feedback = next(
        i
        for i, element in enumerate(children)
        if hasattr(element, "children")
        and any(
            e.value == "Latest design feedback — what to do next" for e in element.get("subheader")
        )
    )
    action = next(
        i
        for i, element in enumerate(children)
        if element.type == "button" and element.label == "Prepare my design"
    )
    assert latest < feedback < action
    assert any(
        e.label == "Check details (for troubleshooting)" for e in children[feedback].get("expander")
    )
    assert any("Keep strict sequential IDs" in e.value for e in a.markdown)
    assert not any("Please answer these questions" in e.value for e in a.markdown)
    assert any(e.label == "Advisory notes — not clarification questions" for e in a.expander)


def test_review_failure_stays_in_chat_with_bottom_feedback(monkeypatch, brief, design):
    import workflow
    from models import Design

    def fake(system, payload, model, role="interviewer", **kw):
        return (
            design
            if model is Design
            else dict(
                summary="Needs correction",
                missing_requirements=[],
                concerns=["R1: wrong calculation; use the approved subtraction."],
                warnings=[],
            )
        ), {}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(workflow, "validate_contract", lambda c, **kw: dict(ok=True, errors=[]))
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.session_state.brief = brief
    a.run()
    button(a, "Prepare my design").click().run()
    a.session_state.design_run.thread.join(timeout=15)
    a.run()
    assert not a.exception and a.session_state.page == 0
    assert any("wrong calculation" in m.value for m in a.markdown)
    assert any(e.label == "Check details (for troubleshooting)" for e in a.expander)
    a.session_state.page = 1
    a.run()
    assert button(a, "Approve and continue").disabled


def test_stop_is_responsive_and_late_answer_not_applied(monkeypatch, brief, design):
    from threading import Event
    import workflow

    entered, release = Event(), Event()

    def fake(*args, **kw):
        entered.set()
        release.wait(timeout=10)
        return design, {}

    monkeypatch.setattr(workflow, "ask", fake)
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.session_state.brief = brief
    a.run()
    button(a, "Prepare my design").click().run()
    run = a.session_state.design_run
    assert entered.wait(timeout=5)
    a.run()
    assert a.chat_input[0].disabled
    button(a, "Stop designing").click().run()
    assert not a.exception and run.snapshot()["status"] == "stopped"
    assert a.chat_input[0].disabled
    release.set()
    run.thread.join(timeout=5)
    a.run()
    assert not a.exception and not a.chat_input[0].disabled
    assert a.session_state.approved is None
    assert not a.session_state.result.get("contract")


def test_internal_metadata_not_scanned_but_message_content_is(monkeypatch, brief):
    import workflow

    calls = []

    def fake(system, payload, model, role="interviewer", **kw):
        calls.append(payload)
        assert all(set(m) == {"role", "content"} for m in payload["messages"])
        return dict(reply="Requirements saved.", brief=brief), {}

    monkeypatch.setattr(workflow, "ask", fake)
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    # Synthetic metadata sentinel that the privacy detector would flag if scanned.
    a.session_state.messages = [
        dict(role="assistant", content="Hello.", message_id="4111111111111111")
    ]
    a.run()
    a.chat_input[0].set_value("A harmless follow-up").run()
    a.session_state.interview_run.thread.join(timeout=5)
    a.run()
    assert not a.exception and len(calls) == 1
    assert a.session_state.failure is None
    a.chat_input[0].set_value("Use 4111111111111111 in the data").run()
    assert not a.exception and len(calls) == 1
    assert "sensitive" in a.session_state.failure["error"]


def test_quick_assumptions_need_explicit_acceptance_and_reset(brief):
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    brief["requirements"][0]["status"] = "proposed"
    brief["assumptions"] = ["Use seed 7 for this illustrative demo."]
    a.session_state.brief = brief
    a.run()
    assert button(a, "Prepare my design").disabled
    next(c for c in a.checkbox if c.label.startswith("I accept the listed")).check().run()
    assert not button(a, "Prepare my design").disabled
    a.session_state.brief = dict(brief, seed=99)
    a.run()
    assert button(a, "Prepare my design").disabled
    assert not next(c for c in a.checkbox if c.label.startswith("I accept the listed")).value


def test_mode_persists_and_reaches_interviewer(monkeypatch, brief):
    import workflow

    seen = []

    def fake(system, payload, model, **kw):
        seen.append((system, payload["interview_mode"]))
        return dict(reply="Ready", brief=brief), {}

    monkeypatch.setattr(workflow, "ask", fake)
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.radio[0].set_value("detailed").run()
    a.session_state.page = 3
    a.run()
    a.session_state.page = 0
    a.run()
    assert a.radio[0].value == "detailed"
    a.chat_input[0].set_value("A demo please").run()
    a.session_state.interview_run.thread.join(timeout=5)
    a.run()
    assert seen[0][1] == "detailed" and "DETAILED MODE" in seen[0][0]


def test_accepted_defaults_remain_accepted_after_preparing(monkeypatch, brief, design):
    import workflow
    from models import Design

    monkeypatch.setattr(
        workflow,
        "ask",
        lambda system, payload, model, *args, **kw: (
            design
            if model is Design
            else dict(summary="OK", missing_requirements=[], concerns=[], warnings=[]),
            {},
        ),
    )
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    brief["requirements"][0]["status"] = "proposed"
    brief["assumptions"] = ["Illustrative demo settings."]
    a.session_state.brief = brief
    a.run()
    next(c for c in a.checkbox if c.label.startswith("I accept the listed")).check().run()
    button(a, "Prepare my design").click().run()
    a.session_state.design_run.thread.join(timeout=20)
    a.run()
    assert not a.exception and a.session_state.result["report"]["ok"]
    a.session_state.page = 0
    a.run()
    assert next(c for c in a.checkbox if c.label.startswith("I accept the listed")).value
    assert a.session_state.brief["requirements"][0]["status"] == "confirmed"
    assert (
        a.session_state.assumption_acceptances[0]["proposed_requirements"][0]["status"]
        == "proposed"
    )


def test_interview_stop_preserves_saved_brief(monkeypatch, brief):
    from threading import Event
    import workflow

    entered, release = Event(), Event()

    def fake(*args, **kw):
        entered.set()
        release.wait(timeout=5)
        return dict(reply="Late reply", brief=dict(brief, seed=99)), {}

    monkeypatch.setattr(workflow, "ask", fake)
    a = app()
    a.checkbox(key="_consent_widget").check().run()
    a.session_state.brief = brief
    a.run()
    a.chat_input[0].set_value("Change my demo").run()
    assert entered.wait(timeout=2)
    run = a.session_state.interview_run
    button(a, "Stop interview").click().run()
    release.set()
    run.thread.join(timeout=5)
    a.run()
    assert not a.exception and a.session_state.brief == brief
    assert not a.session_state.messages
    assert a.session_state.failure["text"] == "Change my demo"
