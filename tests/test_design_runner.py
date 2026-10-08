from threading import Event
import workflow
from design_runner import DesignRun, Control, RunStopped
import pytest


def test_stop_prevents_subsequent_calls(monkeypatch, brief, design):
    entered, release = Event(), Event()
    calls = []

    def fake(*a, **kw):
        calls.append(1)
        entered.set()
        release.wait(timeout=5)
        return design, {}

    monkeypatch.setattr(workflow, "ask", fake)
    run = DesignRun(brief)
    assert entered.wait(timeout=5)
    run.stop()
    assert run.snapshot()["status"] == "stopped"
    assert run.alive()
    release.set()
    run.thread.join(timeout=5)
    assert not run.alive()
    assert len(calls) == 1
    assert not run.snapshot()["state"].get("contract")


def test_cancel_is_checked_before_api_call(monkeypatch, brief):
    calls = []
    monkeypatch.setattr(workflow, "ask", lambda *a, **kw: calls.append(1))
    control = Control()
    control.cancelled.set()
    with pytest.raises(RunStopped):
        workflow.design_graph().invoke(dict(brief=brief, control=control))
    assert not calls


def test_elapsed_time_never_stops_control():
    control = Control()
    control.started -= 100_000
    control.check()
    assert control.snapshot()["elapsed"] >= 100_000
    assert not control.cancelled.is_set()


def test_failure_retains_previous_check_details(monkeypatch, brief, design):
    calls = []

    def fake(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("Claude HTTP 429. Check credits.")
        return design, {}

    monkeypatch.setattr(workflow, "ask", fake)
    monkeypatch.setattr(
        workflow, "validate_contract", lambda c, **kw: {"ok": False, "errors": ["bad rule"]}
    )
    run = DesignRun(brief)
    run.thread.join(timeout=10)
    result = run.snapshot()
    assert result["status"] == "failed"
    assert result["state"]["report"]["errors"] == ["bad rule"]
    assert "429" in result["error"]
