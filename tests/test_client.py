import json
from threading import Thread
import time
import pytest
import requests
import claude_client
import claude_worker
from design_runner import Control, RunStopped
from process_runner import run_worker
from models import Interview


class Response:
    ok = True
    status_code = 200

    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self, **kw):
        for event in self.events:
            yield "data: " + json.dumps(event)
            yield ""


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(claude_client, "load_dotenv", lambda *a: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-not-a-real-key")


def events(text, reason="end_turn"):
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text[:12]}},
        {"type": "ping"},
        {"type": "unknown_future_event"},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text[12:]}},
        {"type": "message_delta", "delta": {"stop_reason": reason}, "usage": {"output_tokens": 4}},
        {"type": "message_stop"},
    ]


def test_streaming_and_complete_validation(monkeypatch, brief):
    def post(url, **kw):
        assert url == "https://api.anthropic.com/v1/messages"
        assert "verify" not in kw and kw["timeout"] == (10, None)
        assert kw["stream"] and kw["json"]["stream"]
        return Response(events(json.dumps(dict(reply="hello", brief=brief))))

    monkeypatch.setattr(claude_worker.requests, "post", post)

    def worker(script, body, control, env):
        assert body["output_config"]["format"]["type"] == "json_schema"
        return claude_worker.stream_response(body, env["ANTHROPIC_API_KEY"], lambda e: None)

    monkeypatch.setattr(claude_client, "run_worker", worker)
    result, usage = claude_client.ask("instructions", {}, Interview)
    assert result["brief"] == brief and usage["input_tokens"] == 3 and usage["output_tokens"] == 4


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "tool_use"])
def test_incomplete_not_accepted(monkeypatch, reason):
    monkeypatch.setattr(
        claude_worker.requests, "post", lambda *a, **kw: Response(events("{}", reason))
    )
    with pytest.raises(ValueError, match="incomplete or refused"):
        claude_worker.stream_response({}, "test", lambda e: None)


def test_truncated_stream_rejected(monkeypatch):
    monkeypatch.setattr(
        claude_worker.requests, "post", lambda *a, **kw: Response(events("{}")[:-1])
    )
    with pytest.raises(ValueError, match="incomplete"):
        claude_worker.stream_response({}, "test", lambda e: None)


def test_stream_error_rejected(monkeypatch):
    monkeypatch.setattr(
        claude_worker.requests,
        "post",
        lambda *a, **kw: Response([{"type": "error", "error": {"message": "SECRET"}}]),
    )
    with pytest.raises(ValueError, match="streaming error") as error:
        claude_worker.stream_response({}, "test", lambda e: None)
    assert "SECRET" not in str(error.value)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        claude_client.ask("", {}, Interview)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 529])
def test_http_errors_do_not_echo_body(monkeypatch, status):
    response = Response([])
    response.ok, response.status_code = False, status
    monkeypatch.setattr(claude_worker.requests, "post", lambda *a, **kw: response)
    with pytest.raises(ValueError, match=str(status)):
        claude_worker.stream_response({}, "test", lambda e: None)


def test_no_insecure_ssl_retry(monkeypatch):
    calls = []

    def post(*a, **kw):
        calls.append(kw)
        raise requests.exceptions.SSLError("failed")

    monkeypatch.setattr(claude_worker.requests, "post", post)
    with pytest.raises(requests.exceptions.SSLError):
        claude_worker.stream_response({}, "test", lambda e: None)
    assert len(calls) == 1 and "verify" not in calls[0]


def test_stop_terminates_worker_blocked_without_output(tmp_path):
    script = tmp_path / "blocked_worker.py"
    script.write_text(
        'import json,sys,time\njson.load(sys.stdin)\nprint(json.dumps({"progress":"connected"}),flush=True)\ntime.sleep(300)\n'
    )
    control, errors = Control(), []

    def run():
        try:
            run_worker(script, {}, control, {})
        except RunStopped:
            errors.append("stopped")

    thread = Thread(target=run)
    thread.start()
    for _ in range(100):
        if control.snapshot()["detail"] == "connected":
            break
        time.sleep(0.02)
    started = time.monotonic()
    control.cancelled.set()
    thread.join(timeout=4)
    assert not thread.is_alive() and errors == ["stopped"]
    assert time.monotonic() - started < 4


def test_schema_failure_no_partial_result(monkeypatch):
    monkeypatch.setattr(
        claude_client, "run_worker", lambda *a: {"text": "User Safety: safe", "usage": {}}
    )
    with pytest.raises(ValueError, match="required structure"):
        claude_client.ask("", {}, Interview)
