"""Build structured Claude requests and validate completed responses locally.

Credentials stay in the worker environment, never in model prompts or command
arguments. Network streaming and cancellation are delegated to process_runner.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ValidationError
from process_runner import run_worker
from design_runner import Control

ROOT = Path(__file__).resolve().parent


def schema_for(model):
    """Adapt a Pydantic schema for the API; retain strict local validation."""
    schema = model.model_json_schema()

    def walk(value):
        if isinstance(value, dict):
            # Claude constrains the shape; Pydantic enforces these locally.
            for key in (
                "minLength",
                "maxLength",
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "pattern",
                "default",
            ):
                value.pop(key, None)
            if value.get("type") == "object":
                value["additionalProperties"] = False
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    return schema


def ask(system, payload, model_class, role="interviewer", *, control=None):
    """Return a validated model dictionary and usage, or raise a safe UI error."""
    load_dotenv(ROOT / ".env")
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Add ANTHROPIC_API_KEY to the project’s .env, save it, and restart Cardinal Studio. See CLAUDE_SETUP.md. Never paste your key into chat."
        )
    default = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    model = os.getenv("ANTHROPIC_DESIGNER_MODEL", default) if role != "interviewer" else default
    body = dict(
        model=model,
        max_tokens=16000 if role == "designer" else 8000,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        output_config={"format": {"type": "json_schema", "schema": schema_for(model_class)}},
    )
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "LANG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(ANTHROPIC_API_KEY=key, PYTHONUTF8="1")
    control = control or Control()
    data = run_worker(ROOT / "claude_worker.py", body, control, env)
    control.check()
    try:
        parsed = model_class.model_validate_json(data["text"])
        usage = data.get("usage", {})
        return parsed.model_dump(), {
            "model": model,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
    except (KeyError, TypeError, AttributeError, ValidationError) as exc:
        raise ValueError(
            "Claude’s response did not match the required structure. Your saved work is unchanged. Retry or report this error without sharing secrets."
        ) from exc
