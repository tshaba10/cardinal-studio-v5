"""Publication guardrails: indexed contents matter, not just the working tree."""

import subprocess
import pytest
from check_repository import allowed_path, inspect_blob, staged_findings


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".venv/secret.py",
        ".runs/demo/contract.json",
        "backups/app.py",
        "app_old.py",
        "notes.txt",
        "dataset.zip",
        "account.csv",
        "vendor/synthgen/synthgen/__pycache__/engine.pyc",
        ".streamlit/secrets.toml",
    ],
)
def test_private_or_unreviewed_paths_rejected(name):
    assert not allowed_path(name)


@pytest.mark.parametrize(
    "name",
    [
        "app.py",
        ".env.example",
        "README.md",
        ".streamlit/config.toml",
        "references/rule_library.json",
        "vendor/synthgen/synthgen/engine/simulate.py",
        "tests/test_repository.py",
    ],
)
def test_current_source_allowed(name):
    assert allowed_path(name)


def test_template_and_secret_detection_do_not_echo_values():
    secret = "sk-ant-" + "x" * 40
    findings = inspect_blob(".env.example", ("ANTHROPIC_API_KEY=" + secret).encode())
    assert len(findings) == 2 and secret not in str(findings)
    assert not inspect_blob(".env.example", b"ANTHROPIC_API_KEY=\nANTHROPIC_MODEL=model-id\n")


def test_scan_reads_staged_secret_after_working_copy_is_cleaned(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    template = tmp_path / ".env.example"
    template.write_text("ANTHROPIC_API_KEY=not-empty\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".env.example"], check=True)
    template.write_text("ANTHROPIC_API_KEY=\n")
    count, findings = staged_findings(tmp_path)
    assert count == 1 and findings[0][0] == ".env.example"
    assert "secret field" in findings[0][1][0]
