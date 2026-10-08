"""Read-only checks of the Git staging area before committing or publishing.

Only staged blob contents are inspected: an unstaged cleanup cannot hide a
previously staged secret. Findings show paths/reasons, never matched values.
This conservative allowlist is a guardrail, not a complete secret/IP scanner.
Run with the project interpreter after `git add .`; nothing is committed/pushed.
"""

from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parent
ROOT_FILES = {
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "DEMO.md",
    "CLAUDE_SETUP.md",
    "ARCHITECTURE.md",
    "PUBLISHING.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    "LICENSE.md",
    "app.py",
    "claude_client.py",
    "claude_worker.py",
    "contract_checks.py",
    "design_runner.py",
    "downloads.py",
    "engine_adapter.py",
    "engine_snapshot.json",
    "export_graphs.py",
    "export_rule_reference.py",
    "file_models.py",
    "generation_worker.py",
    "graphs.py",
    "importer.py",
    "interview_settings.py",
    "jobs.py",
    "models.py",
    "privacy.py",
    "process_runner.py",
    "prompts.py",
    "quality_report.py",
    "quality_review.py",
    "quality_ui.py",
    "safety.py",
    "validator_worker.py",
    "workflow.py",
    "check_repository.py",
    "requirements.txt",
    "requirements-dev.txt",
    "ruff.toml",
    "run.command",
    "run_windows.bat",
}
REFERENCES = {
    "contract_structure.json",
    "dependent_accounts.json",
    "single_grain_accounts.json",
    "credit_card.json",
    "rule_library.json",
}
PRIVATE_PARTS = {
    ".venv",
    "venv",
    "env",
    ".runs",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".vscode",
    ".idea",
    ".history",
    ".change-backups",
    "backup",
    "backups",
    "archive",
    "archives",
    "obsolete",
    "uploads",
    "exports",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-(?:ant-|or-)[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def allowed_path(name):
    """Allow only current source/docs/fixtures; require review for new paths."""
    path = PurePosixPath(name)
    parts = path.parts
    if any(part.lower() in PRIVATE_PARTS for part in parts):
        return False
    if len(parts) == 1:
        return name in ROOT_FILES
    if name == ".streamlit/config.toml":
        return True
    if len(parts) == 2 and parts[0] == "references":
        return parts[1] in REFERENCES
    if len(parts) == 2 and parts[0] == "tests":
        return parts[1] == "conftest.py" or (parts[1].startswith("test_") and path.suffix == ".py")
    if name == "vendor/synthgen/pyproject.toml":
        return True
    if name.startswith("vendor/synthgen/synthgen/"):
        return path.suffix == ".py"
    return name in {
        "vendor/synthgen/LICENSE",
        "vendor/synthgen/LICENSE.md",
        "vendor/synthgen/NOTICE",
        "vendor/synthgen/NOTICE.md",
    }


def inspect_blob(name, data):
    """Return reasons only; never include credential contents in diagnostics."""
    findings = []
    if not allowed_path(name):
        findings.append("not in the reviewed source-file allowlist")
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        findings.append(
            "possible credential or private key; inspect privately and rotate if exposed"
        )
    if name == ".env.example":
        for line in data.decode("utf-8", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator and not key.lstrip().startswith("#"):
                if re.search(r"KEY|TOKEN|SECRET|PASSWORD", key, re.I) and value.strip().strip(
                    "\"'"
                ):
                    findings.append(
                        "setup template contains a nonempty secret field; keep it blank"
                    )
                    break
    if len(data) > 2_000_000:
        findings.append("unexpected source file over 2 MB; inspect for data or packaged output")
    return findings


def git(root, *args):
    """Read Git metadata/blobs without changing the index or working tree."""
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(
            "Git read failed. Run inside the project repository with resolved merge conflicts."
        )
    return result.stdout


def staged_findings(root):
    """Inspect all indexed files, including tracked files excluded by .gitignore."""
    records = git(root, "ls-files", "--stage", "-z").split(b"\0")
    issues, count = [], 0
    for record in filter(None, records):
        metadata, raw_name = record.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        name = raw_name.decode("utf-8", errors="replace")
        count += 1
        if stage != "0" or mode not in ("100644", "100755"):
            issues.append(
                (name, ["unresolved, symbolic-link or submodule entry; review before publication"])
            )
            continue
        data = git(root, "cat-file", "blob", object_id)
        reasons = inspect_blob(name, data)
        if reasons:
            issues.append((name, reasons))
    return count, issues


def main():
    try:
        count, issues = staged_findings(ROOT)
    except (OSError, ValueError):
        print(
            "Create/open the Git repository and stage the intended files first. See PUBLISHING.md."
        )
        return 1
    if not count:
        print("No files in the Git index. Stage intended source files, then run this check again.")
        return 1
    for name, reasons in issues:
        print(f"{name!r}: " + "; ".join(reasons))
    if issues:
        print("Publication check failed. Unstage/correct these files before committing.")
        return 1
    print(f"Checked {count} indexed files: no allowlist or recognized-secret findings.")
    print(
        "Still review the staged diff, repository history, ownership and redistribution permission. This is not a complete secret scan."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
