"""Validate contracts in a stoppable process using the fingerprint-pinned engine."""

import hashlib
import json
import os
from pathlib import Path
from safety import parse_json
from contract_checks import execution_check
from design_runner import Control, RunStopped
from process_runner import run_worker

ROOT = Path(__file__).resolve().parent


def engine_fingerprint():
    """Hash relative paths and source bytes to detect changes to the bundled engine."""
    files = sorted((ROOT / "vendor" / "synthgen" / "synthgen").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT / "vendor")).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_contract(contract, control=None):
    """Check execution policy and engine integrity before isolated compilation."""
    try:
        payload = json.dumps(contract, allow_nan=False)
        execution_check(parse_json(payload))
        expected = json.loads((ROOT / "engine_snapshot.json").read_text())["sha256"]
        actual = engine_fingerprint()
        if actual != expected:
            raise ValueError(
                "The bundled generator changed. Review and update the pinned snapshot before exporting contracts."
            )
    except (ValueError, TypeError, RecursionError, OSError) as exc:
        return dict(ok=False, errors=[str(exc)], entities=[])
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG")
    }
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        report = run_worker(ROOT / "validator_worker.py", contract, control or Control(), env)
        report["engine_sha256"] = actual
        return report
    except RunStopped:
        raise
    except (OSError, ValueError):
        return dict(
            ok=False,
            errors=[
                "Could not read the validator response. Check the project’s Python environment and retry."
            ],
            entities=[],
        )
