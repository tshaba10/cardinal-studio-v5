"""Disk-backed generation, cancellable without blocking Streamlit."""

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from contract_checks import execution_check

ROOT = Path(__file__).resolve().parent


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def start_job(contract):
    """Persist an approved config and launch the local generator without AI calls."""
    execution_check(contract)
    # The worker repeats compilation and fingerprint checks; do not block the UI here.
    folder = ROOT / ".runs" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    (folder / "contract.json").write_text(
        json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG")
    }
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    with (folder / "progress.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", "-B", str(ROOT / "generation_worker.py"), str(folder)],
            cwd=ROOT,
            stdout=log,
            stderr=log,
            env=env,
        )
    return dict(
        folder=str(folder), process=process, fingerprint=fingerprint(contract), cancelled=False
    )


def cancel_job(job):
    process = job["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    job["cancelled"] = True


def job_status(job):
    """Read the worker's atomic final result; partial files are never completion."""
    if job.get("cancelled"):
        return dict(
            status="cancelled",
            error="Generation stopped. Partial files are not approved for download.",
        )
    if job["process"].poll() is None:
        return dict(status="running")
    result = Path(job["folder"]) / "result.json"
    if not result.exists():
        return dict(
            status="failed",
            error="The worker stopped unexpectedly. Check available memory/disk and the local progress log; try a smaller design in chat.",
        )
    return json.loads(result.read_text(encoding="utf-8"))


def progress(job):
    with (Path(job["folder"]) / "progress.log").open("rb") as file:
        file.seek(0, 2)
        file.seek(max(0, file.tell() - 5000))
        return file.read().decode("utf-8", errors="replace")


def preview(job, table):
    """Read at most 100 actual output rows, without loading the entire CSV."""
    folder = Path(job["folder"]) / "data"
    path = folder / (table + ".csv")
    if path.parent.resolve() != folder.resolve():
        raise ValueError("Invalid table name.")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(itertools.islice(reader, 100))
