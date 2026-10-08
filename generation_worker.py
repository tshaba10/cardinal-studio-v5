"""Execute the approved volume, verify the actual CSVs, create a disk-backed ZIP."""

import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "synthgen"))


def run(folder):
    """Compile, generate, verify, audit, then package only a passing local run."""
    from safety import parse_json
    from contract_checks import execution_check
    from engine_adapter import engine_fingerprint
    from synthgen.contract.compiler import compile_contract
    from synthgen.engine.simulate import generate
    from synthgen.verify import verify
    from synthgen.output import emitted_columns
    from quality_report import audit_dataset, sha256_file

    expected = json.loads((ROOT / "engine_snapshot.json").read_text())["sha256"]
    if engine_fingerprint() != expected:
        raise ValueError(
            "Bundled generator fingerprint changed. Restore or review the engine before running."
        )
    contract = parse_json((folder / "contract.json").read_bytes())
    expanded = execution_check(contract)
    print("Validating the approved config before generation…", flush=True)
    plan = compile_contract(expanded)
    if not any(ent.emit and emitted_columns(plan, name) for name, ent in plan.entities.items()):
        raise ValueError(
            "No output tables are enabled. Ask the designer to emit the requested tables."
        )
    output = folder / "data"
    # Only execution location/concurrency change. The approved volumes and semantics do not.
    report = generate(
        plan, out_dir=str(output), workers=1, chunk_size=1000, log=lambda s: print(s, flush=True)
    )
    for name, ent in plan.entities.items():
        if ent.emit and emitted_columns(plan, name):
            if not (output / (name + ".csv")).is_file():
                raise ValueError("Missing expected output: " + name)
            if ent.role == "static" and report["rows"].get(name) != ent.count:
                raise ValueError("Generated static row count differs from contract: " + name)
    print("Verifying generated data with the bundled rules…", flush=True)
    verified = verify(plan, output, log=lambda s: print(s, flush=True))
    print("Checking every emitted CSV row with the independent quality policy…", flush=True)
    quality = audit_dataset(
        plan, output, contract, report, verified, log=lambda s: print(s, flush=True)
    )
    quality["engine_sha256"] = expected
    (output / "quality_report.json").write_text(
        json.dumps(quality, indent=2, allow_nan=False), encoding="utf-8"
    )
    if not quality["gate_passed"]:
        return dict(
            status="failed",
            error="Generated data failed its Python quality gate. Review the failed checks below; correct the design or generator before regenerating. Dataset download is blocked.",
            verification=verified,
            quality=quality,
            run=report,
        )
    print("Packaging verified CSVs and reports…", flush=True)
    with zipfile.ZipFile(
        folder / "dataset.zip", "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        archive.write(folder / "contract.json", "contract.json")
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in (".csv", ".json"):
                archive.write(path, path.name)
    return dict(
        status="complete",
        run=report,
        verification=verified,
        quality=quality,
        dataset_sha256=sha256_file(folder / "dataset.zip"),
        engine_sha256=expected,
    )


def main():
    folder = Path(sys.argv[1]).resolve()
    if folder.parent != (ROOT / ".runs").resolve() or not folder.is_dir():
        raise ValueError("Generation jobs must live in this project’s .runs folder.")
    try:
        result = run(folder)
    except Exception as exc:
        result = dict(
            status="failed",
            error=f"{type(exc).__name__}: {str(exc)[:2000]}. No verified download is available. Review the contract or available resources.",
        )
    temporary = folder / "result.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(folder / "result.json")


if __name__ == "__main__":
    main()
