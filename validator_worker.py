"""Isolated compiler worker. Never calls generate, calibrate or writes a dataset."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "synthgen"))


def main():
    from safety import parse_json
    from contract_checks import execution_check
    from synthgen.contract.loader import resolve_refs, ContractError
    from synthgen.contract.compiler import compile_contract

    try:
        raw = parse_json(sys.stdin.buffer.read(1_000_001))
        execution_check(raw)
        plan = compile_contract(resolve_refs(raw))
        result = dict(
            ok=True,
            errors=[],
            entities=[
                dict(name=e.name, role=e.role, columns=len(e.nodes)) for e in plan.entities.values()
            ],
            invariants=len(plan.invariants),
            targets=len(plan.targets),
            rules=[dict(name=r.name, version=r.version, kind=r.kind) for r in plan.registry.all()],
        )
    except ContractError as exc:
        result = dict(ok=False, errors=[str(p)[:1500] for p in exc.problems[:30]])
    except Exception as exc:
        result = dict(
            ok=False,
            errors=[f"Contract could not be checked: {type(exc).__name__}: {str(exc)[:1500]}"],
        )
    print(json.dumps({"result": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
