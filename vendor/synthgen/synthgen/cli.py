"""
Command line interface.

    python -m synthgen validate  CONTRACT
    python -m synthgen generate  CONTRACT [-o DIR] [--volume account=1000] [--steps 5] ...
    python -m synthgen verify    CONTRACT -o DIR
    python -m synthgen calibrate CONTRACT --write calibrated.json
    python -m synthgen explain   CONTRACT
    python -m synthgen rules     [--contract CONTRACT]

``--volume ENTITY=N`` and ``--steps N`` are shortcuts for
``--set volume.ENTITY=N`` / ``--set volume.steps=N``; ``--set`` accepts any
dotted path in the contract (value parsed as JSON when possible).
"""
from __future__ import annotations

import argparse
import json
import sys

from .contract.compiler import compile_contract, describe_rules
from .contract.loader import ContractError, apply_overrides, load_contract
from .rules import load_packages


def _overrides(args) -> list[str]:
    out = list(args.set or [])
    for v in args.volume or []:
        ent, _, n = v.partition("=")
        out.append(f"volume.{ent}={n}")
    if getattr(args, "steps", None):
        out.append(f"volume.steps={args.steps}")
    if getattr(args, "seed", None) is not None:
        out.append(f"seed={args.seed}")
    return out


def _plan(args):
    return compile_contract(load_contract(args.contract, _overrides(args)))


def cmd_validate(args) -> int:
    plan = _plan(args)
    n_nodes = sum(len(e.nodes) for e in plan.entities.values())
    print(f"contract OK: {len(plan.entities)} entities, {n_nodes} columns, "
          f"{len(plan.invariants)} invariants, {len(plan.targets)} targets")
    for e in plan.entities.values():
        vol = f"{e.count:,} rows" if e.role == "static" else (
            f"per {e.parent} per step" if e.role == "stepped" else f"per {e.parent} row")
        print(f"  {e.name:<16} {e.role:<8} {vol}")
    return 0


def cmd_generate(args) -> int:
    from .engine.simulate import generate
    from .verify import summarize, verify

    plan = _plan(args)
    out = args.output or plan.output["directory"]
    generate(plan, out, workers=args.workers, chunk_size=args.chunk_size)
    if args.no_verify:
        return 0
    result = verify(plan, out)
    print(summarize(result))
    return 0 if result["ok"] else 2


def cmd_verify(args) -> int:
    from .verify import summarize, verify

    plan = _plan(args)
    result = verify(plan, args.output or plan.output["directory"])
    print(summarize(result))
    return 0 if result["ok"] else 2


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate, write_contract

    with open(args.contract, encoding="utf-8") as fh:
        raw = json.load(fh)
    raw = apply_overrides(raw, _overrides(args))
    compile_contract(load_contract(args.contract, _overrides(args)))   # validate first
    doc, report = calibrate(raw)
    for tid, r in report["final"].items():
        print(f"  {tid:<26} measured {r['measured']:.4f}  target {r['target']} +/- {r['tolerance']}")
    target = args.write or args.contract
    write_contract(doc, target)
    print(f"calibrated parameters written to {target}")
    return 0


def cmd_explain(args) -> int:
    plan = _plan(args)
    print("Phase 1 - static entities, execution order:")
    for u in plan.static_order:
        print(f"  {u}" + ("" if u.startswith("joint:") else f"  [{plan.node(u).kind}]"))
    for ent, units in plan.stepped.items():
        print(f"Phase 2 - time loop of {ent}, per-step execution order:")
        for u in units:
            if u.startswith("joint:"):
                print(f"  {u}")
                continue
            n = plan.node(u)
            lag = [r.raw for r in n.refs if r.lag]
            extra = f" rule={n.rule.key}" if n.rule else ""
            extra += f" reads {', '.join(lag)}" if lag else ""
            print(f"  {u:<44} [{n.kind}]{extra}")
        print(f"  state carried to next step: {plan.lagged.get(ent, [])}")
    return 0


def cmd_rules(args) -> int:
    if args.contract:
        plan = _plan(args)
        print(describe_rules(plan.registry))
    else:
        reg = load_packages(["common", "credit_card"])
        print(describe_rules(reg))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="synthgen", description="Deterministic related synthetic data generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, contract_required=True):
        if contract_required:
            sp.add_argument("contract", help="path to the JSON contract")
        sp.add_argument("--set", action="append", metavar="PATH=VALUE", help="override any contract value")
        sp.add_argument("--volume", action="append", metavar="ENTITY=N", help="override an entity volume")
        sp.add_argument("--steps", type=int, help="override the number of output time steps")
        sp.add_argument("--seed", type=int, help="override the seed")

    sp = sub.add_parser("validate", help="validate a contract without generating")
    common(sp)
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("generate", help="generate CSVs (and verify them)")
    common(sp)
    sp.add_argument("-o", "--output", help="output directory (default: output.directory)")
    sp.add_argument("--workers", type=int, help="parallel processes (0 = all cores)")
    sp.add_argument("--chunk-size", type=int, help="roots simulated per chunk")
    sp.add_argument("--no-verify", action="store_true", help="skip post-run verification")
    sp.set_defaults(fn=cmd_generate)

    sp = sub.add_parser("verify", help="verify previously generated CSVs against the contract")
    common(sp)
    sp.add_argument("-o", "--output", help="directory holding the CSVs")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("calibrate", help="tune behaviour parameters to hit targets")
    common(sp)
    sp.add_argument("--write", help="where to write the calibrated contract (default: in place)")
    sp.set_defaults(fn=cmd_calibrate)

    sp = sub.add_parser("explain", help="show the execution plan")
    common(sp)
    sp.set_defaults(fn=cmd_explain)

    sp = sub.add_parser("rules", help="list the rules library")
    sp.add_argument("--contract", help="include the contract's custom rules")
    sp.add_argument("--set", action="append")
    sp.add_argument("--volume", action="append")
    sp.set_defaults(fn=cmd_rules)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ContractError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
