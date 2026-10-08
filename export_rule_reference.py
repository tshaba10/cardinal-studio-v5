"""Build an inspectable AI reference from the bundled rule signatures."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "synthgen"))

if __name__ == "__main__":
    from synthgen.rules import load_packages

    result = [
        {"signature": rule.signature(), "kind": rule.kind, "description": rule.description}
        for rule in load_packages(["common", "credit_card"]).all()
    ]
    (ROOT / "references" / "rule_library.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
