"""Execute the earlier demo's exact math, not just its JSON syntax."""

import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from jobs import start_job, preview
from test_generation import wait

ROOT = Path(__file__).resolve().parents[1]


def test_dependent_demo_generates_correct_values():
    contract = json.loads((ROOT / "references/dependent_accounts.json").read_text())
    job = start_job(contract)
    result = wait(job)
    assert result["status"] == "complete", result
    assert result["run"]["rows"] == {"account": 2000}
    with (Path(job["folder"]) / "data/account.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r["account_id"] for r in rows}) == 2000
    for row in rows:
        limit, balance = Decimal(row["credit_limit"]), Decimal(row["opening_balance"])
        assert limit in ({1000, 3000} if row["segment"] == "standard" else {5000, 10000})
        assert 0 <= balance <= 400
        assert Decimal(row["available_credit"]) == limit - balance
        assert Decimal(row["utilization"]) == (balance / limit).quantize(
            Decimal(".0001"), rounding=ROUND_HALF_UP
        )
        assert all(v != "" for v in row.values())
    assert len(preview(job, "account")[1]) == 100
