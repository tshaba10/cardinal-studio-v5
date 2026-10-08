from pathlib import Path
import sys
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def contract():
    value = json.loads((ROOT / "references" / "single_grain_accounts.json").read_text())
    value["volume"]["account"] = 2000
    return value


@pytest.fixture
def brief():
    return dict(
        name="demo",
        purpose="Fictional learning data",
        requirements=[
            dict(
                id="R1",
                text="2000 accounts with independent and conditional numeric columns and a calculated score band",
                status="confirmed",
            )
        ],
        volumes=[dict(entity="account", count=2000)],
        seed=7,
        questions=[],
        assumptions=[],
    )


@pytest.fixture
def design(contract):
    return dict(
        summary="Proposed accounts",
        contract_json=json.dumps(contract),
        coverage=[
            dict(
                requirement_id="R1",
                explanation="Accounts and their columns.",
                pointers=["/entities/account", "/volume/account"],
            )
        ],
        unresolved=[],
    )
