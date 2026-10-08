from pathlib import Path
import csv
import json
import time
import zipfile
import requests
import pytest
from jobs import start_job, job_status, preview, cancel_job
from downloads import DownloadServer


def wait(job):
    end = time.monotonic() + 40
    while time.monotonic() < end:
        result = job_status(job)
        if result["status"] != "running":
            return result
        time.sleep(0.1)
    cancel_job(job)
    pytest.fail("Worker took too long")


def test_real_full_generation_and_streaming(contract):
    job = start_job(contract)
    result = wait(job)
    assert result["status"] == "complete", result
    assert result["run"]["rows"]["account"] == 2000
    columns, rows = preview(job, "account")
    assert len(rows) == 100 and "score_band" in columns
    with zipfile.ZipFile(Path(job["folder"]) / "dataset.zip") as archive:
        assert json.loads(archive.read("contract.json")) == contract
        assert len(list(csv.DictReader(archive.read("account.csv").decode().splitlines()))) == 2000
    server = DownloadServer()
    try:
        url = server.grant(Path(job["folder"]) / "dataset.zip")
        response = requests.get(url, timeout=5)
        assert response.status_code == 200 and response.content[:2] == b"PK"
        assert requests.get(url, headers={"Host": "evil.example"}, timeout=5).status_code == 403
        server.revoke(url)
        assert requests.get(url, timeout=5).status_code == 404
        with pytest.raises(ValueError):
            server.grant(Path(job["folder"]) / "contract.json")
        result_file = Path(job["folder"]) / "result.json"
        blocked = dict(result, quality=dict(result["quality"], gate_passed=False))
        result_file.write_text(json.dumps(blocked))
        with pytest.raises(ValueError, match="quality gate"):
            server.grant(Path(job["folder"]) / "dataset.zip")
        result_file.write_text(json.dumps(result))
        with (Path(job["folder"]) / "dataset.zip").open("ab") as handle:
            handle.write(b"changed after verification")
        with pytest.raises(ValueError, match="changed after"):
            server.grant(Path(job["folder"]) / "dataset.zip")
    finally:
        server.server.shutdown()
        server.server.server_close()


def test_failed_invariant_blocks_zip(contract):
    contract["invariants"][0]["expression"] = "fico_score < 0"
    job = start_job(contract)
    result = wait(job)
    assert result["status"] == "failed"
    assert not (Path(job["folder"]) / "dataset.zip").exists()


def test_cancellation(contract):
    job = start_job(contract)
    cancel_job(job)
    assert job_status(job)["status"] == "cancelled"
    assert job["process"].poll() is not None


def test_monthly_events_aggregate_state_and_lookup():
    contract = dict(
        contract={"name": "linked_history", "version": "1.0"},
        seed=42,
        volume={"account": 20, "merchant": 3, "steps": 6},
        rule_library={"packages": ["common"]},
        time={
            "calendar": "monthly",
            "output_start": "2026-04-01",
            "output_steps": 6,
            "warmup_max_steps": 0,
            "anchor": {"start_date": "account.opened_on"},
        },
        entities={
            "merchant": {
                "role": "static",
                "primary_key": ["merchant_id"],
                "nodes": {
                    "merchant_id": {
                        "kind": "identifier",
                        "type": "string",
                        "format": {"pattern": "M-[0-9]{3}"},
                    },
                    "category": {
                        "kind": "drawn",
                        "type": "string",
                        "dist": "categorical",
                        "params": {"values": ["grocery", "travel"], "weights": [0.8, 0.2]},
                    },
                },
            },
            "account": {
                "role": "static",
                "primary_key": ["account_id"],
                "nodes": {
                    "account_id": {
                        "kind": "identifier",
                        "type": "string",
                        "format": {"pattern": "A-[0-9]{4}"},
                    },
                    "opened_on": {
                        "kind": "drawn",
                        "type": "date",
                        "dist": "date_uniform",
                        "params": {"start": "2023-01-01", "end": "2023-12-31"},
                    },
                },
            },
            "account_month": {
                "role": "stepped",
                "parent": "account",
                "primary_key": ["account_id", "month_number"],
                "nodes": {
                    "account_id": {
                        "kind": "parent_key",
                        "type": "string",
                        "references": "account.account_id",
                    },
                    "month_number": {
                        "kind": "context",
                        "type": "int16",
                        "source": "step.cycle_number",
                    },
                    "count": {
                        "kind": "drawn",
                        "type": "int32",
                        "dist": "constant",
                        "params": {"value": 2},
                    },
                    "total": {
                        "kind": "aggregate",
                        "type": "decimal(12,2)",
                        "over": "transaction",
                        "fn": "sum",
                        "of": "amount",
                    },
                    "cumulative": {
                        "kind": "state",
                        "type": "decimal(12,2)",
                        "rule": "common.running_total",
                        "init": 0,
                        "inputs": {
                            "previous": "account_month.cumulative@t-1",
                            "value": "account_month.total",
                        },
                    },
                },
            },
            "transaction": {
                "role": "event",
                "parent": "account_month",
                "count": "account_month.count",
                "max_per_parent": 2,
                "primary_key": ["transaction_id"],
                "nodes": {
                    "transaction_id": {
                        "kind": "identifier",
                        "type": "string",
                        "format": {"pattern": "T-[0-9]{10}"},
                    },
                    "account_id": {
                        "kind": "parent_key",
                        "type": "string",
                        "references": "account.account_id",
                    },
                    "month_number": {
                        "kind": "context",
                        "type": "int16",
                        "source": "step.cycle_number",
                    },
                    "merchant_id": {
                        "kind": "reference",
                        "type": "string",
                        "references": "merchant.merchant_id",
                    },
                    "category": {
                        "kind": "lookup",
                        "type": "string",
                        "via": "transaction.merchant_id",
                        "from": "merchant.category",
                    },
                    "amount": {
                        "kind": "drawn",
                        "type": "decimal(12,2)",
                        "dist": "uniform",
                        "params": {"low": 5, "high": 20},
                    },
                },
            },
        },
        invariants=[
            {
                "name": "sum_matches",
                "type": "aggregate",
                "parent": "account_month",
                "child": "transaction",
                "keys": ["account_id", "month_number"],
                "parent_column": "total",
                "fn": "sum",
                "child_column": "amount",
                "tolerance": 0.005,
            },
            {
                "name": "cumulative_grows",
                "type": "monotonic",
                "table": "account_month",
                "column": "cumulative",
                "partition_by": ["account_id"],
                "order_by": "month_number",
                "direction": "non_decreasing",
            },
        ],
    )
    job = start_job(contract)
    result = wait(job)
    assert result["status"] == "complete", result
    assert result["run"]["rows"] == {
        "account": 20,
        "merchant": 3,
        "account_month": 120,
        "transaction": 240,
    }


def test_joint_correlated_columns(contract):
    nodes = contract["entities"]["account"]["nodes"]
    for name in ("a", "b"):
        nodes[name] = {
            "kind": "joint",
            "type": "float",
            "group": "scores",
            "dist": "normal",
            "params": {"mean": 0, "std": 1},
        }
    contract["entities"]["account"]["joint_groups"] = {
        "scores": {
            "method": "gaussian_copula",
            "members": ["a", "b"],
            "correlation": [[1, 0.7], [0.7, 1]],
        }
    }
    job = start_job(contract)
    result = wait(job)
    assert result["status"] == "complete", result
