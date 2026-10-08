import copy
import pytest
from contract_checks import execution_check, intent_errors, brief_errors, resolve_pointer
from engine_adapter import validate_contract
from safety import parse_json


def test_contract_compiles_and_intent_matches(contract, brief, design):
    assert validate_contract(contract)["ok"]
    assert intent_errors(contract, brief, design) == []
    assert brief_errors(brief) == []


@pytest.mark.parametrize("field,value", [("seed", 8), ("volume", {"account": 100})])
def test_confirmed_counts_and_seed_cannot_change(contract, brief, design, field, value):
    contract[field] = value
    assert intent_errors(contract, brief, design)


def test_extra_static_entity_requires_confirmed_volume(contract, brief, design):
    contract["entities"]["merchant"] = copy.deepcopy(contract["entities"]["account"])
    assert intent_errors(contract, brief, design)


@pytest.mark.parametrize("pointer", ["", "/does_not_exist", "/entities/account/nodes/nope"])
def test_bad_coverage(contract, brief, design, pointer):
    design["coverage"][0]["pointers"] = [pointer]
    assert intent_errors(contract, brief, design)


def test_missing_coverage(contract, brief, design):
    design["coverage"] = []
    assert intent_errors(contract, brief, design)


def test_unresolved_blocks_export(contract, brief, design):
    design["unresolved"] = ["Clarify currency"]
    assert "Clarify currency" in intent_errors(contract, brief, design)


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "con", "lpt1", "CamelCase"])
def test_unsafe_output_name(contract, name):
    contract["entities"][name] = contract["entities"].pop("account")
    with pytest.raises(ValueError):
        execution_check(contract)


def test_calibration_not_silently_ignored(contract):
    contract["calibration"] = {"enabled": True}
    with pytest.raises(ValueError, match="calibration"):
        execution_check(contract)


def test_unconfirmed_brief(brief):
    brief["requirements"][0]["status"] = "proposed"
    assert brief_errors(brief)


def test_no_application_volume_cap(contract):
    contract["volume"]["account"] = 2_000_000
    assert execution_check(contract)["volume"]["account"] == 2_000_000


@pytest.mark.parametrize("value", ['{"x":1,"x":2}', '{"x":NaN}', "[]"])
def test_invalid_json(value):
    with pytest.raises(ValueError):
        parse_json(value)


def test_no_external_rule_package(contract):
    contract["rule_library"]["packages"] = ["os"]
    with pytest.raises(ValueError):
        execution_check(contract)


def test_custom_code_rejected(contract):
    contract["rules"]["custom.score_band"]["expression"] = '__import__("os").system("anything")'
    with pytest.raises(ValueError):
        execution_check(contract)


def test_wrong_rule_test_rejected(contract):
    contract["rules"]["custom.score_band"]["tests"][0]["expected"] = "wrong"
    assert not validate_contract(contract)["ok"]


def test_same_step_cycle_rejected(contract):
    node = contract["entities"]["account"]["nodes"]["score_band"]
    node["inputs"]["fico"] = "account.score_band"
    assert not validate_contract(contract)["ok"]


def test_coverage_can_point_at_original_local_ref(contract, brief, design):
    contract["parameters"] = {"count": 2000}
    contract["volume"]["account"] = {"$ref": "#/parameters/count"}
    design["coverage"][0]["pointers"] = ["/volume/account/$ref", "/parameters/count"]
    assert not intent_errors(contract, brief, design)


@pytest.mark.parametrize("pointer", ["/items/-1", "/items/00"])
def test_invalid_array_pointer(pointer):
    with pytest.raises(ValueError):
        resolve_pointer({"items": ["a"]}, pointer)
