"""
synthgen - a deterministic, contract-driven generator of *related* synthetic data.

The contract (JSON) describes entities (grains), columns as nodes in a
dependency graph, the rules that compute derived values, time structure,
shared factors, invariants and calibration targets. The engine validates the
graph, generates the process that produces the data (rather than sampling
each column independently), writes one CSV per grain and verifies the output.

Public API::

    from synthgen import load_contract, compile_contract, generate, verify
    plan = compile_contract(load_contract("contracts/credit_card.json",
                                          ["volume.account=1000"]))
    generate(plan, "out")
    verify(plan, "out")
"""
from .contract.compiler import compile_contract
from .contract.loader import ContractError, load_contract
from .engine.simulate import generate
from .verify import verify

__all__ = ["load_contract", "compile_contract", "generate", "verify", "ContractError"]
__version__ = "0.1.0"
