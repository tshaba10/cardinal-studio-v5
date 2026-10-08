"""Explicit, auditable acceptance of AI-proposed defaults; no invisible approval."""

from copy import deepcopy
from quality_report import config_hash


def needs_acceptance(brief):
    return bool(
        brief.get("assumptions") or any(r["status"] == "proposed" for r in brief["requirements"])
    )


def design_brief(brief, accepted=False):
    result = deepcopy(brief)
    if accepted:
        for requirement in result["requirements"]:
            if requirement["status"] == "proposed":
                requirement["status"] = "confirmed"
    return result


def acceptance_key(brief):
    return "accept_defaults_" + config_hash(brief)
