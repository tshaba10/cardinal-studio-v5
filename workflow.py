"""Actual Claude interviewer/designer/reviewer graphs with bounded repair."""

from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from claude_client import ask
from models import Interview, Design, Review
from prompts import INTERVIEW, INTERVIEW_MODES, DESIGN, REVIEW, reference_context
from contract_checks import intent_errors
from engine_adapter import validate_contract
from safety import parse_json
from time import monotonic


class State(TypedDict, total=False):
    """Internal graph state: intent, candidate, check results, and run telemetry.

    This is not the exported JSON contract. Each node returns only its updates;
    the control object is local cancellation state and is never sent to Claude.
    """

    brief: dict
    messages: list[dict]
    upload: dict
    reply: str
    design: dict
    contract: dict
    report: dict
    review: dict
    attempt: int
    usage: list[dict]
    trace: list[str]
    feedback: dict
    attempts: list[dict]
    control: object
    stop_reason: str
    interview_mode: str


def stage(state, label, node):
    if state.get("control"):
        state["control"].stage(label, node)


def request(state, *args):
    control = state.get("control")
    answer = ask(*args, **({"control": control} if control else {}))
    if control:
        control.check()
    return answer


def interviewer(state):
    """Update the brief while retaining stable requirement IDs across chat turns."""
    stage(state, "Understanding your request and updating the requirements", "interviewer")
    payload = {k: state.get(k, {}) for k in ("brief", "upload", "feedback")}
    payload["messages"] = [
        dict(role=m.get("role", "user"), content=m["content"]) for m in state.get("messages", [])
    ]
    mode = state.get("interview_mode", "detailed")
    payload["interview_mode"] = mode
    answer, usage = request(state, INTERVIEW + INTERVIEW_MODES[mode], payload, Interview)
    old_ids = {r["id"] for r in state["brief"]["requirements"]}
    if not old_ids <= {r["id"] for r in answer["brief"]["requirements"]}:
        raise ValueError(
            "Claude omitted an earlier requirement. Nothing was applied. Retry and ask it to retain every requirement ID, marking explicit deferrals instead of deleting them."
        )
    return {**answer, "usage": state.get("usage", []) + [usage]}


def designer(state):
    """Draft or repair a contract without changing approved user meaning."""
    attempt = state.get("attempt", 0) + 1
    stage(
        state,
        "Drafting your generator config" if attempt == 1 else "Applying the one automatic repair",
        "designer",
    )
    payload = dict(brief=state["brief"], references=reference_context(state["brief"]))
    if state.get("feedback"):
        payload["previous_attempt_feedback"] = state["feedback"]
    if state.get("attempt", 0):
        review = state.get("review", {})
        payload.update(
            previous_design=state.get("design"),
            errors=state.get("report", {}).get("errors", [])
            + review.get("missing_requirements", [])
            + review.get("concerns", []),
        )
    started = monotonic()
    answer, usage = request(state, DESIGN, payload, Design, "designer")
    return dict(
        design=answer,
        review={},
        attempt=attempt,
        attempts=state.get("attempts", [])
        + [dict(attempt=attempt, stage="designer", seconds=round(monotonic() - started, 2))],
        usage=state.get("usage", []) + [usage],
        trace=state.get("trace", []) + ["designer"],
    )


def validator(state):
    """Apply deterministic intent checks and the actual generator compiler."""
    stage(state, "Validating your design with the Python generator", "validator")
    started = monotonic()
    contract = {}
    try:
        contract = parse_json(state["design"]["contract_json"])
        errors = intent_errors(contract, state["brief"], state["design"])
        report = validate_contract(
            contract, **({"control": state["control"]} if state.get("control") else {})
        )
        report["errors"] = list(dict.fromkeys(errors + report.get("errors", [])))
        report["ok"] = report["ok"] and not errors
    except (ValueError, TypeError) as exc:
        report = dict(ok=False, errors=[str(exc)])
    if state.get("control"):
        state["control"].check()
    return dict(
        contract=contract,
        report=report,
        attempts=state.get("attempts", [])
        + [
            dict(
                attempt=state["attempt"],
                stage="validator",
                seconds=round(monotonic() - started, 2),
                errors=report["errors"],
            )
        ],
        trace=state.get("trace", []) + ["validator"],
    )


def route(state):
    if state["report"]["ok"]:
        return "reviewer"
    if state["design"].get("unresolved"):
        return "blocked"
    return "repair" if state["attempt"] < 2 else "blocked"


def reviewer(state):
    """Check semantic coverage independently of compiler success and design claims."""
    stage(state, "Reviewing whether the design matches your requirements", "reviewer")
    started = monotonic()
    review, usage = request(
        state,
        REVIEW,
        dict(
            brief=state["brief"],
            contract=state["contract"],
            coverage=state["design"]["coverage"],
            designer_warnings=state["design"].get("warnings", []),
            suggested_next_steps=state["design"].get("suggested_next_steps", []),
        ),
        Review,
        "reviewer",
    )
    return dict(
        review=review,
        usage=state.get("usage", []) + [usage],
        attempts=state.get("attempts", [])
        + [
            dict(
                attempt=state["attempt"],
                stage="reviewer",
                seconds=round(monotonic() - started, 2),
                errors=review.get("missing_requirements", []) + review.get("concerns", []),
            )
        ],
        trace=state.get("trace", []) + ["reviewer"],
    )


def after_review(state):
    """Spend at most the remaining repair attempt; warnings alone never retry."""
    review = state["review"]
    errors = review.get("missing_requirements", []) + review.get("concerns", [])
    return "repair" if errors and state["attempt"] < 2 else "finished"


def interview_graph():
    graph = StateGraph(State)
    graph.add_node("interviewer", interviewer)
    graph.add_edge(START, "interviewer")
    graph.add_edge("interviewer", END)
    return graph.compile()


def design_graph():
    graph = StateGraph(State)
    graph.add_node("designer", designer)
    graph.add_node("validator", validator)
    graph.add_node("reviewer", reviewer)
    graph.add_edge(START, "designer")
    graph.add_edge("designer", "validator")
    graph.add_conditional_edges(
        "validator", route, {"repair": "designer", "reviewer": "reviewer", "blocked": END}
    )
    graph.add_conditional_edges("reviewer", after_review, {"repair": "designer", "finished": END})
    return graph.compile()
