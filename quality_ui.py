"""Presentation of authoritative Python evidence and strictly optional AI advice."""

import json
import streamlit as st
from quality_report import config_hash
from quality_review import evidence_for, QualityReviewRun


def collect_quality_review():
    run = st.session_state.get("quality_run")
    if not run:
        return
    snapshot = run.snapshot()
    if snapshot["status"] != "running" and st.session_state.get("quality_recorded") != id(run):
        st.session_state.quality_recorded = id(run)
        st.session_state.quality_answer = dict(
            report_sha256=run.evidence["report_sha256"],
            context_sha256=config_hash(run.evidence),
            history=snapshot["history"],
            status=snapshot["status"],
            error=snapshot["error"],
            **snapshot["result"],
        )
        st.session_state.setdefault("quality_history", []).append(st.session_state.quality_answer)
        if snapshot["result"].get("usage"):
            st.session_state.usage.append(snapshot["result"]["usage"])


def quality_panel(quality):
    collect_quality_review()
    st.subheader("Dataset quality report")
    if quality["gate_passed"]:
        st.success(
            "Python quality gate: passed. This covers the checks listed below—not a guarantee of industry realism."
        )
    else:
        st.error(
            "Python quality gate: failed. Dataset download stays blocked, regardless of AI advice."
        )
    st.write(f"{quality['errors']} failed checks · {quality['warnings']} warnings or coverage gaps")
    st.caption(
        "These are counts of checks, not unique affected rows. Every emitted CSV row was scanned; preview rows do not determine this result."
    )
    st.dataframe(
        [
            dict(table=name, rows=info["rows"], role=info["role"])
            for name, info in quality["tables"].items()
        ],
        hide_index=True,
        width="stretch",
    )
    issues = [
        dict(
            table=c["table"],
            check=c["check"],
            status=c["severity"],
            affected=c["violations"],
            details=c.get("detail", ""),
            example_row_numbers=", ".join(map(str, c.get("row_numbers", []))),
        )
        for c in quality["checks"]
        if c["severity"] != "ok"
    ]
    if issues:
        with st.expander("What needs attention", expanded=not quality["gate_passed"]):
            st.dataframe(issues, hide_index=True, width="stretch")
            st.write(
                "For a design mismatch, return to Chat and specify the correct rule or relationship. If the config is correct but the output is wrong, share the report with the developer. Do not relax a rule just to pass."
            )
            st.caption(
                "Example row numbers count data rows from 1, excluding the CSV header. A coverage warning means a check was not independently performed, not that a defect was found."
            )
    with st.expander("Measurements, passed checks and limitations"):
        st.json(quality)
    st.download_button(
        "Download quality report (JSON)",
        json.dumps(quality, indent=2, allow_nan=False),
        file_name="cardinal_quality_report.json",
        mime="application/json",
    )

    with st.expander("Optional AI dataset reviewer"):
        st.write(
            "Check the actual dataset evidence against every approved requirement. This is one paid review, not just a report explanation. It cannot override Python failures or automatically repair data."
        )
        try:
            current = st.session_state.result or {}
            evidence = evidence_for(
                quality, current.get("brief", st.session_state.brief), current.get("contract", {})
            )
        except ValueError as exc:
            st.warning(str(exc))
            return
        if st.checkbox(
            "Show exactly what will be shared with Claude",
            key="quality_evidence_" + evidence["report_sha256"],
        ):
            st.json(evidence)
        share = st.checkbox(
            "Share the approved brief, config, aggregate statistics and checks with Anthropic for dataset review.",
            key="quality_share_" + evidence["report_sha256"],
        )
        st.caption(
            "No raw CSV rows or category-frequency tables are sent. The approved brief/config, numeric/date summaries and check details are sent. Inspect the exact evidence above; use only approved fictional data. Unmeasured requirements must be reported as insufficient evidence."
        )
        run = st.session_state.get("quality_run")
        busy = bool(run and run.alive())
        design = st.session_state.get("design_run")
        interview = st.session_state.get("interview_run")
        if st.button(
            "Review dataset against my requirements",
            disabled=not share
            or not st.session_state.consent
            or busy
            or bool(design and design.alive())
            or bool(interview and interview.alive()),
        ):
            st.session_state.quality_answer = None
            st.session_state.quality_recorded = None
            st.session_state.quality_run = QualityReviewRun(evidence)
            st.rerun()
        if busy:
            st.caption(
                "The current review stage and Stop dataset review are in the bottom panel, on every page."
            )
        answer = st.session_state.get("quality_answer")
        if answer and answer["context_sha256"] == config_hash(evidence):
            if answer["status"] != "complete":
                st.warning(answer["error"])
            else:
                review = answer["review"]
                st.caption(
                    "AI assessment of supplied evidence, not an independent full-row audit. Python remains authoritative."
                )
                st.write(review["summary"])
                requirements = {
                    r["id"]: r["text"] for r in evidence["approved_brief"]["requirements"]
                }
                st.dataframe(
                    [
                        dict(
                            requirement=f["requirement_id"],
                            request=requirements[f["requirement_id"]],
                            verdict=f["verdict"],
                            evidence=", ".join(f["evidence_ids"]),
                            explanation=f["explanation"],
                            next_step=f["suggested_action"],
                        )
                        for f in review["requirements"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                if any(f["verdict"] == "mismatch" for f in review["requirements"]):
                    st.warning(
                        "The reviewer found a possible mismatch with your request. Resolve it in Chat before treating the dataset as compliant."
                    )
                elif any(f["verdict"] == "insufficient_evidence" for f in review["requirements"]):
                    st.warning(
                        "Some requirements could not be verified from the available evidence. This is not a full compliance pass."
                    )
                for limitation in review["limitations"]:
                    st.caption(limitation)
                st.download_button(
                    "Download dataset review (JSON)",
                    json.dumps(answer, indent=2),
                    file_name="cardinal_dataset_review.json",
                    mime="application/json",
                )
