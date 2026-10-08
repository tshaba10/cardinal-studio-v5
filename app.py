"""Streamlit entry point for Cardinal Studio.

The four pages collect intent, approve a checked contract, preview/export data,
and explain the workflow. Background runners keep model calls and generation
out of the UI thread. Never approve a partial result or bypass the Python gate.
Run from this directory with: python -m streamlit run app.py
"""

from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from models import blank_brief
from workflow import interview_graph, design_graph
from contract_checks import brief_errors
from importer import import_draft, sheet_names
from privacy import looks_sensitive
from jobs import start_job, cancel_job, job_status, progress, preview, fingerprint
from downloads import DownloadServer
from graphs import markdown, svg
from design_runner import DesignRun, InterviewRun
from interview_settings import needs_acceptance, design_brief, acceptance_key
from quality_report import download_allowed
from quality_review import quality_graph
from quality_ui import quality_panel, collect_quality_review

ROOT = Path(__file__).resolve().parent
PAGES = ["Chat", "Review", "Preview & export", "Diagrams & session notes"]
ACTIVITY_STYLE = """<style>
.st-key-activity_footer, .st-key-interview_activity {
    position:fixed; bottom:8rem; right:1rem;
    width:min(32rem, calc(100vw - 2rem)); max-height:38vh;
    overflow-y:auto; z-index:990; background:#fff;
    border:2px solid #155e75 !important; border-radius:12px;
    box-shadow:0 4px 18px #172b4d26; padding:0.75rem;
}
[data-testid="stMainBlockContainer"] {padding-bottom:calc(38vh + 7rem);}
</style>"""
st.set_page_config(page_title="Cardinal Studio", page_icon="🗂️", layout="wide")
st.markdown(
    """<style>
.stApp { color:#172b4d; } h1,h2,h3 {color:#102a43;}
[data-testid="stSidebar"] {background:#eef4f8;}
[data-testid="stChatMessage"] {border:1px solid #d9e4ec;border-radius:12px;}
[data-testid="stMainBlockContainer"] {max-width:1100px;}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def download_server():
    return DownloadServer()


def initialize():
    """Create per-browser-session state; no chat is restored from disk on restart."""
    defaults = dict(
        page=0,
        session_id=str(uuid.uuid4()),
        brief=blank_brief(),
        messages=[],
        upload={},
        result=None,
        approved=None,
        job=None,
        url=None,
        consent=False,
        failure=None,
        usage=[],
        design_run=None,
        recorded_run=None,
        run_history=[],
        quality_run=None,
        quality_recorded=None,
        quality_answer=None,
        quality_history=[],
        interview_run=None,
        interview_recorded=None,
        pending_turn=None,
        interview_mode="quick",
        assumption_acceptances=[],
        accepted_brief_hash=None,
    )
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def invalidate():
    """Revoke outputs when requirements change, retaining the conversation itself."""
    if st.session_state.job and job_status(st.session_state.job)["status"] == "running":
        cancel_job(st.session_state.job)
    if st.session_state.url:
        download_server().revoke(st.session_state.url)
    for key in ("result", "approved", "job", "url"):
        st.session_state[key] = None
    if not design_busy():
        st.session_state.design_run = None
        st.session_state.recorded_run = None
        st.session_state.quality_run = None
        st.session_state.quality_recorded = None
        st.session_state.quality_answer = None


def go(page):
    st.session_state.page = page
    st.rerun()


def show_errors(errors):
    for error in errors:
        st.error(error)


def consent():
    # Widget keys disappear when a page does not render them; keep consent separately.
    st.session_state.setdefault("_consent_widget", st.session_state.consent)

    def remember():
        st.session_state.consent = st.session_state._consent_widget
        if not st.session_state.consent and st.session_state.design_run:
            st.session_state.design_run.stop()
        if not st.session_state.consent and st.session_state.quality_run:
            st.session_state.quality_run.stop()
        if not st.session_state.consent and st.session_state.interview_run:
            st.session_state.interview_run.stop()

    st.checkbox(
        "I allow this fictional design, chat, and any attached file summary to be sent to Anthropic.",
        key="_consent_widget",
        on_change=remember,
    )
    st.caption(
        "Claude API usage is paid separately through your Console balance. Use employer-approved access. Never enter customer data, passwords or API keys here."
    )


def attachments():
    with st.expander("Attach CSV or Excel (optional)"):
        file = st.file_uploader("CSV or Excel (.xlsx), under 1 MB", type=["csv", "xlsx"])
        if file:
            try:
                data = file.getvalue()
                sheet = (
                    st.selectbox("Worksheet", sheet_names(data))
                    if file.name.lower().endswith(".xlsx")
                    else None
                )
                separator = (
                    st.selectbox("CSV separator", [",", ";", "\t"]) if sheet is None else ","
                )
                inferred, report = import_draft(data, file.name, sheet, separator)
                st.dataframe(report, hide_index=True)
                share = st.checkbox(
                    "Also include the fictional sample values in the AI context", value=False
                )
                st.caption(
                    "By default only column names and inferred types are attached. Samples do not prove distributions or relationships."
                )
                if st.button("Attach these columns", disabled=design_busy()):
                    context = (
                        {"columns": inferred["elements"]}
                        if share
                        else {
                            "columns": [
                                {"name": x["col_name"], "type": x["type"]}
                                for x in inferred["elements"]
                            ]
                        }
                    )
                    invalidate()
                    st.session_state.upload = context
                    question = "How should the attached columns be used or change the current dataset requirements?"
                    if question not in st.session_state.brief["questions"]:
                        st.session_state.brief["questions"].append(question)
                    st.success(
                        "Columns attached. Describe their meaning and relationships in the chat."
                    )
            except ValueError as exc:
                st.error(str(exc))
    if st.session_state.upload:
        with st.expander("Attached context — this is what Claude will receive"):
            st.json(st.session_state.upload)
            if st.button("Remove attachment", disabled=design_busy()):
                invalidate()
                st.session_state.upload = {}
                st.rerun()


def design_busy():
    run = st.session_state.design_run
    quality = st.session_state.quality_run
    interview = st.session_state.interview_run
    return bool(
        (run and run.alive()) or (quality and quality.alive()) or (interview and interview.alive())
    )


def prepare():
    """Snapshot explicitly accepted requirements and start a paid design attempt."""
    brief = st.session_state.brief
    accepted = st.session_state.accepted_brief_hash == fingerprint(brief)
    candidate = design_brief(brief, accepted)
    errors = brief_errors(candidate)
    if needs_acceptance(brief) and not accepted:
        errors.append("Review and accept the proposed settings and assumptions before designing.")
    if errors or not st.session_state.consent or design_busy():
        raise ValueError(
            "Answer the open questions and enable AI sharing before preparing a design."
        )
    previous = st.session_state.result or {}
    feedback = {key: previous.get(key, {}) for key in ("design", "report", "review", "run_error")}
    invalidate()
    if needs_acceptance(brief):
        st.session_state.assumption_acceptances.append(
            dict(
                brief_sha256=fingerprint(brief),
                assumptions=brief["assumptions"],
                proposed_requirements=[
                    r for r in brief["requirements"] if r["status"] == "proposed"
                ],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
    st.session_state.brief = candidate
    if accepted:
        st.session_state.accepted_brief_hash = fingerprint(candidate)
        st.session_state[acceptance_key(candidate)] = True
    st.session_state.design_run = DesignRun(candidate, feedback=feedback)
    st.session_state.recorded_run = None


def run_progress(run, graph, stop_label):
    snapshot = run.snapshot()
    if st.button(stop_label, disabled=snapshot["status"] != "running"):
        run.stop()
        st.rerun()
    st.info(f"{snapshot['stage']} · {snapshot['elapsed']} seconds elapsed")
    st.caption(f"Graph node: {snapshot['node']} · {snapshot['detail']}")
    st.caption(
        f"Last worker update: {snapshot['quiet_seconds']} seconds ago. No elapsed-time cutoff; use Stop above when needed."
    )
    if snapshot["quiet_seconds"] >= 30:
        st.warning(
            "Still waiting at this stage. No new worker update has arrived; this is not a completion estimate."
        )
    with st.expander("Where am I in the graph?"):
        st.image(svg(graph, active=snapshot["node"]))
        st.dataframe(snapshot["history"], hide_index=True)


def collect_interview():
    """Apply a complete interview once; failures keep the previous brief intact."""
    run = st.session_state.interview_run
    if not run:
        return False
    snapshot = run.snapshot()
    if snapshot["status"] == "running" or st.session_state.interview_recorded == id(run):
        return False
    st.session_state.interview_recorded = id(run)
    turn = st.session_state.pending_turn
    st.session_state.run_history.append(
        dict(
            kind="interview",
            status=snapshot["status"],
            error=snapshot["error"],
            elapsed=snapshot["elapsed"],
            history=snapshot["history"],
        )
    )
    if snapshot["status"] == "complete":
        answer = snapshot["state"]
        invalidate()
        st.session_state.brief = answer["brief"]
        st.session_state.messages += [
            turn,
            dict(
                role="assistant",
                content=answer["reply"],
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        ]
        st.session_state.usage += answer.get("usage", [])
        st.session_state.failure = None
    else:
        st.session_state.failure = dict(text=turn["content"], error=snapshot["error"])
    st.session_state.pending_turn = None
    return True


def design_progress():
    """Render progress, collect completed results, and route only passing designs."""
    run = st.session_state.design_run
    if not run:
        return
    snapshot = run.snapshot()
    if snapshot["status"] == "running":
        run_progress(run, design_graph(), "Stop designing")
        st.caption("At most one automatic repair. No dataset is being generated yet.")
        return
    if st.session_state.recorded_run != id(run):
        state = snapshot["state"]
        state["run_status"] = snapshot["status"]
        state["run_error"] = snapshot["error"]
        st.session_state.result = state
        st.session_state.usage += state.get("usage", [])
        st.session_state.run_history.append(
            dict(
                status=snapshot["status"],
                error=snapshot["error"],
                elapsed=snapshot["elapsed"],
                history=snapshot["history"],
                attempts=state.get("attempts", []),
            )
        )
        st.session_state.recorded_run = id(run)
        review = state.get("review", {})
        ready = (
            snapshot["status"] == "complete"
            and state.get("report", {}).get("ok")
            and review
            and not review.get("missing_requirements")
            and not review.get("concerns")
        )
        if ready:
            st.session_state.page = 1
        st.rerun()
    if run.alive():
        st.warning(
            "Stopping the local worker. Its partial answer will not be applied. The provider may still bill work already started."
        )
    elif snapshot["status"] != "complete":
        st.warning(snapshot["error"])
    # A final full rerun enables controls once a cancelled in-flight request has exited.
    if not run.alive() and st.session_state.get("_awaiting_design_exit"):
        st.session_state._awaiting_design_exit = False
        st.rerun()
    if run.alive():
        st.session_state._awaiting_design_exit = True


@st.fragment(run_every="1s")
def activity_footer():
    """One visible bottom activity panel, shared by every page."""
    if collect_interview():
        st.rerun()
    job = st.session_state.job
    result = job_status(job) if job else None
    generating = bool(result and result["status"] == "running")
    marker = (job["folder"], result["status"]) if job else None
    old_marker = st.session_state.get("_activity_job")
    st.session_state._activity_job = marker
    if old_marker and old_marker != marker:
        st.rerun()  # Refresh buttons and preview when a background worker finishes.
    quality = st.session_state.quality_run
    old_recorded = st.session_state.quality_recorded
    collect_quality_review()
    if old_recorded != st.session_state.quality_recorded:
        st.rerun()
    active = design_busy() or generating
    with st.container(key="activity_footer", border=active):
        if active:
            st.markdown(ACTIVITY_STYLE, unsafe_allow_html=True)
            st.caption("CURRENT ACTIVITY · updates automatically")
        design_progress()
        interview = st.session_state.interview_run
        if interview and interview.alive():
            run_progress(interview, interview_graph(), "Stop interview")
            st.session_state._awaiting_interview_exit = True
        elif st.session_state.get("_awaiting_interview_exit"):
            st.session_state._awaiting_interview_exit = False
            st.rerun()
        if generating:
            st.info(
                "Generating and checking your dataset locally. No AI calls are being made for generation."
            )
            lines = progress(job).strip().splitlines()
            st.caption(
                "Current worker update: "
                + (lines[-1] if lines else "Starting the generation worker")
            )
            with st.expander("Latest generation details"):
                st.code(progress(job), language=None)
            if st.button("Stop generation", key="footer_stop_generation"):
                cancel_job(job)
                st.rerun()
        elif result and st.session_state.page != 2:
            if result["status"] == "complete":
                st.info(
                    "Dataset run finished. Open Preview & export to inspect its quality report and results."
                )
            else:
                st.warning(result.get("error", "Dataset generation did not complete."))
            if st.button("View dataset results", key="footer_dataset_results"):
                go(2)
        if quality and quality.alive():
            if quality.snapshot()["status"] == "running":
                run_progress(quality, quality_graph(), "Stop dataset review")
            else:
                st.warning(
                    "Dataset review stopped. Closing the local worker; the provider may still bill work already started."
                )
                st.button("Stop dataset review", disabled=True)
            st.session_state._awaiting_quality_exit = True
        elif st.session_state.get("_awaiting_quality_exit"):
            st.session_state._awaiting_quality_exit = False
            st.rerun()


def send(text):
    """Screen a user turn and pass the full approved context to the interviewer."""
    if not st.session_state.consent:
        raise ValueError("Enable the Anthropic data-sharing consent before chatting.")
    if len(st.session_state.messages) >= 100:
        raise ValueError(
            "This session has reached 100 messages. Download your notes and start a new chat with the confirmed brief. Earlier requirements will not be silently dropped."
        )
    # Scan user/AI content, not our generated UUIDs/timestamps (which can accidentally pass Luhn).
    content_history = [
        dict(role=m["role"], content=m["content"]) for m in st.session_state.messages
    ]
    payload = json.dumps([text, st.session_state.upload, st.session_state.brief, content_history])
    if looks_sensitive(payload) or "sk-ant-" in payload or "sk-or-" in payload:
        raise ValueError(
            "Possible sensitive data or API key detected. Remove it; use fictional tokens such as ACC-00000001."
        )
    turn = dict(
        role="user",
        content=text,
        message_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    if design_busy():
        raise ValueError("Stop or finish the current AI task before sending another message.")
    st.session_state.pending_turn = turn
    st.session_state.interview_recorded = None
    st.session_state.interview_run = InterviewRun(
        dict(
            brief=st.session_state.brief,
            messages=st.session_state.messages + [turn],
            upload=st.session_state.upload,
            interview_mode=st.session_state.interview_mode,
            feedback={
                key: st.session_state.result.get(key, {}) for key in ("report", "review", "design")
            }
            if st.session_state.result
            else {},
        )
    )
    st.session_state.failure = None


def chat_design_actions():
    """Keep the next action beside the latest reply, not above the transcript."""
    brief = st.session_state.brief
    accepted = st.session_state.accepted_brief_hash == fingerprint(brief)
    errors = brief_errors(design_brief(brief, accepted))
    if brief["requirements"]:
        with st.expander("Your captured requirements", expanded=False):
            st.write(brief["purpose"])
            for requirement in brief["requirements"]:
                st.write(f"{requirement['id']} · {requirement['text']} ({requirement['status']})")
            st.write("Record counts:", {v["entity"]: v["count"] for v in brief["volumes"]})
            st.write("Random seed:", brief["seed"])
            for assumption in brief["assumptions"]:
                st.write("Assumption: " + assumption)
            if errors:
                st.write("Still to clarify:")
                for error in errors:
                    st.write("• " + error)
        if needs_acceptance(brief):
            st.write("Proposed settings to review before designing:")
            for assumption in brief["assumptions"]:
                st.write("• " + assumption)
            for r in brief["requirements"]:
                if r["status"] == "proposed":
                    st.write(f"• {r['id']}: {r['text']}")
            key = acceptance_key(brief)
            st.session_state.setdefault(key, accepted)

            def remember_acceptance():
                st.session_state.accepted_brief_hash = (
                    fingerprint(brief) if st.session_state[key] else None
                )

            st.checkbox(
                "I accept the listed proposed settings and assumptions for this demo.",
                key=key,
                on_change=remember_acceptance,
                disabled=design_busy(),
            )
            if not accepted:
                errors.append("Accept the proposed settings or ask for changes in chat.")
        if not errors:
            previous = st.session_state.result or {}
            review = previous.get("review", {})
            if previous and (
                not previous.get("report", {}).get("ok")
                or review.get("concerns")
                or review.get("missing_requirements")
            ):
                st.caption(
                    "Your brief is saved. Clarify the feedback in chat, or prepare again to request a new attempt with the same approved requirements and check details. This uses paid AI calls; no requirement is automatically relaxed."
                )
            else:
                st.caption(
                    "Ready to design. By preparing, you confirm the captured requirements and assumptions. This uses paid AI calls; it does not generate data."
                )
        else:
            st.caption(
                "Not ready yet? Open Your captured requirements above to see what is missing, then answer in the chat below."
            )
        if st.button(
            "Prepare my design",
            type="primary",
            width="stretch",
            disabled=bool(errors) or not st.session_state.consent or design_busy(),
        ):
            prepare()
            st.rerun()


def design_feedback():
    """Latest failed-design feedback belongs after the transcript, beside the next action."""
    result = st.session_state.result
    if not result:
        return
    design = result.get("design", {})
    report = result.get("report", {})
    review = result.get("review", {})
    concerns = review.get("missing_requirements", []) + review.get("concerns", [])
    failed = result.get("run_status") in ("failed", "stopped") or (
        result.get("run_status", "complete") == "complete" and (not report.get("ok") or concerns)
    )
    if not failed:
        return
    with st.container(border=True, key="latest_design_feedback"):
        st.subheader("Latest design feedback — what to do next")
        st.warning(
            "The design is not ready for approval. Your requirements are saved; no failed config can be approved."
        )
        if result.get("run_error"):
            st.error(result["run_error"])
        unresolved = design.get("unresolved", [])
        if unresolved:
            st.write("Decisions or unresolved items from the designer:")
            for item in unresolved:
                st.write("• " + item)
            st.caption(
                "Reply below to resolve actual choices. Verification notes are not permission to change an accepted requirement."
            )
        if concerns:
            st.write("Requirements that the design reviewer says need correction:")
            for item in concerns:
                st.write("• " + item)
        suggestions = design.get("suggested_next_steps", [])
        if suggestions:
            st.write("Suggested next steps — proposals for your approval:")
            for item in suggestions:
                st.write("• " + item)
        elif unresolved or concerns:
            st.info(
                "In the chat below, you can say: “Explain which items need my decision and propose supported alternatives. Keep my accepted requirements unchanged unless I approve a change.”"
            )
        else:
            st.info(
                "Open the check details below. You do not need to edit JSON: choose Prepare my design to request another attempt with these errors, or clarify your request in chat. Retrying uses paid AI calls."
            )
        notes = design.get("warnings", []) + review.get("warnings", [])
        if notes:
            with st.expander("Advisory notes — not clarification questions"):
                for note in notes:
                    st.write("• " + note)
        with st.expander("Check details (for troubleshooting)"):
            show_errors(report.get("errors", []))
            show_errors(concerns)
            if result.get("run_error"):
                st.write(result["run_error"])
            st.caption(
                "Technical check errors stay blocking. Advisory notes do not bypass Python or requirement review."
            )


def chat():
    """Render history first, then feedback, approvals, and the message composer."""
    st.header("What would you like to generate?")
    st.caption(
        "Describe your fictional dataset naturally. Cardinal asks questions, designs the JSON, and checks it behind the scenes."
    )
    st.session_state.setdefault("_interview_mode", st.session_state.interview_mode)

    def remember_mode():
        st.session_state.interview_mode = st.session_state._interview_mode

    st.radio(
        "How much guidance would you like?",
        ["quick", "detailed"],
        horizontal=True,
        format_func=lambda mode: (
            "Quick — fewer questions, proposed defaults"
            if mode == "quick"
            else "Detailed — clarify more choices"
        ),
        key="_interview_mode",
        on_change=remember_mode,
        disabled=design_busy(),
    )
    st.caption(
        "Quick mode proposes transparent demo assumptions for you to accept together. Detailed mode asks more questions. Neither mode silently changes your instructions."
    )
    if not st.session_state.consent:
        st.info(
            "To start chatting, enable AI sharing under AI connection & privacy in the sidebar. Uploading a file is optional."
        )
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write(
                "Tell me what the dataset is for, the columns you need, and how many records you want. Relationships and calculations are welcome. We will clarify anything missing together."
            )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if st.session_state.pending_turn:
        with st.chat_message("user"):
            st.write(st.session_state.pending_turn["content"])
        st.caption(
            "Interview in progress. Your last saved brief stays unchanged until a complete reply arrives."
        )
    design_feedback()
    failure = st.session_state.failure
    retry = False
    if failure:
        st.error(failure["error"])
        with st.expander("Your unsent message"):
            st.write(failure["text"])
        retry = st.button("Retry message", disabled=not st.session_state.consent or design_busy())
    chat_design_actions()
    text = st.chat_input(
        "Describe, clarify, or change your requirements…",
        max_chars=20000,
        disabled=not st.session_state.consent or design_busy(),
    )
    if text or retry:
        text = text or failure["text"]
        try:
            send(text)
        except (ValueError, OSError) as exc:
            st.session_state.failure = dict(text=text, error=str(exc))
        st.rerun()


def review():
    """Require both passing checks and human approval of this exact contract."""
    st.header("Review your design")
    result = st.session_state.result
    if not result:
        st.info("Describe your dataset in Chat, then choose Prepare my design.")
        if st.button("Go to Chat"):
            go(0)
        return
    st.write(
        result.get("design", {}).get(
            "summary", "No completed design yet. Your requirements are still saved."
        )
    )
    report = result.get("report", {})
    if result.get("run_error"):
        st.warning(result["run_error"])
    if report.get("errors"):
        st.error(
            "The config needs correction before approval. Return to Chat; the designer will receive these check details."
        )
        with st.expander("What needs attention", expanded=True):
            show_errors(report["errors"])
    ai_review = result.get("review", {})
    if ai_review:
        st.write("AI review:", ai_review["summary"])
    concerns = ai_review.get("missing_requirements", []) + ai_review.get("concerns", [])
    show_errors(concerns)
    for warning in ai_review.get("warnings", []):
        st.warning(warning)
    for warning in result.get("design", {}).get("warnings", []):
        st.warning(warning)
    with st.expander("Where each requirement was implemented"):
        st.dataframe(result.get("design", {}).get("coverage", []), hide_index=True, width="stretch")
    with st.expander("Inspect JSON and Python validation details"):
        st.json(result.get("contract", {}))
        st.json(report)
        st.json(result.get("attempts", []))
    if st.button("Return to chat to address feedback"):
        go(0)
    valid = (
        report.get("ok")
        and bool(ai_review)
        and not concerns
        and result.get("run_status", "complete") == "complete"
    )
    if valid:
        st.success(
            "The generator accepts this config, and the AI review found no blocking requirement issues."
        )
    st.caption(
        "These checks do not prove industry realism or every business rule. You approve the design; generated data is checked separately before download."
    )
    accepted = st.checkbox(
        "I reviewed the design and assumptions, and approve this config.",
        key="contract_approved_" + fingerprint(result.get("contract", {})),
    )
    if st.button("Approve and continue", type="primary", disabled=not valid or not accepted):
        st.session_state.approved = fingerprint(result["contract"])
        go(2)


@st.fragment(run_every="2s")
def job_panel():
    """Show measured quality and a preview; offer downloads only after acceptance."""
    job = st.session_state.job
    if not job:
        return
    result = job_status(job)
    if result["status"] == "running":
        st.caption(
            "Generation progress and Stop generation are in the bottom activity panel, on every page."
        )
        return
    if result["status"] != "complete":
        st.error(result["error"])
        if "quality" in result:
            quality_panel(result["quality"])
        elif "verification" in result:
            st.json(result["verification"])
        st.caption(
            "The approved JSON remains downloadable. Partial files are not offered as verified datasets."
        )
        return
    if not download_allowed(result):
        st.error(
            "This run has no passing current Python quality report. Regenerate the dataset before downloading."
        )
        return
    quality_panel(result["quality"])
    table = st.selectbox("Which table would you like to preview?", list(result["run"]["rows"]))
    columns, rows = preview(job, table)
    st.caption(
        "First 100 rows from the actual output, not a separate sample. Fewer are shown if this table has fewer rows."
    )
    st.dataframe(pd.DataFrame(rows, columns=columns), height=450, hide_index=True, width="stretch")
    approved = st.checkbox(
        "I reviewed the preview, Python warnings, and any optional dataset-review findings and accept this dataset.",
        key="dataset_approved_" + job["folder"],
    )
    if not approved and st.session_state.url:
        download_server().revoke(st.session_state.url)
        st.session_state.url = None
    if approved:
        if not st.session_state.url:
            try:
                st.session_state.url = download_server().grant(Path(job["folder"]) / "dataset.zip")
            except ValueError as exc:
                st.error(str(exc))
                return
        st.link_button("Download full dataset (ZIP)", st.session_state.url, type="primary")
        st.caption(
            "The local download streams the ZIP without loading it all into Streamlit memory. Keep Cardinal Studio running until the download finishes."
        )
    st.caption(
        "Files also remain locally at: "
        + job["folder"]
        + ". Close Cardinal Studio before manually removing run folders you no longer need."
    )


def export():
    """Keep JSON as the primary output; generation is a separate opt-in action."""
    st.header("Preview & export")
    result = st.session_state.result
    if not result or st.session_state.approved != fingerprint(result.get("contract", {})):
        st.info("Review and approve the current config first.")
        if st.button("Go to Review"):
            go(1)
        return
    contract = result["contract"]
    st.download_button(
        "Download JSON config — primary output",
        json.dumps(contract, indent=2),
        file_name="cardinal_contract.json",
        mime="application/json",
        type="primary",
    )
    st.write(
        "The config is ready for the downstream generator. To inspect data here, generate the requested volume below."
    )
    st.caption(
        "No fixed row-count cap. Large jobs can exhaust memory or disk. Generation runs locally, not in Claude; start modestly and increase deliberately."
    )
    active = st.session_state.job and job_status(st.session_state.job)["status"] == "running"
    confirmed = st.checkbox(
        "I understand the requested volume and available resources, and want to generate it locally."
    )
    if st.button(
        "Generate the requested dataset", disabled=not confirmed or bool(active) or design_busy()
    ):
        try:
            if st.session_state.url:
                download_server().revoke(st.session_state.url)
                st.session_state.url = None
            st.session_state.job = start_job(contract)
            st.session_state.quality_run = None
            st.session_state.quality_answer = None
            st.session_state.quality_recorded = None
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    job_panel()


def notes_page():
    """Export real graph topology and a private audit record of the session."""
    st.header("Diagrams & session notes")
    graphs = {
        "Interview": interview_graph(),
        "Design, validation and review": design_graph(),
        "Optional dataset requirements review": quality_graph(),
    }
    selected = st.selectbox("Graph", list(graphs))
    st.image(svg(graphs[selected]))
    st.download_button(
        "Download graph (SVG)",
        svg(graphs[selected]),
        file_name="cardinal_graph.svg",
        mime="image/svg+xml",
    )
    st.download_button(
        "Download all graphs (Markdown)", markdown(graphs), file_name="langgraph_diagrams.md"
    )
    notes = {
        key: st.session_state[key]
        for key in (
            "session_id",
            "brief",
            "messages",
            "upload",
            "result",
            "usage",
            "run_history",
            "quality_answer",
            "quality_history",
            "interview_mode",
            "assumption_acceptances",
        )
    }
    notes["active_stages"] = {
        key: st.session_state[key].control.snapshot()
        for key in ("design_run", "interview_run", "quality_run")
        if st.session_state[key]
    }
    if st.session_state.job:
        notes["dataset_quality"] = job_status(st.session_state.job).get("quality")
    st.download_button(
        "Download session notes", json.dumps(notes, indent=2), file_name="cardinal_session.json"
    )
    with st.expander("View session notes"):
        st.json(notes)
    st.caption(
        "Diagrams show the actual LangGraph nodes, not the screen navigation or dataset worker. Notes contain chat, uploaded context and completed check errors; keep them private. API tokens reported here may exclude an interrupted request."
    )


initialize()
collect_interview()
collect_quality_review()
with st.sidebar:
    st.title("Cardinal")
    st.caption("Your dataset design assistant")
    for i, page in enumerate(PAGES):
        if st.button(
            page + (" · current" if i == st.session_state.page else ""),
            type="primary" if i == st.session_state.page else "secondary",
            width="stretch",
        ):
            go(i)
    attachments()
    with st.expander("AI connection & privacy", expanded=not st.session_state.consent):
        consent()
    st.caption(
        "Save session notes before closing. Sessions are not restored on restart. Stop active jobs before closing the app."
    )
    if st.button("New conversation", disabled=design_busy()):
        invalidate()
        st.session_state.clear()
        st.rerun()

[chat, review, export, notes_page][st.session_state.page]()
activity_footer()
