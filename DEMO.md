# Cardinal Studio: UI demonstration guide

Follow this guide after [installing and starting the app](README.md). Everything in the example is fictional. Model responses can vary; inspect the proposal rather than expecting identical wording.

## Before you begin

Confirm approved API access, billing/spend controls and network settings. Start with 100 rows. Interview, design and optional AI dataset review use paid requests; Python generation does not.

The sidebar has four tabs: **Chat**, **Review**, **Preview & export**, and **Diagrams & session notes**. The highlighted tab is the current page.

## 1. Chat: describe the dataset

Open **AI connection & privacy** in the sidebar. Read and enable sharing consent for approved fictional material. Choose **Quick — fewer questions, proposed defaults**.

Paste this into the message box:

```text
Design 100 fictional credit-card accounts for a training demo.

Include exactly these six output columns:
account_id, customer_segment, credit_limit, current_balance,
available_credit, utilization_ratio.

Account IDs must be unique and formatted like ACC-123456.
They do not need to be sequential or ordered.
Use standard and premium segments. Credit limit must depend on segment,
with premium accounts generally having higher limits.
Current balance must be between zero and the account's credit limit.
Available credit = credit limit minus current balance.
Utilization ratio = current balance divided by credit limit, between 0 and 1.
Use USD, two decimal places for money, no missing values, and seed 42.

Propose simple supported demo assumptions for segment probabilities,
credit limits, balances and rounding. Probabilities are not exact quotas;
no exact average utilization is required. Internal helper columns are fine,
but export only the six requested columns.

List proposed assumptions for my approval. Ask only essential questions.
Do not add calibration, sequential IDs or unrelated business rules.
```

Watch the bottom activity panel: current node, elapsed time and last worker update. **Stop interview** cancels local work but does not reverse billing.

Expected outcome: an explanation, captured requirements and proposed defaults. No dataset has been generated yet.

## 2. Chat: clarify and prepare

Open **Your captured requirements** below the latest reply. Check all six names, 100 accounts, USD, seed, formulas and assumptions.

If a default is unsuitable, revise it in chat. For example:

```text
Please revise the proposed settings: standard accounts should have credit
limits of 1000 or 3000 USD, and premium accounts 5000 or 10000 USD.
Use illustrative probabilities, not exact counts. Keep the other requirements.
```

When satisfied, check **I accept the listed proposed settings and assumptions for this demo**, then select **Prepare my design**. A changed brief needs fresh acceptance. Do not accept a contradiction just to proceed.

The design workflow is Designer → Python validator → AI requirements reviewer, with at most one shared automatic repair. Expand **Where am I in the graph?** in the activity panel to see the actual node. Use **Stop designing** if needed. A quiet stage is not a completion estimate.

### If preparation fails

**Latest design feedback — what to do next** appears below the historical chats. It groups unresolved choices, reviewer concerns, proposed next steps and **Check details (for troubleshooting)**.

- Answer real decisions in chat; do not edit JSON merely to bypass checks.
- Advisory notes are limitations, not automatically unanswered questions.
- A new **Prepare my design** attempt receives the failed checks, costs additional API usage, and is not guaranteed to succeed.
- A unique identifier pattern cannot guarantee strictly sequential IDs. If sequencing was accepted, explicitly decide whether it is needed.

Suggested reply:

```text
Explain which items need my decision and propose supported alternatives.
Keep accepted requirements unchanged unless I explicitly approve a change.
```

## 3. Review: approve the config

A passing design opens **Review**. Read the summary and warnings. Open **Where each requirement was implemented** for the mapping from requirement IDs to JSON locations, and **Inspect JSON and Python validation details** for the contract/checks.

Compiler acceptance means the generator understands the config; it does not prove all business meaning or industry realism.

If something is wrong, select **Return to chat to address feedback** and describe the correction. Otherwise, check **I reviewed the design and assumptions, and approve this config**, then **Approve and continue**.

## 4. Preview & export: download JSON

Select **Download JSON config — primary output**. This is the generator configuration. You can stop here if designing a config is the only objective. It is distinct from session notes, quality reports and the dataset ZIP.

## 5. Preview & export: generate and inspect

1. Confirm the requested volume and available resources using the checkbox.
2. Select **Generate the requested dataset**.
3. Watch the bottom activity panel; **Stop generation** remains available.
4. Read **Dataset quality report**. Python failures block dataset download.
5. Select a table and scroll the preview: up to 100 rows from the actual output, not a separate sample.

For this demo, inspect the six columns, 100 accounts, missing-value policy, segment-dependent limits, balances and formulas. Read measured checks and tolerances rather than assuming the preview proves every row.

There is no fixed row-count cap; large jobs can exhaust resources. A requirements change invalidates prior approval and outputs.

### Optional AI dataset reviewer

Expand **Optional AI dataset reviewer**. This is a separate paid call, not a condition for downloading the config.

1. Choose **Show exactly what will be shared with Claude**.
2. Inspect the payload and consent to sharing the brief/config and aggregate evidence.
3. Select **Review dataset against my requirements**.
4. Read every verdict: `supported`, `mismatch`, `insufficient_evidence`, or `deferred`.

Insufficient evidence is not a pass. The reviewer sees measured summaries, not all raw CSV rows, and cannot override Python failures. Resolve mismatches before describing the data as compliant. Download the review JSON separately if useful.

### Download the dataset

Review the preview, warnings and any optional findings, then check the dataset acceptance box. With a passing Python gate, **Download full dataset (ZIP)** becomes available. It contains generated CSVs, the config and generation/check reports. Keep the app running until download completes.

The optional AI review is a separate download, not added retroactively to the ZIP.

## 6. Diagrams & session notes

Select this sidebar tab and choose the Interview, Design/validation/review, or Optional dataset requirements review graph. Download SVG for a browser/presentation, or Markdown for all graph definitions. These depict real LangGraph nodes, not page navigation or the generator worker.

Choose **Download session notes** before closing. Notes include conversation, assumptions, results and diagnostics. Keep them private. They are an audit record, not an automatic restore file.

## Optional demonstrations

**Detailed mode:** save notes, start a new conversation, and select **Detailed — clarify more choices**. Ask for fictional accounts and linked purchases over a specified six-month date window. Agree counts, dates, spending and reconciliation rules. More complex requests may need correction; do not promise arbitrary relationships will pass automatically.

**CSV/Excel:** open **Attach CSV or Excel (optional)**. Upload a fictional `.csv` or `.xlsx` under 1 MB with headers; rows are optional. Select the worksheet/delimiter where applicable, inspect inferred columns, and choose **Attach these columns**. Only names/types are shared by default; sample sharing is separate. Explain meanings and relationships in chat. Samples are not authoritative distributions. Excel formulas/error cells are unsupported; use plain values.

**Finish:** stop active tasks, save notes, and press Ctrl+C in the terminal. Closing only the browser does not reliably stop workers.
