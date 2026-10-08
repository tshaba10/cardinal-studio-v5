# How Cardinal Studio works

## Read the code in this order

Start with `models.py` (data contracts), `workflow.py` (graph nodes/routing), `prompts.py` (AI instructions), then `app.py` (UI/approvals). Read workers and quality modules afterward. Paths resolve relative to source files, not to a particular checkout name or computer.

## End-to-end flow

```text
Chat / optional file context → Interviewer → structured requirements brief
  → user accepts proposals → Designer → candidate JSON + coverage mapping
  → Python validator → AI config reviewer → human config approval
  → JSON download (primary output)
  → optional local generation → Python full-file quality gate
  → preview + optional AI dataset review → human acceptance → ZIP download
```

The interview graph runs once per submitted turn. The design graph has designer, validator and reviewer nodes with one shared automatic repair allowance. A real unanswered decision stops the loop; advisory notes alone do not. Dataset review is a separate optional one-node AI graph.

The generator is ordinary Python, not an LLM agent or LangGraph node. Diagrams reflect actual graph definitions, not every UI/process operation.

## Distinct data structures

| Structure | Purpose | Definition |
|---|---|---|
| Chat message | Role, content, ID and timestamp; session ID lives in session state | `app.py` |
| `Brief` | Purpose, stable requirement IDs/statuses, volumes, seed, questions and assumptions | `models.py` |
| Graph `State` | Internal context, design, reports, usage, trace and local cancellation control | `workflow.py` |
| `Design` | JSON text, requirement-to-JSON-pointer coverage, unresolved decisions and advisories | `models.py` |
| Generator contract | Executable instructions exported as `cardinal_contract.json` | Contract reference and compiler |
| Quality report | Actual counts/profiles/checks, limitations and integrity hashes | `quality_report.py` |
| `DatasetReview` | One evidence-backed advisory finding per requirement | `models.py`, `quality_review.py` |

Do not pass session notes or graph state directly to the generator. `confirmed` means accepted intent; `proposed` requires acceptance; `deferred` is an explicit scope decision. Acceptance is bound to a brief fingerprint and invalidated by changes. Proposed defaults never silently override explicit requirements.

## File map

| Files | Responsibility |
|---|---|
| `app.py` | Four pages, chat history, consent, approvals and bottom activity panel |
| `models.py` | Strict Pydantic brief/response/review schemas |
| `workflow.py` | Nodes, routing, requirement-ID retention and bounded repair |
| `prompts.py` | Interview modes, system instructions and reference selection |
| `interview_settings.py` | Explicit assumption acceptance tied to the brief |
| `design_runner.py` | Background graph runs, completed snapshots, telemetry and Stop |
| `claude_client.py` | Requests, model selection, schema adaptation and response validation |
| `claude_worker.py` | Isolated streaming HTTPS; rejects incomplete/refused/truncated output |
| `process_runner.py` | JSON worker protocol and termination of blocked local workers |
| `contract_checks.py` | Intent anchors, coverage pointers and execution policy |
| `safety.py` | JSON parsing, reference expansion limits and expression preflight |
| `engine_adapter.py`, `validator_worker.py` | Engine fingerprint and isolated compilation |
| `jobs.py`, `generation_worker.py` | Disk-backed generation, cancellation, verification and packaging |
| `quality_report.py` | Independent full-file audit, profiles and integrity hashes |
| `quality_review.py`, `quality_ui.py` | Optional evidence-based AI dataset review and presentation |
| `downloads.py` | Loopback-only token-authorized streaming of verified ZIPs |
| `importer.py`, `file_models.py` | Bounded CSV/XLSX parsing, normalized headers and sample inference |
| `privacy.py` | Conservative sensitive-header/value screening, not complete DLP |
| `graphs.py`, `export_graphs.py` | Mermaid and offline SVG from actual graphs |
| `export_rule_reference.py` | Developer utility to regenerate bundled rule signatures |
| `references/contract_structure.json` | Placeholder-style documentation, not an executable config |
| `references/dependent_accounts.json` | Tested syntax example for dependent numeric columns |
| `references/single_grain_accounts.json` | Executable regression-test fixture |
| `references/credit_card.json` | Historical/transaction syntax supplied when relevant |
| `references/rule_library.json` | Rule signatures supplied to the designer |
| `vendor/synthgen/` | Required compiler/generator source, not a backup |
| `engine_snapshot.json` | Engine fingerprint and provenance metadata |
| `.streamlit/config.toml` | Local address, port, theme and upload/watch settings |
| `.env.example` | Blank-key setup template; `.env` remains private |
| `run.command`, `run_windows.bat` | Optional platform launchers |
| `requirements.txt`, `requirements-dev.txt` | Runtime dependencies and optional tests |
| `ruff.toml` | Consistent application/test formatting and basic lint checks; excludes the pinned vendor |
| `check_repository.py` | Staged-file checks for unwanted paths and recognizable secrets |
| `tests/` | Offline regression suite; maintenance source, not generated history |

## Validation before and after generation

Before generation, Python checks structure, execution boundaries, approved seed/volumes, coverage pointers and compiler acceptance. The AI reviewer examines semantic alignment. Only passing checks plus human approval enable config export/generation. These checks do not prove arbitrary natural-language meaning or industry fidelity.

After generation, the bundled verifier evaluates configured invariants/targets. The independent audit reads every emitted CSV row and checks supported types, finite numbers, precision/scale, nullability, date/identifier formats, static counts, headers, uniqueness and foreign-key membership. Supported categorical/conditional support, literal date ranges and same-table add/subtract/ratio calculations are checked when emitted inputs and supported post-processing permit it.

Custom/state/aggregate/lookup/cross-table formulas and hidden inputs may lack independent formula checks. Gaps are disclosed as warnings; configured invariants still run. Probabilities are not exact quotas. Historical reconciliation needs measured checks. Automatic calibration is not executed.

Failures block ZIP download. Counts of failed checks are not distinct affected rows. Reports record config/output/engine hashes; the download server verifies the ZIP hash too. Human acceptance or AI advice cannot override Python failure.

## Optional dataset-review evidence

The model receives the full approved brief/config and hashes, all table counts, up to 120 checks (failures/warnings first) and up to 80 column profiles. Available summaries include missingness and numeric/date min/max/mean. Selection/truncation is disclosed. Packets exceeding 192 KB fail rather than silently losing requirements.

Raw CSV rows and category-frequency tables are not sent. Config examples and check labels may still contain sensitive values; inspect the sharing preview. Missing measurement coverage must be `insufficient_evidence`, not `supported`.

Python requires every requirement ID exactly once and rejects fabricated evidence IDs or unapproved deferrals. The model can still reason incorrectly. This is advice, not a compliance certificate, automatic repair or authority to download failed data.

## Concurrency and lifecycle

Streamlit reruns the UI; session state retains in-session values. Background graph threads expose snapshots. Network/compiler work uses isolated subprocesses so Stop can terminate local work even while a socket waits.

Progress shows actual node transitions, elapsed time and last worker activity—not guessed percentages or hidden reasoning. There is no overall response/design deadline. Connection errors, provider/output/token limits and the repair budget still apply; no automatic network retries occur.

Only completed, validated model responses are applied. Cancellation discards partial answers but cannot guarantee provider cancellation or avoid charges. Save notes and stop tasks before closing; a disconnected browser is not a reliable worker shutdown.

Requirements changes revoke config approval and download capability. `.runs/` persists until deliberately cleaned while the app is stopped. Notes are a private audit export, not restart recovery.

## Maintenance

Keep the bundled engine and `.gitattributes` intact. LF endings preserve its fingerprint across Windows checkouts. Do not update the hash merely to silence a mismatch; review source changes and test generation/validation first. Rule/package/policy version fields identify technical contracts and must not be removed as cosmetic text.

After installing development dependencies, run `python -m pytest -q` using the project interpreter. Tests mock AI, exercise cancellation/malformed streams, and run real generation/quality/download checks. They do not validate live account access, every request or a corporate network.

`python export_graphs.py` needs no API key; diagrams are generated and ignored by Git. `python export_rule_reference.py` is for intentional engine/rule maintenance: review and commit changed reference signatures when needed.
