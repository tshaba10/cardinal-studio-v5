# Cardinal Studio

Describe a fictional credit-card dataset in plain English. Cardinal Studio uses Claude and LangGraph to clarify your request, design a JSON configuration, and check it against the bundled deterministic generator.

**The approved JSON config is the primary deliverable.** Optionally generate the dataset locally, preview up to 100 actual rows per table, review quality checks, and download the full output.

This is a local learning/demo application, not a production service or a source of real customer data. AI calls require an approved Anthropic API account and may incur charges. Dataset generation runs in Python, not in the model.

## Documentation

- [UI demo walkthrough](DEMO.md): example prompt and every screen, from chat to export.
- [Claude setup](CLAUDE_SETUP.md): environment settings, sharing, billing and connection errors.
- [Architecture and file map](ARCHITECTURE.md): how the code works and what validation can prove.
- [GitHub publishing checklist](PUBLISHING.md): what to include and how to check for unwanted files.

The repository is self-contained: `vendor/synthgen/` contains the required generator. No separate generator checkout or sibling project folder is needed. The checkout folder can have any name.

## Requirements

- Python **3.13**, Git, and optionally VS Code.
- Internet access for installing packages and making Claude calls.
- An Anthropic API key, API billing and a model accessible to your account.
- On a work laptop, employer-approved software, certificates, proxy settings and AI access.

Use fictional data only. Do not enter payment-card details, real customer records, credentials or proprietary data. Privacy screening is a precaution, not comprehensive data-loss prevention.

## Windows: first installation

### 1. Clone and open the project

In VS Code, open the Command Palette, select **Git: Clone**, and paste the HTTPS repository URL from GitHub's **Code** button. Open the cloned folder.

Choose **Terminal → New Terminal** and use PowerShell. Make sure the terminal is in the folder containing `app.py` and `requirements.txt`.

### 2. Check Python

```powershell
py -3.13 --version
```

If no suitable runtime is found, install an approved Python 3.13 distribution or ask IT. Reopen VS Code afterward.

### 3. Create the environment and install packages

Run these one at a time:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.venv` holds this project's packages on this computer. These commands do not require environment activation, Conda or a PowerShell execution-policy change. Software/network access remains subject to your employer's policy.

### 4. Configure the API key

Create `.env` from the template only if it does not exist:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Open `.env` in VS Code, enter your key after `ANTHROPIC_API_KEY=`, and configure accessible model IDs. Follow [CLAUDE_SETUP.md](CLAUDE_SETUP.md). Do not share `.env` or put the key in the app's chat.

### 5. Start Cardinal Studio

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://127.0.0.1:8506** if the browser does not open automatically. Keep the terminal running.

For editor support, use **Python: Select Interpreter → Enter interpreter path** in VS Code and select `.venv\Scripts\python.exe`. The terminal commands above already select the correct interpreter explicitly.

## macOS: first installation

Clone or open the project in VS Code. In a terminal in the folder containing `app.py`, run:

```bash
python3.13 --version
python3.13 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

In VS Code, copy `.env.example` to `.env` if it does not exist. Enter your API key and accessible model settings locally, then run:

```bash
./.venv/bin/python -m streamlit run app.py
```

Open **http://127.0.0.1:8506**. Select `.venv/bin/python` as the VS Code interpreter if needed. Recreate the environment on each computer; never copy a Mac `.venv` to Windows.

## Everyday use

After installation, only the start command is needed. The optional `run_windows.bat` and `run.command` launchers use the same environment.

The four sidebar tabs are:

| Tab | What you do |
|---|---|
| Chat | Enable sharing, choose Quick or Detailed, describe requirements, accept proposals and prepare the design |
| Review | Inspect the checked design, requirement mappings and warnings; approve the exact config |
| Preview & export | Download JSON; optionally generate data, inspect quality/preview, and download the accepted dataset |
| Diagrams & session notes | Inspect real LangGraph diagrams and save a private audit record |

Follow [DEMO.md](DEMO.md) for the complete click-by-click demonstration. File upload is optional; a plain-language request is enough to begin.

### Stop and restart safely

Save session notes if needed. Use the UI Stop buttons for active tasks, then press **Ctrl+C** in the terminal to stop Streamlit. Closing the browser alone does not reliably stop workers.

Restart after changing `.env` or source files; automatic file watching is disabled. Sessions are not restored on restart. Notes are an audit record, not an automatic session-import file.

AI response streaming has no overall waiting deadline. Connection establishment is limited to 10 seconds; provider, authentication and certificate errors still stop requests. Stop terminates local work but cannot guarantee cancellation of provider-side work or charges. There is no built-in overall dollar cap.

## Troubleshooting

| Symptom | Next step |
|---|---|
| `git` not recognized | Install employer-approved Git and reopen the terminal, or ask IT. |
| Python runtime not found | Install approved Python 3.13 and use the explicit commands above. |
| `No module named streamlit` | Install requirements with the same `.venv` interpreter used to start the app. |
| Missing key or HTTP 401 | Check `.env` locally, save and restart. Existing OS environment variables override the file. |
| Model/access/rate error | Check the exact model ID, permissions, billing and limits; see [Claude setup](CLAUDE_SETUP.md). |
| SSL certificate error | Use IT-approved trust settings. Windows dependencies include `pip-system-certs`; never disable TLS verification. |
| Port 8506 unavailable | Stop your other instance with Ctrl+C, or append `--server.port 8507` and open that port. |
| Design validation fails | Read **Latest design feedback** below the chat history. Clarify real choices or retry with the check details. |
| Dataset gate fails | Inspect failed checks. The config remains downloadable; dataset download stays blocked. |
| A stage appears stuck | Inspect the bottom activity panel. Quiet time is not proof of progress; wait or Stop deliberately. |

## Outputs and limitations

- JSON describes generation instructions; it is not chat history or graph state.
- Run files remain in `.runs/`, ignored by Git. They are not automatically deleted.
- Preview is limited to 100 rows per table; the ZIP contains the generated requested volume. There is no fixed row-count cap, but memory, disk and engine limits apply.
- Python checks inspect emitted data, not just the preview. The optional AI dataset reviewer assesses aggregate evidence against the approved brief and cannot override Python failures.
- Compiler success and AI review do not prove industry realism. Unsupported formulas or historical checks remain visible limitations. Automatic calibration is not executed.
- UI and download server bind to `127.0.0.1`. Do not expose this unauthenticated local demo as a public hosted service.

See [ARCHITECTURE.md](ARCHITECTURE.md) for exact responsibilities and evidence boundaries.

## Contributor checks

Install optional test dependencies, then run tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

On macOS, substitute `./.venv/bin/python`. Tests use mocked AI and real local generation; no paid Claude calls are made. Some tests use a loopback-only download server. Passing tests do not replace live account or work-network checks. Dependency ranges are not a fully locked environment.

Before publishing, follow [PUBLISHING.md](PUBLISHING.md). A public GitHub link shares source, not a running app; teammates install and run their own local copy.
