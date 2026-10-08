# Prepare and share the repository

This checklist shares the source project. It does not deploy a public running app. Each teammate clones the repository, creates their own environment and supplies approved local API credentials.

## Before choosing public visibility

Confirm you are allowed to publish all code, references and bundled generator files. No generator redistribution license has been supplied with this checkout; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Do not invent a license or assume public visibility grants permission. Until ownership/permission is confirmed, use an approved private repository and collaborator access.

Review for proprietary business rules and sensitive material as well as keys. `.gitignore` and automated scans cannot prove a repository is safe to publish. If a secret was ever committed or pushed, rotate/revoke it and address repository history; merely deleting the current file is insufficient.

## What belongs in Git

Include current application `.py` files, documentation, blank-key `.env.example`, `.gitignore`, `.gitattributes`, `.streamlit/config.toml`, both requirements files, `ruff.toml`, launchers, `references/`, `vendor/`, `engine_snapshot.json` and `tests/`.

`vendor/` is the required compiler/generator. `references/` supplies design syntax and tests. `tests/` protects behaviour. They are not backups. No separate synthgen folder is required after cloning.

Exclude `.env`, `.venv/`, `.runs/`, caches, logs, uploads, exported datasets, private notes, generated diagrams, editor state, archives and backups. Existing local files need not be deleted to keep them out of Git. The supplied `.gitignore` covers these categories.

## Stage and inspect locally

Open a terminal in the project folder. If this folder has no Git repository yet:

```bash
git init
```

Inspect the candidate list before staging, then stage it:

```bash
git status --short
git add --dry-run .
git add .
git diff --cached --name-only
```

Run the staged-file check on Windows:

```powershell
.\.venv\Scripts\python.exe check_repository.py
```

On macOS:

```bash
./.venv/bin/python check_repository.py
```

The check reads indexed contents, not just working files, so an accidentally staged secret stays detectable even after its local file is edited. It prints paths/reasons, never secret values. It rejects unexpected paths, nonempty secret fields in `.env.example` and some recognizable credentials. It does not scan full Git history or prove absence of confidential information.

Inspect the staged diff **locally** before committing:

```bash
git diff --cached --stat
git diff --cached
```

Do not paste this output into chat if it contains private information. Press `q` to exit Git's pager. `.gitignore` does not untrack files already in the index; if an unwanted file is staged/tracked, remove that exact path from the index before proceeding, without deleting your local copy.

If you intentionally add new source files, review them and update the allowlist in `check_repository.py`; do not bypass the check to include backups or data.

## Commit and publish in VS Code

1. Open **Source Control** and review the staged files.
2. Enter a message such as `Prepare Cardinal Studio for team use`, then choose **Commit**.
3. Use **Publish Branch/Publish to GitHub** if no remote exists, or **Push/Sync** for an already-linked repository. Labels depend on your VS Code setup.
4. Choose visibility only after the ownership/permission checks above. Do not change an existing repository's visibility casually.
5. Open the GitHub repository and verify the README renders and no secrets or generated files appear.

A version-neutral repository name such as `cardinal-studio` is suitable. You do not need to rename your existing local checkout to run it.

## On a teammate's or work laptop

Copy the HTTPS URL from the repository's **Code** button and clone it with VS Code's **Git: Clone**. Follow the fresh-install instructions in [README.md](README.md), create `.env` locally, and use [DEMO.md](DEMO.md).

Do not copy virtual environments between machines. Preserve `.gitattributes`: Python/JSON LF endings keep the engine fingerprint stable on Windows. Keep the app bound to localhost; sharing source is not authorization to expose an unauthenticated service on a work network.
