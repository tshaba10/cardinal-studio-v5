# Claude connection and privacy

Cardinal Studio calls Anthropic's Claude Messages API directly. Use an approved Anthropic API account and key; an OpenRouter key does not work here. API billing is separate from a Claude chat subscription.

## Configure the project

1. Obtain a key through your approved Anthropic Console account. Check billing/spend controls and model access before a demo.
2. Copy `.env.example` to `.env` in the project root if `.env` does not exist. Do not overwrite an existing configuration.
3. Enter the key after `ANTHROPIC_API_KEY=`. Never paste it into chat, screenshots or GitHub.
4. Set `ANTHROPIC_MODEL` to an exact model ID available to your account.
5. Optionally set `ANTHROPIC_DESIGNER_MODEL` for design, config review and dataset review. Omit this line to reuse the interviewer model; do not leave it empty.
6. Save and restart the app. Template model values are starting settings, not guarantees of pricing or account access.

| Setting | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Authentication; local machine or approved secret store only |
| `ANTHROPIC_MODEL` | Interviewer requests |
| `ANTHROPIC_DESIGNER_MODEL` | Designer, config reviewer and optional dataset reviewer |

Keys enter the isolated HTTPS worker through its environment, not command arguments or model prompts. Never disable certificate verification.

## Environment precedence

Existing terminal/OS variables override `.env`. If a stale terminal key overrides the intended file, remove it from the current terminal only and restart:

```powershell
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
```

On macOS/zsh, use `unset ANTHROPIC_API_KEY`. Do not print the key to diagnose it. Each teammate needs their own approved local credentials; never commit a shared key.

## Errors and costs

- **401:** check invalid, incomplete, revoked or expired credentials locally.
- **403/404:** check model ID and account access.
- **429:** check billing, spend and rate limits; retry deliberately, not repeatedly.
- **SSL failure:** use IT-approved trust settings. Windows dependencies include `pip-system-certs`; do not bypass TLS checks or corporate controls.
- **Connection failure:** check approved network/proxy settings. Connection establishment is limited to 10 seconds; response streaming has no overall deadline. A gateway/provider may still terminate a request.
- **Quiet stage:** the panel shows time since the last event, not proof of productive reasoning. Wait or Stop. There is no automatic network retry.

One design action allows at most one automatic repair. Interview turns and optional dataset reviews cost additional requests. Stop terminates local work and rejects partial output, but provider work may still run or be billed. The app has no overall time/dollar cap; configure account-side controls and inspect actual billing.

## What leaves the laptop

- Interview: conversation, requirements, attached names/types and separately approved samples.
- Design: approved brief, bundled contract/rule references and failed-design feedback when applicable.
- Config review: brief, candidate config, coverage mapping and designer advisories.
- Optional dataset review: brief/config and selected aggregate evidence, after separate consent with an exact-payload preview. It sends no raw CSV rows, but config examples and check labels may still be sensitive.

Generation and full-file Python checks run locally. Privacy screening can miss sensitive content; use approved fictional material only. Session notes contain the conversation and should stay private.
