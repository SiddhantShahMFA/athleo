# AGENTS.md

## Purpose
This repository is AI-native by default.
All implementation may use Cursor, Codex, or Antigravity.
No generated output is considered complete until it is verified and documented.

## Required workflow
1. Create or use a branch for every change. Do not work directly on main.
2. Before coding, read this file and summarize the requested change.
3. Make the smallest safe change possible.
4. Before finishing, update docs/releases.md for shipped work.
5. Every commit must use the required commit format below.
6. If any required file or commit field is missing, the work is incomplete.

## Required branch format
Use one of these:
- ai/<short-task>
- fix/<short-task>
- feat/<short-task>
- chore/<short-task>

Examples:
- ai/login-page-validation
- feat/export-csv
- fix/invoice-rounding

## Required commit format
Use this exact template:

<type>: <short summary>

AI-Tool: cursor|codex|antigravity
AI-Usage: design|code|debug|refactor|test|docs
Human-Check: self-verified
Validation: manual
Release-Note: yes|no

Allowed <type> values:
- feat
- fix
- refactor
- docs
- chore
- test

Example:

feat: add customer export endpoint

AI-Tool: codex
AI-Usage: code
Human-Check: self-verified
Validation: manual
Release-Note: yes

## Coding rules
- Reuse existing project patterns before creating new ones.
- Do not add new dependencies unless necessary.
- Keep changes small and focused.
- Do not mix unrelated fixes in one commit.
- Prefer explicit code over clever code.
- If a file is unclear, inspect nearby files before changing structure.

## Security rules
- Never hardcode secrets, tokens, API keys, passwords, certificates, or private URLs.
- Never commit `.env` files, credentials, production data, or customer data.
- Prefer existing environment-variable patterns already used in the repo.
- If a dependency is added or changed, run a security scan before finishing.
- If Snyk is available, run `snyk test` before marking work complete.
- For production-bound changes, run `snyk monitor` after the change is accepted so the project stays continuously monitored.
- If Snyk reports high or critical issues, do not close the task without documenting the finding.
- If security scanning cannot be run, explicitly state that in the commit message or release log.
- Do not suppress or ignore security findings unless a human explicitly approves it.


## Validation rules
- If automated tests exist, run them.
- If automated tests do not exist, do manual validation and state "Validation: manual" in the commit message.
- Do not claim validation that was not actually done.
- If the app cannot be run locally, document the limitation in the commit body.

## Release logging
For every shipped change that affects behavior, add one new line to docs/releases.md.

## Forbidden
- Direct work on main
- Empty release notes for user-visible changes
- Missing commit trailers
- Large mixed-purpose commits
- Claiming review or testing that did not happen
