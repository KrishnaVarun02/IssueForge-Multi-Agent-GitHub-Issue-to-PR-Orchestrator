# Changelog

All notable changes to this project are documented here.

## v1.0.0 — 2026-08-23

### Added

- Complete LangGraph GitHub issue-to-pull-request orchestration flow.
- OpenRouter Code Reader, Planner, Code Writer, and Test Writer agents.
- Strict structured-output validation with bounded corrective retries.
- Repository indexing, safe unified-diff generation, and path validation.
- Restricted Docker pytest sandbox and bounded test-repair loop.
- Human approval, revision feedback, isolated Git worktree preparation, and PR
  creation gates.
- SQLite workflow checkpoints with list and resume commands.
- Secret-safe JSON reports, token and estimated-cost budgets, per-agent usage,
  response traces, response outcomes, and node timings.
- Typed centralized settings and executable security invariants.
- Cost-free end-to-end integration coverage and Python 3.12/3.13 CI.
- Production CLI exposed directly through the `multi_agent_system` package.

### Security

- Treats issue text, repository source, and sandbox output as untrusted data.
- Never executes model output as shell commands.
- Requires passing tests and explicit human approval before delivery.
- Runs generated changes in a non-root, offline, read-only, resource-limited
  container.
- Keeps CI read-only and free of application credentials.
