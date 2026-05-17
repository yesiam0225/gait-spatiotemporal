# Project Rules

This directory is the **single source of truth** for all project rules.

Every AI coding assistant (Cursor, Claude Code, Copilot, etc.) must read and follow the applicable files here. Do not duplicate rule content in assistant-specific config files (`.cursor/rules/`, `CLAUDE.md`, `AGENTS.md`, etc.) — those files may only **point here**.

## Available rules

| File | When to read |
|------|----------------|
| [git.md](git.md) | Commits, pushes, branches, merges, or any git workflow |

Add new rule files to this directory and list them in the table above.
