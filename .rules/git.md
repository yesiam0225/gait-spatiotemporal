# Git Rules

## Commit Guidelines

### Logical Separation
Each commit must represent a single, logical change. Never mix unrelated changes in one commit.
- Separate API changes from UI changes
- Separate feature additions from bug fixes
- Separate configuration changes from code changes

### Commit Message Format
All commit messages must be written in English.
Use [gitmoji](https://gitmoji.dev/) at the start of every commit message.

```
:emoji: Short description of the change

Optional longer description explaining what and why
```

### Common Gitmoji
| Emoji | Code | Use for |
|-------|------|---------|
| ✨ | `:sparkles:` | New features |
| 🐛 | `:bug:` | Bug fixes |
| ♻️ | `:recycle:` | Refactoring |
| 💄 | `:lipstick:` | UI / style changes |
| 📝 | `:memo:` | Documentation |
| 🎨 | `:art:` | Code structure / format |
| ⚡️ | `:zap:` | Performance improvements |
| 🔥 | `:fire:` | Remove code or files |
| 🚑️ | `:ambulance:` | Critical hotfix |
| ✅ | `:white_check_mark:` | Add or update tests |
| 🔧 | `:wrench:` | Configuration files |
| ⬆️ | `:arrow_up:` | Upgrade dependencies |
| ⬇️ | `:arrow_down:` | Downgrade dependencies |
| 💥 | `:boom:` | Breaking changes |
| 🔒️ | `:lock:` | Security fixes |

### Commit Author
Always use the repository owner's identity — never the agent's default.
Never add `Co-authored-by` trailers for AI tools (Claude, Cursor, Copilot, etc.).

```bash
git commit --author="$(git config user.name) <$(git config user.email)>" -m "..."
```

---

## AI Workflow for Git Operations

**Do NOT run `git commit` or `git push` unless explicitly requested by the user.**

When the user asks to "reflect changes in git" (e.g. "깃에 반영해줘"), always commit **and** push.

1. Analyze all pending changes (staged and unstaged)
2. Group changes into distinct logical sets
3. For each group: `git add <files>` → `git commit` with correct author and gitmoji message
4. After all groups are committed, run `git push` once

### Working Directory
Never use `cd` to change directories before running git commands. Use the `-C` flag to target a specific repository path:

```bash
# Wrong
cd backend && git status

# Correct
git -C backend status
git -C /absolute/path/to/repo commit -m "..."
```

---

## Branch Rules

See `README.md` for the full branch naming conventions and version management.

Summary:
- `feature/xxx` → merge into `dev`
- `bugfix/xxx` → created from `unstable`, merge into `unstable` + `dev`
- `hotfix/xxx` → created from `main`, merge into `main` + `dev`
- `unstable` → staging
- `main` → production (authority approval required)
