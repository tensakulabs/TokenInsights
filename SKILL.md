---
name: TokenInsights
description: Claude Code session token usage analytics. Scans ~/.claude/projects/ and prints the Token Operation Table — where tokens are going by command category. USE WHEN token usage, session insights, how many tokens, token stats, token breakdown, usage report, token cost, session cost.
---

## Voice Notification

```bash
curl -s -X POST http://localhost:8888/notify \
  -H "Content-Type: application/json" \
  -d '{"message": "Generating token insights"}' \
  > /dev/null 2>&1 &
```

# TokenInsights Skill

**Purpose:** Scan `~/.claude/projects/` session history and print the Token Operation Table — where tokens are going by command category.

**Script location:** `~/.claude/scripts/token-insights.py`

## Execution

Parse the user's request for any options, then run:

```bash
# Default: last 30 days
python3 ~/.claude/scripts/token-insights.py

# Specific time window
python3 ~/.claude/scripts/token-insights.py --days 7
python3 ~/.claude/scripts/token-insights.py --days 90

# Filter to one project
python3 ~/.claude/scripts/token-insights.py --project <name>

# All-time (slow on large session histories)
python3 ~/.claude/scripts/token-insights.py --all

# JSON output for piping
python3 ~/.claude/scripts/token-insights.py --json

# More parallel workers (faster on multi-core)
python3 ~/.claude/scripts/token-insights.py --workers 8
```

## Trigger → Flag Mapping

| User says | Flag to add |
|-----------|-------------|
| "last week" / "7 days" | `--days 7` |
| "last 3 months" / "90 days" | `--days 90` |
| "all time" / "everything" | `--all` |
| "for [project]" / "in [project]" | `--project <name>` |
| "as JSON" / "json output" | `--json` |
| no time mentioned | (default: `--days 30`) |

## Output

The script prints the Token Operation Table directly to stdout. No further formatting needed.

After output, briefly note the top 1-2 highest-token operations.
