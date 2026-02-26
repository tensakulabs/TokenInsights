# TokenInsights

Token usage analyzer for Claude Code. Scans your session history and shows where tokens are going — by operation, project, and tool.

## What it does

- Breaks down token usage by bash command category (`git diff`, `cat/read`, test runners, etc.)
- Shows per-project cost and cache hit rates
- Identifies low cache-hit projects and optimization opportunities
- Runs in parallel across all your session files (fast even with 15K+ sessions)

## Usage

```bash
# Last 30 days (default)
python3 token-insights.py

# Specific time window
python3 token-insights.py --days 7
python3 token-insights.py --days 90

# Filter to one project
python3 token-insights.py --project myproject

# All time
python3 token-insights.py --all

# JSON output
python3 token-insights.py --json

# More parallel workers
python3 token-insights.py --workers 8
```

## Output

```
📊  SUMMARY  (15,893 sessions · 81,713 API calls)
    Total tokens         393.6M
    ├─ Input (live)       12.0M
    ├─ Output              1.2M
    ├─ Cache writes      380.4M
    └─ Cache reads      6109.7M  (hit rate 94%)
    Estimated cost     $3313.86

⚡  TOKEN OPERATION TABLE

| Operation            | Frequency | Standard |
|----------------------|-----------|----------|
| `git diff`           | 120×      | 61.0K    |
| `npm test` / `cargo test` | 56× | 36.7K   |
| ...                  |           |          |
```

## Requirements

- Python 3.8+
- Claude Code with session history at `~/.claude/projects/`

## Pricing

Defaults to Claude Sonnet 4.x pricing. Edit the `PRICING` dict at the top of the script to match your model.

## License

MIT
