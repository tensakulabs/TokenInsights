# TokenInsights

Claude Code skill — scans your session history and shows the **Token Operation Table**: where tokens are going by command category.

## Installation

1. Copy `SKILL.md` to your Claude skills directory:
   ```bash
   mkdir -p ~/.claude/skills/TokenInsights
   cp SKILL.md ~/.claude/skills/TokenInsights/
   ```

2. Copy the script to your scripts directory:
   ```bash
   mkdir -p ~/.claude/scripts
   cp token-insights.py ~/.claude/scripts/
   ```

## Usage

Once installed, invoke from Claude Code:

```
token usage
show me the token operation table
how many tokens am I using
```

Or run directly:

```bash
python3 ~/.claude/scripts/token-insights.py
python3 ~/.claude/scripts/token-insights.py --days 7
python3 ~/.claude/scripts/token-insights.py --project myproject
python3 ~/.claude/scripts/token-insights.py --all
```

## Output

```
Token Operation Table  ·  last 30 days

| Operation             | Frequency | Standard |
|------------------------|-----------|----------|
| git diff              |      120× |    61.0K |
| npm test / cargo test |       56× |    36.7K |
| git status            |       98× |    12.1K |
| cat / read            |      203× |     9.8K |
| ...                   |           |          |
| Total                 |           |   393.6M |
```

## Requirements

- Python 3.8+
- Claude Code with session history at `~/.claude/projects/`

## Pricing

Defaults to Claude Sonnet 4.x pricing. Edit the `PRICING` dict at the top of the script to match your model.

## License

MIT
