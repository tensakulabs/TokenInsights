#!/usr/bin/env python3
"""
token-insights.py — Token usage analyzer for Claude Code

Scans ~/.claude/projects/ (and any ~/.claude-backup-*/ dirs) to show
where tokens are going across your Claude Code sessions.

Usage:
  python3 token-insights.py
  python3 token-insights.py --days 7
  python3 token-insights.py --project myproject
  python3 token-insights.py --all
  python3 token-insights.py --workers 8
  python3 token-insights.py --json
"""

import json
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

# Claude Sonnet 4.x pricing (per million tokens)
PRICING = {
    "input": 3.00,
    "output": 15.00,
    "cache_write": 3.75,
    "cache_read": 0.30,
}

# Compression ratios per non-Bash tool
TOOL_COMPRESSION = {
    "Read": 0.25,
    "Grep": 0.75,
    "Glob": 0.60,
    "WebFetch": 0.65,
    "WebSearch": 0.55,
    "Task": 0.30,
    "TaskOutput": 0.45,
    "Write": 0.45,
    "mcp__mem0__search_memories": 0.40,
    "mcp__granola__get_transcript": 0.70,
    "default": 0.45,
}

# Compression ratios per Bash command category
CMD_COMPRESSION = {
    "ls / tree":             0.80,
    "cat / read":            0.70,
    "find":                  0.75,
    "grep / rg":             0.80,
    "git status":            0.80,
    "git diff":              0.75,
    "git log":               0.80,
    "git add/commit/push":   0.92,
    "git pull/fetch":        0.80,
    "git branch/checkout":   0.70,
    "git (other)":           0.75,
    "npm/bun test":          0.90,
    "npm/bun install":       0.70,
    "npm/bun run/build":     0.85,
    "cargo test":            0.90,
    "cargo build/run":       0.80,
    "pytest":                0.90,
    "go test":               0.90,
    "lint (ruff/eslint)":    0.80,
    "docker":                0.80,
    "k8s (kubectl/helm)":    0.80,
    "ssh":                   0.50,
    "curl / http":           0.50,
    "python / script":       0.60,
    "terraform":             0.75,
    "make":                  0.80,
    "other bash":            0.70,
}


def simulate_compress(text: str, category: str) -> str:
    """Apply category-specific heuristic compression to tool output text."""
    lines = text.splitlines()

    if category in ("npm/bun test", "cargo test", "pytest", "go test"):
        # Keep only failure lines + final summary
        keep = []
        for line in lines:
            l = line.strip()
            if any(kw in l.lower() for kw in ("fail", "error", "panic", "assert", "FAILED", "passed", "failed", "ok -", "test result")):
                keep.append(line)
        keep = keep or lines[:5]
        return "\n".join(keep[:40])

    if category == "git diff":
        # Keep only +/- changed lines, drop context lines
        keep = [l for l in lines if l.startswith(("+", "-", "@@", "diff", "index", "---", "+++"))]
        return "\n".join(keep[:100])

    if category in ("ls / tree",):
        # Keep first 40 lines — long listings are mostly noise
        return "\n".join(lines[:40])

    if category in ("cat / read",):
        # Keep first 60 + last 10 for large files
        if len(lines) > 80:
            return "\n".join(lines[:60] + ["..."] + lines[-10:])
        return "\n".join(lines)

    if category in ("git log",):
        # Keep first 30 lines (usually 5-6 log entries)
        return "\n".join(lines[:30])

    if category in ("git status",):
        # Already short — just dedup blank lines
        return "\n".join(l for l in lines if l.strip())

    # Default: remove blank lines + deduplicate consecutive identical lines + cap at 100 lines
    seen_last = None
    kept = []
    for line in lines:
        if not line.strip():
            continue
        if line == seen_last:
            continue
        kept.append(line)
        seen_last = line
    return "\n".join(kept[:100])


def categorize_command(cmd: str) -> str:
    """Map a raw bash command string to an operation category."""
    cmd = cmd.strip()
    # Strip leading env vars, sudo, time, etc.
    tokens = cmd.split()
    if not tokens:
        return "other bash"
    i = 0
    while i < len(tokens) and ("=" in tokens[i] or tokens[i] in ("sudo", "time", "env", "exec")):
        i += 1
    if i >= len(tokens):
        return "other bash"
    first = tokens[i].lower()
    rest = tokens[i + 1].lower() if i + 1 < len(tokens) else ""

    if first == "git":
        if rest == "status":              return "git status"
        if rest == "diff":                return "git diff"
        if rest == "log":                 return "git log"
        if rest in ("add", "commit", "push", "amend"): return "git add/commit/push"
        if rest in ("pull", "fetch", "clone"):         return "git pull/fetch"
        if rest in ("checkout", "switch", "branch", "worktree"): return "git branch/checkout"
        return "git (other)"

    if first in ("npm", "bun", "yarn", "pnpm"):
        if rest == "test" or (rest == "run" and len(tokens) > i + 2 and "test" in tokens[i + 2]):
            return "npm/bun test"
        if rest in ("install", "add", "i", "ci"):      return "npm/bun install"
        if rest in ("run", "x", "build", "dev"):       return "npm/bun run/build"
        return f"npm/bun run/build"

    if first == "cargo":
        if rest == "test":    return "cargo test"
        return "cargo build/run"

    if first == "pytest" or (first in ("python", "python3") and "pytest" in cmd):
        return "pytest"

    if first == "go":
        if rest == "test":    return "go test"
        return "cargo build/run"  # reuse bucket

    if first in ("ls", "ll", "exa", "lsd", "tree"):
        return "ls / tree"
    if first == "find":
        return "find"
    if first in ("cat", "head", "tail", "bat"):
        return "cat / read"
    if first in ("grep", "rg", "ag", "ack"):
        return "grep / rg"

    if first in ("ruff", "eslint", "biome", "flake8", "pylint", "mypy", "tsc"):
        return "lint (ruff/eslint)"

    if first == "docker":                 return "docker"
    if first in ("kubectl", "k9s", "helm", "k"):  return "k8s (kubectl/helm)"
    if first == "ssh":                    return "ssh"
    if first in ("curl", "wget", "http", "httpie"): return "curl / http"
    if first in ("python", "python3"):    return "python / script"
    if first in ("terraform", "tofu"):    return "terraform"
    if first == "make":                   return "make"

    return "other bash"


def parse_args():
    p = argparse.ArgumentParser(description="Claude Code session token insights")
    p.add_argument("--days", type=int, default=30, help="Days to analyze (default: 30)")
    p.add_argument("--project", type=str, help="Filter to project name (substring match)")
    p.add_argument("--top", type=int, default=12, help="Top N projects to show (default: 12)")
    p.add_argument("--all", action="store_true", help="Analyze all sessions (ignores --days)")
    p.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel worker processes (default: cpu_count)")
    p.add_argument(
        "--claude-dir",
        action="append",
        dest="claude_dirs",
        metavar="DIR",
        help="Additional claude dirs to scan (auto-detects ~/.claude-backup-*)",
    )
    return p.parse_args()


def fmt_tokens(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def fmt_cost(d):
    if d < 0.001:
        return f"${d:.5f}"
    if d < 0.01:
        return f"${d:.4f}"
    if d < 1:
        return f"${d:.3f}"
    return f"${d:.2f}"


def compute_cost(s):
    return (
        s["input"] * PRICING["input"] / 1_000_000
        + s["output"] * PRICING["output"] / 1_000_000
        + s["cache_write"] * PRICING["cache_write"] / 1_000_000
        + s["cache_read"] * PRICING["cache_read"] / 1_000_000
    )


def analyze_session(filepath, cutoff_ts):
    """Parse one JSONL session. Returns stats dict or None if empty/too old."""
    stats = {
        "input": 0,
        "output": 0,
        "cache_write": 0,
        "cache_read": 0,
        "tool_calls": defaultdict(int),
        "tool_result_bytes": defaultdict(int),
        "cmd_calls": defaultdict(int),   # bash category → call count
        "cmd_bytes": defaultdict(int),   # bash category → result bytes
        "cmd_samples": defaultdict(list),# bash category → [sample texts] (max 5/cat)
        "_id_to_tool": {},               # tool_use_id → tool name
        "_id_to_cmd": {},                # tool_use_id → bash category (Bash only)
        "messages": 0,
        "first_ts": None,
        "last_ts": None,
    }

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts_str = obj.get("timestamp")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if cutoff_ts and ts < cutoff_ts:
                            continue
                        if stats["first_ts"] is None or ts < stats["first_ts"]:
                            stats["first_ts"] = ts
                        if stats["last_ts"] is None or ts > stats["last_ts"]:
                            stats["last_ts"] = ts
                    except ValueError:
                        pass

                msg_type = obj.get("type")

                if msg_type == "assistant":
                    msg = obj.get("message", {})
                    usage = msg.get("usage", {})
                    if usage:
                        stats["input"] += usage.get("input_tokens", 0)
                        stats["output"] += usage.get("output_tokens", 0)
                        stats["cache_write"] += usage.get("cache_creation_input_tokens", 0)
                        stats["cache_read"] += usage.get("cache_read_input_tokens", 0)
                        stats["messages"] += 1

                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if not isinstance(c, dict) or c.get("type") != "tool_use":
                                continue
                            name = c.get("name", "unknown")
                            tid = c.get("id", "")
                            stats["tool_calls"][name] += 1
                            if tid:
                                stats["_id_to_tool"][tid] = name
                                if name == "Bash":
                                    cmd = c.get("input", {}).get("command", "")
                                    cat = categorize_command(cmd)
                                    stats["cmd_calls"][cat] += 1
                                    stats["_id_to_cmd"][tid] = cat

                elif msg_type == "user":
                    content = obj.get("message", {}).get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if not isinstance(c, dict) or c.get("type") != "tool_result":
                                continue
                            tid = c.get("tool_use_id", "")
                            tool_name = stats["_id_to_tool"].get(tid, "unknown")
                            rc = c.get("content", "")
                            size = 0
                            if isinstance(rc, str):
                                size = len(rc)
                            elif isinstance(rc, list):
                                for item in rc:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        size += len(item.get("text", ""))
                            stats["tool_result_bytes"][tool_name] += size
                            if tool_name == "Bash" and tid in stats["_id_to_cmd"]:
                                cat = stats["_id_to_cmd"][tid]
                                stats["cmd_bytes"][cat] += size
                                # Collect sample text (max 5 per category, 4KB each)
                                if len(stats["cmd_samples"][cat]) < 5 and size > 0:
                                    raw = rc if isinstance(rc, str) else "".join(
                                        item.get("text", "") for item in rc
                                        if isinstance(item, dict) and item.get("type") == "text"
                                    )
                                    if raw.strip():
                                        stats["cmd_samples"][cat].append(raw[:4096])

    except (OSError, IOError):
        return None

    if stats["messages"] == 0:
        return None

    del stats["_id_to_tool"]
    del stats["_id_to_cmd"]
    # Convert defaultdicts to plain dicts for pickling
    stats["tool_calls"] = dict(stats["tool_calls"])
    stats["tool_result_bytes"] = dict(stats["tool_result_bytes"])
    stats["cmd_calls"] = dict(stats["cmd_calls"])
    stats["cmd_bytes"] = dict(stats["cmd_bytes"])
    stats["cmd_samples"] = dict(stats["cmd_samples"])
    return stats


# ── Top-level worker (must be module-level for multiprocessing pickle) ────────

def process_project(args_tuple):
    """
    Worker function — processes one project directory.
    args_tuple: (proj_path_str, cutoff_iso_str_or_None)
    Returns (proj_path_str, agg_dict, scanned, skipped) or None.
    """
    proj_path_str, cutoff_iso = args_tuple
    proj_path = Path(proj_path_str)

    cutoff_ts = None
    if cutoff_iso:
        cutoff_ts = datetime.fromisoformat(cutoff_iso)

    sessions = sorted(proj_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        return None

    agg = {
        "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
        "tool_calls": {}, "tool_result_bytes": {},
        "cmd_calls": {}, "cmd_bytes": {}, "cmd_samples": {},
        "session_count": 0, "message_count": 0,
    }
    scanned = 0
    skipped = 0

    for sess in sessions:
        mtime = datetime.fromtimestamp(sess.stat().st_mtime, tz=timezone.utc)
        if cutoff_ts and mtime < cutoff_ts:
            skipped += 1
            continue

        r = analyze_session(sess, cutoff_ts)
        scanned += 1
        if r:
            agg["input"] += r["input"]
            agg["output"] += r["output"]
            agg["cache_write"] += r["cache_write"]
            agg["cache_read"] += r["cache_read"]
            agg["message_count"] += r["messages"]
            agg["session_count"] += 1
            for k, v in r["tool_calls"].items():
                agg["tool_calls"][k] = agg["tool_calls"].get(k, 0) + v
            for k, v in r["tool_result_bytes"].items():
                agg["tool_result_bytes"][k] = agg["tool_result_bytes"].get(k, 0) + v
            for k, v in r["cmd_calls"].items():
                agg["cmd_calls"][k] = agg["cmd_calls"].get(k, 0) + v
            for k, v in r["cmd_bytes"].items():
                agg["cmd_bytes"][k] = agg["cmd_bytes"].get(k, 0) + v
            for k, v in r["cmd_samples"].items():
                existing = agg["cmd_samples"].get(k, [])
                if len(existing) < 20:
                    agg["cmd_samples"][k] = existing + v[:max(0, 20 - len(existing))]

    if agg["session_count"] == 0:
        return None

    agg["_name"] = proj_path.name
    return (proj_path_str, agg, scanned, skipped)


# ── Helpers ───────────────────────────────────────────────────────────────────

def proj_display(dir_name):
    """Convert Claude Code's encoded project dir name to a readable path."""
    name = dir_name
    # Claude encodes paths as hyphen-separated segments, e.g. -Users-alice-Projects-foo
    # Strip the leading home path prefix dynamically
    home_str = str(Path.home()).replace("/", "-")  # e.g. -Users-alice
    for prefix in [
        f"{home_str}-Documents-github-",
        f"{home_str}-Documents-",
        f"{home_str}-Projects-",
        f"{home_str}-",
        "-private-tmp-",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("-", "/", 3)[:35]


def collect_project_dirs(claude_dirs, project_filter):
    for base in claude_dirs:
        proj_root = Path(base) / "projects"
        if not proj_root.is_dir():
            continue
        for p in sorted(proj_root.iterdir()):
            if not p.is_dir():
                continue
            if project_filter and project_filter.lower() not in p.name.lower():
                continue
            yield p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    home = Path.home()
    claude_dirs = [home / ".claude"]
    for backup in sorted(home.glob(".claude-backup-*")):
        if backup.is_dir():
            claude_dirs.append(backup)
    if args.claude_dirs:
        claude_dirs += [Path(d) for d in args.claude_dirs]

    now = datetime.now(timezone.utc)
    cutoff = None if args.all else now - timedelta(days=args.days)
    period_label = "all time" if args.all else f"last {args.days} days"

    all_proj_dirs = list(collect_project_dirs(claude_dirs, args.project))
    cutoff_iso = cutoff.isoformat() if cutoff else None
    work_items = [(str(p), cutoff_iso) for p in all_proj_dirs]
    n_workers = args.workers or min(len(work_items), os.cpu_count() or 4)

    if not args.json:
        print(f"  Scanning {len(work_items)} projects with {n_workers} workers…",
              end="\r", flush=True, file=sys.stderr)

    project_stats = {}
    scanned = skipped = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_project, item): item for item in work_items}
        done = 0
        for future in as_completed(futures):
            done += 1
            if not args.json:
                print(f"  Scanning {len(work_items)} projects with {n_workers} workers… "
                      f"({done}/{len(work_items)})",
                      end="\r", flush=True, file=sys.stderr)
            result = future.result()
            if result:
                proj_path_str, agg, s, sk = result
                scanned += s
                skipped += sk
                project_stats[proj_path_str] = agg

    if not args.json:
        print(" " * 72, end="\r", file=sys.stderr)

    # ── Grand totals ──────────────────────────────────────────────────────────
    grand = {
        "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
        "tool_result_bytes": {}, "tool_calls": {},
        "cmd_calls": {}, "cmd_bytes": {}, "cmd_samples": {},
        "session_count": 0, "message_count": 0,
    }
    for s in project_stats.values():
        for k in ["input", "output", "cache_write", "cache_read", "session_count", "message_count"]:
            grand[k] += s[k]
        for k, v in s["tool_result_bytes"].items():
            grand["tool_result_bytes"][k] = grand["tool_result_bytes"].get(k, 0) + v
        for k, v in s["tool_calls"].items():
            grand["tool_calls"][k] = grand["tool_calls"].get(k, 0) + v
        for k, v in s["cmd_calls"].items():
            grand["cmd_calls"][k] = grand["cmd_calls"].get(k, 0) + v
        for k, v in s["cmd_bytes"].items():
            grand["cmd_bytes"][k] = grand["cmd_bytes"].get(k, 0) + v
        for k, v in s["cmd_samples"].items():
            existing = grand["cmd_samples"].get(k, [])
            if len(existing) < 20:
                grand["cmd_samples"][k] = existing + v[:max(0, 20 - len(existing))]

    # ── JSON mode ─────────────────────────────────────────────────────────────
    if args.json:
        out = {
            "period": period_label,
            "scanned": scanned,
            "skipped": skipped,
            "totals": {k: grand[k] for k in ["input", "output", "cache_write", "cache_read", "session_count", "message_count"]},
            "cost": compute_cost(grand),
            "cmd_breakdown": {
                cat: {"calls": grand["cmd_calls"].get(cat, 0),
                       "bytes": grand["cmd_bytes"].get(cat, 0)}
                for cat in CMD_COMPRESSION
            },
            "projects": [
                {"name": proj_display(s["_name"]),
                 **{k: s[k] for k in ["input", "output", "cache_write", "cache_read", "session_count"]},
                 "cost": compute_cost(s)}
                for s in sorted(project_stats.values(), key=lambda x: -(x["input"] + x["output"]))
            ],
        }
        print(json.dumps(out, indent=2))
        return

    # ── Derived metrics ───────────────────────────────────────────────────────
    total_tokens = grand["input"] + grand["output"] + grand["cache_write"]
    total_cost = compute_cost(grand)
    cacheable = grand["input"] + grand["cache_read"] + grand["cache_write"]
    cache_hit_pct = grand["cache_read"] / cacheable * 100 if cacheable else 0

    W = 68
    print(f"\n{'═'*W}")
    print(f"  Claude Code Token Insights  ·  {period_label}")
    print(f"  Sources: {', '.join(d.name for d in claude_dirs)}")
    print(f"{'═'*W}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"📊  SUMMARY  ({grand['session_count']:,} sessions · {grand['message_count']:,} API calls)")
    print(f"    Total tokens       {fmt_tokens(total_tokens):>8}")
    print(f"    ├─ Input (live)    {fmt_tokens(grand['input']):>8}")
    print(f"    ├─ Output          {fmt_tokens(grand['output']):>8}")
    print(f"    ├─ Cache writes    {fmt_tokens(grand['cache_write']):>8}")
    print(f"    └─ Cache reads     {fmt_tokens(grand['cache_read']):>8}  (hit rate {cache_hit_pct:.0f}%)")
    print(f"    Estimated cost     {fmt_cost(total_cost):>8}")
    if not args.all:
        daily = total_cost / args.days
        print(f"    ≈ {fmt_cost(daily)}/day · {fmt_cost(daily*30)}/mo projected")
    print()

    # ── Empirical compression ratios from sampled content ─────────────────
    empirical_ratios = {}
    for cat, samples in grand["cmd_samples"].items():
        if len(samples) < 3:
            continue
        orig_total = comp_total = 0
        for s in samples:
            orig_total += len(s)
            comp_total += len(simulate_compress(s, cat))
        if orig_total > 0:
            empirical_ratios[cat] = 1.0 - (comp_total / orig_total)

    def get_ratio(cat):
        """Return empirical ratio if available, else hardcoded fallback."""
        return empirical_ratios.get(cat, CMD_COMPRESSION.get(cat, 0.70))

    # ── Operation table ───────────────────────────────────────────────────
    # Rows: Bash command categories first, then other tools, then totals
    rows = []

    # Bash categories (only show ones with data)
    bash_total_calls = bash_total_std = bash_total_compressed = 0
    for cat in CMD_COMPRESSION:
        calls = grand["cmd_calls"].get(cat, 0)
        if calls == 0:
            continue
        ratio = get_ratio(cat)
        total_b = grand["cmd_bytes"].get(cat, 0)
        std_tok = total_b // 4
        avg_tok = std_tok // calls if calls else 0
        compressed_tok = int(std_tok * (1 - ratio))
        rows.append((cat, calls, avg_tok, std_tok, compressed_tok, ratio))
        bash_total_calls += calls
        bash_total_std += std_tok
        bash_total_compressed += compressed_tok

    # Other tools
    other_tool_std = other_tool_compressed = 0
    other_rows = []
    for tool, total_b in sorted(grand["tool_result_bytes"].items(), key=lambda x: -x[1]):
        if tool in ("Bash", "unknown"):
            continue
        ratio = TOOL_COMPRESSION.get(tool, TOOL_COMPRESSION["default"])
        std_tok = total_b // 4
        calls = grand["tool_calls"].get(tool, 0)
        avg_tok = std_tok // calls if calls else 0
        compressed_tok = int(std_tok * (1 - ratio))
        other_rows.append((tool, calls, avg_tok, std_tok, compressed_tok, ratio))
        other_tool_std += std_tok
        other_tool_compressed += compressed_tok

    grand_std = bash_total_std + other_tool_std
    grand_compressed = bash_total_compressed + other_tool_compressed
    grand_saving = (grand_std - grand_compressed) / grand_std if grand_std else 0

    # Build lookup: category → (calls, std_tok, compressed_tok, ratio)
    cat_data = {}
    for cat, calls, avg, std, comp, ratio in rows:
        cat_data[cat] = (calls, std, comp, ratio)

    # Canonical rows (fixed order, matching reference table)
    # Each entry: (display_name, [category_keys_to_merge])
    CANONICAL = [
        ("`ls` / `tree`",             ["ls / tree"]),
        ("`cat` / `read`",            ["cat / read"]),
        ("`grep` / `rg`",             ["grep / rg"]),
        ("`git status`",              ["git status"]),
        ("`git diff`",                ["git diff"]),
        ("`git log`",                 ["git log"]),
        ("`git add/commit/push`",     ["git add/commit/push"]),
        ("`npm test` / `cargo test`", ["npm/bun test", "cargo test"]),
        ("`ruff check`",              ["lint (ruff/eslint)"]),
        ("`pytest`",                  ["pytest"]),
        ("`go test`",                 ["go test"]),
        ("`docker ps`",               ["docker"]),
    ]

    md_header  = "| Operation | Frequency | Standard |"
    md_divider = "|-----------|-----------|----------|"

    print(f"⚡  TOKEN OPERATION TABLE  (based on your {period_label} sessions)\n")
    print(md_header)
    print(md_divider)

    rendered_cats = set()
    canon_total_calls = canon_total_std = canon_total_comp = 0

    for display, cats in CANONICAL:
        c_calls = c_std = c_comp = 0
        is_empirical = False
        for cat in cats:
            if cat in cat_data:
                calls, std, comp, ratio = cat_data[cat]
                c_calls += calls
                c_std += std
                c_comp += comp
                rendered_cats.add(cat)
                if cat in empirical_ratios:
                    is_empirical = True
        if c_calls == 0:
            continue
        saving = (c_std - c_comp) / c_std if c_std else 0
        print(f"| {display} | {c_calls:,}× | {fmt_tokens(c_std)} |")
        canon_total_calls += c_calls
        canon_total_std += c_std
        canon_total_comp += c_comp

    # Extra bash categories not in canonical list
    extras = [(cat, *cat_data[cat]) for cat in sorted(cat_data, key=lambda c: -cat_data[c][1]) if cat not in rendered_cats]
    for cat, calls, std, comp, ratio in extras:
        print(f"| {cat} | {calls:,}× | {fmt_tokens(std)} |")
        canon_total_calls += calls
        canon_total_std += std
        canon_total_comp += comp

    bash_ratio = (canon_total_std - canon_total_comp) / canon_total_std if canon_total_std else 0
    print(f"| **Total** | | **{fmt_tokens(grand_std)}** |")
    print()

    # ── Tool call frequency ───────────────────────────────────────────────────
    print(f"🔧  TOOL CALL FREQUENCY")
    top_calls = sorted(grand["tool_calls"].items(), key=lambda x: -x[1])
    max_count = top_calls[0][1] if top_calls else 1
    for tool, count in top_calls[:10]:
        bar = "█" * max(1, round(count / max_count * 24))
        print(f"    {tool:<28} {count:>5,}  {bar}")
    print()

    # ── Top projects ──────────────────────────────────────────────────────────
    top_n = args.top
    top_projs = sorted(project_stats.values(), key=lambda x: -(x["input"] + x["output"]))[:top_n]
    print(f"🗂️   TOP {top_n} PROJECTS BY TOKEN USAGE\n")
    print(f"    {'Project':<32} {'Sess':>5} {'Input':>7} {'Output':>7} {'Cache%':>7} {'Cost':>8}")
    print(f"    {'─'*32} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for s in top_projs:
        name = proj_display(s["_name"])[:31]
        cap = max(s["input"] + s["cache_read"] + s["cache_write"], 1)
        cpct = s["cache_read"] / cap * 100
        cost = compute_cost(s)
        print(f"    {name:<32} {s['session_count']:>5,} {fmt_tokens(s['input']):>7}"
              f" {fmt_tokens(s['output']):>7} {cpct:>6.0f}% {fmt_cost(cost):>8}")
    print()

    # ── Optimization tips ─────────────────────────────────────────────────────
    print(f"🎯  OPTIMIZATION TIPS\n")
    low_cache = [
        s for s in project_stats.values()
        if s["session_count"] >= 3
        and (s["cache_read"] / max(s["input"] + s["cache_read"] + s["cache_write"], 1)) < 0.25
        and (s["input"] + s["output"]) > 5000
    ]
    if low_cache:
        print(f"    Low cache hit rate (<25%) — longer sessions = more cache reuse:")
        for s in sorted(low_cache, key=lambda x: -(x["input"] + x["output"]))[:4]:
            cap = max(s["input"] + s["cache_read"] + s["cache_write"], 1)
            pct = s["cache_read"] / cap * 100
            print(f"    ├─ {proj_display(s['_name'])}: {pct:.0f}% hit, {s['session_count']} sessions")
        print()

    avg_per_sess = total_cost / grand["session_count"] if grand["session_count"] else 0
    if not args.all:
        daily = total_cost / args.days
        projected_compressed = (daily - (grand_std - grand_compressed) * PRICING["input"] / 1_000_000 / args.days) * 30
        print(f"    Avg cost/session:   {fmt_cost(avg_per_sess)}")
        print(f"    Projected monthly:  {fmt_cost(daily*30)}  →  {fmt_cost(projected_compressed)} with compression")
    print(f"    Sessions scanned:   {scanned:,}  (skipped by age: {skipped:,})")
    print(f"\n{'═'*W}\n")


if __name__ == "__main__":
    main()
