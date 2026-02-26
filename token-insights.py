#!/usr/bin/env python3
"""
token-insights.py — Token operation table for Claude Code

Scans ~/.claude/projects/ session history and prints the Token Operation
Table showing where tokens are going by command category.

Usage:
  python3 token-insights.py
  python3 token-insights.py --days 30
  python3 token-insights.py --days 7
  python3 token-insights.py --project myproject
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
        keep = []
        for line in lines:
            l = line.strip()
            if any(kw in l.lower() for kw in ("fail", "error", "panic", "assert", "FAILED", "passed", "failed", "ok -", "test result")):
                keep.append(line)
        keep = keep or lines[:5]
        return "\n".join(keep[:40])

    if category == "git diff":
        keep = [l for l in lines if l.startswith(("+", "-", "@@", "diff", "index", "---", "+++"))]
        return "\n".join(keep[:100])

    if category in ("ls / tree",):
        return "\n".join(lines[:40])

    if category in ("cat / read",):
        if len(lines) > 80:
            return "\n".join(lines[:60] + ["..."] + lines[-10:])
        return "\n".join(lines)

    if category in ("git log",):
        return "\n".join(lines[:30])

    if category in ("git status",):
        return "\n".join(l for l in lines if l.strip())

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
        return "npm/bun run/build"

    if first == "cargo":
        if rest == "test":    return "cargo test"
        return "cargo build/run"

    if first == "pytest" or (first in ("python", "python3") and "pytest" in cmd):
        return "pytest"

    if first == "go":
        if rest == "test":    return "go test"
        return "cargo build/run"

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

    if first == "docker":                        return "docker"
    if first in ("kubectl", "k9s", "helm", "k"): return "k8s (kubectl/helm)"
    if first == "ssh":                           return "ssh"
    if first in ("curl", "wget", "http", "httpie"): return "curl / http"
    if first in ("python", "python3"):           return "python / script"
    if first in ("terraform", "tofu"):           return "terraform"
    if first == "make":                          return "make"

    return "other bash"


def parse_args():
    p = argparse.ArgumentParser(description="Claude Code token operation table")
    p.add_argument("--days", type=int, default=None, help="Limit to last N days (default: all time)")
    p.add_argument("--project", type=str, help="Filter to project name (substring match)")
    p.add_argument("--all", action="store_true", help="Alias for default all-time behavior")
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
        "cmd_calls": defaultdict(int),
        "cmd_bytes": defaultdict(int),
        "cmd_samples": defaultdict(list),
        "_id_to_tool": {},
        "_id_to_cmd": {},
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
    stats["tool_calls"] = dict(stats["tool_calls"])
    stats["tool_result_bytes"] = dict(stats["tool_result_bytes"])
    stats["cmd_calls"] = dict(stats["cmd_calls"])
    stats["cmd_bytes"] = dict(stats["cmd_bytes"])
    stats["cmd_samples"] = dict(stats["cmd_samples"])
    return stats


# ── Top-level worker (must be module-level for multiprocessing pickle) ────────

def process_project(args_tuple):
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
    home_str = str(Path.home()).replace("/", "-")
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
    cutoff = None if (args.all or args.days is None) else now - timedelta(days=args.days)
    period_label = f"last {args.days} days" if args.days else "all time"

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

    # ── Empirical compression ratios from sampled content ─────────────────────
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
        return empirical_ratios.get(cat, CMD_COMPRESSION.get(cat, 0.70))

    # ── Operation table ────────────────────────────────────────────────────────
    rows = []
    for cat in CMD_COMPRESSION:
        calls = grand["cmd_calls"].get(cat, 0)
        if calls == 0:
            continue
        ratio = get_ratio(cat)
        total_b = grand["cmd_bytes"].get(cat, 0)
        std_tok = total_b // 4
        compressed_tok = int(std_tok * (1 - ratio))
        rows.append((cat, calls, std_tok, compressed_tok))

    other_tool_std = 0
    for tool, total_b in grand["tool_result_bytes"].items():
        if tool in ("Bash", "unknown"):
            continue
        ratio = TOOL_COMPRESSION.get(tool, TOOL_COMPRESSION["default"])
        other_tool_std += total_b // 4

    grand_std = sum(r[2] for r in rows) + other_tool_std

    cat_data = {cat: (calls, std, comp) for cat, calls, std, comp in rows}

    # Canonical rows in reference order; anything extra is appended sorted by tokens
    CANONICAL = [
        ("ls / tree",             ["ls / tree"]),
        ("cat / read",            ["cat / read"]),
        ("grep / rg",             ["grep / rg"]),
        ("git status",            ["git status"]),
        ("git diff",              ["git diff"]),
        ("git log",               ["git log"]),
        ("git add/commit/push",   ["git add/commit/push"]),
        ("npm test / cargo test", ["npm/bun test", "cargo test"]),
        ("ruff check",            ["lint (ruff/eslint)"]),
        ("pytest",                ["pytest"]),
        ("go test",               ["go test"]),
        ("docker ps",             ["docker"]),
    ]

    OP_W = 22
    print(f"\nToken Operation Table  ·  {period_label}\n")
    print(f"| {'Operation':<{OP_W}} | {'Frequency':>9} | {'Standard':>8} |")
    print(f"|{'-'*(OP_W+2)}|{'-'*11}|{'-'*10}|")

    rendered_cats = set()
    for display, cats in CANONICAL:
        c_calls = c_std = 0
        for cat in cats:
            if cat in cat_data:
                calls, std, _ = cat_data[cat]
                c_calls += calls
                c_std += std
                rendered_cats.add(cat)
        if c_calls == 0:
            continue
        freq_str = f"{c_calls:,}×"
        print(f"| {display:<{OP_W}} | {freq_str:>9} | {fmt_tokens(c_std):>8} |")

    extras = sorted(
        [(cat, *cat_data[cat]) for cat in cat_data if cat not in rendered_cats],
        key=lambda x: -x[2],
    )
    for cat, calls, std, _ in extras:
        freq_str = f"{calls:,}×"
        print(f"| {cat:<{OP_W}} | {freq_str:>9} | {fmt_tokens(std):>8} |")

    print(f"| {'Total':<{OP_W}} | {'':>9} | {fmt_tokens(grand_std):>8} |")
    print()


if __name__ == "__main__":
    main()
