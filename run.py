"""
Project task runner.

Usage:
  uv run python run.py <command> [args...]
  uv run python run.py --list
"""

import json
import subprocess
import sys
from pathlib import Path

TASKS = {}


def task(name, help=""):
    def decorator(fn):
        TASKS[name] = {"fn": fn, "help": help}
        return fn
    return decorator


def run(cmd):
    print(f"→ {cmd}")
    return subprocess.run(cmd, shell=True).returncode


# ─── Tasks ───────────────────────────────────────────────────────────


@task("dataset", help="Generate fine-tuning dataset [samples=500]")
def dataset(args):
    samples = args[0] if args else "500"
    return run(f"uv run python data/generate_dataset.py --samples {samples}")


@task("validate", help="Validate dataset — extra args passed through (e.g. --llm-filter --dry-run)")
def validate(args):
    extra = " ".join(args) if args else ""
    return run(f"uv run python data/validate_dataset.py {extra}")


@task("bench", help="Run benchmark — extra args passed through (e.g. --model X --repair 2 --seed 42)")
def bench(args):
    extra = " ".join(args) if args else ""
    return run(f"uv run python eval/run_bench.py {extra}")


@task("bench-dry", help="Show benchmark questions")
def bench_dry(args):
    return run("uv run python eval/run_bench.py --dry-run")


@task("train", help="Fine-tune model [epochs=3]")
def train(args):
    epochs = args[0] if args else "3"
    return run(f"uv run python finetune/train.py --epochs {epochs} --export-gguf")


@task("process", help="Process raw docs into chunks")
def process(args):
    return run("uv run python data/process_docs.py")


@task("fetch", help="Fetch documentation from sources")
def fetch(args):
    return run("uv run python data/fetch_docs.py")


@task("tsc-install", help="Install ts-checker dependencies")
def tsc_install(args):
    return run("cd eval/ts-checker && bun install")


@task("results", help="Show scoreboard from all benchmark runs")
def results(args):
    results_dir = Path("eval/results")
    summaries = sorted(results_dir.glob("*_summary.json"))
    if not summaries:
        print("No results found in eval/results/")
        return 0

    rows = []
    for f in summaries:
        s = json.loads(f.read_text())
        rows.append(s)

    rows.sort(key=lambda s: s.get("rate_at_1", s.get("rate", 0)), reverse=True)

    print(f"\n{'═' * 85}")
    print(f"{'Model':<22} {'pass@1':>12} {'+repair':>12} {'Med.time':>9} {'tok/s':>7} {'temp':>5} {'quant':>8}")
    print(f"{'─' * 85}")
    for s in rows:
        model = s["model"][:21]
        p1 = s.get("passed_at_1", s.get("passed", 0))
        r1 = s.get("rate_at_1", s.get("rate", 0))
        total = s["total"]
        repair = s.get("max_repair", 0)
        if repair > 0:
            pr = s["passed_after_repair"]
            rr = s["rate_after_repair"]
            repair_col = f"{pr}/{total} ({rr:.0f}%)"
        else:
            repair_col = "—"
        med = s.get("median_time", s.get("avg_time", 0))
        tps = s.get("tokens_per_second", 0)
        inf = s.get("inference", {})
        temp = inf.get("temperature", "?")
        quant = inf.get("quantization", "?")[:8]
        ts = s.get("timestamp", "")[:10]
        print(f"  {model:<21} {p1}/{total} ({r1:>3.0f}%) {repair_col:>12} {med:>7.1f}s {tps:>6.1f} {temp:>5} {quant:>8}")
    print(f"{'═' * 85}\n")


# ─── Main ────────────────────────────────────────────────────────────


def print_list():
    print("Available tasks:\n")
    max_name = max(len(n) for n in TASKS)
    for name, info in TASKS.items():
        print(f"  {name:<{max_name + 2}} {info['help']}")
    print(f"\nUsage: uv run python run.py <task> [args...]")


if __name__ == "__main__":
    argv = sys.argv[1:]

    if not argv or argv[0] in ("--list", "-l", "help"):
        print_list()
        sys.exit(0)

    name = argv[0]
    if name not in TASKS:
        print(f"Unknown task: {name}")
        print_list()
        sys.exit(1)

    code = TASKS[name]["fn"](argv[1:])
    sys.exit(code or 0)
