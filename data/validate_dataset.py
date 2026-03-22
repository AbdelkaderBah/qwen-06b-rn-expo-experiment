"""
Dataset validation — 2 layers:
  1. TypeScript check (tsc --noEmit) — catches bad imports, syntax errors, type errors
  2. LLM reviewer via Copilot SDK gpt-5.4 — semantic quality check

Run:
  uv run python data/validate_dataset.py                    # layer 1 only
  uv run python data/validate_dataset.py --llm-review       # both layers
  uv run python data/validate_dataset.py --dry-run          # show results without writing
"""

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from copilot import CopilotClient
from copilot.session import PermissionRequestResult, SessionEventType
from ts_check import check_typescript

DATASET = Path("data/dataset")
INPUT = DATASET / "rn_expo_dataset.jsonl"
OUTPUT = DATASET / "rn_expo_dataset_validated.jsonl"
REJECTED = DATASET / "rn_expo_dataset_rejected.jsonl"


# --- Layer 2: LLM reviewer ---

REVIEW_PROMPT = """You are a senior React Native 0.82 / Expo code reviewer validating a fine-tuning dataset.

For each entry, check:
1. Does the code use REAL, existing APIs? (no hallucinated functions/hooks)
2. Is the code complete and not truncated? (no "..." placeholders)
3. Is the instruction a direct developer question? (not meta like "Create a question about...")
4. Does the code use modern, non-deprecated APIs?

Return ONLY a JSON array:
[{"index": 0, "keep": true, "reason": "ok"}, {"index": 1, "keep": false, "reason": "hallucinated API: useFoo"}]

Entries:
"""


async def llm_review(pairs: list[dict]) -> list[dict]:
    """Review pairs with gpt-5.4."""
    client = CopilotClient()
    await client.start()

    session = await client.create_session(
        on_permission_request=lambda req, inv: PermissionRequestResult(kind="approved"),
        model="gpt-5.4",
        streaming=True,
    )

    entries = []
    for i, p in enumerate(pairs):
        entries.append({
            "index": i,
            "instruction": p["instruction"],
            "output_preview": p["output"][:800],
        })

    prompt = REVIEW_PROMPT + json.dumps(entries, indent=2)

    collected: list[str] = []
    session.on(
        lambda event: collected.append(event.data.delta_content or "")
        if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA else None
    )
    await session.send_and_wait(prompt, timeout=120.0)
    await session.disconnect()
    await client.stop()

    content = "".join(collected).strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"  ⚠️  LLM review parse error, keeping all pairs")
        return [{"index": i, "keep": True, "reason": "parse error"} for i in range(len(pairs))]


# --- Main ---

def load_pairs() -> list[dict]:
    pairs = []
    with INPUT.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


async def validate(llm_review_enabled: bool, dry_run: bool) -> None:
    pairs = load_pairs()
    print(f"📦 Loaded {len(pairs)} pairs from {INPUT}\n")

    results: list[dict] = []

    # Layer 1: tsc
    print("🔍 Layer 1: TypeScript check (tsc --noEmit)")
    auto_passed: list[tuple[int, dict]] = []

    for i, pair in enumerate(pairs):
        ts_ok, ts_err = check_typescript(pair["output"])
        reasons = [] if ts_ok else [ts_err]
        status = "pass" if ts_ok else "reject"
        results.append({"pair": pair, "status": status, "reasons": reasons})

        icon = "✅" if ts_ok else "❌"
        print(f"  [{i+1}/{len(pairs)}] {icon} {pair['instruction'][:70]}...")
        if not ts_ok:
            print(f"         → {ts_err[:120]}")

        if ts_ok:
            auto_passed.append((i, pair))

    reject_count = len(pairs) - len(auto_passed)
    print(f"\n  tsc: {len(auto_passed)} passed, {reject_count} rejected\n")

    # Layer 2: LLM review
    if llm_review_enabled and auto_passed:
        print("🤖 Layer 2: LLM review (gpt-5.4)")
        batch_size = 10
        passed_pairs = [p for _, p in auto_passed]

        for batch_start in range(0, len(passed_pairs), batch_size):
            batch = passed_pairs[batch_start:batch_start + batch_size]
            print(f"  Reviewing batch {batch_start // batch_size + 1} ({len(batch)} pairs)...")

            verdicts = await llm_review(batch)
            for v in verdicts:
                idx = batch_start + v["index"]
                original_idx = auto_passed[idx][0]
                if not v.get("keep", True):
                    results[original_idx]["status"] = "reject"
                    results[original_idx]["reasons"].append(f"LLM: {v.get('reason', '')}")
                    print(f"    ❌ [{idx+1}] {v.get('reason', '')}")
                else:
                    print(f"    ✅ [{idx+1}]")

    # Summary
    kept = [r for r in results if r["status"] == "pass"]
    rejected = [r for r in results if r["status"] == "reject"]
    print(f"\n{'📊 DRY RUN' if dry_run else '📊'} Final: {len(kept)} kept, {len(rejected)} rejected out of {len(pairs)}")

    if not dry_run:
        with OUTPUT.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r["pair"], ensure_ascii=False) + "\n")
        print(f"  ✅ Validated → {OUTPUT}")

        with REJECTED.open("w", encoding="utf-8") as f:
            for r in rejected:
                entry = {**r["pair"], "reject_reasons": r["reasons"]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  ❌ Rejected → {REJECTED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-review", action="store_true", help="Enable gpt-5.4 LLM review (layer 2)")
    parser.add_argument("--dry-run", action="store_true", help="Show results without writing files")
    args = parser.parse_args()
    asyncio.run(validate(args.llm_review, args.dry_run))
