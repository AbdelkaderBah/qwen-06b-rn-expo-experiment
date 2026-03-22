"""
Dataset validation — 3 layers:
  1. TypeScript check (tsc --noEmit) — catches bad imports, syntax errors, type errors
  2. Heuristic filter — rejects non-component outputs (configs, prose, no React imports)
  3. LLM quality filter via Copilot SDK gpt-5.1-mini — semantic quality check

Run:
  uv run python data/validate_dataset.py                    # layers 1+2 only
  uv run python data/validate_dataset.py --llm-filter       # all 3 layers
  uv run python data/validate_dataset.py --dry-run          # show results without writing
"""

import argparse
import asyncio
import json
import re
import subprocess
from pathlib import Path

from copilot import CopilotClient
from copilot.session import PermissionRequestResult, SessionEventType
from ts_check import check_typescript

REQUEST_TIMEOUT = 120.0

DATASET = Path("data/dataset")
INPUT = DATASET / "rn_expo_dataset.jsonl"
OUTPUT = DATASET / "rn_expo_dataset_validated.jsonl"
REJECTED = DATASET / "rn_expo_dataset_rejected.jsonl"


# --- Layer 2: Heuristic filter ---

RN_IMPORT_PATTERN = re.compile(
    r"""import\s+.*from\s+['"](?:react-native|react|expo-|@expo/|@react-navigation/)""",
)

META_PHRASES = re.compile(
    r"(?:create a question|generate a|write an example|coding challenge|explain how)",
    re.IGNORECASE,
)


def heuristic_check(pair: dict) -> tuple[bool, str]:
    output = pair["output"]
    instruction = pair["instruction"]

    if META_PHRASES.search(instruction):
        return False, "meta-phrase in instruction"

    if not RN_IMPORT_PATTERN.search(output):
        return False, "no React/RN/Expo import — likely config or prose"

    if output.count("export default") > 1:
        return False, "multiple default exports"

    if len(output.strip()) < 50:
        return False, "output too short"

    if output.count("...") > 3:
        return False, "too many placeholders (...)"

    return True, ""


# --- Layer 3: LLM quality filter ---

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


async def request_model_output(client: CopilotClient, model: str, prompt: str) -> str:
    session = await client.create_session(
        on_permission_request=lambda req, inv: PermissionRequestResult(kind="approved"),
        model=model,
        streaming=True,
    )
    collected: list[str] = []
    session.on(
        lambda event: collected.append(event.data.delta_content or "")
        if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA else None
    )
    try:
        await session.send_and_wait(prompt, timeout=REQUEST_TIMEOUT)
    finally:
        await session.disconnect()
    return "".join(collected).strip()


def parse_json_response(content: str) -> list[dict] | None:
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    first = content.find("[")
    last = content.rfind("]")
    if first != -1 and last != -1:
        content = content[first:last + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


async def llm_review(client: CopilotClient, pairs: list[dict]) -> list[dict]:
    entries = []
    for i, p in enumerate(pairs):
        entries.append({
            "index": i,
            "instruction": p["instruction"],
            "output_preview": p["output"][:800],
        })

    prompt = REVIEW_PROMPT + json.dumps(entries, indent=2)
    content = await request_model_output(client, "gpt-5.1-mini", prompt)
    verdicts = parse_json_response(content)

    if verdicts is None:
        print(f"  ⚠️  LLM review parse error, keeping all pairs in batch")
        return [{"index": i, "keep": True, "reason": "parse error"} for i in range(len(pairs))]
    return verdicts


# --- Main ---

def load_pairs() -> list[dict]:
    pairs = []
    with INPUT.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


async def validate(llm_filter_enabled: bool, dry_run: bool) -> None:
    pairs = load_pairs()
    print(f"📦 Loaded {len(pairs)} pairs from {INPUT}\n")

    results: list[dict] = []

    # Layer 1: tsc
    print("🔍 Layer 1: TypeScript check (tsc --noEmit)")
    tsc_passed: list[tuple[int, dict]] = []

    for i, pair in enumerate(pairs):
        ts_ok, ts_err = check_typescript(pair["output"])
        reasons = [] if ts_ok else [f"tsc: {ts_err}"]
        status = "pass" if ts_ok else "reject"
        results.append({"pair": pair, "status": status, "reasons": reasons})

        icon = "✅" if ts_ok else "❌"
        print(f"  [{i+1}/{len(pairs)}] {icon} {pair['instruction'][:70]}...")
        if not ts_ok:
            print(f"         → {ts_err[:120]}")

        if ts_ok:
            tsc_passed.append((i, pair))

    tsc_rejected = len(pairs) - len(tsc_passed)
    print(f"\n  Layer 1: {len(tsc_passed)} passed, {tsc_rejected} rejected\n")

    # Layer 2: Heuristic filter
    print("🔍 Layer 2: Heuristic filter")
    heuristic_passed: list[tuple[int, dict]] = []

    for orig_idx, pair in tsc_passed:
        h_ok, h_err = heuristic_check(pair)
        if not h_ok:
            results[orig_idx]["status"] = "reject"
            results[orig_idx]["reasons"].append(f"heuristic: {h_err}")
            print(f"  ❌ [{orig_idx+1}] {h_err} — {pair['instruction'][:60]}...")
        else:
            heuristic_passed.append((orig_idx, pair))

    h_rejected = len(tsc_passed) - len(heuristic_passed)
    print(f"\n  Layer 2: {len(heuristic_passed)} passed, {h_rejected} rejected\n")

    # Layer 3: LLM quality filter
    if llm_filter_enabled and heuristic_passed:
        print("🤖 Layer 3: LLM quality filter (gpt-5.1-mini)")
        client = CopilotClient()
        await client.start()

        batch_size = 10
        llm_rejected = 0

        for batch_start in range(0, len(heuristic_passed), batch_size):
            batch_items = heuristic_passed[batch_start:batch_start + batch_size]
            batch_pairs = [p for _, p in batch_items]
            print(f"  Batch {batch_start // batch_size + 1} ({len(batch_pairs)} pairs)...", end=" ", flush=True)

            verdicts = await llm_review(client, batch_pairs)
            batch_rejected = 0
            for v in verdicts:
                idx = v.get("index", 0)
                if idx >= len(batch_items):
                    continue
                orig_idx = batch_items[idx][0]
                if not v.get("keep", True):
                    results[orig_idx]["status"] = "reject"
                    results[orig_idx]["reasons"].append(f"llm: {v.get('reason', '')}")
                    batch_rejected += 1
                    llm_rejected += 1

            print(f"{'✅' if batch_rejected == 0 else f'❌ {batch_rejected} rejected'}")

        await client.stop()
        print(f"\n  Layer 3: {len(heuristic_passed) - llm_rejected} passed, {llm_rejected} rejected\n")

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
    parser.add_argument("--llm-filter", action="store_true", help="Enable gpt-5.1-mini quality filter (layer 3)")
    parser.add_argument("--dry-run", action="store_true", help="Show results without writing files")
    args = parser.parse_args()
    asyncio.run(validate(args.llm_filter, args.dry_run))
