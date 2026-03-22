"""
Phase 2 — Dataset generation
Reads chunks from data/processed/*.jsonl
Generates (instruction, output) pairs via GitHub Copilot SDK
Saves to data/dataset/rn_expo_dataset.jsonl

Run: uv run python data/generate_dataset.py
Requires: gh copilot CLI signed in (no token needed)
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from copilot import CopilotClient
from copilot.session import PermissionRequestResult, SessionEventType
from ts_check import check_typescript

PROCESSED = Path("data/processed")
DATASET = Path("data/dataset")
DATASET.mkdir(exist_ok=True)
OUT = DATASET / "rn_expo_dataset.jsonl"

MODEL = "gpt-4.1"
REPAIR_MODEL = "gpt-5-mini"
TARGET_PAIRS = 500
SAVE_EVERY = 50
REQUEST_TIMEOUT = 120.0
MAX_REPAIR_CONTEXT_CHARS = 6000

SYSTEM_PROMPT = """You are an expert React Native 0.82 and Expo developer generating a fine-tuning dataset.

Given a documentation excerpt, output ONE entry as JSON:
{"instruction": "<direct developer question>", "output": "<complete code answer>"}

Rules for instruction:
- Must be a direct question a developer would type (e.g. "How do I create a FlatList with pull-to-refresh in React Native?")
- NO meta-phrases like "Create a question about...", "Generate a coding challenge...", "Write an example..."
- Specific and actionable, not vague

Rules for output:
- Complete, runnable TypeScript/JSX code
- Exactly ONE self-contained TypeScript/TSX module, not multiple files
- Functional components and hooks only (no class components)
- Modern APIs only (no deprecated ones)
- No explanations outside the code — inline comments are fine
- No raw JSON, YAML, shell transcript, Markdown, or prose in the output field
- If the excerpt is about config/package.json/YAML/workflows, convert it into valid JS/TS code that still compiles as a single file
- Use only real packages and APIs
- Expo Server imports must use 'expo-server' (never 'expo/server')
- Expo UI imports must use '@expo/ui/swift-ui', '@expo/ui/jetpack-compose', '@expo/ui/datetimepicker', or their '/modifiers' modules when relevant (never 'expo-ui' or 'expo-ui-gauge')
- Do not invent React Native exports or unsupported packages
- Do not emit multiple default exports, file separators, or filename headers
- Do not wrap the JSON object or the output code in Markdown fences

Example of a good entry:
{"instruction": "How do I implement a FlatList with infinite scroll in React Native 0.82?", "output": "import React, { useState, useCallback } from 'react';\\nimport { FlatList, ActivityIndicator, View, Text } from 'react-native';\\n..."}

Respond with valid JSON only."""

REPAIR_PROMPT = """You are repairing a rejected React Native / Expo fine-tuning dataset entry.

Return ONE JSON object only:
{"instruction": "<direct developer question>", "output": "<complete code answer>"}

Repair goals:
- Keep the instruction natural and specific. Rewrite it only if needed.
- The output must be exactly ONE self-contained TypeScript/TSX module.
- The output must compile with TypeScript in a React Native / Expo project.
- Preserve the intent of the original answer, but fix the rejected code completely.
- Never return raw JSON, YAML, shell commands, Markdown, or multiple files.
- Expo Server imports must use 'expo-server' (never 'expo/server').
- Expo UI imports must use '@expo/ui/swift-ui', '@expo/ui/jetpack-compose', '@expo/ui/datetimepicker', or their '/modifiers' modules when relevant (never 'expo-ui' or 'expo-ui-gauge').
- Do not invent packages, APIs, or exports.
- Do not wrap the JSON object or the output code in Markdown fences.

Use the documentation excerpt as the source of truth.
"""


def load_chunks() -> list[dict]:
    chunks = []
    for f in sorted(PROCESSED.glob("*.jsonl")):
        with f.open() as fp:
            for line in fp:
                c = json.loads(line)
                if len(c["text"]) > 300:
                    chunks.append(c)
    return chunks


def load_existing() -> set[str]:
    seen = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                item = json.loads(line)
                seen.add(item["instruction"])
    return seen


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if not lines:
        return text

    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def normalize_pair(raw_pair: dict) -> tuple[dict | None, str]:
    instruction = raw_pair.get("instruction")
    output = raw_pair.get("output")

    if not isinstance(instruction, str) or not isinstance(output, str):
        return None, "missing fields"

    instruction = instruction.strip()
    output = strip_code_fences(output).strip()
    if not instruction or not output:
        return None, "empty fields"

    return {"instruction": instruction, "output": output}, ""


def parse_pair_response(content: str) -> tuple[dict | None, str]:
    cleaned = strip_code_fences(content)
    candidates = [cleaned]

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(cleaned[first_brace:last_brace + 1])

    last_error = "bad JSON"
    seen_candidates = set()

    for candidate in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)

        try:
            raw_pair = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"bad JSON ({exc.msg})"
            continue

        if not isinstance(raw_pair, dict):
            last_error = "response was not a JSON object"
            continue

        pair, pair_error = normalize_pair(raw_pair)
        if pair is not None:
            return pair, ""
        last_error = pair_error

    return None, last_error


def build_source_diverse_queue(chunks: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for chunk in chunks:
        buckets.setdefault(chunk["source"], []).append(chunk)

    sources = list(buckets)
    random.shuffle(sources)
    for source_chunks in buckets.values():
        random.shuffle(source_chunks)

    queue: list[dict] = []
    active_sources = sources
    while active_sources:
        next_round: list[str] = []
        for source in active_sources:
            source_chunks = buckets[source]
            queue.append(source_chunks.pop())
            if source_chunks:
                next_round.append(source)
        random.shuffle(next_round)
        active_sources = next_round

    return queue


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


def build_generation_prompt(chunk: dict) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Source path: {chunk['source']}\n\n"
        f"Documentation excerpt:\n\n{chunk['text']}\n\n"
        "JSON:"
    )


def build_repair_prompt(
    chunk: dict,
    failure_reason: str,
    raw_response: str,
    pair: dict | None,
) -> str:
    repair_context = {
        "source": chunk["source"],
        "failure_reason": failure_reason,
        "previous_raw_response": raw_response[:MAX_REPAIR_CONTEXT_CHARS],
    }
    if pair is not None:
        repair_context["previous_pair"] = pair

    return (
        f"{REPAIR_PROMPT}\n\n"
        f"Source path: {chunk['source']}\n\n"
        f"Documentation excerpt:\n\n{chunk['text']}\n\n"
        f"Rejected candidate context:\n{json.dumps(repair_context, ensure_ascii=False, indent=2)}\n\n"
        "Return repaired JSON:"
    )


async def repair_pair(
    client: CopilotClient,
    chunk: dict,
    failure_reason: str,
    raw_response: str,
    pair: dict | None,
) -> tuple[dict | None, str]:
    repaired_content = await request_model_output(
        client,
        REPAIR_MODEL,
        build_repair_prompt(chunk, failure_reason, raw_response, pair),
    )
    return parse_pair_response(repaired_content)


async def generate_pairs(chunks: list[dict], existing: set[str], requested_pairs: int) -> None:
    client = CopilotClient()
    await client.start()

    starting_total = len(existing)
    generated_new = 0
    target_total = starting_total + requested_pairs
    print(
        f"🚀 Starting from {starting_total} existing pairs, "
        f"target: +{requested_pairs} new ({target_total} total)\n"
    )

    attempted = 0

    with OUT.open("a", encoding="utf-8") as out_file:
        for chunk in chunks:
            if generated_new >= requested_pairs:
                break

            attempted += 1
            prompt = build_generation_prompt(chunk)
            src = chunk["source"][:50]
            print(f"[{attempted}] {src}  ", end="", flush=True)

            # Step 1: generate
            print("generating...", end=" ", flush=True)
            try:
                raw_response = await request_model_output(client, MODEL, prompt)
            except Exception as e:
                print(f"⚠️ {e}")
                continue

            # Step 2: parse
            print("parsing...", end=" ", flush=True)
            if not raw_response:
                print("⏭️ empty reply")
                continue

            pair, parse_error = parse_pair_response(raw_response)
            if pair is None:
                print("repairing...", end=" ", flush=True)
                try:
                    pair, parse_error = await repair_pair(
                        client,
                        chunk,
                        parse_error,
                        raw_response,
                        None,
                    )
                except Exception as e:
                    print(f"⚠️ repair {e}")
                    continue
                if pair is None:
                    print(f"❌ {parse_error}")
                    continue

            instruction = pair["instruction"]
            output = pair["output"]

            if instruction in existing:
                print("⏭️ duplicate")
                continue

            # Step 3: tsc validation
            print("tsc...", end=" ", flush=True)
            ts_ok, ts_err = check_typescript(output)
            if not ts_ok:
                print("repairing...", end=" ", flush=True)
                try:
                    repaired_pair, _ = await repair_pair(
                        client,
                        chunk,
                        ts_err,
                        raw_response,
                        pair,
                    )
                except Exception as e:
                    print(f"⚠️ repair {e}")
                    continue

                if repaired_pair is None:
                    print(f"❌ {ts_err[:70]}")
                    continue

                repaired_instruction = repaired_pair["instruction"]
                repaired_output = repaired_pair["output"]

                if repaired_instruction in existing and repaired_instruction != instruction:
                    print("⏭️ duplicate after repair")
                    continue

                print("tsc...", end=" ", flush=True)
                repaired_ts_ok, repaired_ts_err = check_typescript(repaired_output)
                if not repaired_ts_ok:
                    print(f"❌ {repaired_ts_err[:70]}")
                    continue

                instruction = repaired_instruction
                output = repaired_output

            # Step 4: save
            entry = {
                "instruction": instruction,
                "output": output,
                "source": chunk["source"],
            }
            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_file.flush()
            existing.add(instruction)
            generated_new += 1
            total_pairs = starting_total + generated_new
            print(f"✅ ({generated_new}/{requested_pairs})")

            if total_pairs % SAVE_EVERY == 0:
                print(f"\n💾 Checkpoint: {total_pairs} pairs saved\n")

    await client.stop()
    total_pairs = starting_total + generated_new
    print(f"\n🎉 Done! Added {generated_new} new pairs ({total_pairs} total) → {OUT}")
    if generated_new < requested_pairs:
        print(
            f"⚠️  Requested {requested_pairs} new pairs but only generated {generated_new}. "
            "Some chunks were skipped or produced duplicates."
        )


async def main(samples: int) -> None:
    chunks = load_chunks()
    print(f"📦 {len(chunks)} usable chunks loaded")

    if samples <= 0:
        raise ValueError("--samples must be greater than 0")

    random.seed(42)
    queued_chunks = build_source_diverse_queue(chunks)

    existing = load_existing()
    if existing:
        print(f"▶️  Resuming — {len(existing)} pairs already done")

    if samples > len(queued_chunks):
        print(f"⚠️  Requested {samples} pairs but only {len(queued_chunks)} chunks are available")

    await generate_pairs(queued_chunks, existing, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=TARGET_PAIRS,
                        help=f"Number of pairs to generate (default: {TARGET_PAIRS})")
    args = parser.parse_args()
    asyncio.run(main(args.samples))
