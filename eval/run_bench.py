"""
Phase 3 — Baseline benchmark
Runs RN-Expo-Bench against a model via LM Studio and validates with tsc.

Usage:
  uv run python eval/run_bench.py                          # default: qwen3-0.6b
  uv run python eval/run_bench.py --model qwen3.5-4b       # test another model
  uv run python eval/run_bench.py --repair 2               # up to 2 self-repair attempts
  uv run python eval/run_bench.py --dry-run                # show questions only
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from ts_check import check_typescript

BENCH = Path("eval/rn_expo_bench.jsonl")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(exist_ok=True)

LM_STUDIO_BASE = "http://localhost:1234/v1"
LM_STUDIO_URL = f"{LM_STUDIO_BASE}/chat/completions"
LM_STUDIO_COMPLETIONS_URL = f"{LM_STUDIO_BASE}/completions"

SYSTEM_PROMPT = """You are a React Native 0.82 and Expo expert.
Answer with ONLY complete, runnable TypeScript/JSX code. No explanations, no markdown fences.
Use functional components and hooks only."""

COMPLETION_PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
"""

REPAIR_PROMPT = """The code you wrote has TypeScript errors. Fix the code and return ONLY the corrected complete code.
No explanations, no markdown fences.

Original instruction: {instruction}

Your previous code:
{code}

TypeScript errors:
{errors}

Return ONLY the fixed complete code."""


def preflight_check(model: str) -> bool:
    """Validate model availability and detect capabilities. Returns True if system role is supported."""
    # 1. Check LM Studio is reachable and model is loaded
    try:
        r = httpx.get(f"{LM_STUDIO_BASE}/models", timeout=5.0)
        r.raise_for_status()
    except httpx.ConnectError:
        print("❌ Cannot connect to LM Studio at localhost:1234. Is it running?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to query LM Studio models: {e}")
        sys.exit(1)

    available = [m["id"] for m in r.json().get("data", [])]
    if not available:
        print("❌ No models loaded in LM Studio.")
        sys.exit(1)

    if model not in available:
        print(f"❌ Model '{model}' not found in LM Studio.")
        print(f"   Available models:")
        for m in available:
            print(f"     - {m}")
        sys.exit(1)

    # 2. Probe system role support with a tiny request
    print("🔍 Preflight: checking system role support...", end=" ", flush=True)
    try:
        r = httpx.post(
            LM_STUDIO_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Reply with OK."},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 4,
                "temperature": 0,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        print("✅ system role supported")
        return True
    except httpx.HTTPStatusError:
        print("⚠️  system role not supported — will merge into user prompt")
        return False


def load_bench() -> list[dict]:
    with BENCH.open() as f:
        return [json.loads(line) for line in f]


def query_model(model: str, instruction: str, *, system_role: bool = True,
                 temperature: float = 0.2, max_tokens: int = 2048, top_p: float = 1.0,
                 seed: int | None = None, completion_mode: bool = False,
                 repetition_penalty: float | None = None) -> tuple[str, float, dict]:
    """Send instruction to LM Studio and return (response, duration_seconds, usage)."""
    if completion_mode:
        prompt = COMPLETION_PROMPT_TEMPLATE.format(instruction=instruction)
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["### Instruction:", "\n\n\n"],
        }
        url = LM_STUDIO_COMPLETIONS_URL
    else:
        if system_role:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{instruction}"},
            ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        url = LM_STUDIO_URL
    if seed is not None:
        payload["seed"] = seed
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    start = time.time()
    r = httpx.post(url, json=payload, timeout=120.0)
    duration = time.time() - start
    r.raise_for_status()
    data = r.json()
    if completion_mode:
        content = data["choices"][0]["text"]
    else:
        content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, duration, usage


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        body = lines[1:]
        if body and body[-1].strip() == "```":
            body = body[:-1]
        return "\n".join(body).strip()
    return text


def query_repair(model: str, instruction: str, code: str, errors: str, *, system_role: bool = True,
                  temperature: float = 0.2, max_tokens: int = 2048, top_p: float = 1.0,
                  seed: int | None = None, completion_mode: bool = False,
                  repetition_penalty: float | None = None) -> tuple[str, float, dict]:
    repair_text = REPAIR_PROMPT.format(instruction=instruction, code=code, errors=errors)
    if completion_mode:
        prompt = COMPLETION_PROMPT_TEMPLATE.format(instruction=repair_text)
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["### Instruction:", "\n\n\n"],
        }
        url = LM_STUDIO_COMPLETIONS_URL
    else:
        if system_role:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": repair_text},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{repair_text}"},
            ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        url = LM_STUDIO_URL
    if seed is not None:
        payload["seed"] = seed
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    start = time.time()
    r = httpx.post(url, json=payload, timeout=120.0)
    duration = time.time() - start
    r.raise_for_status()
    data = r.json()
    if completion_mode:
        content = data["choices"][0]["text"]
    else:
        content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, duration, usage


def run_bench(model: str, dry_run: bool, max_repair: int = 0,
              temperature: float = 0.2, max_tokens: int = 2048, top_p: float = 1.0,
              seed: int | None = None, quantization: str = "", completion_mode: bool = False,
              repetition_penalty: float | None = None) -> None:
    questions = load_bench()
    bench_hash = hashlib.md5(BENCH.read_bytes()).hexdigest()[:8]
    print(f"📋 RN-Expo-Bench: {len(questions)} questions (v:{bench_hash})")
    print(f"🤖 Model: {model}")
    print(f"⚙️  temp={temperature} max_tokens={max_tokens} top_p={top_p} seed={seed}")
    system_role = True
    if not dry_run:
        system_role = preflight_check(model)
    if dry_run:
        for q in questions:
            print(f"  [{q['id']}] ({q['difficulty']}) {q['instruction'][:80]}...")
        return

    infer_kwargs = dict(temperature=temperature, max_tokens=max_tokens, top_p=top_p, seed=seed, completion_mode=completion_mode, repetition_penalty=repetition_penalty)
    results = []
    passed_at_1 = 0
    passed_after_repair = 0
    total_time = 0.0
    all_durations = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for q in questions:
        qid = q["id"]
        print(f"[{qid}/{len(questions)}] {q['category']}/{q['difficulty']}  ", end="", flush=True)
        print("generating...", end=" ", flush=True)

        try:
            raw, duration, usage = query_model(model, q["instruction"], system_role=system_role, **infer_kwargs)
            total_time += duration
            all_durations.append(duration)
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
        except Exception as e:
            print(f"⚠️ {e}")
            results.append({**q, "status": "error", "error": str(e), "output": "", "duration": 0, "pass_number": 0, "attempts": 1})
            continue

        code = strip_fences(raw)
        print(f"({duration:.1f}s) tsc...", end=" ", flush=True)

        ts_ok, ts_err = check_typescript(code)
        pass_number = 0
        attempt = 1

        if ts_ok:
            passed_at_1 += 1
            passed_after_repair += 1
            pass_number = 1
            print("✅")
        else:
            repair_succeeded = False
            for repair_i in range(max_repair):
                attempt += 1
                print(f"repair({repair_i + 1})...", end=" ", flush=True)
                try:
                    repair_raw, repair_dur, repair_usage = query_repair(
                        model, q["instruction"], code, ts_err, system_role=system_role, **infer_kwargs,
                    )
                    total_time += repair_dur
                    total_prompt_tokens += repair_usage.get("prompt_tokens", 0)
                    total_completion_tokens += repair_usage.get("completion_tokens", 0)
                except Exception as e:
                    print(f"⚠️ {e} ", end="", flush=True)
                    continue

                code = strip_fences(repair_raw)
                print(f"({repair_dur:.1f}s) tsc...", end=" ", flush=True)
                ts_ok, ts_err = check_typescript(code)

                if ts_ok:
                    passed_after_repair += 1
                    pass_number = attempt
                    repair_succeeded = True
                    print("✅")
                    break

            if not repair_succeeded:
                print(f"❌ {ts_err[:60]}")

        results.append({
            **q,
            "status": "pass" if pass_number > 0 else "fail",
            "output": code,
            "ts_error": ts_err if pass_number == 0 else "",
            "duration": round(duration, 2),
            "pass_number": pass_number,
            "attempts": attempt,
        })

    # Summary
    rate_at_1 = (passed_at_1 / len(questions)) * 100 if questions else 0
    rate_after = (passed_after_repair / len(questions)) * 100 if questions else 0
    avg_time = total_time / len(questions) if questions else 0
    median_time = statistics.median(all_durations) if all_durations else 0
    tok_per_sec = total_completion_tokens / total_time if total_time > 0 else 0

    print(f"\n{'═' * 50}")
    print(f"📊 pass@1:           {passed_at_1}/{len(questions)} ({rate_at_1:.0f}%)")
    if max_repair > 0:
        print(f"📊 pass@1+repair({max_repair}): {passed_after_repair}/{len(questions)} ({rate_after:.0f}%)")
    print(f"⏱️  Avg time: {avg_time:.1f}s | Median: {median_time:.1f}s")
    print(f"🔤 Tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion | {tok_per_sec:.1f} tok/s")

    # Per-category breakdown
    cats = {}
    for r in results:
        cat = r["category"]
        cats.setdefault(cat, {"pass_at_1": 0, "pass_total": 0, "total": 0})
        cats[cat]["total"] += 1
        if r["status"] == "pass":
            cats[cat]["pass_total"] += 1
            if r["pass_number"] == 1:
                cats[cat]["pass_at_1"] += 1

    print(f"\nPer category:")
    for cat, s in sorted(cats.items()):
        r1 = (s['pass_at_1'] / s['total']) * 100
        line = f"  {cat:15s} pass@1: {s['pass_at_1']}/{s['total']} ({r1:.0f}%)"
        if max_repair > 0:
            rt = (s['pass_total'] / s['total']) * 100
            line += f"  +repair: {s['pass_total']}/{s['total']} ({rt:.0f}%)"
        print(line)

    # Per-difficulty breakdown
    diffs = {}
    for r in results:
        d = r["difficulty"]
        diffs.setdefault(d, {"pass_at_1": 0, "pass_total": 0, "total": 0})
        diffs[d]["total"] += 1
        if r["status"] == "pass":
            diffs[d]["pass_total"] += 1
            if r["pass_number"] == 1:
                diffs[d]["pass_at_1"] += 1

    print(f"\nPer difficulty:")
    for d in ["easy", "medium", "hard"]:
        if d in diffs:
            s = diffs[d]
            r1 = (s['pass_at_1'] / s['total']) * 100
            line = f"  {d:15s} pass@1: {s['pass_at_1']}/{s['total']} ({r1:.0f}%)"
            if max_repair > 0:
                rt = (s['pass_total'] / s['total']) * 100
                line += f"  +repair: {s['pass_total']}/{s['total']} ({rt:.0f}%)"
            print(line)

    # Save results
    out_file = RESULTS_DIR / f"{model.replace('/', '_')}.jsonl"
    with out_file.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bench_version": bench_hash,
        "inference": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
            "quantization": quantization or "unknown",
        },
        "total": len(questions),
        "passed_at_1": passed_at_1,
        "rate_at_1": round(rate_at_1, 1),
        "max_repair": max_repair,
        "passed_after_repair": passed_after_repair,
        "rate_after_repair": round(rate_after, 1),
        "avg_time": round(avg_time, 1),
        "median_time": round(median_time, 1),
        "tokens_per_second": round(tok_per_sec, 1),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "categories": cats,
        "difficulties": diffs,
    }
    summary_file = RESULTS_DIR / f"{model.replace('/', '_')}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\n💾 Results → {out_file}")
    print(f"💾 Summary → {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b", help="LM Studio model ID")
    parser.add_argument("--repair", type=int, default=0, help="Max self-repair attempts after tsc failure (default: 0)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max completion tokens (default: 2048)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--quantization", default="", help="Model quantization label (e.g. Q4_K_M, Q8_0, FP16)")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (e.g. 1.15)")
    parser.add_argument("--completion-mode", action="store_true", help="Use /v1/completions with ### Instruction format (for fine-tuned models)")
    parser.add_argument("--dry-run", action="store_true", help="Show questions only")
    args = parser.parse_args()
    run_bench(args.model, args.dry_run, args.repair,
              temperature=args.temperature, max_tokens=args.max_tokens,
              top_p=args.top_p, seed=args.seed, quantization=args.quantization,
              completion_mode=args.completion_mode,
              repetition_penalty=args.repetition_penalty)
