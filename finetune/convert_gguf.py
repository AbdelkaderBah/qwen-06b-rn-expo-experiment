"""
Convert a LoRA adapter to GGUF format.

Usage:
  uv run python finetune/convert_gguf.py
  uv run python finetune/convert_gguf.py --adapter finetune/output/lora-adapter
  uv run python finetune/convert_gguf.py --quant q8_0
  uv run python finetune/convert_gguf.py --output ~/.lmstudio/models/organization/model-name
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel

QUANT_METHODS = ["q4_k_m", "q8_0", "f16"]


def convert(adapter_path: str, output_dir: str, quant: str) -> None:
    adapter = Path(adapter_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] Loading adapter from {adapter}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=2048,
        load_in_4bit=False,
    )

    print(f"[export] Converting to GGUF ({quant})...")
    try:
        model.save_pretrained_gguf(str(out), tokenizer, quantization_method=quant)
        print(f"[done] GGUF saved -> {out}")
    except RuntimeError as e:
        print(f"[warn] GGUF export failed: {e}")
        print("[fallback] Saving merged 16-bit model instead...")
        merged = out.parent / "merged"
        merged.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged), tokenizer)
        print(f"[done] Merged model saved -> {merged}")
        print("[hint] Use llama.cpp manually: llama-quantize model.gguf output.gguf Q4_K_M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="finetune/output/lora-adapter")
    parser.add_argument("--output", default="finetune/output/gguf")
    parser.add_argument("--quant", default="q4_k_m", choices=QUANT_METHODS)
    args = parser.parse_args()
    convert(args.adapter, args.output, args.quant)
