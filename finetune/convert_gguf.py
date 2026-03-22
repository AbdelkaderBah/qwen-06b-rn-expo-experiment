"""
Convert a LoRA adapter or merged model to GGUF format.

Usage:
  uv run python finetune/convert_gguf.py --adapter finetune/output/lora-adapter
  uv run python finetune/convert_gguf.py --merged finetune/output/merged
  uv run python finetune/convert_gguf.py --quant q8_0
  uv run python finetune/convert_gguf.py --output ~/.lmstudio/models/organization/model-name
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

QUANT_METHODS = ["q4_k_m", "q8_0", "f16"]


class _ConfigDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _TokenizerConfigJsonShim:
    def __init__(self, delegate):
        self._delegate = delegate

    def load(self, fp, *args, **kwargs):
        data = self._delegate.load(fp, *args, **kwargs)
        filename = str(getattr(fp, "name", ""))
        if isinstance(data, dict) and (filename.endswith("config.json") or "model_type" in data):
            return _ConfigDict(data)
        return data

    def __getattr__(self, name):
        return getattr(self._delegate, name)


@contextmanager
def _tokenizer_config_workaround():
    import transformers.tokenization_utils_base as _tub

    original_json = _tub.json
    _tub.json = _TokenizerConfigJsonShim(original_json)
    try:
        yield
    finally:
        _tub.json = original_json


def convert(adapter_path: str | None, merged_path: str | None, output_dir: str, quant: str) -> None:
    from unsloth import FastLanguageModel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if merged_path:
        src = Path(merged_path)
        print(f"[load] Loading merged model from {src}...")
    else:
        src = Path(adapter_path or "finetune/output/lora-adapter")
        print(f"[load] Loading adapter from {src}...")

    # transformers 4.57.2 loads local config.json into a plain dict, then reads
    # `_config.model_type` inside tokenizer loading. Patch that JSON boundary only.
    with _tokenizer_config_workaround():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(src),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter", help="Path to LoRA adapter (will merge with base model)")
    group.add_argument("--merged", help="Path to already-merged model (skips merge step)")
    parser.add_argument("--output", default="finetune/output/gguf")
    parser.add_argument("--quant", default="q4_k_m", choices=QUANT_METHODS)
    args = parser.parse_args()
    if not args.adapter and not args.merged:
        args.adapter = "finetune/output/lora-adapter"
    convert(args.adapter, args.merged, args.output, args.quant)
