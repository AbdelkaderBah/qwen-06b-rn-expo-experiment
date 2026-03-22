"""
Convert a HuggingFace model to GGUF and install into LM Studio.

Downloads merged model from HF Hub, converts with llama.cpp, quantizes,
and copies the result to LM Studio's models directory.

Usage:
  uv run python finetune/convert_gguf.py                              # latest run
  uv run python finetune/convert_gguf.py --run 20260322-201920        # specific run
  uv run python finetune/convert_gguf.py --merged finetune/output/merged  # local path
  uv run python finetune/convert_gguf.py --quant q8_0                 # different quant
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

QUANT_METHODS = ["q4_k_m", "q8_0", "f16"]
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
LLAMA_PYTHON = LLAMA_CPP_DIR / ".venv" / "bin" / "python"
LLAMA_CONVERT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
LMSTUDIO_MODELS = Path.home() / ".lmstudio" / "models"


def download_from_hf(repo_id: str, run_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set. Export it or add to .env")
        sys.exit(1)

    pattern = f"runs/{run_id}/merged-16bit/*"
    print(f"[download] {repo_id} / {pattern}")
    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=pattern,
        local_dir=str(dest),
        token=token,
    )
    merged_path = Path(local) / "runs" / run_id / "merged-16bit"
    if not merged_path.exists():
        print(f"❌ merged-16bit not found at {merged_path}")
        sys.exit(1)
    return merged_path


def find_latest_run(repo_id: str) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id)
    runs = sorted(set(
        f.split("/")[1] for f in files if f.startswith("runs/") and len(f.split("/")) > 2
    ))
    if not runs:
        print(f"❌ No runs found in {repo_id}")
        sys.exit(1)
    latest = runs[-1]
    print(f"[info] Latest run: {latest}")
    return latest


def convert_to_gguf(model_path: Path, output_gguf: Path) -> None:
    for dep in [LLAMA_PYTHON, LLAMA_CONVERT]:
        if not dep.exists():
            print(f"❌ {dep} not found. Install llama.cpp at ~/llama.cpp/")
            sys.exit(1)

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    print(f"[convert] {model_path} → {output_gguf}")
    subprocess.run(
        [str(LLAMA_PYTHON), str(LLAMA_CONVERT), str(model_path),
         "--outfile", str(output_gguf), "--outtype", "f16"],
        check=True,
    )


def quantize(input_gguf: Path, output_gguf: Path, quant: str) -> None:
    if not LLAMA_QUANTIZE.exists():
        print(f"❌ {LLAMA_QUANTIZE} not found. Build llama.cpp first.")
        sys.exit(1)

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    print(f"[quantize] {quant.upper()}: {input_gguf} → {output_gguf}")
    subprocess.run(
        [str(LLAMA_QUANTIZE), str(input_gguf), str(output_gguf), quant.upper()],
        check=True,
    )


def convert(merged_path: str | None, run_id: str | None, quant: str, org: str, model_name: str) -> None:
    repo_id = os.environ.get("HF_REPO")
    if not repo_id and not merged_path:
        print("❌ HF_REPO not set. Export it or use --merged for local path")
        sys.exit(1)

    # Resolve merged model path
    if merged_path:
        src = Path(merged_path)
    else:
        if not run_id:
            run_id = find_latest_run(repo_id)
        dl_dest = Path("finetune/output/hf-download")
        src = download_from_hf(repo_id, run_id, dl_dest)

    # Output paths
    lmstudio_dir = LMSTUDIO_MODELS / org / model_name
    lmstudio_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        f16_gguf = Path(tmp) / "model-f16.gguf"
        convert_to_gguf(src, f16_gguf)

        if quant == "f16":
            final = lmstudio_dir / f"{model_name}-f16.gguf"
            f16_gguf.rename(final) if f16_gguf.parent == final.parent else _copy(f16_gguf, final)
        else:
            final = lmstudio_dir / f"{model_name}-{quant.upper()}.gguf"
            quantize(f16_gguf, final, quant)

    print(f"\n✅ Installed to LM Studio: {final}")
    print(f"   Restart LM Studio to see the model.")


def _copy(src: Path, dst: Path) -> None:
    import shutil
    shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", help="Local path to merged model (skips HF download)")
    parser.add_argument("--run", help="HF run ID (e.g. 20260322-201920). Default: latest")
    parser.add_argument("--quant", default="q4_k_m", choices=QUANT_METHODS)
    parser.add_argument("--org", help="Organization name for LM Studio path (default: from HF_REPO)")
    parser.add_argument("--model-name", help="Model name for LM Studio directory (default: from HF_REPO)")
    args = parser.parse_args()

    hf_repo = os.environ.get("HF_REPO", "")
    if not args.org:
        args.org = hf_repo.split("/")[0] if "/" in hf_repo else "local"
    if not args.model_name:
        args.model_name = hf_repo.split("/")[1] if "/" in hf_repo else "model"

    convert(args.merged, args.run, args.quant, args.org, args.model_name)
