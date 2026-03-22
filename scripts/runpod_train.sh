#!/usr/bin/env bash
set -euo pipefail

# ─── RunPod one-shot training script ───
# Trains the model, uploads GGUF to HuggingFace Hub, then stops the pod.
#
# Required env vars (set in RunPod pod config):
#   HF_TOKEN        — HuggingFace write token
#   HF_REPO         — e.g. "your-username/qwen-06b-rn-expo"
#
# Optional env vars:
#   EPOCHS          — default: 5
#   LR              — default: 5e-5
#   BATCH_SIZE      — default: 4
#   RUNPOD_POD_ID   — auto-set by RunPod, used for auto-stop

EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"

echo "═══════════════════════════════════════"
echo "  RunPod Training — Qwen 0.6B LoRA"
echo "  Run ID:  $RUN_ID"
echo "  Epochs: $EPOCHS | LR: $LR | Batch: $BATCH_SIZE"
echo "═══════════════════════════════════════"

cd "$(dirname "$0")/.."

# Train + export GGUF
uv run python finetune/train.py \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --export-gguf

echo "[upload] Pushing artifacts to HuggingFace: $HF_REPO (run: $RUN_ID)"
EPOCHS="$EPOCHS" LR="$LR" BATCH_SIZE="$BATCH_SIZE" RUN_ID="$RUN_ID" \
  uv run python - <<'PYEOF'
import os, json
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO"]
run_id = os.environ["RUN_ID"]
run_prefix = f"runs/{run_id}"

api.create_repo(repo_id, exist_ok=True, repo_type="model")

# Upload LoRA adapter (always available)
api.upload_folder(
    repo_id=repo_id,
    folder_path="finetune/output/lora-adapter",
    path_in_repo=f"{run_prefix}/lora-adapter",
    commit_message=f"[{run_id}] Upload LoRA adapter",
)

# Upload GGUF if it was exported
gguf_path = Path("finetune/output/gguf")
if gguf_path.exists() and any(gguf_path.glob("*.gguf")):
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(gguf_path),
        path_in_repo=f"{run_prefix}/gguf",
        commit_message=f"[{run_id}] Upload GGUF model",
    )

# Upload merged 16-bit if GGUF failed and merged was saved instead
merged_path = Path("finetune/output/merged")
if merged_path.exists():
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(merged_path),
        path_in_repo=f"{run_prefix}/merged-16bit",
        commit_message=f"[{run_id}] Upload merged 16-bit weights",
    )

# Save training params
params = {
    "run_id": run_id,
    "epochs": os.environ.get("EPOCHS", "5"),
    "lr": os.environ.get("LR", "5e-5"),
    "batch_size": os.environ.get("BATCH_SIZE", "4"),
}
api.upload_file(
    repo_id=repo_id,
    path_or_fileobj=json.dumps(params, indent=2).encode(),
    path_in_repo=f"{run_prefix}/training_params.json",
    commit_message=f"[{run_id}] Training params",
)

print(f"[upload] Done! https://huggingface.co/{repo_id}/tree/main/{run_prefix}")
PYEOF

# Auto-stop the pod to stop billing
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[cleanup] Stopping RunPod pod $RUNPOD_POD_ID..."
  runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || \
    echo "[cleanup] Could not auto-stop. Remember to stop the pod manually!"
else
  echo "[cleanup] No RUNPOD_POD_ID set. Stop the pod manually to avoid charges."
fi

echo "═══════════════════════════════════════"
echo "  Done! Model uploaded to HuggingFace."
echo "═══════════════════════════════════════"
