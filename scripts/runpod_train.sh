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
#   EPOCHS          — default: 3
#   LR              — default: 2e-4
#   BATCH_SIZE      — default: 4
#   RUNPOD_POD_ID   — auto-set by RunPod, used for auto-stop

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"

echo "═══════════════════════════════════════"
echo "  RunPod Training — Qwen 0.6B LoRA"
echo "  Run ID:  $RUN_ID"
echo "  Epochs: $EPOCHS | LR: $LR | Batch: $BATCH_SIZE"
echo "═══════════════════════════════════════"

cd /workspace

# Train + export GGUF
uv run python finetune/train.py \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --export-gguf

echo "[upload] Pushing GGUF to HuggingFace: $HF_REPO (run: $RUN_ID)"
pip install -q huggingface_hub
RUN_ID="$RUN_ID" python - <<'PYEOF'
import os, json
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO"]
run_id = os.environ["RUN_ID"]
run_prefix = f"runs/{run_id}"

# Create repo if it doesn't exist
api.create_repo(repo_id, exist_ok=True, repo_type="model")

# Upload GGUF files under runs/<timestamp>/gguf/
api.upload_folder(
    repo_id=repo_id,
    folder_path="finetune/output/gguf",
    path_in_repo=f"{run_prefix}/gguf",
    commit_message=f"[{run_id}] Upload GGUF model",
)

# Upload LoRA adapter under runs/<timestamp>/lora-adapter/
api.upload_folder(
    repo_id=repo_id,
    folder_path="finetune/output/lora-adapter",
    path_in_repo=f"{run_prefix}/lora-adapter",
    commit_message=f"[{run_id}] Upload LoRA adapter",
)

# Save training params as metadata
params = {
    "run_id": run_id,
    "epochs": os.environ.get("EPOCHS", "3"),
    "lr": os.environ.get("LR", "2e-4"),
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
