#!/usr/bin/env bash
set -euo pipefail

# ─── RunPod one-click setup & train ───
# Set these env vars in RunPod pod config:
#   HF_TOKEN   — HuggingFace write token
#   HF_REPO    — e.g. "your-org/your-model"

REPO_URL="${GIT_REPO_URL:?Set GIT_REPO_URL in RunPod pod config}"
REPO_NAME="$(basename "$REPO_URL" .git)"
WORK_DIR="/workspace/$REPO_NAME"

# Clone or pull latest
if [ -d "$WORK_DIR/.git" ]; then
  echo "[setup] Pulling latest changes..."
  git -C "$WORK_DIR" pull --ff-only
else
  echo "[setup] Cloning repo..."
  git clone "$REPO_URL" "$WORK_DIR"
fi

cd "$WORK_DIR"

# Install deps (uv is faster than pip)
pip install -q uv 2>/dev/null || true
uv sync --extra train

# Train
exec bash scripts/runpod_train.sh
