"""
Phase 4 — Fine-tuning Qwen 0.6B with Unsloth + LoRA
Requires NVIDIA GPU (CUDA). Run on RunPod.

Usage:
  python finetune/train.py
  python finetune/train.py --epochs 3 --lr 1e-4
  python finetune/train.py --export-gguf    # export to .gguf after training
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# --- Paths ---
DATASET_PATH = Path("data/dataset/rn_expo_dataset.jsonl")
OUTPUT_DIR = Path("finetune/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model config ---
BASE_MODEL = "unsloth/Qwen3-0.6B"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

SYSTEM_MESSAGE = "You are a React Native 0.82 and Expo expert. Answer with ONLY complete, runnable TypeScript/JSX code. No explanations, no markdown fences. Use functional components and hooks only."


def load_dataset(eval_split: float = 0.15, seed: int = 42) -> tuple[Dataset, Dataset]:
    pairs = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": entry["instruction"]},
                {"role": "assistant", "content": entry["output"]},
            ]
            pairs.append({"messages": messages})
    ds = Dataset.from_list(pairs).shuffle(seed=seed)
    split = ds.train_test_split(test_size=eval_split, seed=seed)
    print(f"[data] Loaded {len(pairs)} examples → {len(split['train'])} train / {len(split['test'])} eval")
    return split["train"], split["test"]


def train(epochs: int, lr: float, batch_size: int, export_gguf: bool) -> None:
    # Load model + tokenizer
    print(f"[model] Loading {BASE_MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    train_dataset, eval_dataset = load_dataset()

    # Training config
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=25,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        assistant_only_loss=True,
        report_to="none",
    )

    # Train
    print(f"\n[train] Training: {epochs} epochs, lr={lr}, batch={batch_size}")
    print(f"   LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"   Dataset: {len(train_dataset)} train / {len(eval_dataset)} eval\n")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    # Save LoRA adapter
    lora_path = OUTPUT_DIR / "lora-adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"\n[save] LoRA adapter saved -> {lora_path}")

    # Export to GGUF
    if export_gguf:
        print("\n[export] Exporting to GGUF (Q4_K_M)...")
        gguf_path = OUTPUT_DIR / "gguf"
        try:
            model.save_pretrained_gguf(
                str(gguf_path),
                tokenizer,
                quantization_method="q4_k_m",
            )
            print(f"[save] GGUF exported -> {gguf_path}")
        except RuntimeError:
            print("[warn] Unsloth GGUF export failed (llama.cpp build issue)")
            print("[export] Saving merged 16-bit model instead...")
            merged_path = OUTPUT_DIR / "merged"
            model.save_pretrained_merged(str(merged_path), tokenizer)
            print(f"[save] Merged model saved -> {merged_path}")
            print("[hint] Convert locally: llama-quantize model.gguf output.gguf Q4_K_M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--export-gguf", action="store_true")
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch_size, args.export_gguf)
