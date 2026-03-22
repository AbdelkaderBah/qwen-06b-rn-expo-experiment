# Qwen 0.6B — React Native & Expo Expert (expertise focalisée)

Fine-tuning a tiny LLM (0.6B) into a niche coding expert for **React Native 0.82 / Expo**, combining LoRA fine-tuning with a hybrid RAG system.

## Goal

Prove that a small, local model can compete with large general-purpose models on a specific niche through:

- **Fine-tuning** (Unsloth + LoRA) on a curated RN/Expo dataset
- **Hybrid RAG** (BM25 + embeddings) backed by official documentation

## Architecture

```
User Query
    │
    ▼
RAG (BM25 + nomic-embed-text)
    │ retrieves relevant doc chunks
    ▼
Prompt = [Context from docs] + [User query]
    │
    ▼
Qwen 0.6B (fine-tuned, via LM Studio .gguf)
    │
    ▼
Clean, modern RN 0.82 code
```

## Stack

| Role                 | Tool                                        |
| -------------------- | ------------------------------------------- |
| Fine-tuning          | Unsloth (LoRA / QLoRA)                      |
| RAG orchestration    | LangChain                                   |
| Retrieval            | BM25 (`rank_bm25`) + `nomic-embed-text`     |
| Local inference      | LM Studio (.gguf)                           |
| Evaluation           | RAGAS, HumanEval (JS), custom RN-Expo-Bench |
| Documentation source | context7                                    |

## Evaluation Strategy

| Metric                                      | What it measures                          |
| ------------------------------------------- | ----------------------------------------- |
| **HumanEval (JS)** — `pass@1`               | General coding ability (regression check) |
| **RAGAS** — Faithfulness, Context Precision | RAG quality (no hallucination)            |
| **RN-Expo-Bench** — 50 questions            | Niche performance vs base model           |

## Project Structure

```
├── data/
│   ├── raw/          # context7 docs
│   ├── processed/    # cleaned chunks
│   └── dataset/      # Q/Code pairs for fine-tuning
├── rag/              # retrieval pipeline
├── finetune/         # Unsloth training scripts
├── eval/             # benchmark scripts + results
├── notebooks/        # exploration & analysis
├── EXPERIMENTS.md    # runs log: params → results
└── TODOS.md
```

## Hardware

Apple M3 Pro — 18GB unified memory
