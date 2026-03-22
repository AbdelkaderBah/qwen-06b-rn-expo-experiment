# TODOs

## Phase 0 — Setup
- [x] Créer la structure de dossiers du repo
- [x] Initialiser l'environnement Python (uv)
- [x] Installer dépendances : langchain, rank-bm25, ragas, datasets, marimo
- [x] Vérifier que LM Studio tourne avec Qwen 0.6B

## Phase 1 — Corpus
- [x] Fetch docs via context7 : `react-native`, `expo`, `react-navigation`, `reanimated`
- [x] Télécharger les docs complètes (llms.txt + git sparse-checkout)
- [x] Nettoyer le corpus et chunker (7047 chunks → data/processed/*.jsonl)
- [ ] Filtrer sur RN 0.82+ / Expo SDK récent (optionnel)

## Phase 2 — Dataset fine-tuning
- [x] Script de génération via Copilot SDK (gpt-5-mini, streaming)
- [x] Validation automatique via `tsc --noEmit` (eval/ts-checker)
- [ ] Générer ~500 paires (en cours)
- [ ] Valider avec tsc → garder ~300+ paires propres
- [ ] Valider manuellement ~50 exemples
- [ ] Convertir au format Unsloth (JSONL)
- [ ] (Optionnel) Ajouter LLM review gpt-5.4 comme Layer 2

## Phase 3 — Baseline
- [ ] Créer RN-Expo-Bench (20-50 questions)
- [ ] Mesurer Qwen 0.6B base → score de référence

## Phase 4 — Fine-tuning
- [x] Script Unsloth + LoRA (finetune/train.py)
- [ ] Entraîner le modèle (RTX 3080, CUDA requis)
- [ ] Exporter en `.gguf` (--export-gguf)
- [ ] Charger dans LM Studio
- [ ] Remesurer sur RN-Expo-Bench → delta vs baseline

## Phase 5 — RAG
- [ ] Pipeline BM25 (`rank_bm25`) sur le corpus
- [ ] Embeddings `nomic-embed-text` via LM Studio
- [ ] Orchestration LangChain
- [ ] Mesurer RAGAS (Faithfulness + Context Precision)

## Phase 6 — Évaluation finale
- [ ] HumanEval JS — pass@1
- [ ] Tableau comparatif : base → fine-tuné → fine-tuné+RAG
- [ ] Graphiques pour Twitter
