# 00 · Roadmap: how to read this repo

**Who this is for:** you can write Python, you've trained a small neural network (or at least know what a loss and a gradient are), and you remember school maths (multiplying, averages, a square root). That's enough.

## The path

```
Karpathy's video ──► v1: 14M GPT ──► v2: 100M GPT ──► Q&A fine-tune ──► Space lore (distillation)
  (attention)       (make it real)   (make it modern)   (half works)      (works!)
```

| # | Read | Then open | Time to read |
|---|---|---|---|
| 0 | [Karpathy: *Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (video, 2 h) | his code: [`ng-video-lecture`](https://github.com/karpathy/ng-video-lecture) | an evening |
| 1 | [01 Tokenizer](01_tokenizer.md) | `notebooks/01`, section 1 | 15 min |
| 2 | [02 From Karpathy's GPT](02_from_karpathy.md) | `notebooks/01` | 30 min |
| 3 | [03 Modern upgrades](03_modern_upgrades.md) | `src/model_v2.py` | 45 min |
| 4 | [04 Pretraining on free GPUs](04_pretraining_free_gpus.md) | `notebooks/02`, `03`, `src/train.py` | 30 min |
| 5 | [05 Fine-tuning](05_finetuning.md) | `notebooks/04` | 20 min |
| 6 | [06 Distillation](06_distillation.md) | `notebooks/05`, `06`, `src/lore_schema.py` | 30 min |
| 7 | [07 Results & lessons](07_results_and_lessons.md) | `assets/` | 15 min |

Stuck on a word? → [glossary](glossary.md).

## Three ways to use this repo

1. **Just read.** Every notebook is saved *with its outputs* (logs, plots, samples), so GitHub shows exactly what happened, without running anything.
2. **Play.** Download the trained model from *Releases* and run `apps/lore_forge.py` on your own PC.
3. **Rebuild.** Run the notebooks on Kaggle in order. Every notebook has a *How to run* cell, pre-flight checks that fail fast with a clear message, and a `SMOKE_TEST=1` mode that runs a tiny version in minutes.

*History note:* the project started as "compare a Transformer, Mamba and Jamba", so some notebook text still mentions them. I dropped that comparison: at 100M parameters on free GPUs, it would have meant a lot of work for small, noisy differences.

## Compute used (all free)

| Step | Hardware | Time |
|---|---|---|
| v1 (notebook 01) | Kaggle, 1× T4 | ~1 h |
| Data prep (02) | Kaggle CPU | a few hours (no GPU quota) |
| Pretraining (03) | Kaggle, 2× T4 | ~12 h, over 2 sessions |
| Q&A fine-tune (04) | Kaggle, 1× T4 | 75 min |
| Lore data (05) | Kaggle, 2× T4 | ~1.5 h |
| Lore fine-tune (06) | Kaggle, 1× T4 | 20 min |
| Running the apps | GTX 1650 laptop (4 GB) | seconds per piece |

Kaggle gives ~30 GPU-hours per week, so the whole project fits into one week's quota.
