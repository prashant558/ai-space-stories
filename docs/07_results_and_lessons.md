# 07 · Results & lessons

## Everything in one table
| Stage | Model | Data | Compute | Key result |
|---|---|---|---|---|
| Karpathy | 0.2M, chars | Shakespeare 1M chars | minutes | val 1.82 |
| v1 | 13.8M | TinyStories 300M tokens | 56 min, 1× T4 | val **1.50**, coherent stories |
| v2 pretrain | 100.7M | 1.5B tokens, 5 sources | ~12 h, 2× T4 | avg val **2.79** (step 2,500), fluent text |
| Q&A SFT | same | 188k conversations | 75 min, 1× T4 | SQuAD **32% EM / 40% F1**, facts from memory unreliable |
| Lore SFT | same | 19.5k distilled pieces | 1.5 h teacher + 19 min | **93%** pass all checks (teacher 100%) |

## What the student still gets wrong
All three examples come from pieces that **passed every automatic check** (see [`assets/lore_samples.md`](../assets/lore_samples.md)):

| Problem | Example |
|---|---|
| Contradiction | Oris: the nightside "glows with eerie green hues", then "remains shrouded in perpetual twilight" |
| Counting | a 2-planet system describes "the first world", then the same planet as a gas giant, then "the fourth planet" |
| Role confusion | a faction "known as the Eskra Minor" (that's its *homeworld*), and a nomad fleet that "does not have a fleet" |

These are **logic** errors, not format errors, and my checks only test format (names, length, links). A 100M model tracks *style* well but loses track of *what it already said*. More epochs won't fix that; more parameters will.

## Lessons
1. **Size sets the ceiling, data sets the floor.** 100M parameters is enough for fluent, on-format text. It's not enough for reliable facts or long-range consistency.
2. **Fine-tuning teaches behaviour, not knowledge.** Q&A got the format right overnight; the facts never came.
3. **Narrow tasks are where small models win.** On *one* well-defined job, a 100M student got within a few points of a 7B teacher, and runs on a laptop GPU.
4. **Automatic checks are worth it, but check the checks.** My ship-log filter silently threw away 85% of the good data.
5. **Write pre-flight checks and smoke tests.** Every notebook fails in seconds with a readable message instead of after 3 hours. That saved more GPU quota than any speed trick.
6. **Free GPUs are enough to learn the whole pipeline**: tokenizer → pretraining → SFT → distillation → app. Total cost: €0.

## What I'd try next
- **Best-of-3 in Lore Forge:** generate three, keep one that passes the checks. Cheap, and should push 93% toward 100%.
- **A consistency check** (e.g. count "planet" mentions vs `planet_count`) to filter or rerank outputs.
- **More pretraining tokens.** The loss curve hadn't flattened at 1.5B.
- **A 300M model.** It fits a T4 with the same code (change `model_v2.py` config), about 3× the time.
