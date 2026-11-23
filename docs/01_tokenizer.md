# 01 · Tokenizer: turning text into numbers

## Plain idea
A neural network only understands numbers, so text must become a list of integers (**tokens**).

- Karpathy uses **characters**: `"hi"` → `[46, 47]`. Simple, but a 1,000-character paragraph becomes 1,000 steps, and the model wastes effort learning to spell.
- We use **BPE (Byte-Pair Encoding)**: frequent chunks become one token. `" planet"` is one token, `" photosynthesis"` maybe three. A paragraph shrinks ~4×, so the model "sees" 4× more text in the same context window.

## Worked example: BPE by hand
Training text (word: how often it appears):

```
low ×3   lower ×1   lowest ×1   new ×1   newer ×1
```

Start with single characters, then repeat: **count every neighbouring pair, merge the most common one into a new token.**

| Round | Most common pair | Count | Words now look like |
|---|---|---|---|
| 0 | — | — | `l o w`, `l o w e r`, `l o w e s t`, `n e w`, `n e w e r` |
| 1 | `l`+`o` | 5 | `lo w`, `lo w e r`, `lo w e s t`, … |
| 2 | `lo`+`w` | 5 | `low`, `low e r`, `low e s t`, … |
| 3 | `n`+`e` | 2 | `low`, `low e r`, …, `ne w`, `ne w e r` |
| … | | | |

After enough rounds, `low` is a single token and `lowest` = `low` + `est`. The model can now guess `lowest` relates to `low` even if it rarely saw `lowest`.

We stop at **32,768 tokens** (`2^15`). Bigger vocab = shorter sequences but a bigger embedding table (32,768 × 768 = **25.2M** of our 100.7M parameters!).

## Our choices
| | Karpathy | v1 | v2 (this repo) |
|---|---|---|---|
| Unit | characters | BPE | BPE (byte-level) |
| Vocab | 65 | 8,192 | 32,768 |
| Trained on | Shakespeare | TinyStories | a sample of *all* 5 data sources + some Q&A |

**Byte-level** means the base alphabet is the 256 possible bytes, so *any* text (emoji, Hindi, code) can be encoded, never "unknown token".

**Why 8k → 32k for v2?** TinyStories uses simple words; Wikipedia doesn't. With 8k tokens, *photosynthesis* splits into many pieces and wastes context.

## Special tokens
Four reserved IDs that never appear in normal text:

| Token | ID | Meaning |
|---|---|---|
| `<\|endoftext\|>` | 0 | document boundary |
| `<\|user\|>` | 1 | a user turn starts |
| `<\|assistant\|>` | 2 | the model's turn starts |
| `<\|end\|>` | 3 | end of a turn: **the model learns to emit this to stop talking** |

## The code
`notebooks/02_data_prep.ipynb`, section 3:

```python
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)   # work on bytes
trainer = trainers.BpeTrainer(vocab_size=32768, special_tokens=SPECIALS, min_frequency=2)
tok.train_from_iterator(sample_for_tokenizer(), trainer=trainer)
```

Then every document is encoded once and saved as `uint16` numbers in `.bin` files (2 bytes per token; that's why the vocab must stay below 65,536). Training later reads random slices of these files with `np.memmap`, without loading 3 GB into RAM.

**Gotcha I hit:** `tok.decode(ids)` silently drops special tokens. Use `skip_special_tokens=False` when you want to *see* them.
