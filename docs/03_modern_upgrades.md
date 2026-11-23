# 03 · Modern upgrades: from GPT-2 parts to Llama parts

The skeleton doesn't change: *embedding → 12 blocks of (attention + MLP) with residual connections → norm → output layer.* We swap five parts for the versions used in Llama, Qwen and Mistral. All of them live in [`src/model_v2.py`](../src/model_v2.py) (~250 lines).

## Where the 100.7M parameters are

| Part | Shape | Parameters |
|---|---|---|
| Token embedding (shared with output) | 32,768 × 768 | 25.2M |
| Per block: attention (q, kv, o) | 768→768, 768→512, 768→768 | 1.57M |
| Per block: SwiGLU MLP (gate, up, down) | 3 × 768 × 2,048 | 4.72M |
| **× 12 blocks** | | 75.5M |
| **Total** | | **100.7M** |

---

## 1. RMSNorm (replaces LayerNorm)
**Idea:** keep each token's vector at a sensible size so numbers don't drift as they pass through 12 layers.

LayerNorm subtracts the mean, then divides by the standard deviation. RMSNorm skips the mean and only divides by the *root mean square*:

```
x = [3, 4]
LayerNorm: mean 3.5, std 0.5        → [-1.0, 1.0]
RMSNorm:   rms = √((9+16)/2) = 3.54 → [0.85, 1.13]
```

One statistic fewer, no bias, and in practice the same quality.

```python
xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
return xf.type_as(x) * self.weight
```
We compute it in float32 even during fp16 training, because squaring big fp16 numbers overflows.

## 2. RoPE: rotary position embeddings (replaces learned positions)
**Idea:** attention itself has no sense of order ("dog bites man" = "man bites dog"). Karpathy *adds* a learned position vector. RoPE instead **rotates** each query and key by an angle proportional to its position.

Take one pair of numbers as a 2-D arrow. A token at position `p` gets rotated by `p × θ`:

```
arrow (1, 0) at position 1, θ = 1 rad  →  (cos 1, sin 1) = (0.54, 0.84)
```

Why that's clever: the dot product of two rotated arrows depends only on the **difference** of their angles. So the attention score between position 7 and position 5 is the same as between 102 and 100. The model learns *relative* distance for free, and nothing is learned or stored per position.

Each of the 32 pairs in a 64-dim head uses a different speed θ (fast pairs for nearby words, slow pairs for far ones).

```python
x1, x2 = x[..., 0::2], x[..., 1::2]                 # split into pairs
out = torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), -1).flatten(-2)   # 2-D rotation
```

## 3. SwiGLU MLP (replaces GELU MLP)
**Idea:** add a "dimmer switch" for every hidden neuron.

```
GPT-2:   down( gelu(up(x)) )
SwiGLU:  down( silu(gate(x)) * up(x) )        silu(z) = z · sigmoid(z)
```
`silu(2) = 1.76`, `silu(0) = 0`, `silu(-2) = -0.24`: the gate mostly passes or blocks, and `up(x)` carries the content. Three matrices instead of two, so the hidden size is 2,048 (≈ ⅔ × 4 × 768) to keep the parameter count equal. Empirically it gives slightly better loss for the same size.

## 4. GQA: grouped-query attention
**Idea:** 12 query heads, but only **4** key/value heads. Each K/V head is shared by 3 query heads.

Why: when generating, the model stores every past key and value (the **KV-cache**). For 1,024 tokens in fp16:

```
2 (K and V) × 12 layers × heads × 64 dims × 1,024 tokens × 2 bytes
  12 heads (normal):  37.7 MB
   4 heads (GQA):     12.6 MB   → 3× smaller, almost no quality loss
```
That's what makes long generations comfortable on a 4 GB GTX 1650.

```python
rep = self.n_head // self.n_kv                      # 3
k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
```

## 5. QK-norm + z-loss: stability on old GPUs
Free Kaggle T4s don't support bfloat16, so we train in **fp16**, whose largest number is **65,504**. If attention scores or output logits grow large, they overflow → `inf` → `NaN` → training is dead.

- **QK-norm:** apply RMSNorm to queries and keys before the dot product, so scores can't explode.
- **z-loss:** a tiny extra penalty `1e-4 × (log Σ e^logits)²` that keeps output logits small.

Result: the whole fp16 run trained without a single NaN, and the gradient norm settled around 0.17.

## Also kept from v1
Pre-norm residual blocks, weight tying, fused attention (SDPA), scaled init for the output projections (`std = 0.02 / √(2 × layers)`, so 12 residual additions don't blow up the signal).

## The tests that saved me
`notebooks/03`, section 4, runs before any training:
1. Output shapes are right.
2. **Causality:** changing token 32 doesn't change predictions for tokens 0–31.
3. **KV-cache = full forward:** generating one token at a time with the cache gives the same numbers (max difference < 0.001).
4. **Initial loss ≈ ln(32,768) = 10.4.**
5. **Overfit one batch:** loss must drop by 70%+ in 100 steps. If a model can't memorise 2 sequences, something is broken.
