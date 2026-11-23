# Glossary

| Term | Plain meaning |
|---|---|
| **Token** | A chunk of text (word, piece of a word, or symbol) mapped to an integer. |
| **BPE** | Byte-Pair Encoding: build a vocabulary by repeatedly merging the most frequent pair of tokens. |
| **Vocabulary (vocab)** | The list of all tokens the model knows. Ours: 32,768. |
| **Embedding** | A learned vector (768 numbers) for each token. |
| **Context / block size** | How many tokens the model sees at once. Ours: 1,024. |
| **Attention** | Each token takes a weighted average of earlier tokens' information; the weights come from query·key scores. |
| **Head** | One independent attention computation. We have 12 query heads. |
| **Causal mask** | Blocks attention to future tokens. |
| **Residual connection** | `x = x + layer(x)`: each layer adds a correction instead of replacing the signal. |
| **MLP / feed-forward** | The per-token "thinking" part of each block. |
| **Logits** | Raw scores for every vocabulary token, before softmax. |
| **Softmax** | Turns scores into probabilities that sum to 1. |
| **Cross-entropy loss** | −log(probability of the correct token). Lower is better; random = ln(vocab). |
| **Perplexity** | e^loss: roughly "how many tokens the model is choosing between". |
| **Validation (val) loss** | Loss on held-out text; the honest measure of learning. |
| **Overfitting** | Train loss falls but val loss rises: the model memorises instead of generalising. |
| **Learning rate (LR)** | Step size of each update. |
| **Warm-up** | Start with a tiny LR and ramp up, so early updates don't wreck random weights. |
| **WSD / cosine** | Learning-rate schedules: warm-up-stable-decay vs smooth cosine curve. |
| **AdamW** | The standard optimizer; "W" = weight decay done properly. |
| **Weight decay** | Gently pulls weights toward 0, a form of regularisation. |
| **Gradient clipping** | Caps how big one update can be. |
| **Gradient accumulation** | Add gradients from several small batches before one update = one big batch. |
| **Mixed precision (fp16 / bf16)** | 16-bit maths for speed. fp16 can overflow above 65,504; bf16 can't, but needs newer GPUs. |
| **GradScaler** | Scales the loss up in fp16 so tiny gradients don't round to zero. |
| **DDP** | DistributedDataParallel: one model copy per GPU, gradients averaged. |
| **Checkpoint** | Saved model + optimizer state to resume training. |
| **KV-cache** | Stored keys/values of past tokens, so generation doesn't recompute them. |
| **SDPA / fused attention** | PyTorch's fast attention kernel that never stores the full T×T matrix. |
| **RMSNorm / LayerNorm** | Rescale each vector to a standard size (RMSNorm skips the mean). |
| **RoPE** | Rotary position embedding: encode position by rotating queries and keys. |
| **SwiGLU** | Gated MLP: `down(silu(gate(x)) * up(x))`. |
| **GQA** | Grouped-query attention: several query heads share one key/value head. |
| **QK-norm** | Normalise queries and keys before attention, for stability. |
| **z-loss** | Small penalty that keeps logits from growing huge. |
| **Weight tying** | Use the same matrix for the input embedding and the output layer. |
| **Base model** | Pretrained only; continues text. |
| **SFT** | Supervised fine-tuning on example conversations. |
| **Loss masking** | Label = −100 for tokens that shouldn't be graded (the user's turn). |
| **Distillation** | A big *teacher* model generates training data for a small *student*. |
| **vLLM** | A fast inference engine that batches many requests on the GPU. |
| **Tensor parallel** | Split one model's layers across several GPUs. |
| **Temperature / top-k / top-p** | Sampling knobs: randomness, and how many candidate tokens are allowed. |
| **Repetition penalty** | Lowers the score of tokens used recently. |
| **SQuAD EM / F1** | Exact-match and word-overlap scores for reading-comprehension answers. |
| **HellaSwag** | Multiple-choice common-sense benchmark (random = 25%). |
| **Chinchilla rule** | ~20 training tokens per parameter is compute-optimal. |
| **Smoke test** | A tiny, fast run that proves the code works end-to-end. |
| **Pre-flight check** | Validate inputs, GPU and files *before* the long job starts. |
