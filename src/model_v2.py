"""model_v2.py — a ~100M GPT with modern (Llama-style) parts.

Shared by the pretraining notebook, the fine-tuning notebook and chat_app.py.
To try another architecture, only this file changes; train.py and the notebooks stay the same.

Parts (each explained in the notebook):
  RMSNorm        normalise by root-mean-square (no mean subtraction, no bias)
  RoPE           rotary position embeddings
  GQA attention  12 query heads share 4 key/value heads -> 3x smaller KV-cache
  QK-norm        RMSNorm on queries and keys -> attention scores can't blow up in fp16
  SwiGLU MLP     gated feed-forward: down( silu(gate(x)) * up(x) )
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    d_model: int = 768
    ffn_hidden: int = 2048
    block_size: int = 1024          # training context length
    max_pos: int = 4096             # RoPE table length (lets benchmarks go past block_size)
    rope_base: float = 10000.0
    qk_norm: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    def validate(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        assert self.max_pos >= self.block_size
        return self

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def build_rope_cache(max_pos, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_pos).float(), inv_freq)        # (max_pos, hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2) — rotate each (even, odd) pair of dims
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.n_kv, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)
        self.o.SCALE_INIT = True
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.dropout = cfg.dropout

    def forward(self, x, rope, kv_cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)              # (B, H, T, hd)
        kv = self.kv(x).view(B, T, 2, self.n_kv, self.hd)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)             # (B, KV, T, hd)
        q, k = self.q_norm(q), self.k_norm(k)

        past = 0 if (kv_cache is None or kv_cache["k"] is None) else kv_cache["k"].size(2)
        assert past == 0 or T == 1, "with a KV-cache: feed the prompt once, then one token at a time"
        cos, sin = rope
        q = apply_rope(q, cos[past:past + T], sin[past:past + T])
        k = apply_rope(k, cos[past:past + T], sin[past:past + T])
        if kv_cache is not None:                       # the cache stores only the 4 KV heads
            if past:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        rep = self.n_head // self.n_kv                 # each KV head serves `rep` query heads
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(past == 0 and T > 1),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.down.SCALE_INIT = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model, cfg.norm_eps), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model, cfg.norm_eps), SwiGLU(cfg)

    def forward(self, x, rope, kv_cache=None):
        x = x + self.attn(self.norm1(x), rope, kv_cache)     # communication
        x = x + self.mlp(self.norm2(x))                       # computation
        return x


class GPT(nn.Module):
    arch = "transformer_v2"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                  # weight tying
        cos, sin = build_rope_cache(cfg.max_pos, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02 * ((2 * self.cfg.n_layer) ** -0.5 if getattr(m, "SCALE_INIT", False) else 1.0)
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def new_cache(self):
        return [{"k": None, "v": None} for _ in self.blocks]

    def forward(self, idx, targets=None, kv_caches=None, last_only=False, z_loss=0.0):
        B, T = idx.shape
        past = 0 if (kv_caches is None or kv_caches[0]["k"] is None) else kv_caches[0]["k"].size(2)
        assert past + T <= self.cfg.max_pos, f"sequence {past + T} > max_pos {self.cfg.max_pos}"
        x = self.drop(self.tok_emb(idx))
        rope = (self.rope_cos, self.rope_sin)
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope, None if kv_caches is None else kv_caches[i])
        x = self.norm_f(x)
        if targets is None:
            return self.lm_head(x[:, -1:, :] if last_only else x), None
        logits = self.lm_head(x).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        if z_loss > 0:   # keeps logits small -> stable fp16 training (PaLM trick)
            mask = (targets.view(-1) != -100)
            lse = torch.logsumexp(logits.view(-1, logits.size(-1)), dim=-1)
            loss = loss + z_loss * (lse[mask] ** 2).mean()
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, betas=(0.9, 0.95), fused=False):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=fused)

    # ---------------------------------------------------------------- generation
    @staticmethod
    def sample_next(logits, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, history=None):
        logits = logits.float()
        if repetition_penalty != 1.0 and history is not None and history.numel():
            prev = logits.gather(1, history)
            logits = logits.scatter(1, history, torch.where(prev > 0, prev / repetition_penalty,
                                                            prev * repetition_penalty))
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cut = (sp.cumsum(-1) - sp) > top_p
            sp = sp.masked_fill(cut, 0.0)
            probs = torch.zeros_like(probs).scatter(1, si, sp)
            probs = probs / probs.sum(-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=None,
                        repetition_penalty=1.0, stop_ids=()):
        """Yields one new token id at a time (batch size 1), using the KV-cache."""
        assert idx.size(0) == 1, "generate_stream is for a single sequence"
        max_new_tokens = min(max_new_tokens, self.cfg.max_pos - idx.size(1))
        caches = self.new_cache()
        logits, _ = self(idx, kv_caches=caches, last_only=True)          # prefill the prompt
        generated = idx.new_zeros((1, 0))
        for _ in range(max_new_tokens):
            nxt = self.sample_next(logits[:, -1, :], temperature, top_k, top_p, repetition_penalty,
                                   generated[:, -64:])
            tid = int(nxt)
            if tid in stop_ids:
                return
            yield tid
            generated = torch.cat([generated, nxt], dim=1)
            logits, _ = self(nxt, kv_caches=caches, last_only=True)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=None,
                 repetition_penalty=1.0, stop_ids=(), use_cache=True):
        """Batched generation. Returns idx with new tokens appended (stops early only if all rows hit a stop id)."""
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.max_pos:], last_only=True)
                idx = torch.cat([idx, self.sample_next(logits[:, -1, :], temperature, top_k, top_p)], dim=1)
            return idx
        max_new_tokens = min(max_new_tokens, self.cfg.max_pos - idx.size(1))
        caches = self.new_cache()
        logits, _ = self(idx, kv_caches=caches, last_only=True)
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        stop = torch.tensor(list(stop_ids) or [-1], device=idx.device)
        for _ in range(max_new_tokens):
            nxt = self.sample_next(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, nxt], dim=1)
            done |= torch.isin(nxt.squeeze(1), stop)
            if done.all():
                break
            logits, _ = self(nxt, kv_caches=caches, last_only=True)
        return idx


def build_model(cfg_dict):
    return GPT(ModelConfig(**cfg_dict))


def load_exported(path, device="cpu", dtype=None):
    """Load a file written by the fine-tuning notebook's export step."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
        model.rope_cos, model.rope_sin = model.rope_cos.float(), model.rope_sin.float()   # keep RoPE precise
    return model.eval(), ck
