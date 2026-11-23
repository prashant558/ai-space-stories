"""train.py — pretraining with DDP, a mixture of data sources, WSD schedule, checkpoint/resume.

Run (2 GPUs):   torchrun (or python -m torch.distributed.run) --nproc_per_node=2 train.py --config run_config.json
Speed test:     torchrun --nproc_per_node=2 train.py --config run_config.json --speed_test 30
Single process: python train.py --config run_config.json
"""
import argparse, contextlib, csv, glob, json, math, os, time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model_v2 import build_model


# ------------------------------------------------------------------ data
class MixtureLoader:
    """Random windows from several sources; each sequence picks its source by the mixing weight."""

    def __init__(self, data_dir, manifest, split, seed):
        self.names = list(manifest["sources"])
        w = np.array([manifest["sources"][n]["weight"] for n in self.names], dtype=np.float64)
        self.weights = w / w.sum()
        self.arrays = {}
        for n in self.names:
            s = manifest["sources"][n]
            files = s["train_shards"] if split == "train" else [s["val"]]
            arrs = [np.memmap(os.path.join(data_dir, s["dir"], f), dtype=np.uint16, mode="r") for f in files]
            lens = np.array([len(a) for a in arrs], dtype=np.float64)
            self.arrays[n] = (arrs, lens / lens.sum())
        self.rng = np.random.default_rng(seed)

    def _window(self, name, T):
        arrs, p = self.arrays[name]
        a = arrs[self.rng.choice(len(arrs), p=p)]
        i = int(self.rng.integers(0, len(a) - T - 1))
        w = torch.from_numpy(a[i:i + T + 1].astype(np.int64))
        return w[:-1], w[1:]

    def batch(self, B, T, device, source=None):
        names = [source] * B if source else self.rng.choice(self.names, size=B, p=self.weights)
        xs, ys = zip(*(self._window(n, T) for n in names))
        x, y = torch.stack(xs), torch.stack(ys)
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y


# ------------------------------------------------------------------ helpers
def lr_at(step, C):
    """Warmup -> Stable -> Decay (WSD). The decay phase is the last `decay_frac` of training."""
    if step < C["warmup_steps"]:
        return C["peak_lr"] * (step + 1) / C["warmup_steps"]
    decay_start = int(C["max_steps"] * (1 - C["decay_frac"]))
    if step < decay_start:
        return C["peak_lr"]
    frac = (step - decay_start) / max(1, C["max_steps"] - decay_start)
    return C["min_lr"] + (C["peak_lr"] - C["min_lr"]) * (1 - frac)


def find_checkpoint(C, rank):
    cands = [os.path.join(C["out_dir"], "ckpt_latest.pt")]
    cands += glob.glob("/kaggle/input/**/ckpt_latest.pt", recursive=True)
    best = None
    for c in cands:
        if not os.path.exists(c):
            continue
        meta_path = c.replace("ckpt_latest.pt", "ckpt_meta.json")
        if os.path.exists(meta_path):            # cheap check without loading 1 GB
            meta = json.load(open(meta_path))
            if meta.get("run_name") != C["run_name"] or meta.get("model") != C["model"]:
                continue
            if best is None or meta["step"] > best[1]:
                best = (c, meta["step"])
    if best and rank == 0:
        print(f"[resume] found {best[0]} at step {best[1]}")
    return best[0] if best else None


def save_checkpoint(C, raw_model, optimizer, scaler, step, tokens, train_time):
    out = C["out_dir"]
    latest, prev = os.path.join(out, "ckpt_latest.pt"), os.path.join(out, "ckpt_prev.pt")
    state = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "step": step, "tokens": tokens, "train_time": train_time,
             "run_name": C["run_name"], "model_config": C["model"]}
    tmp = latest + ".tmp"
    torch.save(state, tmp)
    if os.path.exists(latest):
        os.replace(latest, prev)                  # keep one older copy in case the newest is bad
    os.replace(tmp, latest)
    json.dump({"run_name": C["run_name"], "model": C["model"], "step": step, "tokens": tokens},
              open(os.path.join(out, "ckpt_meta.json"), "w"))


def unwrap(m):
    m = m.module if isinstance(m, DDP) else m
    return getattr(m, "_orig_mod", m)             # undo torch.compile


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--speed_test", type=int, default=0, help="run N steps, report speed, save nothing")
    args = ap.parse_args()
    C = json.load(open(args.config))

    ddp = "RANK" in os.environ
    cuda = torch.cuda.is_available()
    if ddp:
        dist.init_process_group("nccl" if cuda else "gloo")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    master = rank == 0
    device = f"cuda:{local}" if cuda else "cpu"
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = None
    if cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext

    # ---- derived sizes
    T, B = C["model"]["block_size"], C["micro_batch"]
    per_step = B * T * world
    assert C["tokens_per_step"] % per_step == 0, \
        f"tokens_per_step {C['tokens_per_step']} must be a multiple of micro_batch*block*world = {per_step}"
    accum = C["tokens_per_step"] // per_step
    C["max_steps"] = C["max_tokens"] // C["tokens_per_step"]
    os.makedirs(C["out_dir"], exist_ok=True)

    manifest = json.load(open(os.path.join(C["data_dir"], "manifest.json")))
    assert manifest["vocab_size"] == C["model"]["vocab_size"], "vocab mismatch between data and model"
    train_loader = MixtureLoader(C["data_dir"], manifest, "train", C["seed"] + 1000 * rank)
    val_loader = MixtureLoader(C["data_dir"], manifest, "val", C["seed"] + 7 + 1000 * rank)

    torch.manual_seed(C["seed"])                   # same init on every rank
    model = build_model(C["model"]).to(device)
    optimizer = unwrap(model).configure_optimizer(C["peak_lr"], C["weight_decay"], fused=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    step, tokens, train_time = 0, 0, 0.0
    ck_path = None if args.speed_test else find_checkpoint(C, rank)
    if ck_path:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, tokens, train_time = ck["step"], ck["tokens"], ck["train_time"]
        del ck
    if C.get("compile") and cuda:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local] if cuda else None)
    raw = unwrap(model)

    if master:
        print(f"[setup] world={world} device={device} amp={amp_dtype} params={raw.num_params()/1e6:.2f}M | "
              f"micro_batch={B} x {T} x accum {accum} x {world} GPUs = {C['tokens_per_step']:,} tok/step | "
              f"steps {step}->{C['max_steps']:,}", flush=True)

    log_path = os.path.join(C["out_dir"], "log.csv")
    val_cols = [f"val_{n}" for n in manifest["sources"]]
    if master and not args.speed_test and not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens", "lr", "train_loss", *val_cols, "val_avg", "tok_per_s", "train_time_s"])

    @torch.no_grad()
    def evaluate():
        raw.eval()
        res = {}
        for n in manifest["sources"]:
            tot = torch.zeros(1, device=device)
            for _ in range(C["eval_batches"]):
                x, y = val_loader.batch(B, T, device, source=n)
                with autocast():
                    _, loss = raw(x, y)
                tot += loss.float()
            if ddp:
                dist.all_reduce(tot)
            res[n] = (tot / (C["eval_batches"] * world)).item()
        raw.train()
        return res

    session_start, last_ckpt = time.time(), time.time()
    t_log, tok_log, last_tps, last_loss = time.time(), 0, 0.0, float("nan")
    speed_t0, speed_tokens = None, 0
    model.train()
    while True:
        is_last = step >= C["max_steps"] or (args.speed_test and step >= args.speed_test)
        if not args.speed_test and (step % C["eval_interval"] == 0 or is_last):
            val = evaluate()
            if master:
                avg = sum(val.values()) / len(val)
                print(f"[eval] step {step:,} | " + " | ".join(f"{k} {v:.3f}" for k, v in val.items())
                      + f" | avg {avg:.3f}", flush=True)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, tokens, f"{lr_at(step, C):.2e}", f"{last_loss:.4f}",
                                            *[f"{val[n]:.4f}" for n in manifest["sources"]], f"{avg:.4f}",
                                            f"{last_tps:.0f}", f"{train_time:.0f}"])
            t_log = time.time()
        if is_last:
            break

        # ---- stop flag (time limit / periodic checkpoint) decided on rank 0, shared with all
        flags = torch.tensor([0, 0], device=device)
        if master and not args.speed_test:
            if time.time() - session_start > C["time_limit_hours"] * 3600:
                flags[0] = 1
            if time.time() - last_ckpt > C["ckpt_minutes"] * 60:
                flags[1] = 1
        if ddp:
            dist.broadcast(flags, src=0)
        if flags[1] or flags[0]:
            if master:
                save_checkpoint(C, raw, optimizer, scaler, step, tokens, train_time)
                print(f"[ckpt] saved at step {step:,}", flush=True)
            last_ckpt = time.time()
            if ddp:
                dist.barrier()
        if flags[0]:
            if master:
                print("[time] session time limit reached — re-run to resume", flush=True)
            break

        # ---- one optimizer step
        t0 = time.time()
        lr = lr_at(step, C)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for micro in range(accum):
            x, y = train_loader.batch(B, T, device)
            sync = (not ddp) or micro == accum - 1
            with (contextlib.nullcontext() if sync else model.no_sync()):
                with autocast():
                    _, loss = model(x, y, z_loss=C["z_loss"])
                scaler.scale(loss / accum).backward()
            loss_acc += loss.item() / accum
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), C["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0
        step += 1
        tokens += C["tokens_per_step"]
        train_time += dt
        tok_log += C["tokens_per_step"]
        last_loss = loss_acc

        if args.speed_test and step == 3:            # skip compile / warm-up steps
            speed_t0, speed_tokens = time.time(), 0
        elif speed_t0 is not None:
            speed_tokens += C["tokens_per_step"]

        if master and step % C["log_interval"] == 0:
            last_tps = tok_log / (time.time() - t_log)
            eta_h = (C["max_steps"] - step) * C["tokens_per_step"] / max(last_tps, 1) / 3600
            mem = f" | mem {torch.cuda.max_memory_allocated(device)/1e9:.1f}GB" if cuda else ""
            print(f"step {step:>6,}/{C['max_steps']:,} | loss {loss_acc:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | "
                  f"{last_tps/1e3:.1f}k tok/s | ETA {eta_h:.1f} h{mem}", flush=True)
            t_log, tok_log = time.time(), 0

    if master and args.speed_test and speed_t0 is not None:
        tps = speed_tokens / (time.time() - speed_t0)
        mem = torch.cuda.max_memory_allocated(device) / 1e9 if cuda else 0.0
        hours = C["max_tokens"] / tps / 3600
        print(f"\n[speed test] {tps/1e3:.1f}k tokens/s on {world} device(s) | peak memory {mem:.1f} GB/GPU")
        print(f"[speed test] full run of {C['max_tokens']/1e9:.2f}B tokens ≈ {hours:.1f} h "
              f"≈ {math.ceil(hours / C['time_limit_hours'])} Kaggle session(s)")
        json.dump({"tok_per_s": tps, "peak_mem_gb": mem, "hours_full_run": hours},
                  open(os.path.join(C["out_dir"], "speed_test.json"), "w"))
    elif master and step >= C["max_steps"]:
        save_checkpoint(C, raw, optimizer, scaler, step, tokens, train_time)
        torch.save({"model": raw.state_dict(), "config": C["model"], "step": step, "tokens": tokens},
                   os.path.join(C["out_dir"], "final.pt"))
        print(f"[done] {tokens/1e9:.2f}B tokens in {train_time/3600:.2f} h — saved final.pt", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
