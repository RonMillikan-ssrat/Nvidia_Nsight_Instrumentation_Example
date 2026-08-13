"""
vit_weather_train.py  v3 — Vision Transformer on synthetic weather-field patches.
Multi-node DDP training baseline for NOAA RDHPCS H100 nodes.

New in v3:
  - GPU utilization + memory per step via pynvml (pip install pynvml)
  - Per-rank metrics written to separate JSONL files (rank0.jsonl, rank1.jsonl, ...)
  - Throughput roofline: achieved vs theoretical peak FLOP/s
  - Epoch timing breakdown: data-load / h2d / forward / backward / optim / nccl / idle
  - NCCL communication timing: per-step barrier timing via CUDA events

Launch via torchrun (see run_vit.slurm):
  torchrun --nproc_per_node=2 vit_weather_train.py [args]

NVTX ranges active only when --nsys is passed.
"""

import argparse
import os
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# NVTX
try:
    import torch.cuda.nvtx as nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False

import subprocess


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="ViT weather DDP training baseline v3")
    # Data
    p.add_argument("--img-size",      type=int,   default=128)
    p.add_argument("--in-channels",   type=int,   default=4)
    p.add_argument("--patch-size",    type=int,   default=16)
    p.add_argument("--dataset-size",  type=int,   default=4096)
    # Model
    p.add_argument("--embed-dim",     type=int,   default=512)
    p.add_argument("--depth",         type=int,   default=8)
    p.add_argument("--num-heads",     type=int,   default=8)
    p.add_argument("--mlp-ratio",     type=float, default=4.0)
    p.add_argument("--dropout",       type=float, default=0.1)
    # Training
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--warmup-steps",  type=int,   default=100)
    p.add_argument("--dtype",         choices=["fp32", "bf16"], default="bf16")
    # Infrastructure
    p.add_argument("--log-dir",       type=str,   default="logs")
    p.add_argument("--ckpt-dir",      type=str,   default="checkpoints")
    p.add_argument("--log-interval",  type=int,   default=10)
    p.add_argument("--nsys",          action="store_true")
    p.add_argument("--profile-steps", type=int,   default=20)
    p.add_argument("--run-tag",       type=str,   default="")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# NVTX context manager
# ──────────────────────────────────────────────────────────────────────────────
class nvtx_range:
    _COLOR_MAP = {
        "green":   0xFF00B050, "blue":    0xFF0070C0,
        "red":     0xFFFF0000, "yellow":  0xFFFFFF00,
        "cyan":    0xFF00FFFF, "magenta": 0xFFFF00FF,
        "orange":  0xFFFF8C00, "purple":  0xFF7030A0,
        "white":   0xFFFFFFFF,
    }
    def __init__(self, name, color="blue", active=True):
        self.name   = name
        self.active = active and _NVTX_AVAILABLE
    def __enter__(self):
        if self.active:
            nvtx.range_push(self.name)
        return self
    def __exit__(self, *_):
        if self.active:
            nvtx.range_pop()


# ──────────────────────────────────────────────────────────────────────────────
# pynvml GPU poller
# ──────────────────────────────────────────────────────────────────────────────
class GpuPoller:
    """
    Polls per-GPU utilization and memory via subprocess nvidia-smi query.
    Avoids all pynvml/nvml init conflicts with torchrun's forked processes.

    nvidia-smi is always available on RDHPCS GPU nodes and has no init
    state — each call is fully independent across ranks.

    Returns:
      gpu_util_pct      — SM utilization 0-100
      gpu_mem_used_gb   — driver-reported used memory in GB
      gpu_mem_total_gb  — total physical memory in GB
    """
    _CMD_TEMPLATE = (
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits --id={gpu_id}"
    )

    def __init__(self, local_rank: int):
        self.gpu_id = local_rank
        self.cmd    = self._CMD_TEMPLATE.format(gpu_id=local_rank).split()
        # Verify nvidia-smi is reachable at init time
        try:
            subprocess.run(self.cmd, capture_output=True, check=True, timeout=5)
            self.available = True
        except Exception:
            self.available = False

    def sample(self) -> dict:
        if not self.available:
            return {"gpu_util_pct": None,
                    "gpu_mem_used_gb": None,
                    "gpu_mem_total_gb": None}
        try:
            out = subprocess.run(
                self.cmd, capture_output=True, text=True,
                check=True, timeout=5,
            ).stdout.strip()
            # output: "util, mem_used_MiB, mem_total_MiB"
            util_pct, mem_used_mib, mem_total_mib = [
                float(v.strip()) for v in out.split(",")
            ]
            return {
                "gpu_util_pct":    util_pct,
                "gpu_mem_used_gb": mem_used_mib  / 1024.0,
                "gpu_mem_total_gb":mem_total_mib / 1024.0,
            }
        except Exception:
            return {"gpu_util_pct": None,
                    "gpu_mem_used_gb": None,
                    "gpu_mem_total_gb": None}


# ──────────────────────────────────────────────────────────────────────────────
# Roofline calculator
# ──────────────────────────────────────────────────────────────────────────────
def compute_vit_flops(img_size, patch_size, in_channels, embed_dim,
                      depth, num_heads, mlp_ratio, batch_size) -> int:
    """
    Estimates forward-pass FLOPs for WeatherViT using standard ViT accounting.
    Multiply by 3 for forward + backward (2× backward rule of thumb).

    Returns total FLOPs per batch (int).
    """
    N  = (img_size // patch_size) ** 2   # number of patch tokens
    E  = embed_dim
    H  = num_heads
    Dh = E // H                          # head dim
    M  = int(E * mlp_ratio)             # MLP hidden dim
    B  = batch_size

    flops = 0

    # Patch embedding: Conv2d (in_channels→E, kernel patch×patch)
    flops += B * in_channels * E * (img_size // patch_size) ** 2 * patch_size ** 2 * 2

    # Transformer blocks
    for _ in range(depth):
        # QKV projections: 3 × (B, N, E) × (E, E)
        flops += B * N * E * E * 2 * 3
        # Attention scores: (B, H, N, Dh) × (B, H, Dh, N)
        flops += B * H * N * Dh * N * 2
        # Attention weighted sum: (B, H, N, N) × (B, H, N, Dh)
        flops += B * H * N * N * Dh * 2
        # Output projection: (B, N, E) × (E, E)
        flops += B * N * E * E * 2
        # MLP: two linear layers
        flops += B * N * E * M * 2
        flops += B * N * M * E * 2

    # Decoder head: (B, N, E) × (E, P*P)
    P = patch_size
    flops += B * N * E * (P * P) * 2

    return flops


H100_BF16_TFLOPS = 989.0   # H100 SXM5 tensor core BF16 peak (TFLOP/s)
H100_FP32_TFLOPS = 67.0    # H100 FP32 peak


# ──────────────────────────────────────────────────────────────────────────────
# Step timer (CUDA events, v2 carry-over)
# ──────────────────────────────────────────────────────────────────────────────
class StepTimer:
    """
    CUDA-event-based timing for each named phase.
    v3 adds 'nccl' and 'dataload' phases.
    """
    PHASES = ["dataload", "h2d", "forward", "backward", "nccl", "optim"]

    def __init__(self, device):
        self.device  = device
        self._starts = {p: torch.cuda.Event(enable_timing=True) for p in self.PHASES}
        self._ends   = {p: torch.cuda.Event(enable_timing=True) for p in self.PHASES}
        self._ms     = {p: 0.0 for p in self.PHASES}

    def start(self, phase: str):
        self._starts[phase].record()

    def stop(self, phase: str):
        self._ends[phase].record()

    def sync_and_collect(self):
        torch.cuda.synchronize()
        for p in self.PHASES:
            try:
                self._ms[p] = self._starts[p].elapsed_time(self._ends[p])
            except RuntimeError:
                self._ms[p] = 0.0

    def get_ms(self) -> dict:
        return dict(self._ms)

    def all_reduce_ms(self, world_size: int) -> dict:
        """Returns mean ms per phase across all ranks."""
        t = torch.tensor(
            [self._ms[p] for p in self.PHASES],
            dtype=torch.float32, device=self.device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
        return {p: t[i].item() for i, p in enumerate(self.PHASES)}


# ──────────────────────────────────────────────────────────────────────────────
# Epoch timer — wall-clock breakdown accumulated over all steps
# ──────────────────────────────────────────────────────────────────────────────
class EpochTimer:
    """
    Accumulates per-phase ms totals across all steps in an epoch.
    Reports wall-clock seconds at epoch end.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._totals = {p: 0.0 for p in StepTimer.PHASES}
        self._t0     = time.perf_counter()

    def accumulate(self, step_ms: dict):
        for p in StepTimer.PHASES:
            self._totals[p] += step_ms.get(p, 0.0)

    def summary(self) -> dict:
        """Returns epoch wall time and per-phase seconds + percentages."""
        wall_s   = time.perf_counter() - self._t0
        total_ms = sum(self._totals.values()) + 1e-9
        out = {"epoch_wall_s": wall_s}
        for p, ms in self._totals.items():
            out[f"epoch_s_{p}"]   = ms / 1000.0
            out[f"epoch_pct_{p}"] = 100.0 * ms / total_ms
        # idle = wall time not accounted for by instrumented phases
        accounted_s         = total_ms / 1000.0
        out["epoch_s_idle"] = max(0.0, wall_s - accounted_s)
        out["epoch_pct_idle"] = 100.0 * out["epoch_s_idle"] / (wall_s + 1e-9)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-rank metrics logger
# ──────────────────────────────────────────────────────────────────────────────
class RankLogger:
    """
    Every rank writes its own JSONL file: logs/rank0.jsonl, rank1.jsonl, ...
    Rank 0 additionally writes the shared train_metrics.jsonl (global view).
    """
    def __init__(self, log_dir: str, rank: int):
        self.rank = rank
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # Per-rank file — all ranks write
        rank_path       = Path(log_dir) / f"rank{rank}.jsonl"
        self._rank_fh   = open(rank_path, "w")
        # Global file — rank 0 only
        self._global_fh = None
        if rank == 0:
            self._global_fh = open(Path(log_dir) / "train_metrics.jsonl", "w")
        self._t0 = time.perf_counter()

    def log_rank(self, **kwargs):
        """Write a record to this rank's own JSONL."""
        kwargs["rank"]      = self.rank
        kwargs["wall_time"] = time.perf_counter() - self._t0
        self._rank_fh.write(json.dumps(kwargs) + "\n")
        self._rank_fh.flush()

    def log_global(self, **kwargs):
        """Write a record to the shared JSONL (rank 0 only)."""
        if self._global_fh is None:
            return
        kwargs["wall_time"] = time.perf_counter() - self._t0
        self._global_fh.write(json.dumps(kwargs) + "\n")
        self._global_fh.flush()

    def close(self):
        self._rank_fh.close()
        if self._global_fh:
            self._global_fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic weather dataset (unchanged from v2)
# ──────────────────────────────────────────────────────────────────────────────
class SyntheticWeatherDataset(Dataset):
    def __init__(self, size, img_size, in_channels, seed=42):
        self.size        = size
        self.img_size    = img_size
        self.in_channels = in_channels
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.seeds = torch.randint(0, 2**31, (size,), generator=rng).tolist()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = torch.Generator()
        rng.manual_seed(self.seeds[idx])
        H = W = self.img_size
        K = 8
        x = torch.zeros(self.in_channels, H, W)
        freqs = torch.randint(1, 8, (K, 2), generator=rng).float()
        amps  = torch.randn(self.in_channels, K, generator=rng) * 0.5
        grid_h = torch.linspace(0, 2 * math.pi, H)
        grid_w = torch.linspace(0, 2 * math.pi, W)
        gh, gw = torch.meshgrid(grid_h, grid_w, indexing="ij")
        for k in range(K):
            phase = torch.rand(1, generator=rng).item() * 2 * math.pi
            mode  = torch.sin(freqs[k, 0] * gh + freqs[k, 1] * gw + phase)
            for c in range(self.in_channels):
                x[c] += amps[c, k] * mode
        target = (x[0] * 0.6 + x[1] * 0.3 + x[2] * 0.1).unsqueeze(0)
        target += 0.05 * torch.empty_like(target).normal_(generator=rng)
        return x, target


# ──────────────────────────────────────────────────────────────────────────────
# Vision Transformer (unchanged from v2)
# ──────────────────────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        return x + self.mlp(self.norm2(x))


class WeatherViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels,
                 embed_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.patch_size  = patch_size
        self.img_size    = img_size
        num_patches      = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm        = nn.LayerNorm(embed_dim)
        self.decoder     = nn.Linear(embed_dim, patch_size * patch_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        H = W = self.img_size
        P = self.patch_size
        nH = nW = H // P
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 1:, :]
        x = self.decoder(x)
        x = x.reshape(B, nH, nW, P, P).permute(0, 1, 3, 2, 4).contiguous()
        return x.reshape(B, 1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────
def get_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # device_id must be set before init_process_group in multi-node jobs
    # so NCCL uses the correct GPU-to-rank mapping across heterogeneous nodes.
    dist.init_process_group(backend="nccl", device_id=device)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank0   = rank == 0

    if is_rank0:
        print(f"[vit_train] v3  world_size={world_size}  "
              f"rank={rank}  local_rank={local_rank}  device={device}")

    # ── pynvml handle for this rank's GPU ─────────────────────────────────────
    gpu_poller = GpuPoller(local_rank)
    if is_rank0:
        status = "available" if gpu_poller.available else "unavailable (nvidia-smi not found)"
        print(f"[vit_train] gpu_poller: {status}")

    # ── Roofline: compute per-batch FLOPs once ────────────────────────────────
    fwd_flops   = compute_vit_flops(
        args.img_size, args.patch_size, args.in_channels,
        args.embed_dim, args.depth, args.num_heads,
        args.mlp_ratio, args.batch_size,
    )
    # fwd + bwd ≈ 3× forward FLOPs
    step_flops  = fwd_flops * 3
    # Peak TFLOP/s depends on dtype
    peak_tflops = H100_BF16_TFLOPS if args.dtype == "bf16" else H100_FP32_TFLOPS
    if is_rank0:
        print(f"[vit_train] step FLOPs: {step_flops/1e12:.3f} TFLOP/batch  "
              f"H100 peak ({args.dtype}): {peak_tflops:.0f} TFLOP/s")

    # ── dtype ─────────────────────────────────────────────────────────────────
    use_amp   = args.dtype == "bf16"
    amp_dtype = torch.bfloat16

    # ── Dataset & loader ──────────────────────────────────────────────────────
    with nvtx_range("Dataset init", color="yellow", active=args.nsys):
        dataset = SyntheticWeatherDataset(args.dataset_size, args.img_size,
                                          args.in_channels)
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True, drop_last=True)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             sampler=sampler, num_workers=4,
                             pin_memory=True, drop_last=True,
                             persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    with nvtx_range("Model init", color="yellow", active=args.nsys):
        model = WeatherViT(
            img_size=args.img_size, patch_size=args.patch_size,
            in_channels=args.in_channels, embed_dim=args.embed_dim,
            depth=args.depth, num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        ).to(device)

    # Barrier outside nvtx_range — ensure all ranks have completed model
    # construction and GPU allocation before DDP's cross-process parameter
    # shape verification. Without this, fast ranks can reach DDP.__init__
    # while slow ranks are still allocating, causing the monitoredBarrier
    # timeout and SIGSEGV on the lagging node.
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_rank0:
        print(f"[vit_train] model params: {n_params:.1f}M")

    # ── Optimizer / scaler / criterion ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    #scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.MSELoss()

    total_steps  = args.epochs * len(loader)
    logger       = RankLogger(args.log_dir, rank)
    timer        = StepTimer(device)
    epoch_timer  = EpochTimer()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0

    with nvtx_range("Training", color="green", active=args.nsys):
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            epoch_timer.reset()

            with nvtx_range(f"Epoch {epoch}", color="blue", active=args.nsys):

                # ── Data prefetch timing: time the iterator's __next__ ────────
                data_iter   = iter(loader)
                step        = 0

                while True:
                    # ── Data loading ──────────────────────────────────────────
                    timer.start("dataload")
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        timer.stop("dataload")
                        break
                    timer.stop("dataload")

                    nvtx_active = args.nsys and global_step < args.profile_steps

                    # LR
                    lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    # ── H2D ───────────────────────────────────────────────────
                    with nvtx_range(f"step={global_step} h2d", color="red",
                                    active=nvtx_active):
                        timer.start("h2d")
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        timer.stop("h2d")

                    optimizer.zero_grad(set_to_none=True)

                    # ── Forward ───────────────────────────────────────────────
                    with nvtx_range(f"step={global_step} forward", color="cyan",
                                    active=nvtx_active):
                        timer.start("forward")
                        with torch.autocast(device_type="cuda",
                                            dtype=amp_dtype, enabled=use_amp):
                            pred = model(x)
                            loss = criterion(pred, y)
                        timer.stop("forward")

                    # ── Backward ──────────────────────────────────────────────
                    with nvtx_range(f"step={global_step} backward", color="purple",
                                    active=nvtx_active):
                        timer.start("backward")
                        scaler.scale(loss).backward()
                        # DDP all-reduce fires inside .backward() above.
                        # We bracket the explicit loss all-reduce separately
                        # below as 'nccl' to isolate that collective.
                        timer.stop("backward")

                    # ── NCCL: explicit loss all-reduce (per-step barrier) ─────
                    # This times the cost of synchronising a scalar across ranks.
                    # The DDP gradient all-reduce is already captured inside
                    # the backward timer; this adds the logging collective cost.
                    with nvtx_range(f"step={global_step} nccl", color="magenta",
                                    active=nvtx_active):
                        timer.start("nccl")
                        loss_reduced = loss.detach().clone()
                        dist.all_reduce(loss_reduced, op=dist.ReduceOp.SUM)
                        loss_reduced /= world_size
                        timer.stop("nccl")

                    # ── Optimizer ─────────────────────────────────────────────
                    with nvtx_range(f"step={global_step} optim", color="orange",
                                    active=nvtx_active):
                        timer.start("optim")
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        ).item()
                        scaler.step(optimizer)
                        scaler.update()
                        timer.stop("optim")

                    # ── Sync CUDA events ──────────────────────────────────────
                    timer.sync_and_collect()
                    step_ms = timer.get_ms()
                    epoch_timer.accumulate(step_ms)

                    # ── pynvml sample (this rank's GPU) ───────────────────────
                    gpu_stats = gpu_poller.sample()

                    # ── Roofline: achieved TFLOP/s this step ─────────────────
                    step_total_ms = sum(step_ms.values()) + 1e-9
                    achieved_tflops = (step_flops / 1e12) / (step_total_ms / 1000.0)
                    roofline_pct    = 100.0 * achieved_tflops / peak_tflops

                    epoch_loss += loss_reduced.detach()

                    # ── Per-step logging (every rank, every log_interval) ─────
                    if global_step % args.log_interval == 0:
                        loss_val = loss_reduced.item()
                        elapsed  = epoch_timer._totals.get("forward", 1e-9) * \
                                   (step + 1) / max(step + 1, 1)
                        wall_elapsed = time.perf_counter() - epoch_timer._t0 + 1e-9
                        sps      = (step + 1) * args.batch_size * world_size / wall_elapsed

                        # All-reduce timings for global (rank-0) log
                        mean_ms = timer.all_reduce_ms(world_size)

                        # ── Per-rank JSONL (all ranks) ────────────────────────
                        logger.log_rank(
                            epoch=epoch, step=global_step,
                            loss=loss_val,
                            lr=lr,
                            grad_norm=grad_norm,
                            samples_per_sec=sps,
                            # pynvml GPU stats for this rank's physical GPU
                            **gpu_stats,
                            # torch allocator view
                            torch_mem_alloc_gb=torch.cuda.memory_allocated(device) / 1e9,
                            torch_mem_reserved_gb=torch.cuda.memory_reserved(device) / 1e9,
                            # per-rank step timing (not all-reduced)
                            **{f"ms_{k}": v for k, v in step_ms.items()},
                            # roofline
                            achieved_tflops=achieved_tflops,
                            roofline_pct=roofline_pct,
                            run_tag=args.run_tag,
                        )

                        # ── Global JSONL (rank 0, all-reduced values) ─────────
                        if is_rank0:
                            print(
                                f"epoch={epoch:03d}  step={global_step:05d}  "
                                f"loss={loss_val:.4f}  lr={lr:.2e}  "
                                f"grad_norm={grad_norm:.3f}  "
                                f"samples/s={sps:.1f}  "
                                f"util={gpu_stats['gpu_util_pct']}%  "
                                f"roof={roofline_pct:.1f}%  "
                                f"fwd={mean_ms['forward']:.1f}ms  "
                                f"bwd={mean_ms['backward']:.1f}ms  "
                                f"nccl={mean_ms['nccl']:.1f}ms"
                            )
                            logger.log_global(
                                epoch=epoch, step=global_step,
                                loss=loss_val, lr=lr,
                                grad_norm=grad_norm,
                                samples_per_sec=sps,
                                gpu_mem_gb=torch.cuda.memory_allocated(device) / 1e9,
                                gpu_mem_reserved_gb=torch.cuda.memory_reserved(device) / 1e9,
                                achieved_tflops=achieved_tflops,
                                roofline_pct=roofline_pct,
                                **{f"ms_{k}": v for k, v in mean_ms.items()},
                                run_tag=args.run_tag,
                            )

                    global_step += 1
                    step        += 1

            # ── End of epoch ──────────────────────────────────────────────────
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            epoch_loss_avg = epoch_loss.item() / (len(loader) * world_size)
            ep_summary     = epoch_timer.summary()

            if is_rank0:
                print(
                    f"── epoch {epoch:03d} done  "
                    f"avg_loss={epoch_loss_avg:.4f}  "
                    f"wall={ep_summary['epoch_wall_s']:.1f}s  "
                    f"dataload={ep_summary['epoch_pct_dataload']:.1f}%  "
                    f"fwd={ep_summary['epoch_pct_forward']:.1f}%  "
                    f"bwd={ep_summary['epoch_pct_backward']:.1f}%  "
                    f"nccl={ep_summary['epoch_pct_nccl']:.1f}%  "
                    f"idle={ep_summary['epoch_pct_idle']:.1f}% ──"
                )
                logger.log_global(
                    epoch=epoch, step=global_step,
                    epoch_avg_loss=epoch_loss_avg,
                    **ep_summary,
                )

            # Per-rank epoch summary
            logger.log_rank(
                epoch=epoch, step=global_step,
                epoch_avg_loss=epoch_loss_avg,
                **ep_summary,
            )

            # Checkpoint every 10 epochs (rank 0)
            if is_rank0 and (epoch + 1) % 10 == 0:
                ckpt_path = Path(args.ckpt_dir) / f"vit_epoch{epoch:03d}.pt"
                torch.save({
                    "epoch": epoch, "step": global_step,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"[vit_train] checkpoint saved: {ckpt_path}")

    logger.close()
    if is_rank0:
        print(f"[vit_train] complete — {global_step} steps")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
