"""
plot_metrics.py  v3 — Plot training metrics from vit_weather_train.py v3.

Single-run mode:
  python plot_metrics.py --metrics-file logs/train_metrics.jsonl --output-dir logs

Multi-run comparison:
  python plot_metrics.py \
      --metrics-files logs/runA/train_metrics.jsonl logs/runB/train_metrics.jsonl \
      --run-labels "depth=8" "depth=12" \
      --output-dir logs/comparison

Per-rank analysis (reads rankN.jsonl files automatically when --rank-dir set):
  python plot_metrics.py --rank-dir logs --output-dir logs

New in v3:
  gpu_utilization.png      SM utilization % from pynvml over training
  roofline.png             achieved vs theoretical TFLOP/s + roofline % curve
  epoch_breakdown.png      stacked bar of epoch wall time by phase
  nccl_timing.png          per-step NCCL all-reduce ms over training
  rank_comparison.png      per-rank loss / timing / utilization spread
"""

import argparse
import json
from pathlib import Path
from itertools import cycle
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--metrics-file",  default=None)
p.add_argument("--metrics-files", nargs="+", default=None)
p.add_argument("--run-labels",    nargs="+", default=None)
p.add_argument("--rank-dir",      default=None,
               help="Directory containing rankN.jsonl files for per-rank plots")
p.add_argument("--output-dir",    default="logs")
args = p.parse_args()

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
})
PALETTE = ["#0070C0","#FF8C00","#00B050","#C00000",
           "#7030A0","#00B0F0","#FF0000","#92D050"]
BLUE, ORANGE, GREEN, RED, PURPLE = PALETTE[:5]

PHASE_COLORS = {
    "dataload": "#92D050", "h2d":      "#00B0F0",
    "forward":  "#0070C0", "backward": "#7030A0",
    "nccl":     "#FF00FF", "optim":    "#FF8C00",
    "idle":     "#D9D9D9",
}
PHASES     = ["dataload", "h2d", "forward", "backward", "nccl", "optim"]
ALL_PHASES = PHASES + ["idle"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def smooth(y, w=15):
    y = np.array(y, dtype=float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def savefig(fig, name):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def load_jsonl(path):
    step_recs, epoch_recs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "epoch_avg_loss" in r:
                epoch_recs.append(r)
            elif "loss" in r:
                step_recs.append(r)
    return step_recs, epoch_recs


def get(recs, key):
    return [r.get(key) for r in recs]


def valid(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return [], []
    a, b = zip(*pairs)
    return list(a), list(b)


# ── Resolve file list ─────────────────────────────────────────────────────────
if args.metrics_files:
    files  = args.metrics_files
    labels = args.run_labels or [Path(f).parent.name for f in files]
elif args.metrics_file:
    files  = [args.metrics_file]
    labels = [Path(args.metrics_file).parent.name or "run0"]
else:
    files  = ["logs/train_metrics.jsonl"]
    labels = ["run0"]

multi = len(files) > 1

all_step  = []
all_epoch = []
for f in files:
    sr, er = load_jsonl(f)
    all_step.append(sr)
    all_epoch.append(er)

step_recs  = all_step[0]
epoch_recs = all_epoch[0]

if not step_recs:
    print("No step records found in primary file.")
    raise SystemExit(0)

# ── If gpu_util_pct missing from primary file, merge from rank0.jsonl ─────────
# log_global() didn't include gpu_util_pct in older runs — rank0.jsonl has it.
if not any(r.get("gpu_util_pct") is not None for r in step_recs):
    rank0_path = Path(files[0]).parent / "rank0.jsonl"
    if rank0_path.exists():
        rank0_steps, _ = load_jsonl(str(rank0_path))
        # Build a step→record lookup from rank0
        rank0_by_step = {r["step"]: r for r in rank0_steps if "step" in r}
        for r in step_recs:
            if r.get("step") in rank0_by_step:
                r0 = rank0_by_step[r["step"]]
                for key in ("gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_total_gb"):
                    if r0.get(key) is not None:
                        r[key] = r0[key]
        print("  [info] gpu_util_pct merged from rank0.jsonl into primary metrics")

steps      = get(step_recs, "step")
losses     = get(step_recs, "loss")
lrs        = get(step_recs, "lr")
grad_norms = get(step_recs, "grad_norm")
sps        = get(step_recs, "samples_per_sec")
gpu_mem    = get(step_recs, "gpu_mem_gb")
gpu_res    = get(step_recs, "gpu_mem_reserved_gb")
wall_times = get(step_recs, "wall_time")
gpu_util   = get(step_recs, "gpu_util_pct")
achieved_tf= get(step_recs, "achieved_tflops")
roof_pct   = get(step_recs, "roofline_pct")
ms         = {ph: get(step_recs, f"ms_{ph}") for ph in PHASES}

ep_steps   = get(epoch_recs, "step")
ep_losses  = get(epoch_recs, "epoch_avg_loss")


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN PLOTS (v2 carry-overs, updated for new fields)
# ═════════════════════════════════════════════════════════════════════════════

# ── Loss curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, losses, alpha=0.2, color=BLUE, linewidth=0.8, label="Step loss")
ax.plot(steps, smooth(losses), color=BLUE, linewidth=1.8, label="Smoothed")
if ep_losses:
    ax.plot(ep_steps, ep_losses, "o--", color=ORANGE, linewidth=1.5,
            markersize=5, label="Epoch avg")
ax.set_xlabel("Global step"); ax.set_ylabel("MSE loss")
ax.set_title("Training loss — WeatherViT v3")
ax.legend(framealpha=0.8)
savefig(fig, "loss_curve.png")

# ── Gradient norm ─────────────────────────────────────────────────────────────
gn_x, gn_y = valid(steps, grad_norms)
if gn_x:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(gn_x, gn_y, alpha=0.2, color=PURPLE, linewidth=0.8)
    axes[0].plot(gn_x, smooth(gn_y), color=PURPLE, linewidth=1.8, label="Grad norm")
    axes[0].axhline(1.0, color=RED, linestyle="--", linewidth=1.0,
                    alpha=0.7, label="Clip threshold")
    axes[0].set_ylabel("Gradient L2 norm"); axes[0].legend(framealpha=0.8)
    axes[0].set_title("Gradient norm (norm > 1 = clipped)")
    clipped  = [1.0 if v > 1.0 else 0.0 for v in gn_y]
    clip_pct = smooth(clipped, w=20) * 100
    axes[1].fill_between(gn_x, clip_pct, alpha=0.35, color=RED)
    axes[1].plot(gn_x, clip_pct, color=RED, linewidth=1.2, label="% clipped (w=20)")
    axes[1].set_ylim(0, 105); axes[1].set_ylabel("% clipped")
    axes[1].set_xlabel("Global step"); axes[1].legend(framealpha=0.8)
    fig.tight_layout(); savefig(fig, "grad_norm.png")

# ── Timing breakdown ──────────────────────────────────────────────────────────
ms_valid = {ph: [] for ph in PHASES}
ms_steps = []
for i, r in enumerate(step_recs):
    vals = [r.get(f"ms_{ph}") for ph in PHASES]
    if all(v is not None for v in vals):
        ms_steps.append(steps[i])
        for ph, v in zip(PHASES, vals):
            ms_valid[ph].append(v)

if ms_steps:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    bottom = np.zeros(len(ms_steps))
    for ph in PHASES:
        y = np.array(ms_valid[ph])
        axes[0].fill_between(ms_steps, bottom, bottom + y,
                             alpha=0.75, color=PHASE_COLORS[ph], label=ph)
        bottom += y
    axes[0].set_ylabel("ms / step"); axes[0].legend(loc="upper right", framealpha=0.8)
    axes[0].set_title("Step timing breakdown (stacked, mean across ranks)")
    fwd = np.array(ms_valid["forward"]); bwd = np.array(ms_valid["backward"])
    ratio = np.where(fwd > 0, bwd / fwd, np.nan)
    axes[1].plot(ms_steps, smooth(ratio, w=10), color=PURPLE, linewidth=1.5,
                 label="backward / forward ratio")
    axes[1].axhline(2.0, color=PURPLE, linestyle="--", linewidth=1.0,
                    alpha=0.5, label="Expected ~2×")
    axes[1].set_ylabel("bwd/fwd"); axes[1].set_xlabel("Global step")
    axes[1].legend(framealpha=0.8); fig.tight_layout()
    savefig(fig, "timing_breakdown.png")

# ── Throughput ────────────────────────────────────────────────────────────────
sp_x, sp_y = valid(steps, sps)
if sp_x:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sp_x, sp_y, alpha=0.25, color=GREEN, linewidth=0.8)
    ax.plot(sp_x, smooth(sp_y), color=GREEN, linewidth=2.0, label="Samples/s (smoothed)")
    ax.axhline(np.mean(sp_y), color=GREEN, linestyle=":",
               label=f"Mean: {np.mean(sp_y):.0f}")
    ax.set_xlabel("Global step"); ax.set_ylabel("Samples / second")
    ax.set_title("Training throughput — all GPUs combined")
    ax.legend(framealpha=0.8); savefig(fig, "throughput.png")

# ── LR ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(steps, lrs, color=ORANGE, linewidth=1.8)
ax.set_xlabel("Global step"); ax.set_ylabel("LR")
ax.set_title("LR schedule (warmup → cosine)")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
savefig(fig, "lr_schedule.png")

# ── GPU memory ────────────────────────────────────────────────────────────────
mx, my = valid(steps, gpu_mem)
if mx:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(mx, my, alpha=0.2, color=RED, label="Allocated")
    ax.plot(mx, my, color=RED, linewidth=1.5)
    rx, ry = valid(steps, gpu_res)
    if rx:
        ax.fill_between(rx, ry, alpha=0.1, color=ORANGE)
        ax.plot(rx, ry, color=ORANGE, linewidth=1.0, linestyle="--",
                label="Reserved")
    ax.axhline(80.0, color=RED, linestyle=":", alpha=0.4, label="H100 cap (80 GB)")
    ax.set_xlabel("Global step"); ax.set_ylabel("GB")
    ax.set_title("GPU memory — rank 0 (allocated vs reserved)")
    ax.legend(framealpha=0.8); savefig(fig, "gpu_memory.png")

# ── Loss vs wall time ─────────────────────────────────────────────────────────
wt_x, wt_y = valid(wall_times, losses)
if wt_x:
    wt_x = [t / 60 for t in wt_x]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wt_x, wt_y, alpha=0.2, color=BLUE, linewidth=0.8)
    ax.plot(wt_x, smooth(list(wt_y)), color=BLUE, linewidth=1.8)
    ax.set_xlabel("Wall time (min)"); ax.set_ylabel("MSE loss")
    ax.set_title("Loss vs wall time"); savefig(fig, "loss_vs_time.png")


# ═════════════════════════════════════════════════════════════════════════════
# NEW v3 PLOTS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. GPU utilization (pynvml) ───────────────────────────────────────────────
ux, uy = valid(steps, gpu_util)
if ux:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ux, uy, alpha=0.25, color=GREEN, linewidth=0.8)
    ax.plot(ux, smooth(list(uy)), color=GREEN, linewidth=2.0,
            label="SM utilization % (smoothed)")
    ax.axhline(np.mean(uy), color=GREEN, linestyle=":",
               label=f"Mean: {np.mean(uy):.1f}%")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Global step"); ax.set_ylabel("SM utilization (%)")
    ax.set_title("GPU SM utilization — rank 0 (nvidia-smi driver view)\n"
                 "< 80% sustained suggests CPU bottleneck or idle gaps")
    ax.legend(framealpha=0.8); savefig(fig, "gpu_utilization.png")
else:
    print("  [skip] gpu_utilization.png — no gpu_util_pct data in JSONL"
          " (GpuPoller may have failed on compute node; check nvidia-smi PATH)")


# ── 2. Roofline ───────────────────────────────────────────────────────────────
tf_x, tf_y = valid(steps, achieved_tf)
rf_x, rf_y = valid(steps, roof_pct)
if tf_x:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    # Top: achieved TFLOP/s over training
    axes[0].plot(tf_x, tf_y, alpha=0.25, color=BLUE, linewidth=0.8)
    axes[0].plot(tf_x, smooth(list(tf_y)), color=BLUE, linewidth=2.0,
                 label="Achieved TFLOP/s")
    # Annotate the peak line — read from first record if available
    peak_val = step_recs[0].get("peak_tflops", None)
    if peak_val:
        axes[0].axhline(peak_val, color=RED, linestyle="--",
                        linewidth=1.0, alpha=0.6,
                        label=f"H100 peak ({peak_val:.0f} TFLOP/s)")
    axes[0].set_ylabel("TFLOP/s"); axes[0].legend(framealpha=0.8)
    axes[0].set_title("Throughput roofline — achieved vs H100 theoretical peak\n"
                       "(estimates fwd+bwd FLOPs from model config)")

    # Bottom: % of peak
    axes[1].plot(rf_x, rf_y, alpha=0.25, color=ORANGE, linewidth=0.8)
    axes[1].plot(rf_x, smooth(list(rf_y)), color=ORANGE, linewidth=2.0,
                 label="% of peak TFLOP/s")
    axes[1].axhline(np.mean(rf_y), color=ORANGE, linestyle=":",
                    label=f"Mean: {np.mean(rf_y):.1f}%")
    axes[1].set_ylim(0, max(110, max(rf_y) * 1.05))
    axes[1].set_ylabel("% of peak"); axes[1].set_xlabel("Global step")
    axes[1].legend(framealpha=0.8)

    fig.tight_layout(); savefig(fig, "roofline.png")


# ── 3. Epoch timing breakdown (stacked bar per epoch) ─────────────────────────
ep_phase_data = {ph: [] for ph in ALL_PHASES}
ep_nums = []
for r in epoch_recs:
    ep_wall = r.get("epoch_wall_s")
    if ep_wall is None:
        continue
    ep_nums.append(r["epoch"])
    for ph in ALL_PHASES:
        ep_phase_data[ph].append(r.get(f"epoch_s_{ph}", 0.0))

if ep_nums:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked bar — seconds per phase
    bottom = np.zeros(len(ep_nums))
    for ph in ALL_PHASES:
        y = np.array(ep_phase_data[ph])
        axes[0].bar(ep_nums, y, bottom=bottom,
                    color=PHASE_COLORS.get(ph, "#AAAAAA"),
                    label=ph, alpha=0.85)
        bottom += y
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Wall time (s)")
    axes[0].set_title("Epoch wall time by phase (seconds)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.8)

    # Right: stacked bar — percentage per phase
    bottom = np.zeros(len(ep_nums))
    for ph in ALL_PHASES:
        y = np.array([r.get(f"epoch_pct_{ph}", 0.0) for r in epoch_recs
                      if r.get("epoch_wall_s") is not None])
        axes[1].bar(ep_nums, y, bottom=bottom,
                    color=PHASE_COLORS.get(ph, "#AAAAAA"),
                    label=ph, alpha=0.85)
        bottom += y
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("% of epoch wall time")
    axes[1].set_title("Epoch time distribution (%)\n"
                       "idle = wall time not captured by instrumented phases")
    axes[1].set_ylim(0, 110)
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.8)

    fig.tight_layout(); savefig(fig, "epoch_breakdown.png")


# ── 4. NCCL timing ────────────────────────────────────────────────────────────
nc_x, nc_y = valid(steps, ms.get("nccl", [None]*len(steps)))
if nc_x:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(nc_x, nc_y, alpha=0.25, color=PURPLE, linewidth=0.8)
    axes[0].plot(nc_x, smooth(list(nc_y)), color=PURPLE, linewidth=2.0,
                 label="NCCL all-reduce ms (smoothed)")
    axes[0].axhline(np.mean(nc_y), color=PURPLE, linestyle=":",
                    label=f"Mean: {np.mean(nc_y):.2f} ms")
    axes[0].set_ylabel("ms"); axes[0].legend(framealpha=0.8)
    axes[0].set_title("NCCL per-step barrier timing (explicit loss all-reduce)\n"
                       "Spikes indicate network contention or rank straggler")

    # NCCL as % of total step time
    total_ms_per_step = []
    nccl_pct_per_step = []
    for i, r in enumerate(step_recs):
        if steps[i] not in nc_x:
            continue
        total = sum(r.get(f"ms_{ph}", 0.0) for ph in PHASES)
        nccl  = r.get("ms_nccl", 0.0)
        if total > 0:
            total_ms_per_step.append(steps[i])
            nccl_pct_per_step.append(100.0 * nccl / total)

    if nccl_pct_per_step:
        axes[1].fill_between(total_ms_per_step, nccl_pct_per_step,
                             alpha=0.3, color=PURPLE)
        axes[1].plot(total_ms_per_step,
                     smooth(nccl_pct_per_step, w=10), color=PURPLE, linewidth=1.5)
        axes[1].set_ylabel("NCCL % of step"); axes[1].set_xlabel("Global step")
        axes[1].set_ylim(0, max(110, max(nccl_pct_per_step) * 1.1))

    fig.tight_layout(); savefig(fig, "nccl_timing.png")


# ── 5. Per-rank comparison ────────────────────────────────────────────────────
rank_dir = Path(args.rank_dir) if args.rank_dir else Path(files[0]).parent
rank_files = sorted(rank_dir.glob("rank*.jsonl"))

if rank_files:
    rank_data = {}
    for rf in rank_files:
        rank_id = rf.stem   # "rank0", "rank1", ...
        sr, _   = load_jsonl(rf)
        if sr:
            rank_data[rank_id] = sr

if rank_files and rank_data:
    rank_ids   = sorted(rank_data.keys())
    rank_colors = {rid: PALETTE[i % len(PALETTE)]
                   for i, rid in enumerate(rank_ids)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-rank comparison — load imbalance diagnostics", fontsize=14)

    # Top-left: loss per rank
    for rid, color in rank_colors.items():
        sr = rank_data[rid]
        sx = get(sr, "step"); sy = get(sr, "loss")
        vx, vy = valid(sx, sy)
        if vx:
            axes[0, 0].plot(vx, smooth(list(vy)), color=color,
                            linewidth=1.5, label=rid, alpha=0.85)
    axes[0, 0].set_title("Loss per rank")
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("MSE loss")
    axes[0, 0].legend(framealpha=0.8)

    # Top-right: forward ms per rank
    for rid, color in rank_colors.items():
        sr = rank_data[rid]
        sx = get(sr, "step"); sy = get(sr, "ms_forward")
        vx, vy = valid(sx, sy)
        if vx:
            axes[0, 1].plot(vx, smooth(list(vy)), color=color,
                            linewidth=1.5, label=rid, alpha=0.85)
    axes[0, 1].set_title("Forward pass ms per rank\n(spread = compute imbalance)")
    axes[0, 1].set_xlabel("Step"); axes[0, 1].set_ylabel("ms")
    axes[0, 1].legend(framealpha=0.8)

    # Bottom-left: GPU utilization per rank
    any_util = False
    for rid, color in rank_colors.items():
        sr = rank_data[rid]
        sx = get(sr, "step"); sy = get(sr, "gpu_util_pct")
        vx, vy = valid(sx, sy)
        if vx:
            axes[1, 0].plot(vx, smooth(list(vy)), color=color,
                            linewidth=1.5, label=rid, alpha=0.85)
            any_util = True
    axes[1, 0].set_title("GPU SM utilization % per rank")
    axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("util %")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].legend(framealpha=0.8)
    if not any_util:
        axes[1, 0].text(0.5, 0.5, "pynvml data not available",
                        ha="center", va="center", transform=axes[1, 0].transAxes)

    # Bottom-right: backward ms per rank (straggler detection)
    for rid, color in rank_colors.items():
        sr = rank_data[rid]
        sx = get(sr, "step"); sy = get(sr, "ms_backward")
        vx, vy = valid(sx, sy)
        if vx:
            axes[1, 1].plot(vx, smooth(list(vy)), color=color,
                            linewidth=1.5, label=rid, alpha=0.85)
    axes[1, 1].set_title("Backward ms per rank\n(outlier rank = DDP straggler)")
    axes[1, 1].set_xlabel("Step"); axes[1, 1].set_ylabel("ms")
    axes[1, 1].legend(framealpha=0.8)

    fig.tight_layout(); savefig(fig, "rank_comparison.png")

    # Summary table
    print("\n── Per-rank summary ──────────────────────────────────")
    print(f"  {'rank':<8} {'loss_mean':>10} {'fwd_ms':>8} {'bwd_ms':>8} {'util%':>7}")
    for rid in rank_ids:
        sr = rank_data[rid]
        lo = [r["loss"]       for r in sr if "loss"       in r]
        fm = [r["ms_forward"] for r in sr if "ms_forward" in r]
        bm = [r["ms_backward"]for r in sr if "ms_backward" in r]
        ut = [r["gpu_util_pct"] for r in sr if r.get("gpu_util_pct") is not None]
        print(f"  {rid:<8} "
              f"{np.mean(lo) if lo else 0:>10.4f} "
              f"{np.mean(fm) if fm else 0:>8.1f} "
              f"{np.mean(bm) if bm else 0:>8.1f} "
              f"{np.mean(ut) if ut else 0:>7.1f}")
    print("─────────────────────────────────────────────────────")
else:
    print("  [skip] rank_comparison.png — no rankN.jsonl files found "
          f"in {rank_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-RUN COMPARISON (unchanged from v2, extended with roofline)
# ═════════════════════════════════════════════════════════════════════════════
if multi:
    colors     = cycle(PALETTE)
    run_colors = [next(colors) for _ in files]

    # Loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for sr, er, label, color in zip(all_step, all_epoch, labels, run_colors):
        s = get(sr, "step"); lo = get(sr, "loss")
        wt = get(sr, "wall_time")
        axes[0].plot(s, smooth(lo), color=color, linewidth=1.8, label=label)
        wx, wy = valid([t/60 for t in wt if t], [l for t,l in zip(wt,lo) if t])
        if wx:
            axes[1].plot(wx, smooth(list(wy)), color=color, linewidth=1.8, label=label)
    for ax, xl in zip(axes, ["Step", "Wall time (min)"]):
        ax.set_xlabel(xl); ax.set_ylabel("MSE loss"); ax.legend(framealpha=0.8)
    axes[0].set_title("Loss vs step"); axes[1].set_title("Loss vs wall time")
    fig.suptitle("Multi-run loss comparison", fontsize=13, y=1.02)
    fig.tight_layout(); savefig(fig, "compare_loss.png")

    # Roofline % comparison
    fig, ax = plt.subplots(figsize=(10, 4))
    for sr, label, color in zip(all_step, labels, run_colors):
        rx, ry = valid(get(sr, "step"), get(sr, "roofline_pct"))
        if rx:
            ax.plot(rx, smooth(list(ry)), color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("% of peak TFLOP/s")
    ax.set_title("Roofline efficiency comparison across runs")
    ax.legend(framealpha=0.8); savefig(fig, "compare_roofline.png")

    # Phase timing grouped bar
    run_means = []
    for sr in all_step:
        means = {}
        for ph in PHASES:
            vals = [r.get(f"ms_{ph}") for r in sr if r.get(f"ms_{ph}") is not None]
            means[ph] = float(np.mean(vals)) if vals else 0.0
        run_means.append(means)
    x     = np.arange(len(PHASES)); w = 0.8 / len(files)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (means, label, color) in enumerate(zip(run_means, labels, run_colors)):
        offsets = x + (i - len(files)/2 + 0.5) * w
        bars = ax.bar(offsets, [means[ph] for ph in PHASES],
                      width=w*0.9, color=color, alpha=0.8, label=label)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(PHASES)
    ax.set_ylabel("Mean ms / step")
    ax.set_title("Step phase timing comparison"); ax.legend(framealpha=0.8)
    savefig(fig, "compare_timing.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Global summary ────────────────────────────────────")
for label, sr, er in zip(labels, all_step, all_epoch):
    lo = [r["loss"] for r in sr if "loss" in r]
    gn = [r["grad_norm"] for r in sr if "grad_norm" in r]
    sp = [r["samples_per_sec"] for r in sr if r.get("samples_per_sec")]
    ut = [r["gpu_util_pct"] for r in sr if r.get("gpu_util_pct") is not None]
    rf = [r["roofline_pct"] for r in sr if r.get("roofline_pct") is not None]
    tf = [r["achieved_tflops"] for r in sr if r.get("achieved_tflops") is not None]
    print(f"\n  Run: {label}")
    if lo:
        print(f"    Loss      initial={lo[0]:.4f}  final={lo[-1]:.4f}  min={min(lo):.4f}")
    if gn:
        print(f"    Grad norm mean={np.mean(gn):.3f}  max={max(gn):.3f}  "
              f"clipped={100*sum(v>1 for v in gn)/len(gn):.1f}%")
    if sp:
        print(f"    Throughput  mean={np.mean(sp):.0f}  peak={max(sp):.0f} samples/s")
    if ut:
        print(f"    GPU util    mean={np.mean(ut):.1f}%  min={min(ut):.1f}%")
    if tf:
        print(f"    TFLOP/s     mean={np.mean(tf):.1f}  peak={max(tf):.1f}")
    if rf:
        print(f"    Roofline    mean={np.mean(rf):.1f}%  peak={max(rf):.1f}%")
print("\n─────────────────────────────────────────────────────")
