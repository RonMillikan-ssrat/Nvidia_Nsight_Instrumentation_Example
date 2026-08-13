#!/bin/bash
#SBATCH --job-name=vit_weather
#SBATCH --account=nesccmgmt
#SBATCH --partition=u1-h100
#SBATCH --qos=admin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1             # one srun task per node; torchrun forks GPUs
#SBATCH --cpus-per-task=16             # all CPUs for that task (torchrun uses them)
#SBATCH --gres=gpu:2                   # 2 GPUs per node
#SBATCH --time=00:45:00
#SBATCH --output=logs/vit_%j.out
#SBATCH --error=logs/vit_%j.err
#SBATCH --exclusive

# Uncomment to debug NCCL issues:
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
#export PYTHONFAULTHANDLER=1

# ── Sanity checks ─────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs checkpoints profiles

SCRIPT_DIR=/scratch3/SYSADMIN/nesccmgmt/Ron.Millikan/devl/nsite3
TRAIN_SCRIPT=${SCRIPT_DIR}/vit_weather_train.py
PLOT_SCRIPT=${SCRIPT_DIR}/plot_metrics.py
#NSYS=/tds_scratch2/SYSADMIN/nesccmgmt/Ron.Millikan/tools/bin/nsys
NSYS=/scratch3/SYSADMIN/nesccmgmt/Ron.Millikan/devl/nsite3/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys
for f in "${TRAIN_SCRIPT}" "${PLOT_SCRIPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: not found: ${f}"
        exit 1
    fi
done

if [[ ! -x "${NSYS}" ]]; then
    echo "WARNING: nsys not found at ${NSYS} — RUN B will fail if enabled"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source ${SCRIPT_DIR}/nsite-files-env/bin/activate

echo "=== Environment ==="
which python
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "import pynvml; print('pynvml ok')" 2>/dev/null || echo "pynvml not found (ok — using nvidia-smi)"
"${NSYS}" --version | head -1
echo ""

# ── Node / GPU inventory ──────────────────────────────────────────────────────
echo "=== GPU inventory ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader
echo ""

# ── nvidia-smi path check across nodes ───────────────────────────────────────
echo "=== nvidia-smi path check ==="
which nvidia-smi || echo "nvidia-smi NOT IN PATH on login/head node"
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "echo node \$(hostname): \$(which nvidia-smi 2>/dev/null || echo NOT FOUND)"
echo ""

# ── Distributed setup ─────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 2))    # 2 GPUs per node × N nodes

echo "=== Distributed config ==="
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}  WORLD_SIZE=${WORLD_SIZE}"
echo "Nodes: $(scontrol show hostnames ${SLURM_NODELIST} | tr '\n' ' ')"
echo ""

# ── Common training args ──────────────────────────────────────────────────────
RUN_TAG="job${SLURM_JOB_ID}"

TRAIN_ARGS=(
    --img-size      128
    --in-channels   4
    --patch-size    16
    --dataset-size  8192
    --embed-dim     512
    --depth         8
    --num-heads     8
    --mlp-ratio     4.0
    --epochs        40
    --batch-size    16
    --lr            1e-4
    --warmup-steps  200
    --dtype         bf16
    --log-dir       logs
    --ckpt-dir      checkpoints
    --log-interval  5
    --run-tag       "${RUN_TAG}"
)

# ── Shared torchrun launcher ──────────────────────────────────────────────────
run_torchrun() {
    srun --ntasks="${SLURM_NNODES}" \
         --ntasks-per-node=1 \
         --cpus-per-task=16 \
         torchrun \
             --nnodes="${SLURM_NNODES}" \
             --nproc_per_node=2 \
             --rdzv_id="${SLURM_JOB_ID}" \
             --rdzv_backend=c10d \
             --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
             "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# RUN A — baseline training (no profiler overhead)
# Produces:
#   logs/train_metrics.jsonl   global (rank-0, all-reduced) metrics
#   logs/rank{0..3}.jsonl      per-rank metrics including gpu_util_pct
#   checkpoints/vit_epoch*.pt  every 10 epochs
# ─────────────────────────────────────────────────────────────────────────────
echo "=== RUN A: baseline training ==="
run_torchrun "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
echo ""
echo "=== RUN A complete ==="
echo "  global metrics : logs/train_metrics.jsonl"
echo "  per-rank files : logs/rank{0..$(( WORLD_SIZE - 1 ))}.jsonl"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# RUN B — nsys profiled run (comment out to disable)
#
# nsys wraps torchrun on each node via srun bash -c.
# Produces one .nsys-rep per node named by SLURM_NODEID.
# --nsys activates NVTX ranges inside the Python script.
# Runs only 2 epochs to keep profile file size manageable.
# ─────────────────────────────────────────────────────────────────────────────
echo "=== RUN B: nsys profile ==="
srun --ntasks="${SLURM_NNODES}" \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     bash -c "
         ${NSYS} profile \
             --output=${PWD}/profiles/vit_${SLURM_JOB_ID}_node\${SLURM_NODEID} \
             --trace=cuda,nvtx,osrt,nccl \
             --force-overwrite=true \
             --cudabacktrace=false \
             --cpuctxsw=none \
         torchrun \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=2 \
             --rdzv_id=${SLURM_JOB_ID} \
             --rdzv_backend=c10d \
             --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
             ${TRAIN_SCRIPT} ${TRAIN_ARGS[*]} --nsys --profile-steps 30 --epochs 2
     "
echo "=== RUN B complete ==="
echo "  profiles: profiles/vit_${SLURM_JOB_ID}_node{0,1}.nsys-rep"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# train_metrics.jsonl is the primary source; rank0.jsonl is used automatically
# as fallback for gpu_util_pct if missing from global file (older runs).
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Plotting metrics ==="
python "${PLOT_SCRIPT}" \
    --metrics-file  logs/train_metrics.jsonl \
    --rank-dir      logs \
    --output-dir    logs

echo ""
echo "=== Plots written to logs/ ==="
ls -lh logs/*.png 2>/dev/null || echo "  (no PNG files found)"

# ─────────────────────────────────────────────────────────────────────────────
# Multi-run comparison (run manually after two or more jobs):
#
# python plot_metrics.py \
#     --metrics-files logs_job1/train_metrics.jsonl \
#                     logs_job2/train_metrics.jsonl \
#     --run-labels    "depth=8 bs=16" "depth=12 bs=16" \
#     --output-dir    logs/comparison
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "=== Job ${SLURM_JOB_ID} complete ==="
echo "  Metrics:      logs/train_metrics.jsonl"
echo "  Per-rank:     logs/rank{0..$(( WORLD_SIZE - 1 ))}.jsonl"
echo "  Plots:        logs/*.png"
echo "  Checkpoints:  checkpoints/"
echo "  Profiles:     profiles/vit_${SLURM_JOB_ID}_node{0,1}.nsys-rep"
