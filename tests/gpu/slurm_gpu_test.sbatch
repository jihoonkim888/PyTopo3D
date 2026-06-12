#!/bin/bash
# Run the GPU test suite on a Slurm cluster with NVIDIA GPU(s).
#
# Usage (from the repo root on the cluster):
#     sbatch tests/gpu/slurm_gpu_test.sbatch
#
# Adjust the #SBATCH lines and the env-var defaults below to match your cluster
# (partition name, GPU type, CUDA version).
#
#SBATCH --job-name=pytopo3d-gpu
#SBATCH --gres=gpu:1          # pin a type if you like, e.g. gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=pytopo3d-gpu-%j.log
# #SBATCH --partition=...     # uncomment and set if your cluster needs an explicit partition

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
echo "repo: $REPO"

# Dedicated venv for the GPU run. The [ctk] extra pulls the CUDA toolkit (libraries
# AND headers) as pip wheels, so CuPy works without a system CUDA install -- it needs
# the headers to JIT-compile its kernels at runtime. Swap the wheel (e.g.
# cupy-cuda11x[ctk]) to match your cluster's CUDA major version.
VENV="${VENV:-$HOME/.cache/pytopo3d-gpu-venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "creating venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip wheel
    "$VENV/bin/pip" install numpy scipy matplotlib pytest "cupy-cuda12x[ctk]"
fi

echo "--- GPU visible to the job ---"
nvidia-smi || echo "(nvidia-smi unavailable)"

echo "--- running GPU tests ---"
PYTHONPATH="$REPO" "$VENV/bin/python" -m pytest tests/gpu -v
