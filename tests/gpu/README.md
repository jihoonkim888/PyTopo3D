# GPU tests (run on a CUDA machine)

The tests under `tests/gpu/` need an NVIDIA GPU with CuPy installed, so they are
**skipped automatically** on CPU-only machines (including the GitHub Actions CI
runners). They have to be run by hand on a GPU box.

`test_cpu_gpu_parity.py` runs the same small problem on the CPU and GPU paths and
asserts the density fields agree within tolerance. Because the two paths are
independent implementations of the same algorithm, agreement is strong evidence
that **both** are correct — it is the closest thing to a correctness check we have
without the MATLAB reference.

## Running locally on a GPU machine

```bash
python -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install numpy scipy matplotlib pytest cupy-cuda12x   # match the wheel to your CUDA version
PYTHONPATH=. pytest tests/gpu -v
```

## Running on a Slurm cluster

A ready-to-edit batch script is provided at `slurm_gpu_test.sbatch`. From the repo
root on the cluster:

```bash
sbatch tests/gpu/slurm_gpu_test.sbatch
squeue --me
tail -f pytopo3d-gpu-<jobid>.log
```

The script creates a throwaway venv, installs `cupy-cuda12x`, and runs the GPU
tests. Edit the `#SBATCH` lines (partition, GPU type/count, time) and the
`cupy-cuda12x` wheel to match your cluster's scheduler and CUDA toolkit.

## Why this is not in CI

GitHub's hosted runners have no GPU, and wiring a self-hosted GPU runner into CI is
a larger piece of infrastructure. For now GPU parity is a **manual, periodic**
check — run it after changes that touch the GPU code paths (the `*_gpu` branches in
`optimizer.py`, `oc_update.py`, `filter.py`, `compliance.py`, `solver.py`).
