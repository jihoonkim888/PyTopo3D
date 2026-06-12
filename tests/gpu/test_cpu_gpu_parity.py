"""GPU <-> CPU cross-validation.

Skipped automatically unless CuPy is importable (i.e. on a CUDA box). This is the
one test that anchors *correctness* rather than just no-regression: the CPU and GPU
code paths are independent implementations of the same algorithm, so on identical
inputs they must produce the same density field. Agreement is strong evidence both
paths are right; disagreement localizes a bug to one of them.

Run on a GPU machine, or on a Slurm cluster -- see tests/gpu/README.md and
tests/gpu/slurm_gpu_test.sbatch.
"""

import numpy as np
import pytest

# Skip the whole module unless a usable CUDA device is present. `importorskip` alone is
# not enough: the cupy wheel can be installed on a CPU-only box (it imports fine) and
# would then NOT skip, failing later inside top3d. Check the device count too.
cp = pytest.importorskip("cupy", reason="CuPy not installed (needs a CUDA GPU)")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device available", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from pytopo3d.core.optimizer import top3d  # noqa: E402
from tests.cases import SMALL_CASE  # noqa: E402

VOXEL_DELTA = 0.05
MAX_FRAC_DISAGREE = 0.02
MEAN_TOL = 5e-3


def test_cpu_gpu_density_parity():
    cpu_case = dict(SMALL_CASE, use_gpu=False)
    gpu_case = dict(SMALL_CASE, use_gpu=True)

    cpu = np.asarray(top3d(**cpu_case))
    gpu = np.asarray(top3d(**gpu_case))

    assert cpu.shape == gpu.shape
    frac_disagree = float(np.mean(np.abs(cpu - gpu) > VOXEL_DELTA))
    assert frac_disagree <= MAX_FRAC_DISAGREE, (
        f"CPU and GPU disagree on {frac_disagree:.1%} of voxels "
        f"(allowed {MAX_FRAC_DISAGREE:.0%})"
    )
    assert abs(float(cpu.mean()) - float(gpu.mean())) < MEAN_TOL
