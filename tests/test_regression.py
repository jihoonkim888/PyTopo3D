"""Golden-master regression test.

Locks the *current* CPU result of the small case as a no-regression floor: a saved
reference density field. Any future change that silently alters the numerical output
trips this test. It does NOT certify the result is physically correct (that is the
MATLAB/paper cross-validation, a separate, larger effort) -- it only catches drift.

Comparison is structural (fraction of voxels that move materially) rather than strict
elementwise equality, so platform/BLAS noise does not cause false alarms while a real
algorithmic regression -- which moves the whole pattern -- still fails.

Regenerate intentionally after a *blessed* algorithm change:
    python tests/_generate_golden.py
"""

from pathlib import Path

import numpy as np
import pytest

from pytopo3d.core.optimizer import top3d

GOLDEN = Path(__file__).parent / "data" / "golden_small_case.npz"

# Tolerances: a genuine regression moves many voxels; BLAS noise moves a few boundary ones.
VOXEL_DELTA = 0.05          # a voxel "moved materially" if |Δdensity| exceeds this
MAX_FRAC_MOVED = 0.02       # at most 2% of voxels may move materially
MEAN_TOL = 5e-3             # mean density (= volume fraction) must match closely


@pytest.mark.skipif(
    not GOLDEN.exists(),
    reason="golden file missing -- run `python tests/_generate_golden.py`",
)
def test_small_case_matches_golden(small_case):
    ref = np.load(GOLDEN)
    ref_density = ref["density"]

    xPhys = top3d(**small_case)

    assert xPhys.shape == ref_density.shape
    frac_moved = float(np.mean(np.abs(xPhys - ref_density) > VOXEL_DELTA))
    assert frac_moved <= MAX_FRAC_MOVED, (
        f"{frac_moved:.1%} of voxels moved by more than {VOXEL_DELTA} vs golden "
        f"(allowed {MAX_FRAC_MOVED:.0%}). If this change was intentional, regenerate "
        f"the golden file."
    )
    assert abs(float(xPhys.mean()) - float(ref["mean"])) < MEAN_TOL
