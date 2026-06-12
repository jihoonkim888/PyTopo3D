"""CPU-path smoke tests and physical invariants.

The targeted regressions here are issue #7: on a CPU-only machine the optimizer
crashed with `NameError: name 'cusp' is not defined` (and, once that was fixed, a
second `name 'cp' is not defined` surfaced in apply_filter's CPU branch). These
tests exercise exactly those code paths end to end.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pytopo3d.core.optimizer import HAS_CUPY, top3d
from pytopo3d.utils.filter import apply_filter
from pytopo3d.utils.oc_update import optimality_criteria_update

# The issue-#7 NameError (`cp`/`cusp` undefined) can only occur when CuPy is absent;
# with CuPy installed those names exist, so the crash is unreproducible. Skip -- not
# fail -- those two regressions on a CuPy machine. The invariant tests further down run
# everywhere: they validate CPU behaviour regardless of whether CuPy is present.
cpu_only_regression = pytest.mark.skipif(
    HAS_CUPY, reason="issue-#7 NameError is only reproducible without CuPy installed"
)


@cpu_only_regression
def test_oc_update_cpu_runs():
    """Regression for issue #7: OC update must not touch the undefined name `cusp` on CPU."""
    n = 4
    nele = n ** 3
    x = np.full((n, n, n), 0.3)
    dc = -np.ones((n, n, n))
    dv = np.ones((n, n, n))
    obstacle = np.zeros((n, n, n), dtype=bool)
    H = sp.identity(nele, format="csr")
    Hs = np.ones(nele)

    xnew, change = optimality_criteria_update(
        x, dc, dv, 0.3, H, Hs, nele, obstacle, nele, use_gpu=False
    )

    assert xnew.shape == (n, n, n)
    assert np.isfinite(change)


@cpu_only_regression
def test_apply_filter_cpu_runs():
    """Regression for the second (previously unreachable) bug: apply_filter's CPU
    branch referenced the undefined name `cp`."""
    n = 4
    nele = n ** 3
    H = sp.identity(nele, format="csr")
    Hs = np.ones(nele)
    x = np.full((n, n, n), 0.3)

    out = apply_filter(H, x, Hs, x.shape, use_gpu=False)

    assert out.shape == (n, n, n)
    assert np.allclose(out, x)  # identity filter is a no-op


def test_top3d_cpu_end_to_end(small_case):
    """The full optimization loop must run to completion on CPU (the path #7 broke)."""
    xPhys = top3d(**small_case)
    expected = (small_case["nely"], small_case["nelx"], small_case["nelz"])
    assert xPhys.shape == expected
    assert np.isfinite(xPhys).all()


def test_density_bounds_and_volume(small_case):
    """Physical invariants, independent of platform / BLAS."""
    xPhys = top3d(**small_case)
    assert xPhys.min() >= -1e-9
    assert xPhys.max() <= 1.0 + 1e-9
    # Volume constraint is enforced on the design domain (no obstacles here).
    assert abs(float(xPhys.mean()) - small_case["volfrac"]) < 0.02


def test_optimization_actually_progresses(small_case):
    """A working SIMP run pushes densities toward 0/1, away from the uniform guess."""
    xPhys = top3d(**small_case)
    assert xPhys.std() > 1e-3
    assert xPhys.min() < small_case["volfrac"] < xPhys.max()


def test_determinism(small_case):
    """No hidden randomness: identical inputs give a bit-identical result (same platform)."""
    a = top3d(**small_case)
    b = top3d(**small_case)
    assert np.array_equal(a, b)
