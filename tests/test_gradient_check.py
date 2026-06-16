"""R1 -- finite-difference gradient check for the compliance evaluator.

The foundational test for a gradient-based optimizer (plan P4.1.1): the analytic
sensitivity ``dc`` that ``ComplianceEvaluator`` returns must equal the gradient of the
compliance it returns, verified against central finite differences of the *full*
design -> density-filter -> physics chain. This catches the sign / scaling / missing
filter-chain-rule errors that the higher-level feasibility and monotonicity checks would
silently pass over (a wrong gradient still converges to a wrong-but-feasible point).
"""

import numpy as np
import scipy.sparse as sp

from pytopo3d.core.evaluator import ComplianceEvaluator
from pytopo3d.core.optimizer import _make_scatter_map
from pytopo3d.utils.assembly import build_edof, build_force_vector, build_supports
from pytopo3d.utils.filter import build_filter
from pytopo3d.utils.solver import get_solver
from pytopo3d.utils.stiffness import lk_H8


def _build(nelx, nely, nelz, rmin):
    """Assemble the same static context the optimizer builds, with no obstacles."""
    nu = 0.3
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    F = build_force_vector(nelx, nely, nelz, ndof, None)
    freedofs0, _ = build_supports(nelx, nely, nelz, ndof, None)
    KE = lk_H8(nu)
    edofMat, iK, jK = build_edof(nelx, nely, nelz)
    i_u, j_u, dup2uniq = _make_scatter_map(iK - 1, jK - 1, ndof)
    K = sp.csr_matrix((np.zeros(len(i_u)), (i_u, j_u)), shape=(ndof, ndof))
    H, Hs = build_filter(nelx, nely, nelz, rmin)
    shape = (nely, nelx, nelz)
    ev = ComplianceEvaluator(
        KE=KE, dup2uniq=dup2uniq, K=K, freedofs0=freedofs0, F=F, ndof=ndof,
        solver_func=get_solver(False)[0], edofMat=edofMat, shape=shape,
        H=H, Hs=Hs, obstacle_mask=np.zeros(shape, dtype=bool),
    )
    return ev, H, Hs, shape


def _max_rel_error(seed=0, h=1e-6, n_samples=10):
    nelx, nely, nelz, rmin, penal = 4, 3, 2, 1.5, 3.0
    ev, H, Hs, shape = _build(nelx, nely, nelz, rmin)

    # Density filter, matching optimizer.py exactly: xPhys = (H @ x) / Hs.
    def xphys(xd):
        return (H * xd.ravel(order="F") / Hs).reshape(shape, order="F")

    def compliance(xd):
        return ev.evaluate(xphys(xd), penal)[0]

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 0.7, size=shape)          # interior, so x +/- h stays in (0, 1)
    _, dc, _ = ev.evaluate(xphys(x), penal)        # analytic d(compliance)/d(x_design)

    worst = 0.0
    for k in rng.choice(x.size, size=n_samples, replace=False):
        idx = np.unravel_index(k, x.shape)
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        fd = (compliance(xp) - compliance(xm)) / (2 * h)        # central difference
        rel = abs(fd - dc[idx]) / max(abs(dc[idx]), 1e-30)
        worst = max(worst, rel)
    return worst


def test_compliance_gradient_matches_finite_difference():
    # Tolerance derived from the measured central-difference error window (plan R3):
    # the best h sits where truncation O(h^2) and round-off O(eps/h) balance; well within
    # 1e-4 relative here. A sign error makes fd and dc differ by ~2x -> ~1.0 relative, so
    # this gate is sharp (mutation: flip the dc sign and it fails by orders of magnitude).
    assert _max_rel_error(h=1e-6) < 1e-4


def test_gradient_check_is_consistent_across_h():
    # The agreement is not a fluke of one step size: central differences converge.
    assert _max_rel_error(h=1e-5) < 1e-4
    assert _max_rel_error(h=1e-6) < 1e-4
