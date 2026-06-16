"""
Main optimizer for 3-D topology optimization – fixed CSR scatter version
(24 April 2025).

Changes relative to the previous refactor
-----------------------------------------
* Keep the 576 × nele COO coordinates (with duplicates) **and** build a
  mapping `dup2uniq` → CSR.data so we can scatter-add each iteration.
* Still reuse the CSR structure – no allocation inside the loop.
* Compatible with the CG + Jacobi GPU solver provided in solver.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from pytopo3d.core.compliance import element_compliance
from pytopo3d.core.evaluator import ComplianceEvaluator
from pytopo3d.utils.assembly import build_edof, build_force_vector, build_supports
from pytopo3d.utils.filter import HAS_CUPY, apply_filter, build_filter
from pytopo3d.utils.logger import get_logger
from pytopo3d.utils.oc_update import optimality_criteria_update
from pytopo3d.utils.solver import get_solver
from pytopo3d.utils.stiffness import lk_H8
from pytopo3d.visualization.display import display_3D

logger = get_logger(__name__)

if HAS_CUPY:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    logger.info("CuPy is available for GPU acceleration")


# ──────────────────────────────────────────────────────────────────────────
def _make_scatter_map(i_full, j_full, ndof):
    """Memory– and speed-optimised scatter map builder."""
    linear = i_full.astype(np.int64) * ndof + j_full        # 1-D key
    uniq_lin, dup2uniq = np.unique(linear, return_inverse=True)
    i_uniq = uniq_lin // ndof
    j_uniq = uniq_lin % ndof
    return i_uniq, j_uniq, dup2uniq


# ──────────────────────────────────────────────────────────────────────────
def top3d(
    nelx: int,
    nely: int,
    nelz: int,
    volfrac: float,
    penal: float,
    rmin: float,
    disp_thres: float,
    obstacle_mask: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    tolx: float = 0.01,
    maxloop: int = 2000,
    save_history: bool = False,
    history_frequency: int = 10,
    use_gpu: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    # ─────────────────────── setup
    gpu = HAS_CUPY and use_gpu
    if use_gpu and not HAS_CUPY:
        logger.warning("GPU requested, but CuPy not found – falling back to CPU.")
    elif gpu:
        logger.info("Using GPU acceleration with CuPy.")

    E0, Emin, nu = 1.0, 1e-9, 0.3
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    obstacle_mask = (
        np.zeros((nely, nelx, nelz), dtype=bool)
        if obstacle_mask is None
        else obstacle_mask
    )
    design_nele = nele - obstacle_mask.sum()
    logger.debug(f"design elements: {design_nele}/{nele}")

    # History dict
    history = (
        {"density_history": [], "iteration_history": [], "compliance_history": []}
        if save_history
        else None
    )

    # Force / supports
    F = build_force_vector(nelx, nely, nelz, ndof, force_field)
    freedofs0, _ = build_supports(nelx, nely, nelz, ndof, support_mask)

    # Element stiffness
    KE = lk_H8(nu)
    edofMat, iK, jK = build_edof(nelx, nely, nelz)
    iK0, jK0 = iK - 1, jK - 1

    # Solver
    solver_func, solver_name = get_solver(use_gpu)
    logger.info(f"Linear solver: {solver_name}")

    # GPU transfers
    if gpu:
        KE_gpu = cp.asarray(KE)
        iK0_gpu, jK0_gpu = cp.asarray(iK0), cp.asarray(jK0)
        F_gpu = cp.asarray(F)
        freedofs0_gpu = cp.asarray(freedofs0)
        obstacle_gpu = cp.asarray(obstacle_mask)
        U_gpu = cp.zeros(ndof)

    # Filter
    H, Hs = build_filter(nelx, nely, nelz, rmin)
    if gpu:
        H_gpu = cusp.csr_matrix(
            (cp.asarray(H.data), cp.asarray(H.indices), cp.asarray(H.indptr)),
            shape=H.shape,
        )
        Hs_gpu = cp.asarray(Hs)

    # Initial design
    if gpu:
        x_gpu = cp.full((nely, nelx, nelz), volfrac)
        x_gpu[obstacle_gpu] = 0.0
        xPhys_gpu = apply_filter(H_gpu, x_gpu, Hs_gpu, x_gpu.shape, use_gpu=True)
        xPhys_gpu[obstacle_gpu] = 0.0
    else:
        x = np.full((nely, nelx, nelz), volfrac)
        x[obstacle_mask] = 0.0
        xPhys = (H * x.ravel(order="F") / Hs).reshape((nely, nelx, nelz), order="F")
        xPhys[obstacle_mask] = 0.0

    # ─────────────────────── sparsity pattern + scatter map
    logger.debug("Building global stiffness pattern & scatter map")
    i_unique, j_unique, dup2uniq = _make_scatter_map(iK0, jK0, ndof)

    if gpu:
        dup2uniq_gpu = cp.asarray(dup2uniq)
        K_gpu = cusp.csr_matrix(
            (cp.zeros(len(i_unique)), (cp.asarray(i_unique), cp.asarray(j_unique))),
            shape=(ndof, ndof),
        )
    else:
        K = sp.csr_matrix((np.zeros(len(i_unique)), (i_unique, j_unique)), shape=(ndof, ndof))
        # Physics evaluator (the swappable seam): assembles K(xPhys), solves, and returns
        # (c, dc, dv). The CPU loop below calls it; the GPU branch keeps its own inline
        # physics for now. Returns the clean dv (no OC +1e-9 guard) — added at the OC call.
        cpu_evaluator = ComplianceEvaluator(
            KE=KE,
            dup2uniq=dup2uniq,
            K=K,
            freedofs0=freedofs0,
            F=F,
            ndof=ndof,
            solver_func=solver_func,
            edofMat=edofMat,
            shape=(nely, nelx, nelz),
            H=H,
            Hs=Hs,
            obstacle_mask=obstacle_mask,
            E0=E0,
            Emin=Emin,
        )

    # ─────────────────────── main loop
    loop, change, c_prev = 0, 1.0, np.inf
    if save_history:
        # Cap the interval at 500 so even long runs save frames, while still honouring
        # a smaller requested frequency. `max(...)` here forced a *minimum* interval of
        # 500, so any run converging in <500 iterations saved only the first and last
        # frame (issue #8: "GIF animation limited to 2 frames").
        history_frequency = min(history_frequency, 500)
        if gpu:
            history["density_history"].append(cp.asnumpy(xPhys_gpu))
        else:
            history["density_history"].append(xPhys.copy())
        history["iteration_history"].append(0)
        history["compliance_history"].append(0.0)

    while change > tolx and loop < maxloop:
        loop += 1
        t0 = time.time()

        # ================================================= GPU
        if gpu:
            # Element-wise stiffness coefficients
            stiff_gpu = Emin + (xPhys_gpu.ravel(order="F") ** penal) * (E0 - Emin)
            elem_vals_gpu = cp.kron(stiff_gpu, KE_gpu.ravel())  # 576×nele

            # Scatter-add into CSR.data
            K_gpu.data.fill(0.0)
            cp.add.at(K_gpu.data, dup2uniq_gpu, elem_vals_gpu)

            # Solve
            Kff_gpu = K_gpu[freedofs0_gpu, :][:, freedofs0_gpu]
            Uf_gpu = solver_func(Kff_gpu, F_gpu[freedofs0_gpu])
            U_gpu.fill(0)
            U_gpu[freedofs0_gpu] = Uf_gpu

            # Compliance & sensitivities
            ce_flat_gpu = element_compliance(U_gpu, cp.asarray(edofMat), KE_gpu)
            ce_gpu = ce_flat_gpu.reshape(nely, nelx, nelz, order="F")
            c = cp.sum((Emin + xPhys_gpu ** penal * (E0 - Emin)) * ce_gpu).item()

            dc_gpu = -penal * (E0 - Emin) * xPhys_gpu ** (penal - 1) * ce_gpu
            dv_gpu = cp.ones_like(xPhys_gpu)

            dc_gpu = apply_filter(H_gpu, dc_gpu, Hs_gpu, xPhys_gpu.shape, use_gpu=True)
            dv_gpu = apply_filter(H_gpu, dv_gpu, Hs_gpu, xPhys_gpu.shape, use_gpu=True)
            dc_gpu[obstacle_gpu] = dv_gpu[obstacle_gpu] = 0.0
            dv_gpu += 1e-9

            xnew_gpu, change = optimality_criteria_update(
                x_gpu,
                dc_gpu,
                dv_gpu,
                volfrac,
                H_gpu,
                Hs_gpu,
                nele,
                obstacle_gpu,
                design_nele,
                use_gpu=True,
            )
            xnew_gpu[obstacle_gpu] = 0.0
            xPhys_gpu = apply_filter(H_gpu, xnew_gpu, Hs_gpu, xnew_gpu.shape, use_gpu=True)
            xPhys_gpu[obstacle_gpu] = 0.0
            x_gpu = xnew_gpu
            current_vol = cp.mean(xPhys_gpu[~obstacle_gpu]).item()

        # ================================================= CPU
        else:
            c, dc, dv = cpu_evaluator.evaluate(xPhys, penal)
            dv += 1e-9  # OC divides by dv (oc_update); MMA/NLopt would use the clean dv

            xnew, change = optimality_criteria_update(
                x,
                dc,
                dv,
                volfrac,
                H,
                Hs,
                nele,
                obstacle_mask,
                design_nele,
                use_gpu=False,
            )
            xnew[obstacle_mask] = 0.0
            xPhys = (H * xnew.ravel(order="F") / Hs).reshape((nely, nelx, nelz), order="F")
            xPhys[obstacle_mask] = 0.0
            x = xnew
            current_vol = xPhys[~obstacle_mask].mean()

        # ------------------------------------------------ logging / history
        c_delta, c_prev = c - c_prev, c
        iter_t = time.time() - t0

        if save_history and (loop % history_frequency == 0 or change <= tolx):
            history["density_history"].append(
                cp.asnumpy(xPhys_gpu) if gpu else xPhys.copy()
            )
            history["iteration_history"].append(loop)
            history["compliance_history"].append(c)

        logger.info(
            f"Iter {loop:4d}: Obj={c:9.4f}, ΔObj={c_delta:9.4f}, "
            f"Vol={current_vol:6.3f}, change={change:6.3f}, "
            f"time={iter_t:5.2f}s"
        )

    # ─────────────────────── final output
    final_xPhys = cp.asnumpy(xPhys_gpu) if gpu else xPhys
    if save_history:
        return final_xPhys, history
    return final_xPhys