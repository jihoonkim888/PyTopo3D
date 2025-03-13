"""
Main optimizer for 3D topology optimization.

This module contains the main top3d function that performs the optimization.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from pytopo3d.core.compliance import element_compliance
from pytopo3d.utils.assembly import build_edof, build_force_vector, build_supports
from pytopo3d.utils.filter import build_filter
from pytopo3d.utils.logger import get_logger
from pytopo3d.utils.oc_update import optimality_criteria_update
from pytopo3d.utils.solver import solver, solver_name
from pytopo3d.utils.stiffness import lk_H8
from pytopo3d.visualization.display import display_3D

# Create a module-specific logger
logger = get_logger(__name__)


def top3d(
    nelx,
    nely,
    nelz,
    volfrac,
    penal,
    rmin,
    disp_thres,
    obstacle_mask=None,
    tolx: float = 0.01,
    maxloop: int = 2000,
):
    """
    Accelerated 3D Topology Optimization with optional obstacle region.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    volfrac : float
        Volume fraction target for the design domain (excludes obstacles).
    penal : float
        Penalization exponent for SIMP.
    rmin : float
        Filter radius (for sensitivity filtering).
    disp_thres : float
        Display threshold for 3D visualization. Elements with density >
        disp_thres are plotted.
    obstacle_mask : ndarray of bool, shape (nely, nelx, nelz), optional
        If provided, wherever obstacle_mask==True, we force the density to 0
        (no material can occupy those cells). They are excluded from the
        volume constraint.
    tolx : float, optional
        Convergence tolerance for optimization. The algorithm stops when the
        maximum change in design variables is less than this value. Default is 0.01.
    maxloop : int, optional
        Maximum number of optimization iterations. Default is 2000.

    Returns
    -------
    ndarray
        Optimized physical density array.
    """

    # ---------------------------
    # USER-DEFINED PARAMETERS
    displayflag = False  # Only final display by default

    E0 = 1.0
    Emin = 1e-9
    nu = 0.3

    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    # If no obstacle mask is provided, make it all False
    # (meaning no obstacle anywhere).
    if obstacle_mask is None:
        obstacle_mask = np.zeros((nely, nelx, nelz), dtype=bool)

    # Number of "free" design elements (excluded obstacles)
    design_nele = nele - np.count_nonzero(obstacle_mask)
    logger.debug(f"Design elements: {design_nele}/{nele} ({design_nele / nele:.1%})")

    # ---------------------------
    # Build force vector & supports
    logger.debug("Building force vector and supports")
    F = build_force_vector(nelx, nely, nelz, ndof)
    freedofs0, fixeddof0 = build_supports(nelx, nely, nelz, ndof)
    U = np.zeros(ndof)

    # ---------------------------
    # Element stiffness matrix (24x24)
    logger.debug("Computing element stiffness matrix")
    KE = lk_H8(nu)

    # ---------------------------
    # Build edofMat, and global indexing arrays iK, jK
    logger.debug("Building element DOF matrix and indexing arrays")
    edofMat, iK, jK = build_edof(nelx, nely, nelz)
    # Precompute the mapping from the full (iK,jK) to unique global indices.
    pairs = np.vstack(((iK - 1), (jK - 1))).T  # shape (576*nele, 2)
    unique_pairs, assembly_map = np.unique(pairs, axis=0, return_inverse=True)
    nUnique = unique_pairs.shape[0]
    K_template = sp.csr_matrix(
        (np.zeros(nUnique, dtype=np.float64), (unique_pairs[:, 0], unique_pairs[:, 1])),
        shape=(ndof, ndof),
    )

    # ---------------------------
    # Build filter matrix H (once) and Hs
    logger.debug(f"Building filter with radius {rmin}")
    H, Hs = build_filter(nelx, nely, nelz, rmin)

    # ---------------------------
    # INITIALIZE design variable
    # Start with uniform distribution = volfrac in the *design domain*.
    logger.debug(f"Initializing design with volume fraction {volfrac}")
    x = np.full((nely, nelx, nelz), volfrac)
    # Force obstacle elements to 0 from the start:
    x[obstacle_mask] = 0.0
    # Apply filter
    xPhys = (H * x.ravel(order="F") / Hs).reshape((nely, nelx, nelz), order="F")

    loop = 0
    change = 1.0
    # Initialize previous objective value for delta tracking
    c_prev = float("inf")

    logger.info(f"Using solver: {solver_name}")
    logger.info(
        f"Starting optimization with tolerance {tolx} and max iterations {maxloop}"
    )
    pbar = tqdm(
        total=maxloop, desc="Optimizing", ncols=100
    )  # Increased width for more info

    # ---------------------------
    # START ITERATION
    while change > tolx and loop < maxloop:
        loop += 1
        t_start = time.time()

        # 1) Assemble stiffness values for each element
        xFlat = xPhys.ravel(order="F")  # length = nele
        stiff_vals = Emin + (xFlat**penal) * (E0 - Emin)  # shape (nele,)
        # Each element => 576 values; sK_full has shape (576*nele,)
        sK_full = np.kron(stiff_vals, KE.ravel())

        # 2) Sum duplicates via precomputed assembly_map:
        K_data = np.bincount(assembly_map, weights=sK_full, minlength=nUnique)
        K_template.data[:] = K_data

        # 3) Extract reduced K and solve
        K_ff = K_template[freedofs0, :][:, freedofs0]
        F_f = F[freedofs0]
        U_f = solver(K_ff, F_f)
        U[:] = 0.0
        U[freedofs0] = U_f

        # 4) Compute compliance and sensitivities
        ce_flat = element_compliance(U, edofMat, KE)  # shape (nele,)
        ce = ce_flat.reshape(nely, nelx, nelz, order="F")
        c = ((Emin + xPhys**penal * (E0 - Emin)) * ce).sum()

        # Calculate delta of objective function
        c_delta = c - c_prev
        c_prev = c

        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones_like(xPhys)

        # 5) Filter sensitivities
        dc = (H * (dc.ravel(order="F") / Hs)).reshape((nely, nelx, nelz), order="F")
        dv = (H * (dv.ravel(order="F") / Hs)).reshape((nely, nelx, nelz), order="F")

        # Force zero sensitivities in obstacle region, so they remain at x=0.
        dc[obstacle_mask] = 0.0
        dv[obstacle_mask] = 0.0

        # 6) Optimality Criteria update via bisection
        xnew, change = optimality_criteria_update(
            x, dc, dv, volfrac, H, Hs, nele, obstacle_mask, design_nele
        )

        # Force obstacle region to remain zero
        xnew[obstacle_mask] = 0.0

        # Recompute physical densities
        xPhys = (H * xnew.ravel(order="F") / Hs).reshape((nely, nelx, nelz), order="F")
        xPhys[obstacle_mask] = 0.0

        x = xnew

        iter_time = time.time() - t_start

        # Compute the volume fraction only over the design domain
        current_volume_fraction = xPhys[~obstacle_mask].mean()

        # Log detailed iteration information at DEBUG level
        logger.debug(
            f"Iteration {loop}: Obj={c:.4f}, ΔObj={c_delta:.4f}, Vol={current_volume_fraction:.3f}, "
            f"change={change:.3f}, time={iter_time:.2f}s"
        )

        pbar.update(1)
        pbar.set_postfix(
            {
                "Obj": f"{c:.4f}",
                "ΔObj": f"{c_delta:.4e}",
                "Vol": f"{current_volume_fraction:.3f}",
                "change": f"{change:.3f}",
            }
        )

        if displayflag:
            plt.clf()
            display_3D(xPhys, disp_thres)
            plt.pause(0.01)

    # Close progress bar
    pbar.close()

    # Log final results
    if loop >= maxloop:
        logger.warning(
            f"Optimization reached maximum iterations ({maxloop}) without converging"
        )
    else:
        logger.info(f"Optimization converged in {loop} iterations")

    logger.info(f"Final objective value: {c:.6f}")
    logger.info(
        f"Final volume fraction: {current_volume_fraction:.6f} (target: {volfrac:.6f})"
    )

    return xPhys
