"""Physics evaluator: compliance and its design sensitivities.

This is the swappable seam (see ``docs/literature-reviews/0.3.0-implementation-plan.md``,
P3) between the finite-element physics and the optimizer. Given the physical (filtered)
density field ``xPhys`` it assembles ``K(xPhys)``, solves ``K U = F``, and returns the
compliance objective together with its sensitivities. Both integration modes call it --
the framework-owns-loop OC update today, and the solver-owns-loop NLopt MMA backend later
-- so no optimizer owns the physics and the optimizer kernel stays swappable.

CPU path only. The optimizer keeps its own CuPy GPU branch for now; a GPU evaluator
follows when the two loops are unified.

The returned ``dv`` omits OC's ``+1e-9`` division guard (the OC call site adds that itself).
NOTE: with obstacles it is *not* yet the exact volume-constraint gradient -- obstacle cells
are zeroed *after* the filter rather than excluded before it, so their sensitivity leaks into
neighbours. This is carried over verbatim from the original loop and is fine for the OC
bisection, but a future MMA/NLopt backend should exclude obstacle cells before the filter
chain rule for obstacle problems.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp

from pytopo3d.core.compliance import element_compliance


class ComplianceEvaluator:
    """Assemble, solve, and return ``(c, dc, dv)`` on the CPU.

    Static problem data (element stiffness, the CSR scatter map and its preallocated
    structure, loads, supports, the density filter, and the obstacle mask) is built once
    and held. ``evaluate`` is then called per iteration with the current physical density
    and penalization.
    """

    def __init__(
        self,
        *,
        KE: np.ndarray,
        dup2uniq: np.ndarray,
        K: sp.csr_matrix,
        freedofs0: np.ndarray,
        F: np.ndarray,
        ndof: int,
        solver_func,
        edofMat: np.ndarray,
        shape: Tuple[int, int, int],  # (nely, nelx, nelz) -- y first, per the array layout
        H: sp.csr_matrix,
        Hs: np.ndarray,
        obstacle_mask: np.ndarray,
        E0: float = 1.0,
        Emin: float = 1e-9,
    ) -> None:
        self.KE = KE
        self.dup2uniq = dup2uniq
        self.K = K
        self.freedofs0 = freedofs0
        self.F = F
        self.ndof = ndof
        self.solver_func = solver_func
        self.edofMat = edofMat
        self.shape = shape
        self.H = H
        self.Hs = Hs
        self.obstacle_mask = obstacle_mask
        self.E0 = E0
        self.Emin = Emin

    def evaluate(
        self, xPhys: np.ndarray, penal: float
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return ``(c, dc, dv)`` for physical density ``xPhys`` at penalization ``penal``.

        ``dc`` and ``dv`` are filtered (the density-filter chain rule) and zeroed on
        obstacle elements. ``dv`` is the clean volume sensitivity (no OC ``+1e-9`` guard).
        """
        nely, nelx, nelz = self.shape
        E0, Emin = self.E0, self.Emin

        # Assemble the global stiffness, reusing the preallocated CSR structure.
        stiff = Emin + (xPhys.ravel(order="F") ** penal) * (E0 - Emin)
        elem_vals = np.kron(stiff, self.KE.ravel())
        self.K.data[:] = 0.0
        np.add.at(self.K.data, self.dup2uniq, elem_vals)

        # Solve on the free DOFs.
        Kff = self.K[self.freedofs0, :][:, self.freedofs0]
        Uf = self.solver_func(Kff, self.F[self.freedofs0])
        U = np.zeros(self.ndof)
        U[self.freedofs0] = Uf

        # Compliance and raw (unfiltered) sensitivities.
        ce = element_compliance(U, self.edofMat, self.KE).reshape(
            nely, nelx, nelz, order="F"
        )
        c = ((Emin + xPhys ** penal * (E0 - Emin)) * ce).sum()
        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones_like(xPhys)

        # Density-filter chain rule, then zero obstacle elements.
        dc = (self.H * (dc.ravel(order="F") / self.Hs)).reshape(
            (nely, nelx, nelz), order="F"
        )
        dv = (self.H * (dv.ravel(order="F") / self.Hs)).reshape(
            (nely, nelx, nelz), order="F"
        )
        dc[self.obstacle_mask] = 0.0
        dv[self.obstacle_mask] = 0.0
        return c, dc, dv
