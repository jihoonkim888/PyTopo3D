"""R2 -- closed-form absolute anchor for the physics (plan P4.1.2).

The relative checks (gradient consistency in R1, cross-solver agreement later) are blind
to a *shared/systematic* error: a globally mis-scaled stiffness keeps compliance and its
gradient mutually consistent, so R1 still passes. This test pins the physics to analytic
ground truth.

Axis convention: after the 0.3.0 fix to `build_edof`, the element's local axes align with
the grid axes -- a node at grid index ``(ix, iy, iz)`` sits at physical ``(x, y, z) =
(ix, iy, iz)`` and its DOFs ``3*nid + {0,1,2}`` are displacements along ``(x, y, z)``.
(`test_force_direction.py` locks the user-facing half of this; before the fix x and y were
transposed.)

Two anchors:
  * energy patch test -- imposes the analytic uniaxial-stress field and checks the strain
    energy ``U = 1/2 * V * sigma:eps`` exactly. Anchors ``lk_H8`` + assembly; no solve.
  * uniaxial bar -- runs the full ``ComplianceEvaluator`` (assemble -> solve -> compliance)
    on a solid Nx1x1 bar (long along x) under uniaxial tension with symmetry (roller) BCs,
    a constant-strain patch the trilinear hex represents exactly. Closed form:
    ``C = P^2 L /(A E) = N``.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pytopo3d.core.evaluator import ComplianceEvaluator
from pytopo3d.core.optimizer import _make_scatter_map
from pytopo3d.utils.assembly import build_edof
from pytopo3d.utils.filter import build_filter
from pytopo3d.utils.solver import get_solver
from pytopo3d.utils.stiffness import lk_H8

NU = 0.3


def _assemble_solid_K(nelx, nely, nelz):
    """Assemble the global stiffness of the fully-solid grid (density == 1)."""
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    KE = lk_H8(NU)
    _, iK, jK = build_edof(nelx, nely, nelz)
    i_u, j_u, dup2uniq = _make_scatter_map(iK - 1, jK - 1, ndof)
    K = sp.csr_matrix((np.zeros(len(i_u)), (i_u, j_u)), shape=(ndof, ndof))
    np.add.at(K.data, dup2uniq, np.kron(np.ones(nelx * nely * nelz), KE.ravel()))
    return K, ndof


def test_single_element_energy_patch():
    """A uniaxial-stress field on one solid unit element has the exact analytic energy.

    eps = (1, -nu, -nu) gives sigma = (1, 0, 0); energy density 1/2*sigma:eps = 1/2 over a
    unit volume, so U = 1/2 exactly (a constant-strain patch test). Physical axes = grid axes.
    """
    K, ndof = _assemble_solid_K(1, 1, 1)
    Kd = K.toarray()

    def grid(nid):  # nid = iy + ix*2 + iz*4  ->  (ix, iy, iz)
        return ((nid // 2) % 2, nid % 2, nid // 4)

    u = np.zeros(ndof)
    for nid in range(8):
        ix, iy, iz = grid(nid)
        u[3 * nid:3 * nid + 3] = [1.0 * ix, -NU * iy, -NU * iz]  # physical (x, y, z) = (ix, iy, iz)

    energy = 0.5 * u @ Kd @ u
    assert abs(energy - 0.5) <= 1e-12, f"strain energy {energy} != closed-form 0.5"


def _uniaxial_bar_evaluator(N):
    """Solid Nx1x1 bar along x, symmetry BCs, unit axial load P=1.

    A = 1 (one element across each lateral axis), L = N, E = 1 -> C = N exactly.
    """
    nelx, nely, nelz = N, 1, 1
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    KE = lk_H8(NU)
    edofMat, iK, jK = build_edof(nelx, nely, nelz)
    i_u, j_u, dup2uniq = _make_scatter_map(iK - 1, jK - 1, ndof)
    K = sp.csr_matrix((np.zeros(len(i_u)), (i_u, j_u)), shape=(ndof, ndof))
    H, Hs = build_filter(nelx, nely, nelz, 1.5)

    def nid(ix, iy, iz):
        return iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)

    fixed = set()
    for iy in range(nely + 1):
        for iz in range(nelz + 1):
            fixed.add(3 * nid(0, iy, iz) + 0)        # x=0 face: fix u_x (DOF 0)
    for ix in range(nelx + 1):
        for iz in range(nelz + 1):
            fixed.add(3 * nid(ix, 0, iz) + 1)        # y=0 face: fix u_y (DOF 1)
    for ix in range(nelx + 1):
        for iy in range(nely + 1):
            fixed.add(3 * nid(ix, iy, 0) + 2)        # z=0 face: fix u_z (DOF 2)
    freedofs0 = np.setdiff1d(np.arange(ndof), np.array(sorted(fixed)))

    F = np.zeros(ndof)
    n_load = (nely + 1) * (nelz + 1)                 # nodes on the loaded end face (x = L)
    for iy in range(nely + 1):
        for iz in range(nelz + 1):
            F[3 * nid(nelx, iy, iz) + 0] = 1.0 / n_load   # total +x force P = 1, consistent split

    ev = ComplianceEvaluator(
        KE=KE, dup2uniq=dup2uniq, K=K, freedofs0=freedofs0, F=F, ndof=ndof,
        solver_func=get_solver(False)[0], edofMat=edofMat, shape=(nely, nelx, nelz),
        H=H, Hs=Hs, obstacle_mask=np.zeros((nely, nelx, nelz), dtype=bool),
    )
    return ev, (nely, nelx, nelz)


@pytest.mark.parametrize("N", [1, 2, 4])
def test_uniaxial_bar_compliance_matches_closed_form(N):
    ev, shape = _uniaxial_bar_evaluator(N)
    c, _, _ = ev.evaluate(np.ones(shape), penal=3.0)   # solid: x**penal = 1 for any penal
    assert abs(c - N) <= 1e-9 * N, (
        f"compliance {c} != closed-form {N} for an {N}-element uniaxial bar"
    )
