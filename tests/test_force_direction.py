"""Locks the public (x, y, z) force-direction convention.

`force_field` component 0 ("Fx") must act along the physical x (= `nelx`) axis, component 1
along y, component 2 along z. A rewrite of `build_edof` in commit 75174dd silently swapped
the element's first two local axes, so an "Fx" load acted along the `nely` (y) axis instead
of x -- a regression invisible to the golden master (default load is -z, unaffected) and to
the STL-orientation tests (which check geometry positions, not force directions).

Test: on a beam long along x, an axial Fx load is stiff (small compliance) while transverse
Fy / Fz loads bend the beam (much larger compliance). If Fx is NOT the smallest, x and y are
transposed.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pytopo3d.utils.assembly import build_edof, build_force_vector, build_supports
from pytopo3d.utils.stiffness import lk_H8


def _beam_compliances():
    """Solid beam long along x (16x4x4), fixed at x=0; unit load per force component."""
    nelx, nely, nelz = 16, 4, 4
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    KE = lk_H8(0.3)
    edofMat, _, _ = build_edof(nelx, nely, nelz)
    d0 = edofMat - 1
    rows = np.repeat(d0, 24, axis=1).ravel()
    cols = np.tile(d0, (1, 24)).ravel()
    vals = np.tile(KE.ravel(), d0.shape[0])
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    free, _ = build_supports(nelx, nely, nelz, ndof, None)  # default: fix the x=0 face
    Kff = K[free][:, free]

    compliances = []
    for comp in range(3):
        ff = np.zeros((nely, nelx, nelz, 3))
        ff[:, nelx - 1, :, comp] = -1.0  # unit load on the far-x face, this component
        F = build_force_vector(nelx, nely, nelz, ndof, ff)
        u = np.zeros(ndof)
        u[free] = spsolve(Kff, F[free])
        compliances.append(float(F @ u))
    return compliances  # [Fx, Fy, Fz]


def test_fx_is_axial_fy_fz_are_transverse():
    cx, cy, cz = _beam_compliances()
    assert cx < 0.2 * cy and cx < 0.2 * cz, (
        f"force_field component 0 (Fx) must act along x (axial, stiff) on an x-long beam, "
        f"but compliances are Fx={cx:.1f}, Fy={cy:.1f}, Fz={cz:.1f}. If Fx is one of the "
        f"large (transverse) values, the x<->y force convention is transposed (regression)."
    )


def test_force_xy_notice_fires_once_for_fx_load():
    """An Fx/Fy force_field triggers the one-time 0.3.0 transpose-fix notice, once only."""
    import warnings

    import pytest

    import pytopo3d.utils.axis_convention as ac

    nelx, nely, nelz = 4, 3, 2
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    ff = np.zeros((nely, nelx, nelz, 3))
    ff[0, nelx - 1, 0, 0] = -1.0  # an Fx (component 0) load

    original = ac._FORCE_WARNED
    ac._FORCE_WARNED = False
    try:
        with pytest.warns(UserWarning, match="x/y transpose"):
            build_force_vector(nelx, nely, nelz, ndof, ff)
        # A second force_field call must NOT warn again (once per session).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_force_vector(nelx, nely, nelz, ndof, ff)
    finally:
        ac._FORCE_WARNED = original


def test_force_z_only_does_not_warn():
    """An Fz-only force_field is unaffected by the x/y fix, so it must stay silent."""
    import warnings

    import pytopo3d.utils.axis_convention as ac

    nelx, nely, nelz = 4, 3, 2
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    ff = np.zeros((nely, nelx, nelz, 3))
    ff[0, nelx - 1, 0, 2] = -1.0  # Fz only

    original = ac._FORCE_WARNED
    ac._FORCE_WARNED = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            build_force_vector(nelx, nely, nelz, ndof, ff)  # must not warn
    finally:
        ac._FORCE_WARNED = original
