"""STL axis-convention tests.

Lock the ``(x, y, z)`` <-> ``(nely, nelx, nelz)`` mapping at the STL boundary: an STL's
x-axis must map to the domain's x (``nelx``) on import, and back to the mesh's x on export.
These guard the 0.2.0 fix from silently regressing to the old transposed behaviour. They
need ``trimesh`` (import) and ``scikit-image`` (export), both core dependencies; the test
is skipped if either is unavailable, mirroring the HAS_CUPY guard used for the GPU tests.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("skimage")

from pytopo3d.utils.export import voxel_to_stl, voxel_to_stl_tpms
from pytopo3d.utils.import_design_space import stl_to_design_space

# An intentionally asymmetric box so every axis has a distinct length:
# x is the longest, y the shortest, z in between.
BOX_EXTENTS = [24, 6, 12]


def test_stl_import_maps_x_to_nelx(tmp_path):
    """A box longer in x than y must voxelize to a mask whose nelx axis is the long one."""
    stl = tmp_path / "xlong.stl"
    trimesh.creation.box(extents=BOX_EXTENTS).export(stl)

    mask = stl_to_design_space(str(stl), pitch=1.0)
    nely, nelx, nelz = mask.shape

    # x is the longest CAD dimension -> nelx (axis 1) is the largest count;
    # y is the shortest -> nely (axis 0) is the smallest.
    assert nelx > nelz > nely


def test_stl_export_maps_nelx_to_x():
    """Exporting a (nely, nelx, nelz) block must put the long nelx axis on the mesh's x."""
    # nelx (axis 1) is the long axis; nely (axis 0) is the short one.
    block = np.ones((6, 24, 12), dtype=float)
    mesh = voxel_to_stl(block, output_file=None, smooth_mesh=False, fix_mesh=False)

    ex, ey, ez = mesh.extents  # trimesh extents are in CAD (x, y, z) order
    assert ex > ez > ey


def test_export_places_content_on_correct_axis():
    """Content-level check: a cube-shaped array carries no shape information, so only a
    correct axis mapping puts a bar that runs along nelx onto the mesh's x-axis. Guards
    against a 'right shape, wrong content' regression (e.g. reshape instead of swapaxes)
    that the solid-box tests above cannot catch."""
    block = np.zeros((6, 6, 6), dtype=float)
    block[0:2, :, 0:2] = 1.0  # solid bar along the nelx axis, at the y=0,z=0 corner
    mesh = voxel_to_stl(block, output_file=None, smooth_mesh=False, fix_mesh=False)
    ex, ey, ez = mesh.extents
    assert ex > 2 * ey and ex > 2 * ez


def test_tpms_export_places_content_on_correct_axis():
    """voxel_to_stl_tpms applies the same axis swap; a bar along nelx must export x-long.
    Cube-shaped array so only the axis mapping (not the shape) can satisfy this."""
    block = np.zeros((6, 6, 6), dtype=float)
    block[0:2, :, 0:2] = 1.0
    mesh = voxel_to_stl_tpms(block, output_stl_path=None, res_factor=2, plot_mapping=False)
    ex, ey, ez = mesh.extents
    assert ex > ey and ex > ez


def test_stl_round_trip_preserves_orientation(tmp_path):
    """Import then export must return a mesh with the same axis-aspect order (no net flip).

    Note: this only checks import/export *consistency*. It cannot catch a regression that
    drops the swap in BOTH directions (those cancel out, leaving the round trip unchanged) --
    the standalone import/export tests above cover that case.
    """
    stl = tmp_path / "block.stl"
    trimesh.creation.box(extents=BOX_EXTENTS).export(stl)

    mask = stl_to_design_space(str(stl), pitch=1.0).astype(float)
    mesh = voxel_to_stl(mask, output_file=None, smooth_mesh=False, fix_mesh=False)

    ex, ey, ez = mesh.extents
    # Same ordering as the original box extents (x > z > y).
    assert ex > ez > ey


def test_axis_change_notice_fires_once(tmp_path):
    """The 0.2.0 transition notice must fire on first STL use, then stay silent."""
    import warnings

    import pytopo3d.utils.axis_convention as ac

    stl = tmp_path / "notice.stl"
    trimesh.creation.box(extents=BOX_EXTENTS).export(stl)

    original = ac._WARNED
    ac._WARNED = False  # reset session-once flag for a deterministic check
    try:
        with pytest.warns(UserWarning, match="axis convention"):
            stl_to_design_space(str(stl), pitch=1.0)

        # A second STL operation must NOT warn again (once per session).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stl_to_design_space(str(stl), pitch=1.0)
    finally:
        ac._WARNED = original  # leave no global state behind for other tests
