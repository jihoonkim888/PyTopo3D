"""Regression tests for #16: an STL design space defines the working grid.

`load_geometry_data` voxelizes the STL to a resolution set by the mesh + pitch, which can
differ from the nelx/nely/nelz arguments. The returned masks must all share that resolution
so downstream code (which adopts ``design_space_mask.shape``) never hits a shape mismatch --
the crash #16 reported when an STL was combined with default dims or a JSON obstacle. Needs
``trimesh``; skipped if unavailable (mirrors the other STL tests).
"""

from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")

from pytopo3d.preprocessing.geometry import load_geometry_data

# Deliberately mismatched: the STL voxelizes to dims far from these args.
ARGS_NELX, ARGS_NELY, ARGS_NELZ = 60, 30, 20
OBSTACLE_CFG = "examples/obstacles_config_cylinder.json"


def _write_box_stl(path):
    # 36 x 9 x 9 -> voxelizes to a non-cube grid that is nowhere near (30, 60, 20).
    trimesh.creation.box(extents=[36, 9, 9]).export(path)


def test_stl_design_space_overrides_mismatched_dims(tmp_path):
    """An STL voxelizing to dims != args must still return self-consistent masks."""
    stl = tmp_path / "bar.stl"
    _write_box_stl(stl)

    design, obstacle, combined = load_geometry_data(
        nelx=ARGS_NELX, nely=ARGS_NELY, nelz=ARGS_NELZ,
        design_space_stl=str(stl), pitch=1.0,
    )
    # The STL set the grid, not the args.
    assert design.shape != (ARGS_NELY, ARGS_NELX, ARGS_NELZ)
    # Every mask shares the STL-derived grid -> downstream code cannot hit a shape mismatch.
    assert obstacle.shape == design.shape
    assert combined.shape == design.shape


def test_stl_with_obstacle_config_does_not_mismatch(tmp_path):
    """STL design space + JSON obstacle (the #16 crash site) must combine cleanly."""
    stl = tmp_path / "bar.stl"
    _write_box_stl(stl)
    assert Path(OBSTACLE_CFG).exists(), "example obstacle config is missing"

    design, obstacle, combined = load_geometry_data(
        nelx=ARGS_NELX, nely=ARGS_NELY, nelz=ARGS_NELZ,
        design_space_stl=str(stl), pitch=1.0, obstacle_config=OBSTACLE_CFG,
    )
    assert obstacle.shape == design.shape == combined.shape
    # The JSON obstacle actually placed material on the STL-derived grid.
    assert combined.sum() > 0
