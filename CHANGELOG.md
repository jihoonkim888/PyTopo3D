# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-13

### Fixed (breaking)

- **STL import and export now use a consistent `x`/`y` axis convention.** `trimesh`
  voxel grids come in CAD axis order `(x, y, z)`, but they were stored directly as the
  solver's `(nely, nelx, nelz)` array, so an STL's x-axis became the domain's y-axis.
  For any non-cube STL this placed the default supports/loads and JSON obstacles on the
  wrong faces, and made the matplotlib view disagree with the exported STL. STL import
  (`stl_to_design_space`) and STL export (`voxel_to_stl`, `voxel_to_stl_tpms`) now swap
  the first two axes so an STL's x-axis maps to the domain's x-axis (`nelx`), matching
  the `nelx`/`nely` arguments, JSON obstacle `center = [x, y, z]`, and `force_field`
  everywhere else in the library.

  **Impact:** any workflow that imports or exports STL is affected. The orientation now
  follows the `x = nelx` convention, so a result is identical to 0.1.x only for parts
  that are symmetric under an `x` <-> `y` swap (a cube *bounding box* is not enough — an
  asymmetric part inside it still transposes). Non-STL (pure-array) usage — passing
  `nelx`/`nely`/`nelz` and NumPy masks directly, with no STL — is unchanged: the
  numerical core and the golden-master result are byte-identical.

  **Migration:** to reproduce pre-0.2.0 results, pin `pytopo3d==0.1.2`. As a transitional
  aid, 0.2.0 emits a one-time notice the first time an STL is imported or exported; it is
  removed in 0.3.0.

### Changed

- Corrected the misleading default-load comment in `build_force_vector`: the default
  load is a downward (`-z`) force along the line `x = nelx, z = 0` spanning all `y`, not
  "`y = 0`".

### Removed

- Removed the unused `calculate_boundary_positions` helper — dead code that still carried
  the old MATLAB `y`-flip visualization convention.

### Added

- A "Coordinate system and axis conventions" section in the README.
- Documentation of the default load direction (`-z`, z-up), how it differs from MATLAB
  `top3d`'s `-y`, and a `force_field` recipe for reproducing the canonical `top3d` cantilever.
- A one-time runtime notice on first STL import/export about the axis-convention change
  (transitional; removed in 0.3.0).
- `tests/test_stl_orientation.py`, which locks the STL import/export axis convention.

## [0.1.2]

- Last release before the coordinate-convention fix. STL design spaces are loaded with
  their `x` and `y` axes transposed relative to the rest of the API.
