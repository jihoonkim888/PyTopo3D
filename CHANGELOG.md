# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-16

### Fixed

- **Force-direction x/y transpose.** `force_field` components `[Fx, Fy, Fz]` now act along
  the correct axes. The `build_edof` rewrite in the commit that added custom `force_field`
  support (the same change that moved the default load from `-y` to `-z`) silently swapped
  the H8 element's first two local axes, so a user-supplied `Fx` acted along the **y**
  (`nely`) axis and `Fy` along **x**. This was invisible to the default `-z` load, every
  shipped cantilever example, and the golden master (all `-z` / `x`<->`y`-symmetric), so only
  `force_field` loads with distinct `Fx` vs `Fy` were affected.

  **Impact:** non-symmetric `force_field` loads carrying both x and y components change. The
  default load, `-z`-only loads, supports, obstacle/geometry positions, and the golden-master
  result are byte-identical (verified: max abs difference ~4e-13). The README's `force_field`
  cantilever recipe now behaves as documented (a `-y` tip load bending in the x-y plane). To
  reproduce the old (transposed) behavior, pin the previous release.

### Added

- `tests/test_force_direction.py` — locks the public `(x, y, z)` force-direction convention
  (on an x-long beam, `Fx` is axial while `Fy`/`Fz` bend), so the transpose cannot regress.

## [0.2.2] - 2026-06-15

A maintenance release with no functional code changes. Cut as a one-off — outside the
normal "milestone or urgent fix" cadence — to bootstrap Zenodo archival (the project's
first software DOI) and to ship the documentation and project-metadata work that has
accumulated since 0.2.1.

### Added

- `CITATION.cff` (GitHub's "Cite this repository") and status badges in the README.
- `AGENTS.md` / `CLAUDE.md` contributor & agent guide, and a `RELEASING.md` release guide.
- `.zenodo.json` so that published GitHub Releases are archived to Zenodo with a citable DOI.

### Fixed

- README citation: a malformed BibTeX entry (missing comma after the cite key) and the
  prose publication year (2024 → 2025).

### Removed

- **Removed the two dormant, broken alternate entry points** `pytopo3d/runners/pipeline.py`
  (`run_optimization_pipeline` / `run_batch_optimization`) and `pytopo3d/cli/command.py`. Both
  carried call signatures that no longer matched their helpers, by different routes: `pipeline.py`
  still passed a single `args` object positionally to functions that now take explicit keyword
  scalars, and `command.py` omitted the now-required `loads_array` / `constraints_array` arguments
  to `visualize_initial_setup` — so each raised a `TypeError` when invoked. Nothing imported them,
  they were not wired to a console script (`pyproject.toml` has no `[project.scripts]`), and
  `main.py` is the only maintained entry point, so no working code path is affected and the
  numerical core (golden-master result) is unchanged. This resolves the 0.2.1 "Known limitations"
  note about these files (issue #20). A programmatic batch / parameter-sweep API, if wanted later,
  should be written fresh against `main.py`'s current flow rather than revived from these.

## [0.2.1]

### Fixed

- **An STL design space now sets the working resolution (#16).** When a design space is
  loaded from an STL, its voxelized grid (`mesh size ÷ pitch`) defines `nelx`/`nely`/`nelz`;
  the CLI used to keep the `--nelx/--nely/--nelz` arguments instead, so unless they happened
  to equal the voxelized shape the run crashed with a shape mismatch (e.g. combining the STL
  with default dims or a JSON obstacle). `main.py` now adopts the STL-derived resolution and
  **warns** that `--nelx/--nely/--nelz` are ignored when `--design-space-stl` is set, and
  obstacles are built against the design-space grid. Pure-array and non-STL CLI runs are
  unaffected.

### Known limitations

- The auto-generated experiment-directory name reflects the requested CLI dims, not the
  STL-derived ones (the run itself and the saved `config.json` use the correct dims).
- The dormant `pytopo3d.cli.command` and `pytopo3d.runners.pipeline` entry points have
  pre-existing, unrelated breakage (stale call signatures) and are not wired to a console
  script; they are tracked separately, not addressed here.

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
