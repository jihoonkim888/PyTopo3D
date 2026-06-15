# AGENTS.md

Guidance for AI coding agents (and human contributors) working in this repository. This is the single source of truth; `CLAUDE.md` just imports it.

## What this is

PyTopo3D is a 3D structural **topology optimization** framework implementing the **SIMP** (Solid Isotropic Material with Penalization) method in pure Python. It is a port and extension of the MATLAB `top3d` reference, adding obstacle regions, STL import/export, and optional CUDA (CuPy) GPU acceleration.

It is a **public** open-source package (published to PyPI, backed by an arXiv paper). Treat everything committed here as world-readable.

## Repository layout

Flat layout — the package and its packaging files both live at the repo root:

- `pytopo3d/` — the importable package
  - `core/` — the optimizer entry point `top3d` (`optimizer.py`) and compliance
  - `cli/` — argument parsing (`parser.py`); `command.py` is a stale alternate driver (see **Entry points** below)
  - `preprocessing/` — geometry / design-space loading
  - `runners/` — experiment setup / orchestration (`experiment.py`, the path `main.py` actually uses); `pipeline.py` is a stale alternate driver (see below)
  - `utils/` — assembly, solver, filter, stiffness, OC update, obstacles, export, boundary, metrics, `axis_convention`, results manager, logger
  - `visualization/` — 3D display and animation generation
- `main.py` — the **only working, maintained entry point** (`python main.py ...`)
- `tests/` — pytest suite (CPU); `tests/gpu/` holds the manual GPU parity tests
- `examples/`, `scripts/`, `assets/` — sample configs, run scripts, images
- `pyproject.toml` — **all** package metadata (PEP 621); `setup.py` is a shim only; `MANIFEST.in`, `environment.yml` for conda

> **Entry points.** `main.py` is the only working, maintained driver. Two alternate drivers — `runners/pipeline.py` (`run_optimization_pipeline`) and `cli/command.py` — are currently **stale**: they call `setup_experiment` / `visualize_initial_setup` with an outdated argument shape, and nothing in the repo imports them, so they raise if invoked. Mirror `main.py`, not those.

## Running

```bash
# CLI
python main.py --nelx 32 --nely 16 --nelz 16 --volfrac 0.3 --penal 3.0 --rmin 3.0

# Python API
python -c "from pytopo3d.core.optimizer import top3d"
```

Key parameters: `nelx/nely/nelz` (element counts), `volfrac` (volume fraction), `penal` (SIMP penalization), `rmin` (filter radius). See the README for the full set.

> **STL design spaces override the grid.** If you load the design space from an STL (`--design-space-stl`), the resolution comes from the STL's voxelization (mesh size / pitch) and your `--nelx/--nely/--nelz` are **ignored** (a warning is logged). The adopted shape follows the internal `(nely, nelx, nelz)` order — see **Axis / array layout** below.

## Tests

```bash
pytest          # from the repo root
```

`pytest.ini` sets `pythonpath = .`, so the suite runs **against the source tree** — no install step. What each test guards:

- `test_imports.py` — package imports cleanly
- `test_cpu_smoke.py` — a small CPU `top3d` run (regression for issue #7)
- `test_regression.py` — tolerant golden-master comparison against `tests/data/golden_small_case.npz` (structural: fails if >2% of voxels move materially, not strict elementwise)
- `test_packaging.py` — package discovery finds `pytopo3d` (regression for issue #6)
- `test_stl_orientation.py` — STL axis-convention round-trip
- `tests/gpu/` — CPU/GPU parity. **Not part of CI** — it needs a CUDA GPU and is run manually (see `tests/gpu/README.md`).

CI is `.github/workflows/ci.yml`: CPU only, Python 3.10–3.12, plus a real `pip install .` + import to guard the packaging.

## Conventions and invariants you must respect

### 1. Axis / array layout — the most common source of bugs
- The **public surface** is `(x, y, z)`, z-up: `nelx/nely/nelz`, JSON obstacle `center`/`size`, and `force_field` components `[Fx, Fy, Fz]` are all in `(x, y, z)` order.
- **But internal density/mask arrays are stored `(nely, nelx, nelz)` — y first**, inherited from the MATLAB `top3d` reference. Index a hand-built mask as `mask[y, x, z]`.
- STL import/export transpose the first two axes for you (fixed in 0.2.0).
- Before touching anything geometry-related, read the README section **"Coordinate System and Axis Conventions."**

### 2. GPU guard idiom
The package runs **CPU-only by default** and uses the GPU only when CuPy is importable (`HAS_CUPY`) **and** the GPU path is explicitly requested (`use_gpu=True` / the `--gpu` flag, which defaults off). Every reference to `cp`/`cusp` must be guarded so short-circuit evaluation never touches them on a CPU-only machine. The idiom is:

```python
if HAS_CUPY and (... cp / cusp references ...):
```

Breaking it (a stray `or`, a missing guard in a CPU branch) crashes CPU-only runs with `NameError` — this was issue #7. Keep the idiom in any new GPU code.

### 3. Determinism / golden master
`top3d` is deterministic (uniform initialization, no RNG), which is what makes `test_regression.py` possible. The CPU **solver**, though, is not fixed by default: `utils/solver.py` uses `pypardiso.spsolve` whenever `pypardiso` is importable and falls back to scipy `spsolve` otherwise. `pypardiso` is currently a hard dependency in `pyproject.toml`, so a normal install runs PyPardiso. CI deliberately installs deps **without** `pypardiso`, so the golden tests run on scipy `spsolve` and match the saved reference — do **not** add `pypardiso` to the CI test job, or the solver changes and the golden-master output drifts.

### 4. Packaging
All metadata lives in `pyproject.toml`; `setup.py` is a no-metadata shim (do not duplicate fields into it — setuptools errors on duplicates). Package discovery is explicit (`[tool.setuptools.packages.find] include = ["pytopo3d*"]`) because of the flat layout; without it the build ships an **empty** package (issue #6).

## Public-repo hygiene

Everything committed here is public. Keep machine- and environment-specific details out of committed files: no personal usernames, no absolute home paths, no internal cluster/host names, no SSH key names, no hardware inventories. The `tests/gpu/` scaffolding and any CI config must stay **generic** — use env-var placeholders like `$REPO` and `$HOME/.cache/...`. GPU *type* names alone (e.g. `gpu:a100:1`) are fine.

## Releases

Release cadence, versioning (SemVer), the per-release checklist, and the publish steps live in **`RELEASING.md`** — follow it when cutting a release. The one rule worth repeating here: **publishing is irreversible** (a PyPI version can never be re-uploaded or fully deleted), so the version bump + GitHub Release is a confirm-once step.

## Citation

PyTopo3D is research software. If you use it — or build a downstream project on it — please cite the paper: Kim & Kang, *PyTopo3D: A Python Framework for 3D SIMP-based Topology Optimization*, arXiv:2504.05604 (2025). See `CITATION.cff` (GitHub's "Cite this repository") or the README *Citation* section for BibTeX.
