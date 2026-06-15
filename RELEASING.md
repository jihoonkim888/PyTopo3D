# Releasing PyTopo3D

PyTopo3D publishes to [PyPI](https://pypi.org/project/pytopo3d/) automatically when a GitHub Release is published — see `.github/workflows/publish.yml` (PyPI Trusted Publishing via OIDC, no stored token).

## When to release

Cut a release only when one of these holds; otherwise let changes accumulate on `main`:

- **Milestone** — a roadmap milestone's work has landed (see the Roadmap in the README).
- **Urgent user-facing fix** — install-broken, a crash, data loss, or a security issue.

**Merge ≠ release.** Merge PRs as soon as they are ready, but do not cut a release per PR. Batch related changes — especially fixes in the *same subsystem* — into one release. (For example, the `0.2.0` STL axis-convention fix and the `0.2.1` STL-resolution fix touched the same subsystem and ideally would have shipped together as one release.)

## Versioning (SemVer)

`MAJOR.MINOR.PATCH`. While the project is in `0.x`:

- **PATCH** (`0.2.0 → 0.2.1`) — backwards-compatible bug fixes.
- **MINOR** (`0.1.x → 0.2.0`) — new features, **or a breaking change** (permitted in `0.x`).
- A release number should map to a meaningful unit of change. Maintenance/correctness releases take numbers independently of the Roadmap's feature-milestone numbers; on a collision, push the roadmap milestone up rather than renaming the release.

## Checklist for every release

- [ ] Bump `version` in `pyproject.toml` (PyPI rejects re-uploading an existing version).
- [ ] Add a `CHANGELOG.md` section; put **breaking changes** at the top with a migration note.
- [ ] For a silently-changed behaviour, emit a one-time transition warning for one release and remove it the next (see the `0.2.0` STL axis-convention change for the pattern).
- [ ] CI (the CPU test matrix) is green on `main`.
- [ ] For a risky breaking change, consider shipping a release candidate (e.g. `0.3.0rc1`) first.

## How to publish

1. Make sure `pyproject.toml` `version` and `CHANGELOG.md` are updated and merged to `main`.
2. Create a GitHub Release whose tag is `vX.Y.Z`, targeting `main`. Publishing the release triggers `publish.yml`, which builds the sdist + wheel and uploads them via Trusted Publishing.
3. Verify the new version is live at <https://pypi.org/simple/pytopo3d/>. **Do not** trust the JSON API at `/pypi/pytopo3d/json` for this — it is heavily cached and can lag for minutes.

## Reproducibility

- **Never yank old versions.** PyTopo3D is used in research; `pip install pytopo3d==<old>` must keep working so past results remain reproducible.

> Publishing is irreversible: a PyPI version can never be re-uploaded or fully deleted.
