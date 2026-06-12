"""Shared, deterministic test problems.

Kept tiny and short so the whole CPU suite runs in a few seconds. `top3d` has no
random component (uniform `volfrac` initial guess, deterministic solver/OC), so a
fixed parameter set yields a fixed result -- which is what makes golden-master and
CPU/GPU-parity testing meaningful.
"""

# Small cantilever-style box. Used by smoke, invariant, regression and GPU-parity tests.
SMALL_CASE = dict(
    nelx=12,
    nely=6,
    nelz=6,
    volfrac=0.3,
    penal=3.0,
    rmin=1.5,
    disp_thres=0.5,
    tolx=0.01,
    maxloop=12,
    use_gpu=False,
)
