"""(Re)generate golden-master reference files.

Run this *intentionally*, only after a deliberate, reviewed change to the algorithm
whose new output you want to bless as the reference:

    python tests/_generate_golden.py

It is NOT run by the test suite. Commit the regenerated tests/data/*.npz alongside
the change that justified it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from pytopo3d.core.optimizer import top3d  # noqa: E402
from tests.cases import SMALL_CASE  # noqa: E402


def main():
    # Force the same CPU solver CI uses (scipy spsolve). Otherwise, regenerating the
    # golden on a machine that happens to have pypardiso installed would bless a
    # different solver's output, permanently mismatching CI (which has no pypardiso).
    import pytopo3d.utils.solver as _solver

    _solver.solvers["cpu"] = _solver.seq_solve
    _solver.solvers["cpu_name"] = "SciPy spsolve (forced for golden)"

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    xPhys = top3d(**SMALL_CASE)
    out_file = out_dir / "golden_small_case.npz"
    np.savez_compressed(
        out_file,
        density=xPhys,
        mean=np.float64(xPhys.mean()),
        params=str(SMALL_CASE),
    )
    print(f"wrote {out_file}")
    print(f"  shape={xPhys.shape} mean={xPhys.mean():.6f} "
          f"range=[{xPhys.min():.4f}, {xPhys.max():.4f}]")


if __name__ == "__main__":
    main()
