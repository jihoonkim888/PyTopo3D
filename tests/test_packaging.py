"""Packaging guards for issue #6.

The published 0.1.0 build shipped a package containing NO modules (and therefore no
`gpu` extra): the packaging files lived in `build-tools/`, so `find_packages()` ran
there and discovered nothing. These tests guard both halves of that regression.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def test_build_discovers_the_package():
    """Regression for #6: a build must actually include the pytopo3d package, not ship
    an empty distribution."""
    from setuptools import find_packages

    packages = find_packages(where=str(ROOT), include=["pytopo3d*"])
    assert "pytopo3d" in packages, (
        f"build would ship an empty package (issue #6); discovered {packages}"
    )
    for sub in ("pytopo3d.core", "pytopo3d.utils"):
        assert sub in packages, f"missing subpackage {sub} (issue #6)"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires Python 3.11+")
def test_gpu_extra_declared():
    """The [gpu] extra must stay declared so `pip install pytopo3d[gpu]` resolves."""
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text())
    extras = data["project"]["optional-dependencies"]
    assert "gpu" in extras, "the [gpu] extra regressed (issue #6)"
    assert any("cupy" in dep for dep in extras["gpu"]), "gpu extra must require cupy"
