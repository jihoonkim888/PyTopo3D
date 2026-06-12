"""Packaging guard for issue #6: the `[gpu]` install extra must stay declared.

The PyPI 0.1.0 build shipped without a `gpu` extra, so `pip install pytopo3d[gpu]`
warned and installed nothing. The extra is declared in the source packaging files;
this test keeps it from silently disappearing again.

This is a string-level check on purpose: `tomllib` only exists on Python >= 3.11,
and we want this guard to run on the whole CI matrix (3.10+).
"""

from pathlib import Path

import pytest

BUILD_TOOLS = Path(__file__).parent.parent / "build-tools"
PYPROJECT = BUILD_TOOLS / "pyproject.toml"
SETUP_PY = BUILD_TOOLS / "setup.py"


@pytest.mark.skipif(not PYPROJECT.exists(), reason="build-tools/pyproject.toml not found")
def test_gpu_extra_declared_in_pyproject():
    text = PYPROJECT.read_text()
    assert "[project.optional-dependencies]" in text
    assert "gpu" in text and "cupy" in text, "the [gpu] extra regressed (issue #6)"


@pytest.mark.skipif(not SETUP_PY.exists(), reason="build-tools/setup.py not found")
def test_gpu_extra_declared_in_setup_py():
    text = SETUP_PY.read_text()
    assert "extras_require" in text
    assert '"gpu"' in text and "cupy" in text, "the [gpu] extra regressed (issue #6)"
