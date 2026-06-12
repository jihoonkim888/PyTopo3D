"""Import smoke tests.

Every module must import on a CPU-only machine (no CuPy installed). GPU-only
names such as `cp` / `cusp` are defined only inside a `try: import cupy` block, so
any code that references them outside a `HAS_CUPY` guard fails to import (or, worse,
crashes at call time -- see issue #7). Importing every module is the cheapest guard
against that whole class of regression.
"""

import importlib
import pkgutil

import pytest

import pytopo3d


def _all_submodules():
    return sorted(
        info.name
        for info in pkgutil.walk_packages(pytopo3d.__path__, prefix="pytopo3d.")
    )


@pytest.mark.parametrize("module_name", _all_submodules())
def test_module_imports_on_cpu(module_name):
    importlib.import_module(module_name)
