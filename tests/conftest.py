"""Shared pytest fixtures and headless setup."""

# Force a non-interactive matplotlib backend BEFORE anything imports pyplot
# (pytopo3d.core.optimizer imports matplotlib.pyplot at module top).
import matplotlib

matplotlib.use("Agg")

import pytest

from tests.cases import SMALL_CASE


@pytest.fixture(scope="session")
def small_case():
    """A fresh copy of the small deterministic problem parameters."""
    return dict(SMALL_CASE)
