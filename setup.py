#!/usr/bin/env python3
# All package metadata lives in pyproject.toml (PEP 621). This shim only exists so
# that legacy `python setup.py ...` invocations still work; it must NOT duplicate the
# metadata, or setuptools errors on fields defined in both places.
from setuptools import setup

setup()
