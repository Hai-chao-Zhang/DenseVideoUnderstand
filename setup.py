"""Setuptools compatibility shim for generated wheel resources."""

import sys
from pathlib import Path

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_support.release_data import BuildPyWithReleaseData

setup(cmdclass={"build_py": BuildPyWithReleaseData})
