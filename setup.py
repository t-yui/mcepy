from __future__ import annotations

import os
import sysconfig
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

PACKAGE_ROOT = Path(__file__).parent.resolve()
SOURCE_ROOT = PACKAGE_ROOT / "src" / "mcepy"
USE_CYTHON = os.environ.get("MCEPY_USE_CYTHON", "").lower() in {"1", "true", "yes"}
PYTHON_INCLUDE = Path(sysconfig.get_paths().get("include", ""))
PYTHON_HEADER = PYTHON_INCLUDE / "Python.h"


def _build_extensions():
    if not PYTHON_HEADER.exists():
        print("[mcepy] Python development headers were not found. Building without the optional extension.")
        return []

    if USE_CYTHON:
        try:
            from Cython.Build import cythonize
        except ImportError as exc:
            raise RuntimeError(
                "Cython is required when MCEPY_USE_CYTHON=1. Install the dev dependencies first."
            ) from exc
        source_file = Path("src/mcepy/_speedups.pyx")
    else:
        source_file = Path("src/mcepy/_speedups.c")
        if not (PACKAGE_ROOT / source_file).exists():
            source_file = Path("src/mcepy/_speedups.pyx")
            try:
                from Cython.Build import cythonize
            except ImportError as exc:
                raise RuntimeError(
                    "Neither src/mcepy/_speedups.c nor Cython are available."
                ) from exc
        else:
            cythonize = None

    extensions = [
        Extension(
            "mcepy._speedups",
            sources=[str(source_file)],
            include_dirs=[np.get_include()],
        )
    ]

    if source_file.suffix == ".pyx":
        extensions = cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            },
        )

    return extensions


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as exc:
            print(f"\n[mcepy] optional extension build failed: {exc}\n")
            self.extensions = []

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            print(f"\n[mcepy] optional extension build failed for {ext.name}: {exc}\n")


ext_modules = _build_extensions()
cmdclass = {"build_ext": OptionalBuildExt} if ext_modules else {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
