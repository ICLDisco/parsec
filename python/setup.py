# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Build script of the PaRSEC Python bindings.

The PaRSEC headers and library are located, in order, from:

* ``PARSEC_INCLUDE_DIRS`` / ``PARSEC_LIBRARY_DIRS`` (os.pathsep separated
  lists), which the PaRSEC CMake build sets for the in-tree build;
* ``pkg-config parsec``, looking first in ``$PARSEC_ROOT/lib*/pkgconfig``.

``CC`` selects the compiler. The extension modules link ``libparsec`` only,
with a run path to ``PARSEC_RUNTIME_LIBRARY_DIRS`` if set (the CMake install
step links against the build tree but runs from the install prefix), else to
the library directory.
"""

import os
import shlex
import subprocess
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def _split(var):
    return [p for p in os.environ.get(var, "").split(os.pathsep) if p]


def _pkg_config():
    """Include directories, other C flags and library directory of PaRSEC."""
    env = dict(os.environ)
    root = os.environ.get("PARSEC_ROOT")
    if root:
        dirs = [str(Path(root) / d / "pkgconfig") for d in ("lib", "lib64")]
        env["PKG_CONFIG_PATH"] = os.pathsep.join(dirs + _split("PKG_CONFIG_PATH"))

    def query(option):
        return subprocess.run(["pkg-config", option, "parsec"], env=env, check=True,
                              capture_output=True, text=True).stdout

    try:
        cflags = shlex.split(query("--cflags"))
        libdir = query("--variable=libdir").strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit("PaRSEC not found: set PARSEC_ROOT to the PaRSEC installation prefix "
                         f"(pkg-config parsec failed: {e})") from None
    return ([f[2:] for f in cflags if f.startswith("-I")], [libdir],
            [f for f in cflags if not f.startswith("-I")])


include_dirs = _split("PARSEC_INCLUDE_DIRS")
library_dirs = _split("PARSEC_LIBRARY_DIRS")
extra_compile_args = []
if not include_dirs:
    include_dirs, library_dirs, extra_compile_args = _pkg_config()

common = dict(
    include_dirs=[str(SRC / "py_parsec"), numpy.get_include()] + include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=_split("PARSEC_RUNTIME_LIBRARY_DIRS") or library_dirs,
    libraries=["parsec"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=extra_compile_args,
    depends=[str(SRC / "py_parsec" / "pyparsec.h")],
)

extensions = [
    Extension(f"py_parsec.{name}", [str(Path("src") / "py_parsec" / f"{name}.pyx")], **common)
    for name in ("runtime", "matrix", "dtd")
]

setup(
    ext_modules=cythonize(
        extensions,
        include_path=[str(SRC)],
        compiler_directives={"language_level": 3, "embedsignature": True},
    ),
)
