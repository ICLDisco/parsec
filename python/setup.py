#!/usr/bin/env python3
"""
Setup script for Py_PaRSEC - Python interface for PaRSEC

This script lives inside the PaRSEC source tree at python/setup.py.
It expects the PaRSEC C library to have been built (via CMake) in the
parent directory's build/ tree, with an install prefix at ../build/install.
"""

import os
import sys
from pathlib import Path

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

project_root = Path(__file__).parent.resolve()
parsec_repo_root = project_root.parent


def get_long_description():
    """Get the long description from README.md"""
    readme_path = project_root / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


def find_parsec():
    """Find PaRSEC installation paths.

    Search order:
      1. PARSEC_ROOT environment variable (explicit override)
      2. ../build/install  (in-tree CMake build — the normal case)
      3. System-wide locations
    """
    parsec_root = os.environ.get("PARSEC_ROOT")
    if parsec_root:
        lib_dirs = []
        for lib_dir in ["lib64", "lib"]:
            lib_path = f"{parsec_root}/{lib_dir}"
            if os.path.exists(lib_path):
                lib_dirs.append(lib_path)

        return {
            "include_dirs": [f"{parsec_root}/include"],
            "library_dirs": lib_dirs,
            "libraries": ["parsec"],
        }

    common_paths = [
        str(parsec_repo_root / "build" / "install"),
        "/usr/local",
        "/opt/parsec",
        "/usr",
        str(Path.home() / ".local" / "parsec"),
    ]

    for path in common_paths:
        include_path = f"{path}/include"
        for lib_subdir in ["lib64", "lib"]:
            lib_path = f"{path}/{lib_subdir}"
            if os.path.exists(f"{include_path}/parsec.h"):
                lib_files = []
                if sys.platform == "darwin":
                    lib_files = ["libparsec.dylib", "libparsec.4.dylib", "libparsec.4.1.0.dylib"]
                elif sys.platform.startswith("linux"):
                    lib_files = ["libparsec.so", "libparsec.so.4", "libparsec.so.4.1.0"]
                elif sys.platform.startswith("win"):
                    lib_files = ["parsec.dll", "libparsec.dll"]

                lib_found = any(os.path.exists(f"{lib_path}/{lib_file}") for lib_file in lib_files)

                if lib_found:
                    return {
                        "include_dirs": [include_path],
                        "library_dirs": [lib_path],
                        "libraries": ["parsec"],
                    }

    print("Warning: PaRSEC not found.")
    print("Please build PaRSEC first (cmake --build ../build && cmake --install ../build)")
    print("or set the PARSEC_ROOT environment variable.")
    return {
        "include_dirs": [],
        "library_dirs": [],
        "libraries": [],
    }


parsec_config = find_parsec()


def get_mpi_include_dirs():
    try:
        import subprocess
        result = subprocess.run(['mpicc', '--showme:compile'],
                              capture_output=True, text=True, check=True)
        return [flag[2:] for flag in result.stdout.strip().split() if flag.startswith('-I')]
    except:
        return []


if get_mpi_include_dirs():
    parsec_config.setdefault("include_dirs", []).extend(get_mpi_include_dirs())


def get_extra_link_args():
    """Get extra link arguments for dynamic linking"""
    extra_args = []
    default_lib = str(parsec_repo_root / "build" / "install" / "lib64")

    if sys.platform in ("darwin",) or sys.platform.startswith("linux"):
        if parsec_config.get("library_dirs"):
            lib_dir = parsec_config["library_dirs"][0]
            extra_args.append(f"-Wl,-rpath,{lib_dir}")
        else:
            extra_args.append(f"-Wl,-rpath,{default_lib}")

    return extra_args


extra_link_args = get_extra_link_args()

# Detect CUDA and cuBLAS
cuda_libs = []
cuda_lib_dirs = []
cuda_include_dirs = []

cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_ROOT")
if not cuda_path:
    for possible_path in ["/usr/local/cuda", "/opt/cuda"]:
        if os.path.exists(possible_path):
            cuda_path = possible_path
            break

if cuda_path and os.path.exists(cuda_path):
    cuda_include = f"{cuda_path}/include"
    cuda_lib = f"{cuda_path}/lib64" if os.path.exists(f"{cuda_path}/lib64") else f"{cuda_path}/lib"

    if os.path.exists(cuda_include) and os.path.exists(cuda_lib):
        cuda_include_dirs = [cuda_include]
        cuda_lib_dirs = [cuda_lib]
        cuda_libs = ["cublas", "cudart"]
        print(f"Found CUDA at {cuda_path}")

# Paths to PaRSEC test-app C sources (now siblings via the parent repo).
# Use relative paths (setuptools requires them for sources), but absolute
# for include_dirs (which are passed as -I flags and don't have that restriction).
stencil_src_rel = os.path.join("..", "tests", "apps", "stencil")
stencil_build_rel = os.path.join("..", "build", "tests", "apps", "stencil")
merge_sort_src_rel = os.path.join("..", "tests", "apps", "merge_sort")
merge_sort_build_rel = os.path.join("..", "build", "tests", "apps", "merge_sort")

stencil_src_abs = str(parsec_repo_root / "tests" / "apps" / "stencil")
stencil_build_abs = str(parsec_repo_root / "build" / "tests" / "apps" / "stencil")
merge_sort_src_abs = str(parsec_repo_root / "tests" / "apps" / "merge_sort")
merge_sort_build_abs = str(parsec_repo_root / "build" / "tests" / "apps" / "merge_sort")

extensions = [
    Extension(
        "py_parsec.core",
        sources=["src/py_parsec/core.pyx"],
        **parsec_config,
        language="c",
        extra_link_args=extra_link_args,
    ),
    Extension(
        "py_parsec.runtime",
        sources=["src/py_parsec/runtime.pyx"],
        **parsec_config,
        language="c",
        extra_link_args=extra_link_args,
    ),
    Extension(
        "py_parsec.tasks",
        sources=["src/py_parsec/tasks.pyx"],
        **parsec_config,
        language="c",
        extra_link_args=extra_link_args,
    ),
    Extension(
        "py_parsec.dtd",
        sources=["src/py_parsec/dtd.pyx"],
        include_dirs=parsec_config.get("include_dirs", []) + cuda_include_dirs,
        library_dirs=parsec_config.get("library_dirs", []) + cuda_lib_dirs,
        libraries=parsec_config.get("libraries", []) + cuda_libs,
        language="c",
        extra_link_args=extra_link_args,
    ),
    Extension(
        "py_parsec.matrix",
        sources=["src/py_parsec/matrix.pyx"],
        **parsec_config,
        language="c",
        extra_link_args=extra_link_args,
    ),
    Extension(
        "py_parsec.stencil_core",
        sources=[
            "src/py_parsec/stencil_core.pyx",
            os.path.join(stencil_src_rel, "stencil_internal.c"),
            os.path.join(stencil_build_rel, "stencil_1D.c"),
        ],
        include_dirs=parsec_config.get("include_dirs", []) + [
            str(project_root),
            stencil_src_abs,
            stencil_build_abs,
        ],
        library_dirs=parsec_config.get("library_dirs", []),
        libraries=parsec_config.get("libraries", []),
        language="c",
        extra_link_args=extra_link_args,
    ),
    Extension(
        "py_parsec.merge_sort_core",
        sources=[
            "src/py_parsec/merge_sort_core.pyx",
            os.path.join(merge_sort_src_rel, "merge_sort_wrapper.c"),
            os.path.join(merge_sort_src_rel, "sort_data.c"),
            os.path.join(merge_sort_build_rel, "merge_sort.c"),
        ],
        include_dirs=parsec_config.get("include_dirs", []) + [
            str(project_root),
            merge_sort_src_abs,
            merge_sort_build_abs,
        ],
        library_dirs=parsec_config.get("library_dirs", []),
        libraries=parsec_config.get("libraries", []),
        language="c",
        extra_link_args=extra_link_args,
    ),
]

compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
}

if __name__ == "__main__":
    setup(
        packages=find_packages("src"),
        package_dir={"": "src"},
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=True,
        ),
        zip_safe=False,
    )
