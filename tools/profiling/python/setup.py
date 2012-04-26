#!/usr/bin/env python
# build script for dbpreader_py
# run python setup.py build_ext --inplace to build


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('dbpreader_py', ['dbpreader_py.pyx', '../dbpreader.c'], include_dirs=['../../../include', '../'], depends=['dbpreader_py.pxd', '../dbpreader.h', '../../../include/dbp.h', '../../../include/os-spec-timing.h'], extra_compile_args=['-O0'])]

setup(
  name = 'dbpreader python interface',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
