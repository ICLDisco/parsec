#!/usr/bin/env python
# build script for dbpreader_py
# run python setup.py build_ext --inplace to build

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext, extension
import os.path
import sys
import socket

build_dir_name = 'build' # change this for local usage
build_dir = '../../../../' + build_dir_name
if build_dir and os.path.isdir(build_dir):
    config_h = build_dir + '/include/dague_config.h'
    libdaguebase = build_dir + '/dague/libdague-base.a'
else: # for old-style, hacky "in-place" build
    build_dir = '.'
    config_h = 'build/include/dague_config.h'
    libdaguebase = 'build/libdague-base.a'

hostname = socket.gethostname()

if 'rebel' in hostname or 'peter' in hostname:
    lib_dirs = ['/opt/local/lib']
    run_dirs = ['/opt/local/lib']
    libs = ['openblas']
elif 'icl' in hostname or 'thog' in hostname:
    lib_dirs = ['/mnt/scratch/sw/intel/composer_xe_2013/lib/intel64/']
    run_dirs = ['/mnt/scratch/sw/intel/composer_xe_2013/lib/intel64/']
    libs = ['irc', 'imf']
else:
    lib_dirs = []
    run_dirs = []
    libs = []

ext_modules = [Extension('py_dbpreader', 
                         ['py_dbpreader.pyx', '../dbpreader.c'], 
                         extra_objects=[libdaguebase],
                         include_dirs=['../../../include', '../../..', '../', '/home/pgaultne/include/', build_dir + '/include'], 
                         depends=['setup.py', 'py_dbpreader.pyx', 'py_dbpreader.pxd', '../dbpreader.h', '../../../include/dbp.h', '../../../include/os-spec-timing.h', 
                                  '../../../dague/class/dague_object.h', config_h, '../../../dague/mca/pins/papi_socket/pins_papi_socket.h'], 
                         extra_compile_args=['-O0', '-g3'],
                         extra_link_args=["-g"],
#                          pyrex_gdb=True
                         library_dirs=lib_dirs,
                         runtime_library_dirs=run_dirs,
                         libraries=libs
                         )
               ]

setup(
  name = 'dbpreader python interface',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
