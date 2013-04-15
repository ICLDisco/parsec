#!/usr/bin/env python
# build script for dbpreader_py
# run python setup.py build_ext --inplace to build

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext, extension
import os.path
import sys

build_dir_name = 'build' # change this for local usage
build_dir = '../../../../' + build_dir_name
if build_dir and os.path.isdir(build_dir):
    config_h = build_dir + '/include/dague_config.h'
    libdaguebase = build_dir + '/dague/libdague-base.a'
else:
    build_dir = '.'
    config_h = '../../../include/dague_config.h'
    libdaguebase = 'build/libdague-base.a'
    
ext_modules = [Extension('dbpreader_py', 
                         ['dbpreader_py.pyx', '../dbpreader.c'], 
                         extra_objects=[libdaguebase],
                         include_dirs=['../../../include', '../../..', '../', '/home/pgaultne/include/', build_dir + '/include'], 
                         depends=['dbpreader_py.pxd', '../dbpreader.h', '../../../include/dbp.h', '../../../include/os-spec-timing.h', 
                                  '../../../dague/class/dague_object.h', config_h, '../../../dague/mca/pins/papi_socket/pins_papi_socket.h'], 
                         extra_compile_args=['-O0', '-g3'],
                         extra_link_args=["-g"],
                         libraries=['irc', 'imf'],
                         library_dirs=['/mnt/scratch/sw/intel/composer_xe_2013/lib/intel64/'],
                         runtime_library_dirs=['/mnt/scratch/sw/intel/composer_xe_2013/lib/intel64/']
                         )
               ]
# ext_modules = [Extension('dbpreader_py', 
#                          ['dbpreader_py.pyx', '../dbpreader.c', 'pins_cachemiss_info.c'], 
#                          extra_objects=['build/libdague-base.a'],
#                          include_dirs=['../../../include', '../../..', '../', '/home/pgaultne/include/'], 
#                          depends=['dbpreader_py.pxd', '../dbpreader.h', '../../../include/dbp.h', '../../../include/os-spec-timing.h', 
#                                   '../../../dague/class/dague_object.h', '../../../include/dague_config.h', '../../../dague/pins/papi/cachemiss.h'], 
#                          extra_compile_args=['-O0']
#                          )
#                ]

setup(
  name = 'dbpreader python interface',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
