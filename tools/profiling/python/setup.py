#!/usr/bin/env python
# build script for dbpreader_py
# run python setup.py build_ext --inplace to build

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('dbpreader_py', 
                         ['dbpreader_py.pyx', '../dbpreader.c'], 
                         extra_objects=['build/libdague-base.a'],
                         include_dirs=['../../../include', '../../..', '../', '/home/pgaultne/include/'], 
                         depends=['dbpreader_py.pxd', '../dbpreader.h', '../../../include/dbp.h', '../../../include/os-spec-timing.h', 
                                  '../../../dague/class/dague_object.h', '../../../include/dague_config.h', '../../../dague/pins/papi/cachemiss.h'], 
                         extra_compile_args=['-O0'],
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
