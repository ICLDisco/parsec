##############################################################################
# Copyright (c) 2017-2020  The University of Tennessee and the University
#                          of Tennessee Research Foundation.  All rights
#                          reserved.
#
# $COPYRIGHT$
#
##############################################################################
from spack import *

class Parsec(CMakePackage):
    """PaRSEC: the Parallel Runtime Scheduler and Execution Controller for micro-tasks on distributed heterogeneous systems"""

    homepage = "https://icl.utk.edu/dte"
    url      = "https://bitbucket.org/icldistcomp/parsec/get/parsec-3.0.2012.tar.bz2"
    list_url = "https://bitbucket.org/icldistcomp/parsec/downloads/?tab=tags"
    git      = "https://bitbucket.org/icldistcomp/parsec.git"

    version('master', branch='master')
    version('3.0.2012-rc1', sha256='a0f013bd5a2c44c61d3d76bab102e3ca3bab68ef2e89d7b5f544b9c1a6fde475')
    version('1.1.0', '6c8b2b8d6408004bdb4c6d9134da74a4', url='https://bitbucket.org/icldistcomp/parsec/get/v1.1.0.tar.bz2')

    variant('build_type', default='RelWithDebInfo', description='CMake build type', values=('Debug', 'Release',' RelWithDebInfo' ))
    variant('shared', default=True, description='Build a shared library')
    # We always need MPI for now.
    #variant('mpi', default=True, description='Use MPI for dependency transport between nodes')
    variant('cuda', default=True, description='Use CUDA for GPU acceleration')
    variant('profile', default=False, description='Generate profiling data')
    variant('debug_verbose', default=False, description='Debug version with verbose and paranoid (incurs performance overhead!)')
    conflicts('+debug_verbose build_type=Release', msg='You need to set build_type=Debug for +debug_verbose')
    conflicts('+debug_verbose build_type=RelWithDebInfo', msg='You need to set build_type=Debug for +debug_verbose')
    #SPack does not handle cross-compilation atm
    #variant('xcompile', default=False, description='Cross compile')

    depends_on('cmake@3.16.0:', type='build')
    depends_on('python', type='build')
    depends_on('hwloc')
    depends_on('mpi')     #depends_on('mpi', when='+mpi')
    depends_on('cuda', when='+cuda')
    depends_on('papi', when='+profile')
    depends_on('python', when='+profile')
    depends_on('py-cython', when='+profile')
    depends_on('py-pandas', when='+profile')
    depends_on('py-matplotlib', when='+profile')
    depends_on('py-tables', when='+profile')

    def cmake_args(self):
        args = [
            self.define_from_variant('BUILD_SHARED_LIBS', 'shared'),
            #self.define_from_variant('PARSEC_DIST_WITH_MPI', 'mpi'),
            self.define_from_variant('PARSEC_GPU_WITH_CUDA', 'cuda'),
            self.define_from_variant('PARSEC_PROF_TRACE', 'profile'),
            self.define_from_variant('PARSEC_DEBUG_HISTORY', 'debug_verbose'),
            self.define_from_variant('PARSEC_DEBUG_PARANOID', 'debug_verbose'),
        ]
        return args
