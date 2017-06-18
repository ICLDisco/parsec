##############################################################################
# Copyright (c) 2017       The University of Tennessee and the University
#                          of Tennessee Research Foundation.  All rights
#                          reserved.
#
# $COPYRIGHT$
#
##############################################################################
from spack import *

class Parsec(CMakePackage):
    """PaRSEC: the Parallel Runtime Scheduler and Execution Controller for micro-tasks on distributed heterogeneous systems"""

    homepage = "https://bitbucket.org/icldistcomp/parsec"
    url      = "https://bitbucket.org/icldistcomp/parsec/get/v1.1.0.tar.bz2"

    version('devel', git='https://bitbucket.org/icldistcomp/parsec/git', branch='master')
    version('1.1.0', '6c8b2b8d6408004bdb4c6d9134da74a4')

    # We always need MPI for now.
    #variant('mpi', default=True, description='Use MPI for dependency transport between nodes')
    variant('cuda', default=True, description='Use CUDA for GPU acceleration')
    variant('profile', default=False, description='Generate profiling data')
    variant('debug', default=False, description='Debug version (incurs performance overhead!)')
    #variant('xcompile', default=False, description='Cross compile')

    depends_on('hwloc')
    depends_on('mpi')
    #depends_on('mpi', when='+mpi')
    depends_on('cuda', when='+cuda')
    depends_on('papi', when='+profile')

    def configure_args(self):
        spec = self.spec
        return [
            '-DCMAKE_BUILD_TYPE=%s' % ('Debug' if '+debug' in spec else 'RelWithDebInfo'),
            '-DPARSEC_DEBUG_HISTORY=%s' % ('YES' if '+debug' in spec else 'NO'),
            '-DPARSEC_DEBUG_PARANOID=%s' % ('YES' if '+debug' in spec else 'NO'),
            '-DPARSEC_DEBUG_NOISIER=%s' % ('YES' if '+debug' in spec else 'NO'),
            '-DPARSEC_GPU_WITH_CUDA=%s' % ('YES' if '+cuda' in spec else 'NO'),
#            '-DCUDA_TOOLKIT_ROOT_DIR=%s' %
#            '-DPARSEC_DIST_WITH_MPI=%s' % ('YES' if '-mpi' in spec else 'NO'),
            '-DPARSEC_PROF_TRACE=%s' % ('YES' if '+profile' in spec else 'NO'),
#            '-DMPI_C_COMPILER=%s'
#            '-DMPI_CXX_COMPILER=%s'
#            '-DMPI_Fortran_COMPILER=%s'
        ]

