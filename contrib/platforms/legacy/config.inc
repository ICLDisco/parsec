#!/bin/bash

# Parse command line arguments
if [ "$1" = "debug" ]; then 
    shift
    DEBUG="-DCMAKE_BUILD_TYPE=Debug -DPARSEC_DEBUG_HISTORY=ON -DPARSEC_DEBUG_PARANOID=ON -DPARSEC_DEBUG_NOISIER=ON"
fi

if [ "$1" = "profile" ]; then
    shift
    PROFILE="-DPARSEC_PROF_TRACE=ON"
fi

function dirname_bin {
    local _progname=$(which $1 2>/dev/null)
    if [ -z "$_progname" ]; then
        echo ""
    else
        echo $(dirname $(dirname $_progname))
    fi
}


## If not overriden, try to guess some meaningful defaults. For the compilers
## start with Intel and if not found fallback to GNU set of compilers.
function guess_defaults {
    CC=${CC:=$(which icc 2>/dev/null)}
    CC=${CC:=$(which gcc 2>/dev/null)}
    CXX=${CXX:=$(which icpc 2>/dev/null)}
    CXX=${CXX:=$(which g++ 2>/dev/null)}
    FC=${FC:=$(which ifort 2>/dev/null)}
    FC=${FC:=$(which gfortran 2>/dev/null)}
    MPI_DIR=${MPI_DIR:=$(dirname_bin mpicc)}
    HWLOC_DIR=${HWLOC_DIR:=$(dirname_bin hwloc-ls)}
    GTG_DIR=${GTG_DIR:=$(pkg-config gtg --variable=prefix)}
    CUDA_DIR=${CUDA_DIR:=$(dirname_bin nvcc)}
    OMEGA_DIR=${OMEGA_DIR:=$(dirname_bin petit)}
    PAPI_DIR=${PAPI_DIR:=$(dirname_bin papi_avail)}
    #PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:=$(which python3 2>/dev/null)}
    PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:=$(which python 2>/dev/null)}
}

function run_cmake() {
export CC CXX FC

#####
## Cmake does not have a clean interface for FindXXX modules, everyone has a different flavor. Reconciliation.

if [ -n "$INSTALL_PREFIX" ]; then
    USER_OPTIONS+=" -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
fi

if [ -n "$MPI_DIR" -a $(expr "${USER_OPTIONS}" : ".*PARSEC_DIST_WITH_MPI=OFF.*") -eq 0 ]; then
    MPI=""
  # Detect MPI C compiler
    if [ -x $MPI_DIR/bin/mpicc ]; then
        MPI+="-DMPI_C_COMPILER=${MPI_DIR}/bin/mpicc "
    elif [ -x $MPI_DIR/bin64/mpicc ]; then
        MPI+="-DMPI_C_COMPILER=${MPI_DIR}/bin64/mpicc "
    elif [ -x $(command -v mpicxx 2>/dev/null) ]; then
        MPI+="-DMPI_C_COMPILER=`command -v mpicc` "
    fi
  # Detect MPI C++ compiler
    if [ -x $MPI_DIR/bin/mpicxx ]; then
        MPI+="-DMPI_CXX_COMPILER=${MPI_DIR}/bin/mpicxx "
    elif [ -x $MPI_DIR/bin64/mpicxx ]; then
        MPI+="-DMPI_CXX_COMPILER=${MPI_DIR}/bin64/mpicxx "
    elif [ -x $(command -v mpicxx 2>/dev/null) ]; then
        MPI+="-DMPI_CXX_COMPILER=`command -v mpicxx` "
    fi
  # Detect MPI Fortran 90 compiler
    if [ -x $MPI_DIR/bin/mpif90 ]; then
        MPI+="-DMPI_Fortran_COMPILER=${MPI_DIR}/bin/mpif90 "
    elif [ -x $MPI_DIR/bin64/mpif90 ]; then
        MPI+="-DMPI_Fortran_COMPILER=${MPI_DIR}/bin64/mpif90 "
    elif [ -x $(command -v mpif90 2>/dev/null) ]; then
        MPI+="-DMPI_Fortran_COMPILER=`command -v mpif90` "
    fi
fi

# Make sure to always set all three compilers at the same time. The name of the wrapper may vary on your system
if [ -n "$HWLOC_DIR" -a $(expr "${USER_OPTIONS}" : ".*PARSEC_WITH_HWLOC=OFF.*") -eq 0 ]; then
    HWLOC="-DHWLOC_DIR=${HWLOC_DIR}"
fi

if [ -f "$GTG_DIR/include/GTG.h" -a -f "$GTG_DIR/lib/libgtg.so" ]; then
    GTG="-DGTG_DIR=${GTG_DIR}"
fi

if [ -n "${CUDA_DIR}" -a $(expr "${USER_OPTIONS}" : ".*PARSEC_GPU_WITH_CUDA=OFF.*") -eq 0 ]; then
    CUDA="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_DIR} -DCUDA_HOST_COMPILER=$(which \"${CC}\" 2>/dev/null)"
fi

if [ -n "${OMEGA_DIR}" -a $(expr "${USER_OPTIONS}" : ".*PARSEC_Q2J=OFF.*") -eq 0 ]; then
    Q2J="-DOMEGA_DIR=${OMEGA_DIR}"
fi

if [ -n "${PAPI_DIR}" -a -f "$PAPI_DIR/lib/libpapi.so" ]; then
    PROFILE="${PROFILE} -DPAPI_DIR=${PAPI_DIR}"
fi

if [ -n "${PYTHON_EXECUTABLE}" -a -x "${PYTHON_EXECUTABLE}" ]; then
    PYTHON="-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
fi

# Done with variable allocation, do the thing with Cmake
rm -rf CMakeCache.txt CMakeFiles
SRC_LOCATION=${SRC_LOCATION:=`dirname $0`/../../}
CMAKE_COMMAND="cmake -G 'Unix Makefiles' ${MPI} ${HWLOC} ${Q2J} ${CUDA} ${GTG} ${PROFILE} ${PYTHON} ${DEBUG} ${USER_OPTIONS} ${TESTCMD} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}
