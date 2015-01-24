#!/bin/bash

# Parse command line arguments
if [ "$1" = "debug" ]; then 
    shift
    DEBUG="-DDPLASMA_PRECISIONS=s -DCMAKE_BUILD_TYPE=Debug -DDAGUE_DEBUG_ENABLE=ON -DDAGUE_DEBUG_HISTORY=ON"
fi

if [ "$1" = "profile" ]; then
    shift
    PROFILE="-DDAGUE_PROF_TRACE=ON"
fi

function dirname_bin {
    local _progname=$(which $1)
    if [ -z "$_progname" ]; then
        echo ""
    else
        echo $(dirname $(dirname $_progname))
    fi
}


## If not overriden, try to guess some meaningful defaults
function guess_defaults {
    CC=${CC:=$(which icc)}
    CXX=${CXX:=$(which icpc)}
    FC=${FC:=$(which ifort)}
    MPI_DIR=${MPI_DIR:=$(dirname_bin mpicc)}
    HWLOC_DIR=${HWLOC_DIR:=$(dirname_bin hwloc-ls)}
    GTG_DIR=${GTG_DIR:=$(pkg-config gtg --variable=prefix)}
    CUDA_DIR=${CUDA_DIR:=$(dirname_bin nvcc)}
    OMEGA_DIR=${OMEGA_DIR:=/opt/omega} # wild guess, but no better way
    PLASMA_DIR=${PLASMA_DIR:=$(pkg-config plasma --variable=prefix)}
    PAPI_DIR=${PAPI_DIR:=$(dirname_bin papi_avail)}
    PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:=$(which python)}
}

function run_cmake() {
export CC CXX FC

#####
## Cmake does not have a clean interface for FindXXX modules, everyone has a different flavor. Reconciliation.

if [ -n "$INSTALL_PREFIX" ]; then
  USER_OPTIONS+=" -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
fi

if [ -n "$MPI_DIR" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_DIST_WITH_MPI=OFF.*") -eq 0 ]; then
  MPI="-DMPI_C_COMPILER=${MPI_DIR}/bin/mpicc -DMPI_CXX_COMPILER=${MPI_DIR}/bin/mpicxx -DMPI_Fortran_COMPILER=${MPI_DIR}/bin/mpif90"
# Make sure to always set all three compilers at the same time. The name of the wrapper may vary on your system
fi

if [ -n "$HWLOC_DIR" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_WITH_HWLOC=OFF.*") -eq 0 ]; then
    HWLOC="-DHWLOC_DIR=${HWLOC_DIR}"
fi

if [ -f "$GTG_DIR/include/GTG.h" -a -f "$GTG_DIR/lib/libgtg.so" ]; then
    GTG="-DGTG_DIR=${GTG_DIR}"
fi

if [ -n "${CUDA_DIR}" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_GPU_WITH_CUDA=OFF.*") -eq 0 ]; then
    CUDA="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_DIR} -DCUDA_HOST_COMPILER=${CC}"
fi

if [ -n "${OMEGA_DIR}" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_Q2J=OFF.*") -eq 0 ]; then
    Q2J="-DOMEGA_DIR=${OMEGA_DIR}"
fi

if [ -n "${PLASMA_DIR}" ]; then
  PLASMA="-DPLASMA_DIR=${PLASMA_DIR}"
fi

if [ -n "${PAPI_DIR}" -a -f "$PAPI_DIR/lib/libpapi.so" -a -n "${PROFILE}" ]; then
    PROFILE="${PROFILE} -DPAPI_DIR=${PAPI_DIR}"
fi

if [ -n "${PYTHON_EXECUTABLE}" -a -x "${PYTHON_EXECUTABLE}" ]; then
    PYTHON="-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
fi

# Done with variable allocation, do the thing with Cmake
rm -rf CMakeCache.txt CMakeFiles
SRC_LOCATION=${SRC_LOCATION:=`dirname $0`/../../}
CMAKE_COMMAND="cmake -G 'Unix Makefiles' ${MPI} ${HWLOC} ${Q2J} ${PLASMA} ${CUDA} ${GTG} ${PROFILE} ${PYTHON} ${DEBUG} ${USER_OPTIONS} ${TESTCMD} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}
