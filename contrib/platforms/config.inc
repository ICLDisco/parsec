
# Parse command line arguments
if [ "$1" = "debug" ]; then 
  shift
  DEBUG="-DDPLASMA_PRECISIONS=s -DCMAKE_BUILD_TYPE=Debug -DDAGUE_DEBUG=ON -DDAGUE_DEBUG_HISTORY=ON"
fi

## If not overriden, try to guess some meaningful defaults
function guess_defaults {
CC=${CC:="$(which icc)"}
CXX=${CXX:="$(which icpc)"}
FC=${FC:="$(which ifort)"}
MPI_DIR=${MPI_DIR:="$(dirname $(dirname $(which mpicc)))"}
HWLOC_DIR=${HWLOC_DIR:="$(dirname $(dirname $(which hwloc-ls)))"}
GTG_DIR=${GTG_DIR:="$(pkg-config gtg --variable=prefix)"}
CUDA_DIR=${CUDA_DIR:="$(dirname $(dirname $(which nvcc)))"}
OMEGA_DIR=${OMEGA_DIR:="/opt/omega"} # wild guess, but no better way
PLASMA_DIR=${PLASMA_DIR:="$(pkg-config plasma --variable=prefix)"}
}

function run_cmake() {
export CC CXX FC

#####
## Cmake does not have a clean interface for FindXXX modules, everyone has a different flavor. Reconciliation.

if [ -n "$INSTALL_PREFIX" ]; then 
  USER_OPTIONS+=" -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
fi

if [ -n "$MPI_DIR" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_DIST_WITH_MPI=OFF.*") -eq 0 ]; then
  MPI="-DMPI_C_COMPILER=${MPI_DIR}/bin/mpicc -DMPI_CXX_COMPILER=${MPI_DIR}/bin/mpic++ -DMPI_Fortran_COMPILER=${MPI_DIR}/bin/mpif90"
# Make sure to always set all three compilers at the same time. The name of the wrapper may vary on your system
fi

if [ -n "$HWLOC_DIR" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_WITH_HWLOC=OFF.*") -eq 0 ]; then
    HWLOC="-DHWLOC_DIR=${HWLOC_DIR}"
fi

if [ -f "$GTG_DIR/include/GTG.h" -a -f "$GTG_DIR/lib/libgtg.so" ]; then
    GTG="-DGTG_DIR=${GTG_DIR}"
fi

if [ -n "${CUDA_DIR}" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_GPU_WITH_CUDA=OFF.*") -eq 0 ]; then 
    CUDA="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_DIR}"
fi

if [ -n "${OMEGA_DIR}" -a $(expr "${USER_OPTIONS}" : ".*DAGUE_Q2J=OFF.*") -eq 0 ]; then 
    Q2J="-DOMEGA_DIR=${OMEGA_DIR}"
fi

if [ -n "${PLASMA_DIR}" ]; then
  PLASMA="-DPLASMA_DIR=${PLASMA_DIR}"
fi

# Done with variable allocation, do the thing with Cmake
rm -rf CMakeCache.txt CMakeFiles
SRC_LOCATION=`dirname $0`/../../
CMAKE_COMMAND="cmake -G 'Unix Makefiles' ${MPI} ${HWLOC} ${Q2J} ${PLASMA} ${CUDA} ${GTG} ${DEBUG} ${USER_OPTIONS} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}
