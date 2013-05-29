#!/bin/bash

# Adding "debug" on the command line adds the appropriate debug flags

echo "### Defaults for ICL Linux machine"
echo "# Many parameters can be tuned (command line, env, etc)"
echo "# Open this file to see what common variables are available"
echo "#"
# These are override variables you can set (here or in the env) to alter defaults
CC=${CC:="/mnt/scratch/sw/intel/bin/icc"}
CXX=${CXX:="/mnt/scratch/sw/intel/bin/icpc"}
FC=${FC:="/mnt/scratch/sw/intel/bin/ifort"}
#MPI_DIR=${MPI_DIR:="/path/mpi"}
#HWLOC_DIR=${HWLOC_DIR:="/path/hwloc"}
#GTG_DIR=${GTG_DIR:="/path/gtg"}
CUDA_DIR=${CUDA_DIR:="/mnt/scratch/cuda"}
#OMEGA_DIR=${OMEGA_DIR:="/path/omega"}
PLASMA_DIR=${PLASMA_DIR:="/home/bosilca/unstable/dplasma/PLASMA/build"}

# This can be used to control auto detection of some packages
#PKG_CONFIG_PATH=/some/package/lib/pkgconfig:$PKG_CONFIG_PATH

# This option permits setting arbitrary options to cmake
# Options passed on the command line are appended to this variable
USER_OPTIONS+=""

if [ "x${USER}" = "xsmoreaud" ]; then
  unset CUDA_DIR
  USER_OPTIONS+="-DDAGUE_GPU_WITH_CUDA=OFF "

  PAPI_DIR=${PAPI_DIR:="/mnt/scratch/sw/papi-5.0.1"}
  USER_OPTIONS+="-DPINS_ENABLE=ON "
  USER_OPTIONS+="-DPAPI_DIR=$PAPI_DIR "
fi

. $(dirname $0)/config.inc
guess_defaults
run_cmake $*

