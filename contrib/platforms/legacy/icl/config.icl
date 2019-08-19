#!/bin/bash

# Adding "debug" on the command line adds the appropriate debug flags

echo "### Defaults for ICL Linux machine"
echo "# Many parameters can be tuned (command line, env, etc)"
echo "# Open this file to see what common variables are available"
echo "#"

# if you don't like icc, comment this out.
. /mnt/scratch/sw/intel/composerxe/bin/compilervars.sh intel64
# if you don't like intel MPI, comment this out.
. /mnt/scratch/sw/intel/impi/4.1.1.036/bin64/mpivars.sh intel64

# These are override variables you can set (here or in the env) to alter defaults
#CXX=${CXX:="/mnt/scratch/sw/intel/bin/icpc"}
#FC=${FC:="/mnt/scratch/sw/intel/bin/ifort"}
#MPI_DIR=${MPI_DIR:="/path/mpi"}
#HWLOC_DIR=${HWLOC_DIR:="/path/hwloc"}
#GTG_DIR=${GTG_DIR:="/path/gtg"}
#CUDA_DIR=${CUDA_DIR:="/mnt/scratch/cuda"}
#OMEGA_DIR=${OMEGA_DIR:="/path/omega"}

# This can be used to control auto detection of some packages
#PKG_CONFIG_PATH=/some/package/lib/pkgconfig:$PKG_CONFIG_PATH

# This option permits setting arbitrary options to cmake
# Options passed on the command line are appended to this variable
USER_OPTIONS+=""

. $(dirname $0)/config.inc
guess_defaults
run_cmake $*

