#!/bin/sh                                                                       
LOCATION=`dirname $0`/../../
export CC=/mnt/scratch/sw/intel/2011.6.233/bin/icc
export CXX=/mnt/scratch/sw/intel/2011.6.233/bin/icpc
export F77=/mnt/scratch/sw/intel/2011.6.233/bin/ifort
export MKL="/mnt/scratch/sw/intel/2011.6.233/mkl/"
PLASMADIR="/home/bosilca/unstable/dplasma/PLASMA/build/"

cmake -G "Unix Makefiles" ./ -DPLASMA_DIR=${PLASMADIR} -DHWLOC_DIR=/home/bosilca/opt/64/ -DDPLASMA_SCHED_HWLOC=ON ${LOCATION}
