#!/bin/sh                                                                       
LOCATION=`dirname $0`
echo ${LOCATION}
export CC=/mnt/scratch/sw/intel/composer_xe_2013.2.146/bin/intel64/icc
export CXX=/mnt/scratch/sw/intel/composer_xe_2013.2.146/bin/intel64/icpc
export F77=/mnt/scratch/sw/intel/composer_xe_2013.2.146/bin/intel64/ifort
export MKL="/mnt/scratch/sw/intel/composer_xe_2013.2.146/mkl"
export CFLAGS="-g3 -fPIC"
export CMAKE_PREFIX_PATH=$HOME/enthought_7.3.2_x64/
export CMAKE_FIND_ROOT_PATH=$HOME/enthought_7.3.2_x64/
PLASMADIR="/home/bosilca/unstable/dplasma/PLASMA/build/"

echo "cmake -G "Unix Makefiles" ./ -DPLASMA_DIR=${PLASMADIR} -DHWLOC_DIR=/home/bosilca/opt/64/ -DDPLASMA_SCHED_HWLOC=ON ${LOCATION} -DDAGUE_DIST_WITH_MPI=OFF -DPAPI_DIR=/home/pgaultne"
cmake -G "Unix Makefiles" ./ -DPLASMA_DIR=${PLASMADIR} -DHWLOC_DIR=/home/bosilca/opt/64/ -DDPLASMA_SCHED_HWLOC=ON ${LOCATION} -DDAGUE_DIST_WITH_MPI=OFF -DPAPI_DIR=/home/pgaultne -DPYTHON_EXECUTABLE:FILEPATH=/home/pgaultne/enthought_7.3.2_x64/bin/python -DPYTHON_INCLUDE_DIRS:FILEPATH=/home/pgaultne/enthought_7.3.2_x64/include/ -DPYTHON_LIBRARIES:FILEPATH=/home/pgaultne/enthought_7.3.2_x64/lib/libpython2.7.so
