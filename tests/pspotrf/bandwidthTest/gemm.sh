#!/bin/sh

export PLASMA_DIR=/home/bosilca/unstable/dplasma/plasma-installer-2.1.0/build/plasma_2.1.0

echo gcc -o gemm -Wall -DHAVE_SCHED_SETAFFINITY -D__USE_GNU -I${PLASMA_DIR}/cblas gemm.c -lpthread -lrt -L${PLASMA_DIR}/lib/ -lcoreblas -lcblas -L/opt/mkl/lib/em64t -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm -O3
gcc -o gemm -Wall -DHAVE_SCHED_SETAFFINITY -D__USE_GNU -I${PLASMA_DIR}/cblas gemm.c -lpthread -lrt -L${PLASMA_DIR}/lib/ -lcoreblas -lcblas -L/opt/mkl/lib/em64t -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm -O3
