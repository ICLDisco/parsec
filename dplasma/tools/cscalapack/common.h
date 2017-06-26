/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#ifndef _SCALAPACK_COMMON_H_
#define _SCALAPACK_COMMON_H_

#include "../../testing/flops.h"


static int i0=0, i1=1;
static double m1=-1e0, p0=0e0, p1=1e0;

typedef enum {
    PARAM_BLACS_CTX,
    PARAM_RANK,
    PARAM_M,
    PARAM_N,
    PARAM_NB,
    PARAM_SEED,
    PARAM_VALIDATE,
    PARAM_NRHS
} params_enum_t;

#ifndef max
#define max(_a, _b) ( (_a) < (_b) ? (_b) : (_a) )
#define min(_a, _b) ( (_a) > (_b) ? (_b) : (_a) )
#endif

void setup_params( int params[], int argc, char* argv[] );

void scalapack_pdplrnt( double *A,
                        int m, int n,
                        int mb, int nb,
                        int myrow, int mycol,
                        int nprow, int npcol,
                        int mloc,
                        int seed );

void scalapack_pdplghe( double *A,
                        int m, int n,
                        int mb, int nb,
                        int myrow, int mycol,
                        int nprow, int npcol,
                        int mloc,
                        int seed );

#endif /* _SCALAPACK_COMMON_H_ */
