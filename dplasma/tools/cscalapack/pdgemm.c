/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>
#include "myscalapack.h"
#include "common.h"

static double check_solution( int params[], double *Alu, double *tau );

int main( int argc, char **argv ) {
    int params[8];
    int info;
    int ictxt, nprow, npcol, myrow, mycol, iam;
    int m, n, nb, s, mloc, nloc, verif, iseed;
    double *A=NULL, *B=NULL, *C=NULL; int descA[9];
    double resid = NAN;
    double telapsed, gflops, pgflops;

    setup_params( params, argc, argv );
    ictxt = params[PARAM_BLACS_CTX];
    iam   = params[PARAM_RANK];
    m     = params[PARAM_M];
    n     = params[PARAM_N];
    nb    = params[PARAM_NB];
    s     = params[PARAM_NRHS];
    iseed = params[PARAM_SEED];
    verif = params[PARAM_VALIDATE];

    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    mloc = numroc_( &m, &nb, &myrow, &i0, &nprow );
    nloc = numroc_( &n, &nb, &mycol, &i0, &npcol );
    descinit_( descA, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
    assert( 0 == info );
    A = malloc( sizeof(double)*mloc*nloc );
    B = malloc( sizeof(double)*mloc*nloc );
    C = malloc( sizeof(double)*mloc*nloc );

    scalapack_pdplrnt(A,
                       m, n,
                       nb, nb,
                       myrow, mycol,
                       nprow, npcol,
                       mloc,
                       iseed);
    scalapack_pdplrnt(B,
                       m, n,
                       nb, nb,
                       myrow, mycol,
                       nprow, npcol,
                       mloc,
                       iseed);
    scalapack_pdplrnt(C,
                       m, n,
                       nb, nb,
                       myrow, mycol,
                       nprow, npcol,
                       mloc,
                       iseed);

    {
        double t1, t2;
        t1 = MPI_Wtime();
        pdgemm_( "n", "n", &m, &n, &n, &p1, A, &i1, &i1, descA,
                                            B, &i1, &i1, descA,
                                       &p1, C, &i1, &i1, descA );
        t2 = MPI_Wtime();
        telapsed = t2-t1;
    }

    if( 0 != iam ) {
        MPI_Reduce( &telapsed, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    }
    else {
        MPI_Reduce( MPI_IN_PLACE, &telapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        gflops = FLOPS_DGEMM((double)m, (double)n, (double)n)/1e+9/telapsed;
        pgflops = gflops/(((double)nprow)*((double)npcol));
        printf( "### PDGEMM ###\n"
                "#%4sx%-4s %7s %7s %4s %4s # %10s %10s %10s %11s\n", "P", "Q", "M", "N", "NB", "NRHS", "resid", "time(s)", "gflops", "gflops/pxq" );
        printf( " %4d %-4d %7d %7d %4d %4d   %10.3e %10.3g %10.3g %11.3g\n", nprow, npcol, m, n, nb, s, resid, telapsed, gflops, pgflops );
    }

    free( A ); A = NULL;
    free( B ); B = NULL;
    free( C ); C = NULL;

    Cblacs_exit( 0 );
    return 0;
}
