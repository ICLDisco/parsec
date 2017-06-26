/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>
#include "myscalapack.h"
#include "common.h"

static double check_solution( int params[], double *W );

int main( int argc, char **argv ) {
    int params[8];
    int info;
    int ictxt, nprow, npcol, myrow, mycol, iam;
    int m, n, nb, s, mloc, nloc, verif, iseed;
    int descA[9];
    double *A = NULL;
    double *D = NULL;
    double *E = NULL;
    double *TAUQ = NULL;
    double *TAUP = NULL;
    double resid, telapsed, gflops, pgflops;
    int minmn;

    setup_params( params, argc, argv );
    ictxt = params[PARAM_BLACS_CTX];
    iam   = params[PARAM_RANK];
    m     = params[PARAM_M];
    n     = params[PARAM_N];
    nb    = params[PARAM_NB];
    s     = params[PARAM_NRHS];
    iseed = params[PARAM_SEED];
    verif = params[PARAM_VALIDATE];
    minmn = min(m, n);

    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    mloc = numroc_( &m, &nb, &myrow, &i0, &nprow );
    nloc = numroc_( &n, &nb, &mycol, &i0, &npcol );
    descinit_( descA, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
    assert( 0 == info );
    A = malloc( sizeof(double)*mloc*nloc );
    scalapack_pdplghe( A,
                       m, n,
                       nb, nb,
                       myrow, mycol,
                       nprow, npcol,
                       mloc,
                       iseed );
    D = malloc( sizeof(double)*minmn );
    E = malloc( sizeof(double)*minmn );
    TAUP = malloc( sizeof(double)*minmn );
    TAUQ = malloc( sizeof(double)*minmn );

    {
        double *work=NULL; int lwork=-1; double getlwork;
        double t1, t2;
        pdgebrd_( &m, &n, A, &i1, &i1, descA, D, E, TAUQ, TAUP, &getlwork, &lwork, &info );
        assert( 0 == info );
        lwork = (int)getlwork;
        work = malloc( sizeof(double)*lwork );
        t1 = MPI_Wtime();
        pdgebrd_( &m, &n, A, &i1, &i1, descA, D, E, TAUQ, TAUP, work, &lwork, &info );
        assert( 0 == info );
        t2 = MPI_Wtime();
        telapsed = t2-t1;
        free(work);
    }
    if ( verif ) {
        resid = check_solution( params, A );
    } else {
        resid = -1;
    }

    if( 0 != iam ) {
        MPI_Reduce( &telapsed, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    }
    else {
        MPI_Reduce( MPI_IN_PLACE, &telapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        gflops = FLOPS_DGEBRD((double)m, (double)n)/1e+9/telapsed;
        pgflops = gflops/(((double)nprow)*((double)npcol));
        printf( "### PDGEBRD ###\n"
                "#%4sx%-4s %7s %7s %4s %4s # %10s %10s %10s %11s\n", "P", "Q", "M", "N", "NB", "NRHS", "resid", "time(s)", "gflops", "gflops/PxQ" );
        printf( " %4d %-4d %7d %7d %4d %4d   %10.3e %10.3g %10.3g %11.3g\n", nprow, npcol, m, n, nb, s, resid, telapsed, gflops, pgflops );
    }

    free( A ); A = NULL;
    Cblacs_exit( 0 );
    return 0;
}


static double check_solution( int params[], double* W ) {
    double resid = NAN;
    /* This check is not correct, we need to compute something completely different for Aw=wz */
#if 0
    int info;
    int ictxt = params[PARAM_BLACS_CTX],
        iam   = params[PARAM_RANK];
    int m     = params[PARAM_M],
        n     = params[PARAM_N],
        nb    = params[PARAM_NB],
        s     = params[PARAM_NRHS];
    int nprow, npcol, myrow, mycol;
    int mloc, nloc, sloc;
    double *A=NULL; int descA[9];
    double *B=NULL; int descB[9];
    double *X=NULL;
    double eps, AnormF, XnormF, RnormF;

    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    mloc = numroc_( &m, &nb, &myrow, &i0, &nprow );
    nloc = numroc_( &n, &nb, &mycol, &i0, &npcol );
    sloc = numroc_( &s, &nb, &mycol, &i0, &npcol );

    /* recreate A */
    descinit_( descA, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
    assert( 0 == info );
    A = malloc( sizeof(double)*mloc*nloc );
    random_matrix( A, descA, iam*n*max(m,s), n );
    /* create B and copy it to X */
    descinit_( descB, &n, &s, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
    assert( 0 == info );
    B = malloc( sizeof(double)*mloc*sloc );
    X = malloc( sizeof(double)*mloc*sloc );
    random_matrix( B, descB, -1, 0e0 );
    pdlacpy_( "All", &n, &s, B, &i1, &i1, descB, X, &i1, &i1, descB );
    /* Compute X from Alu */
    pdpotrs_( "L", &n, &s, Allt, &i1, &i1, descA, X, &i1, &i1, descB, &info );
    assert( 0 == info );
    /* Compute B-AX */
    pdsymm_( "L", "L", &n, &s, &m1, A, &i1, &i1, descA, X, &i1, &i1, descB,
             &p1, B, &i1, &i1, descB);
    AnormF = pdlansy_( "F", "L", &n, A, &i1, &i1, descA, NULL );
    XnormF = pdlange_( "F", &n, &s, X, &i1, &i1, descB, NULL );
    RnormF = pdlange_( "F", &n, &s, B, &i1, &i1, descB, NULL );
    eps = pdlamch_( &ictxt, "Epsilon" );
    resid = RnormF / ( AnormF * XnormF * eps );
    free( A ); free( B ); free( X );
#endif
    return resid;
}
