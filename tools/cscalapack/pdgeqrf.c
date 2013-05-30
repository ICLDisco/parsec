/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
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
#include "../../dplasma/testing/flops.h"

#ifndef max
#define max(_a, _b) ( (_a) < (_b) ? (_b) : (_a) )
#define min(_a, _b) ( (_a) > (_b) ? (_b) : (_a) )
#endif

static int i0=0, i1=1;
static double m1=-1e0, p1=1e0;

typedef enum {
    PARAM_BLACS_CTX, 
    PARAM_RANK, 
    PARAM_P, 
    PARAM_Q, 
    PARAM_LOCP, 
    PARAM_LOCQ, 
    PARAM_M, 
    PARAM_N, 
    PARAM_NB, 
    PARAM_SEED, 
    PARAM_VALIDATE, 
    PARAM_NRHS
} params_enum_t;

static void setup_params( int params[], int argc, char* argv[] );
static void random_matrix( double *M, int mloc, int nloc, int seed );
static double check_solution( int params[], double *Aqr, double *tau );


int main( int argc, char **argv ) {
    int params[12];
    int info;
    int ictxt, nprow, npcol, myrow, mycol, iam;
    int m, n, nb, nrhs, mloc, nloc;
    double *A=NULL; int descA[9];
    double *tau=NULL;
    double residF;
    double telapsed, gflops, pgflops;

    setup_params( params, argc, argv );
    ictxt   = params[PARAM_BLACS_CTX];
    iam     = params[PARAM_RANK];
    m       = params[PARAM_M];
    n       = params[PARAM_N];
    nb      = params[PARAM_NB];
    nrhs    = params[PARAM_NRHS];
    
    
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    mloc = numroc_( &m, &nb, &myrow, &i0, &nprow );
    nloc = numroc_( &n, &nb, &mycol, &i0, &npcol );
    descinit_( descA, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
    assert( 0 == info );
    A = malloc( sizeof(double)*mloc*nloc );
    random_matrix( A, mloc, nloc, iam*n*max(m,nrhs) );
    tau = malloc( sizeof(double)*min(m,n) );

    { double *work=NULL; int lwork=-1; double getlwork;
      double t1, t2;
      pdgeqrf_( &m, &n, A, &i1, &i1, descA, tau, &getlwork, &lwork, &info );
      assert( 0 == info );
      lwork = (int)getlwork;
      work = malloc( sizeof(double)*lwork );
      t1 = MPI_Wtime();
      pdgeqrf_( &m, &n, A, &i1, &i1, descA, tau, work, &lwork, &info );
      assert( 0 == info );
      t2 = MPI_Wtime();
      telapsed = t2-t1;
      free(work);
    }
    residF = check_solution( params, A, tau );

    if( 0 != iam ) 
        MPI_Reduce( &telapsed, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    else {
        MPI_Reduce( MPI_IN_PLACE, &telapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        gflops = FLOPS_DGEQRF((double)m, (double)n)/1e+9/telapsed;
        pgflops = gflops/(((double)nprow)*((double)npcol));
        printf( "### PDGEQRF ###\n"
                "#%4sx%-4s %7s %7s %4s %4s # %4s %10s %10s %10s %11s\n", "P", "Q", "M", "N", "NB", "NRHS", "info", "resid", "time(s)", "gflops", "gflops/pxq" );
        printf( " %4d %-4d %7d %7d %4d %4d   %-4d %10.3e %10.3g %10.3g %11.3g\n", nprow, npcol, m, n, nb, nrhs, info, residF, telapsed, gflops, pgflops );
    }

    free( A ); A = NULL;
    free( tau ); tau = NULL;

    Cblacs_exit( 0 );
    return 0;
}


static double check_solution( int params[], double* Aqr, double *tau ) {
    double residF = NAN;

    if( params[PARAM_VALIDATE] ) {
        int info;
        int ictxt   = params[PARAM_BLACS_CTX],
            nprow   = params[PARAM_P], 
            npcol   = params[PARAM_Q],
            myrow   = params[PARAM_LOCP],
            mycol   = params[PARAM_LOCQ],
            iam     = params[PARAM_RANK];
        int m       = params[PARAM_M],
            n       = params[PARAM_N],
            nb      = params[PARAM_NB],
            nrhs    = params[PARAM_NRHS];
        int mloc, nloc, nrhsloc;
        double *A=NULL; int descA[9];
        double *B=NULL; int descB[9];
        double *X=NULL;
        double eps, AnormF, XnormF, RnormF;
        
        mloc = numroc_( &m, &nb, &myrow, &i0, &nprow );
        nloc = numroc_( &n, &nb, &mycol, &i0, &npcol );
        nrhsloc = numroc_( &nrhs, &nb, &mycol, &i0, &npcol );

        /* recreate A */
        descinit_( descA, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
        assert( 0 == info );
        A = malloc( sizeof(double)*mloc*nloc );
        random_matrix( A, mloc, nloc, iam*n*max(m,nrhs) );
        /* create B and copy it to X */
        descinit_( descB, &n, &nrhs, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
        assert( 0 == info );
        B = malloc( sizeof(double)*mloc*nrhsloc );
        X = malloc( sizeof(double)*mloc*nrhsloc );
        random_matrix( B, mloc, nrhsloc, 0 );
        pdlacpy_( "All", &n, &nrhs, B, &i1, &i1, descB, X, &i1, &i1, descB );

        /* Compute X from Aqr */
        { double *work=NULL; int lwork=-1; double getlwork;
          pdormqr_( "L", "T", &n, &nrhs, &n, Aqr, &i1, &i1,
                    descA, tau, X, &i1, &i1, descB,
                    &getlwork, &lwork, &info );
          lwork = (int)getlwork;
          work = malloc( sizeof(double)*lwork );
          pdormqr_( "L", "T", &n, &nrhs, &n, Aqr, &i1, &i1,
                    descA, tau, X, &i1, &i1, descB,
                    work, &lwork, &info );
          free(work);
        }
        pdtrsm_( "L", "U", "N", "N", &n, &nrhs, &p1, Aqr, &i1, &i1, descA, X, &i1, &i1, descB );
        /* Compute B-AX */
        pdgemm_( "N", "N", &n, &nrhs, &n, &p1, A, &i1, &i1, descA, X, &i1, &i1, descB,
                 &m1, B, &i1, &i1, descB );
        AnormF = pdlange_( "F", &n, &n, A, &i1, &i1, descA, NULL );
        XnormF = pdlange_( "F", &n, &nrhs, X, &i1, &i1, descB, NULL );
        RnormF = pdlange_( "F", &n, &nrhs, B, &i1, &i1, descB, NULL );
        eps = pdlamch_( &ictxt, "Epsilon" );
        residF = RnormF / ( AnormF * XnormF * eps );
        free( A ); free( B ); free( X );
    }
    return residF;
}



static void random_matrix( double* M, int mloc, int nloc, int seed ) {
    int i, j, k = 0;
    if( seed ) srand( seed );
    for( i = 0; i < mloc; i++ ) {
        for( j = 0; j < nloc; j++ ) {
            M[k] = ((double)rand()) / ((double)RAND_MAX) - 0.5;
            k++;
        }
    }
}


static void setup_params( int params[], int argc, char* argv[] ) {
    int i;
    int ictxt, iam, nprocs, p, q;
    MPI_Init( &argc, &argv );
    Cblacs_pinfo( &iam, &nprocs );
    Cblacs_get( -1, 0, &ictxt );

    p = 1;
    q = 1;
    params[PARAM_M]         = 0;
    params[PARAM_N]         = 1000;
    params[PARAM_NB]        = 64;
    params[PARAM_SEED]      = 0;
    params[PARAM_VALIDATE]  = 1;
    params[PARAM_NRHS]      = 1;

    for( i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-p" ) == 0 ) {
            p = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-q" ) == 0 ) {
            q = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-m" ) == 0 ) {
            params[PARAM_M] = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-n" ) == 0 ) {
            params[PARAM_N] = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-nb" ) == 0 ) {
            params[PARAM_NB] = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-x" ) == 0 ) {
            params[PARAM_VALIDATE] = 0;
        }
        if( strcmp( argv[i], "-nrhs" ) == 0 ) {
            params[PARAM_NRHS] = atoi(argv[i+1]);
            i++;
        }
    }
    /* Validity checks etc. */
    if( params[PARAM_NB] > params[PARAM_N] )
        params[PARAM_NB] = params[PARAM_N];
    if( 0 == params[PARAM_M] )
        params[PARAM_M] = params[PARAM_N];
    if( params[PARAM_P] * params[PARAM_Q] > nprocs ) {
        if( 0 == iam )
            fprintf( stderr, "### ERROR: we do not have enough processes available to make a p-by-q process grid ###\n"
                             "###   Bye-bye                                                                      ###\n" );
        Cblacs_abort( ictxt, 1 );
    }
    if( params[PARAM_VALIDATE] && (params[PARAM_M] != params[PARAM_N]) ) {
        if( 0 == iam )
            fprintf( stderr, "### WARNING: Unable to validate on a non-square matrix. Canceling validation.\n" );
        params[PARAM_VALIDATE] = 0;
    }
    Cblacs_gridinit( &ictxt, "Row", p, q );
    params[PARAM_BLACS_CTX] = ictxt;
    params[PARAM_RANK] = iam;
    params[PARAM_P] = p;
    params[PARAM_Q] = q;
}
