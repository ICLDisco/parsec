/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2010      University of Denver, Colorado.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include "myscalapack.h"
#include "../../dplasma/testing/flops.h"

#ifndef max
#define max(_a, _b) ( (_a) < (_b) ? (_b) : (_a) )
#define min(_a, _b) ( (_a) > (_b) ? (_b) : (_a) )
#endif

int main(int argc, char **argv) {
    int iam, nprocs, do_validation = 0;
    int myrank_mpi, nprocs_mpi;
    int ictxt, nprow, npcol, myrow, mycol;
    int mloc, nloc, n, m = 0, nb, nqrhs, nrhs;
    int i, j, k, info=0, seed;
    int descA[9], descB[9];
    double *A=NULL, *Acpy=NULL, *B=NULL, *X=NULL, *R=NULL, eps, *work=NULL;
    double AnormF, XnormF, RnormF, BnormF, residF=-1.0e+00;
    double *tau=NULL;
    int lwork;
    int izero=0,ione=1;
    double mone=(-1.0e0),pone=(1.0e0);
    /**/
    double MPIt1, MPIt2, MPIelapsed, GFLOPS, GFLOPS_per_proc ;
    /**/
    MPI_Init( &argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
    /**/
    n = 100; nrhs = 1; nprow = 1; npcol = 1; nb = 64;
    for( i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-n" ) == 0 ) {
            n      = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-m" ) == 0 ) {
            m      = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-nrhs" ) == 0 ) {
            nrhs   = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-p" ) == 0 ) {
            nprow  = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-q" ) == 0 ) {
            npcol  = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-nb" ) == 0 ) {
            nb     = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-v" ) == 0 ) {
            do_validation = 1;
        }
    }
    /**/
    if (nb>n)
        nb = n;
    if(m==0)
        m = n;
    if(do_validation && (m != n) ) {
        fprintf(stderr, "Unable to validate on a non-square matrix. Cancelling validation.\n");
        do_validation = 0;
    }
    if (nprow*npcol>nprocs_mpi){
        if (myrank_mpi==0)
            printf(" **** ERROR : we do not have enough processes available to make a p-by-q process grid ***\n");
        printf(" **** Bye-bye                                                                         ***\n");
        MPI_Finalize(); exit(1);
    }
    /**/
    /* no idea why I have problem with Cblacs on my computer, I am using blacsF77 interface here .... */
    blacs_pinfo_( &iam, &nprocs ) ;
    { int im1 = -1; int i0 = 0; blacs_get_( &im1, &i0, &ictxt ); }
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    //Cblacs_pinfo( &iam, &nprocs ) ;
    //Cblacs_get( -1, 0, &ictxt );
    //Cblacs_gridinit( &ictxt, "Row", nprow, npcol );
    //Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );

    /**/

    { int i0=0; mloc = numroc_( &m, &nb, &myrow, &i0, &nprow ); }
    { int i0=0; nloc = numroc_( &n, &nb, &mycol, &i0, &npcol ); }

    { int i0=0; descinit_( descA, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info ); }

    seed = iam*n*max(m, nrhs); srand(seed);

    A = (double *)malloc(mloc*nloc*sizeof(double)) ;

    k = 0;
    for (i = 0; i < mloc; i++) {
        for (j = 0; j < nloc; j++) {
            A[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5 ;
            k++;    
        }
    }

    descinit_( descA, &m, &n, &nb, &nb, &izero, &izero, &ictxt, &mloc, &info );

    if( do_validation ) {

        { int i0=0; nqrhs = numroc_( &nrhs, &nb, &mycol, &i0, &npcol ); }

        if (mloc*nqrhs>0) {
            B = (double *)malloc(mloc*nqrhs*sizeof(double)) ;
            if (B==NULL){ printf("error of memory allocation B on proc %dx%d\n",myrow,mycol); exit(0); }
        }

        k = 0;
        for (i = 0; i < mloc; i++) {
            for (j = 0; j < nqrhs; j++) {
                B[k] = ((double) rand()) / ((double) RAND_MAX) - 0.5 ;
                k++;    
            }
        }

        descinit_( descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &mloc, &info );

        Acpy = (double *)malloc(mloc*nloc*sizeof(double)) ;
        if (Acpy==NULL){ printf("error of memory allocation Acpy on proc %dx%d\n",myrow,mycol); exit(0); }

        if (mloc*nqrhs>0) {
            R = (double *)malloc(mloc*nqrhs*sizeof(double)) ;
            if (R==NULL){ printf("error of memory allocation R on proc %dx%d\n",myrow,mycol); exit(0); }
        }

        if (mloc*nqrhs>0) {
            X = (double *)malloc(mloc*nqrhs*sizeof(double)) ;
            if (X==NULL){ printf("error of memory allocation X on proc %dx%d\n",myrow,mycol); exit(0); }
        }

        pdlacpy_( "All", &n, &n, A, &ione, &ione, descA, Acpy, &ione, &ione, descA );
        pdlacpy_( "All", &n, &nrhs, B, &ione, &ione, descB, X, &ione, &ione, descB );

    }

    /**/
    tau = (double *)malloc(min(m, n)*sizeof(double)) ;
    lwork = -1;
    work = (double *)malloc(sizeof(double)) ;
    if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
    pdgeqrf_( &m, &n, A, &ione, &ione, descA, tau, work, &lwork, &info );
    lwork = (int) work[0];
    free(work); work = NULL;
    work = (double *)malloc(lwork*sizeof(double)) ;
    if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
    /**/        
    MPIt1 = MPI_Wtime();
    pdgeqrf_( &m, &n, A, &ione, &ione, descA, tau, work, &lwork, &info );
    /**/
    MPIt2 = MPI_Wtime();
    MPIelapsed=MPIt2-MPIt1;
    free(work); work = NULL;

    if( do_validation ) {

        lwork = -1;
        work = (double *)malloc(sizeof(double)) ;
        if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
        pdormqr_( "L", "T", &n, &nrhs, &n, A, &ione, &ione,
                  descA, tau, X, &ione, &ione, descB,
                  work, &lwork, &info );
        lwork = (int) work[0];
        free(work); work = NULL;
        work = (double *)malloc(lwork*sizeof(double)) ;
        if (work==NULL){ printf("error of memory allocation WORK on proc %dx%d\n",myrow,mycol); exit(0); }
        pdormqr_( "L", "T", &n, &nrhs, &n, A, &ione, &ione,
                  descA, tau, X, &ione, &ione, descB,
                  work, &lwork, &info );
        free(work); work=NULL;
        
        pdtrsm_( "L", "U", "N", "N", &n, &nrhs, &pone, A, &ione, &ione, descA, X, &ione, &ione, descB );
        
        pdlacpy_( "All", &n, &nrhs, B, &ione, &ione, descB, R   , &ione, &ione, descB );
        eps = pdlamch_( &ictxt, "Epsilon" );
        pdgemm_( "N", "N", &n, &nrhs, &n, &pone, Acpy, &ione, &ione, descA, X, &ione, &ione, descB,
                 &mone, R, &ione, &ione, descB);
        AnormF = pdlange_( "F", &n, &n, Acpy, &ione, &ione, descA, work);
        XnormF = pdlange_( "F", &n, &nrhs, X, &ione, &ione, descB, work);
        RnormF = pdlange_( "F", &n, &nrhs, R, &ione, &ione, descB, work);
        residF = RnormF / ( AnormF * XnormF * eps );
        
        free(Acpy);
        if ( B!=NULL ) free(B);
        if ( X!=NULL ) free(X);
        if ( R!=NULL ) free(R);
    }

    GFLOPS = FLOPS_DGEQRF((double)m, (double)n)/1e+9/MPIelapsed;
    GFLOPS_per_proc = GFLOPS / (((double) nprow)*((double) npcol));

    if ( iam==0 ){
        printf("m\tn\tnrhs\tnb\tp\tq\tinfo\tresid\ttime(s)  \tGFLOPS/sec\tGFLOPS/sec/proc\n");
        if( do_validation ) {
            printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%1.1f\t%f\t%f\t%f\n\n",
                   m, n, nrhs, nb, nprow, npcol, info, residF, MPIelapsed, GFLOPS, GFLOPS_per_proc);
        } else {
            printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\tx.x\t%f\t%f\t%f\n\n",
                   m, n, nrhs, nb, nprow, npcol, info, MPIelapsed, GFLOPS, GFLOPS_per_proc);
        }
    }

    free(A); A = NULL;
    free(tau); tau = NULL;

    { int i0=0; blacs_gridexit_( &i0 ); }
    //{ int i0=0; blacs_exit_( &i0 ); } // OK, so that should be done, nevermind ...
    MPI_Finalize();
    exit(0);
}
