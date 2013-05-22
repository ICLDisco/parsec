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

#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20
#define NBELEM 1

static unsigned long long int
Rnd64_jump(unsigned long long int n, unsigned long long int seed ) {
    unsigned long long int a_k, c_k, ran;
    int i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    ran = seed;
    for (i = 0; n; n >>= 1, ++i) {
        if (n & 1)
            ran = a_k * ran + c_k;
        c_k *= (a_k + 1);
        a_k *= a_k;
    }

    return ran;
}

void CORE_dplrnt( int m, int n, double *A, int lda,
                  int bigM, int m0, int n0, unsigned long long int seed )
{
    double *tmp = A;
    int64_t i, j;
    unsigned long long int ran, jump;

    jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)bigM;

    for (j=0; j<n; ++j ) {
        ran = Rnd64_jump( NBELEM*jump, seed );
        for (i = 0; i < m; ++i) {
            *tmp = 0.5f - ran * RndF_Mul;
            ran  = Rnd64_A * ran + Rnd64_C;
            tmp++;
        }
        tmp  += lda-i;
        jump += bigM;
    }
}

static void init_random_matrix(double *A,
                               int m, int n,
                               int mb, int nb,
                               int myrow, int mycol,
                               int nprow, int npcol,
                               int mloc,
                               int seed)
{
    int i, j;
    int idum1, idum2, iloc, jloc, i0=0;
    int tempm, tempn;
    double *Ab;

    for (i = 1; i <= m; i += mb) {
        for (j = 1; j <= n; j += nb) {
            if ( ( myrow == indxg2p_( &i, &mb, &idum1, &i0, &nprow ) ) &&
                 ( mycol == indxg2p_( &j, &nb, &idum1, &i0, &npcol ) ) ){
                iloc = indxg2l_( &i, &mb, &idum1, &idum2, &nprow );
                jloc = indxg2l_( &j, &nb, &idum1, &idum2, &npcol );

                Ab =  &A[ (jloc-1)*mloc + (iloc-1) ];
                tempm = (i+mb > m) ? (m%mb) : (mb);
                tempn = (j+nb > n) ? (n%nb) : (nb);
                tempm = (m - i +1) > mb ? mb : (m-i + 1);
                tempn = (n - j +1) > nb ? nb : (n-j + 1);
                CORE_dplrnt( tempm, tempn, Ab, mloc,
                             m, mb*( (i-1)/mb ), nb*( (j-1)/nb ), seed);
            }
        }
    }
}


int main(int argc, char **argv) {
    int iam, nprocs, do_validation = 0;
    int myrank_mpi, nprocs_mpi;
    int ictxt, nprow, npcol, myrow, mycol;
    int mloc, nloc, n, m = 0, nb, nqrhs, nrhs;
    int i, info, iseed, verif,s;
    int descA[9], descB[9];
    double *A=NULL, *Acpy=NULL, *B=NULL, *X=NULL, eps, *work=NULL;
    double XnormI, AnormI, RnormI, BnormI, resid = -1.0e+00;
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
    n = 1000; nprow = 1; npcol = 1; nb = 64; s = 1; verif = 0; iseed = 3872;
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
        if( strcmp( argv[i], "-verif" ) == 0 ) {
            verif = 1;
        }
        if( strcmp( argv[i], "-seed" ) == 0 ) {
            iseed = atoi(argv[i+1]);
            i++;
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

    A = (double *)malloc(mloc*nloc*sizeof(double)) ;

    init_random_matrix(A,
                       n, n,
                       nb, nb,
                       myrow, mycol,
                       nprow, npcol,
                       mloc,
                       iseed);

    descinit_( descA, &m, &n, &nb, &nb, &izero, &izero, &ictxt, &mloc, &info );

    if( verif == 1 ) {

        { int i0=0; nqrhs = numroc_( &nrhs, &nb, &mycol, &i0, &npcol ); }

        if (mloc*nqrhs>0) {
            B = (double *)malloc(mloc*nqrhs*sizeof(double)) ;
            if (B==NULL){ printf("error of memory allocation B on proc %dx%d\n",myrow,mycol); exit(0); }
        }

	init_random_matrix(B,
                           n, s,
                           nb, nb,
                           myrow, mycol,
                           nprow, npcol,
                           mloc,
                           iseed + 1);


        descinit_( descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &mloc, &info );

       {
            /* For Norm Inf, LWORK >= Mp0, where Mp0 = ... */
            int i1=1;
            int i0=0;
            int iarow = indxg2p_( &i1, &nb, &myrow, &i0, &nprow);
            int Mp0 = numroc_( &n, &nb, &myrow, &iarow, &nprow );
            work = (double*)malloc(Mp0 * sizeof(double));
        }

        { int i1=1; BnormI = pdlange_( "I", &n, &s, B, &i1, &i1, descB, work); }
        if(iam == 0)
            printf("||B||oo = %e, ", BnormI);

        Acpy = (double *)malloc(mloc*nloc*sizeof(double)) ;
        if (Acpy==NULL){ printf("error of memory allocation Acpy on proc %dx%d\n",myrow,mycol); exit(0); }
        pdlacpy_( "All", &n, &n, A, &ione, &ione, descA, Acpy, &ione, &ione, descA );

        if (mloc*nqrhs>0) {
            X = (double *)malloc(mloc*nqrhs*sizeof(double)) ;
            if (X==NULL){ printf("error of memory allocation X on proc %dx%d\n",myrow,mycol); exit(0); }
        }
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

    if( verif == 1 ) {

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
        
        eps = pdlamch_( &ictxt, "Epsilon" );

	pdgemm_( "N", "N", &n, &nrhs, &n, &pone, Acpy, &ione, &ione, descA, X, &ione, &ione, descB,
                 &mone, B, &ione, &ione, descB);

       {
            /* For Norm Inf, LWORK >= Mp0, where Mp0 = ... */
            int i1=1;
            int i0=0;
            int iarow = indxg2p_( &i1, &nb, &myrow, &i0, &nprow);
            int Mp0 = numroc_( &n, &nb, &myrow, &iarow, &nprow );
            work = (double*)malloc(Mp0 * sizeof(double));
        }

        { int i1=1; AnormI = pdlange_( "I", &n, &n, Acpy, &i1, &i1, descA, work); }
        { int i1=1; XnormI = pdlange_( "I", &n, &s, X, &i1, &i1, descB, work); }
        { int i1=1; RnormI = pdlange_( "I", &n, &s, B, &i1, &i1, descB, work); }

 	if(iam == 0)
            printf("||A||oo = %e, ||X||oo = %e, ||R||oo = %e\n", AnormI, XnormI, RnormI);


        resid = RnormI / ( ( AnormI * XnormI + BnormI ) * n * eps );
        
        free(Acpy);
        if ( B!=NULL ) free(B);
        if ( X!=NULL ) free(X);
    }

    GFLOPS = FLOPS_DGEQRF((double)m, (double)n)/1e+9/MPIelapsed;
    GFLOPS_per_proc = GFLOPS / (((double) nprow)*((double) npcol));

    if ( iam==0 ){
        printf("M\tN\tNRHS\tNB\tP\tQ\tinfo\tresid\ttime(s)  \tGFLOPS/sec\tGFLOPS/sec/proc\n");
        if( verif == 1 ) {
            printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%10e\t%f\t%f\t%f\n\n",
                   m, n, nrhs, nb, nprow, npcol, info, resid, MPIelapsed, GFLOPS, GFLOPS_per_proc);
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
