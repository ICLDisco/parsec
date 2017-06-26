/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
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
#include "common.h"
#include "myscalapack.h"

void setup_params( int params[], int argc, char* argv[] )
{
    int i;
    int ictxt, iam, nprocs, p, q;
    int provided;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided );

#if 0
    fprintf(stderr, "Level of thread provided is %s\n",
	    provided == MPI_THREAD_MULTIPLE   ? "MPI_THREAD_MULTIPLE" :
	    provided == MPI_THREAD_SERIALIZED ? "MPI_THREAD_SERIALIZED" :
	    provided == MPI_THREAD_FUNNELED   ? "MPI_THREAD_FUNNELED" :
	    provided == MPI_THREAD_SINGLE     ? "MPI_THREAD_SINGLE" : "UNKNOWN" );
#endif

    Cblacs_pinfo( &iam, &nprocs );
    Cblacs_get( -1, 0, &ictxt );

    p = 1;
    q = 1;
    params[PARAM_M]         = 0;
    params[PARAM_N]         = 1000;
    params[PARAM_NB]        = 64;
    params[PARAM_SEED]      = 3872;
    params[PARAM_VALIDATE]  = 1;
    params[PARAM_NRHS]      = 1;

    for( i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-p" ) == 0 ) {
            p = atoi(argv[i+1]);
            i++;
            continue;
        }
        if( strcmp( argv[i], "-q" ) == 0 ) {
            q = atoi(argv[i+1]);
            i++;
            continue;
        }
        if( strcmp( argv[i], "-m" ) == 0 ) {
            params[PARAM_M] = atoi(argv[i+1]);
            i++;
            continue;
        }
        if( strcmp( argv[i], "-n" ) == 0 ) {
            params[PARAM_N] = atoi(argv[i+1]);
            i++;
            continue;
        }
        if( strcmp( argv[i], "-b" ) == 0 ) {
            params[PARAM_NB] = atoi(argv[i+1]);
            i++;
            continue;
        }
        if( strcmp( argv[i], "-x" ) == 0 ) {
            params[PARAM_VALIDATE] = 0;
            continue;
        }
        if(( strcmp( argv[i], "-s" ) == 0 ) ||
           ( strcmp( argv[i], "-nrhs" ) == 0 )) {
            params[PARAM_NRHS] = atoi(argv[i+1]);
            i++;
            continue;
        }
        if( strcmp( argv[i], "-seed" ) == 0 ) {
            params[PARAM_SEED] = atoi(argv[i+1]);
            i++;
        }
        fprintf( stderr, "### USAGE: %s [-p NUM][-q NUM][-m NUM][-n NUM][-b NUM][-x][-s NUM]\n"
                         "#     -p         : number of rows in the PxQ process grid\n"
                         "#     -q         : number of columns in the PxQ process grid\n"
                         "#     -m         : dimension of the matrix (MxN)\n"
                         "#     -n         : dimension of the matrix (MxN)\n"
                         "#     -b         : block size (NB)\n"
                         "#     -s | -nrhs : number of right hand sides for backward error computation (NRHS)\n"
                         "#     -x         : disable verification\n"
                         "#          -seed : Change the seed\n", argv[0] );
        Cblacs_abort( ictxt, i );
    }
    /* Validity checks etc. */
    if( params[PARAM_NB] > params[PARAM_N] )
        params[PARAM_NB] = params[PARAM_N];
    if( 0 == params[PARAM_M] )
        params[PARAM_M] = params[PARAM_N];
    if( p*q > nprocs ) {
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
}

/**
 * Matrix generations
 */
#define Rnd64_A  6364136223846793005ULL
#define Rnd64_C  1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static inline unsigned long long int
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


static inline void
CORE_dplrnt( int m, int n, double *A, int lda,
             int bigM, int m0, int n0, unsigned long long int seed )
{
    double *tmp = A;
    int64_t i, j;
    unsigned long long int ran, jump;

    jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)bigM;

    for (j=0; j<n; ++j ) {
        ran = Rnd64_jump( jump, seed );
        for (i = 0; i < m; ++i) {
            *tmp = 0.5f - ran * RndF_Mul;
            ran  = Rnd64_A * ran + Rnd64_C;
            tmp++;
        }
        tmp  += lda-i;
        jump += bigM;
    }
}

static inline void
CORE_dplghe( double bump, int m, int n, double *A, int lda,
             int gM, int m0, int n0, unsigned long long int seed )
{
    double *tmp = A;
    int64_t i, j;
    unsigned long long int ran, jump;

    jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)gM;

    /*
     * Tile diagonal
     */
    if ( m0 == n0 ) {
        for (j = 0; j < n; j++) {
            ran = Rnd64_jump( jump, seed );

            for (i = j; i < m; i++) {
                *tmp = 0.5f - ran * RndF_Mul;
                ran  = Rnd64_A * ran + Rnd64_C;
                tmp++;
            }
            tmp  += (lda - i + j + 1);
            jump += gM + 1;
        }

        for (j = 0; j < n; j++) {
            A[j+j*lda] += bump;

            for (i=0; i<j; i++) {
                A[lda*j+i] = A[lda*i+j];
            }
        }
    }
    /*
     * Lower part
     */
    else if ( m0 > n0 ) {
        for (j = 0; j < n; j++) {
            ran = Rnd64_jump( jump, seed );

            for (i = 0; i < m; i++) {
                *tmp = 0.5f - ran * RndF_Mul;
                ran  = Rnd64_A * ran + Rnd64_C;
                tmp++;
            }
            tmp  += (lda - i);
            jump += gM;
        }
    }
    /*
     * Upper part
     */
    else if ( m0 < n0 ) {
        /* Overwrite jump */
        jump = (unsigned long long int)n0 + (unsigned long long int)m0 * (unsigned long long int)gM;

        for (i = 0; i < m; i++) {
            ran = Rnd64_jump( jump, seed );

            for (j = 0; j < n; j++) {
                A[j*lda+i] = 0.5f - ran * RndF_Mul;
                ran = Rnd64_A * ran + Rnd64_C;
            }
            jump += gM;
        }
    }
}

void scalapack_pdplrnt( double *A,
                        int m, int n,
                        int mb, int nb,
                        int myrow, int mycol,
                        int nprow, int npcol,
                        int mloc,
                        int seed )
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
                tempm = (m - i +1) > mb ? mb : (m-i + 1);
                tempn = (n - j +1) > nb ? nb : (n-j + 1);
                CORE_dplrnt( tempm, tempn, Ab, mloc,
                             m, (i-1), (j-1), seed );
            }
        }
    }
}

void scalapack_pdplghe( double *A,
                        int m, int n,
                        int mb, int nb,
                        int myrow, int mycol,
                        int nprow, int npcol,
                        int mloc,
                        int seed )
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
                tempm = (m - i +1) > mb ? mb : (m-i + 1);
                tempn = (n - j +1) > nb ? nb : (n-j + 1);
                CORE_dplghe( (double)m, tempm, tempn, Ab, mloc,
                             m, (i-1), (j-1), seed );
            }
        }
    }
}

