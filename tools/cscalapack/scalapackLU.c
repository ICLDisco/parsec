// /usr/local/bin/mpirun -np 8 ./scalapackChInv -p 2 -q 4 -n 8000 -nb 200

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "myscalapack.h"

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
                CORE_dplrnt( tempm, tempn, Ab, mloc,
                             m, mb*( (i-1)/mb ), nb*( (j-1)/nb ), seed);
            }
        }
    }
}

int main(int argc, char **argv) {
    int iam, nprocs;
    int myrank_mpi, nprocs_mpi;
    int ictxt, nprow, npcol, myrow, mycol;
    int nb, n, s, mloc, nloc, sloc;
    int i, info_facto, info_solve, info, iseed, verif;
    int my_info_facto, my_info_solve;
    int *ippiv;
    int descA[9], descB[9];
    double *A=NULL, *B=NULL, *Acpy=NULL, *X=NULL, *work=NULL;
    double XnormI, AnormI, RnormI, BnormI, resid = -1.0e+00;
    double eps;
    /**/
    double elapsed, GFLOPS;
    double my_elapsed = 0.0e+00;
    /**/

    MPI_Init( &argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    n = 1000; nprow = 1; npcol = 1; nb = 64; s = 1; verif = 0; iseed = 3872;
    for( i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-n" ) == 0 ) {
            n = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-nrhs" ) == 0 ) {
            s = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-p" ) == 0 ) {
            nprow = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-q" ) == 0 ) {
            npcol = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-nb" ) == 0 ) {
            nb = atoi(argv[i+1]);
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

    /* no idea why I have problem with Cblacs on my computer, I am using blacsF77 interface here .... */
    blacs_pinfo_( &iam, &nprocs ) ;
    { int im1 = -1; int i0 = 0; blacs_get_( &im1, &i0, &ictxt ); }
    blacs_gridinit_( &ictxt, "R", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

    { int i0=0; mloc = numroc_( &n, &nb, &myrow, &i0, &nprow ); }
    { int i0=0; nloc = numroc_( &n, &nb, &mycol, &i0, &npcol ); }

    { int i0=0; descinit_( descA, &n, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info ); }
    ippiv = calloc(n+n, sizeof(int));

    A = (double *)malloc(mloc*nloc*sizeof(double)) ;

    init_random_matrix(A,
                       n, n,
                       nb, nb,
                       myrow, mycol,
                       nprow, npcol,
                       mloc,
                       iseed);

#if 0 /* A is not diagonal dominant in LU, lets the pivoting do its office */
    /* not that smart..., could used pdgeadd and pdlaset as well or pdmatgen */
    for (i = 1; i <= n; i++) {
        int idum1, idum2, iloc, jloc, i0=0;
        if ( ( myrow == indxg2p_( &i, &nb, &idum1, &i0, &nprow ) )
             &&   ( mycol == indxg2p_( &i, &nb, &idum1, &i0, &npcol ) ) ){
            iloc = indxg2l_( &i, &nb, &idum1, &idum2, &nprow );
            jloc = indxg2l_( &i, &nb, &idum1, &idum2, &npcol );
            A[ (jloc-1)*mloc + (iloc-1) ] += ((double) n);
        }
    }
#endif

#if 0    /* PDMATGEN is part of the testing, it is not in the general lib...*/
    {int i0 = 0; int seed = 800;
        pdmatgen_ (&ictxt, "N", "N", &n, &n, &nb, &nb, A, descA+8, descA+6, descA+7, &seed,
                   &i0, &mloc, &i0, &nloc, &myrow, &mycol, &nprow, &npcol); }
#endif

    if (verif==1){
        Acpy = (double *)malloc(mloc*nloc*sizeof(double)) ;
        { int i1=1; pdlacpy_( "A", &n, &n, A, &i1, &i1, descA, Acpy, &i1, &i1, descA ); }
    }


    my_elapsed =- MPI_Wtime();
    { int i1=1; pdgetrf_( &n, &n, A, &i1, &i1, descA, ippiv, &my_info_facto ); }
    my_elapsed += MPI_Wtime();

    if( my_info_facto < 0 ) {
        fprintf(stderr, "Factorization failed on proc (%d, %d): info = %d\n", myrow, mycol, my_info_facto);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Allreduce( &my_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce( &my_info_facto, &info_facto, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if( my_info_facto != 0 ) {
        printf("***Facto failed: info = %d\n", my_info_facto);
        MPI_Finalize();
    }

    GFLOPS = 2.0*(((double) n)*((double) n)*((double) n))/1e+9/elapsed/3.0;

    if (verif == 1)
    {
        { int i0=0; sloc = numroc_( &s, &nb, &mycol, &i0, &npcol ); }
        { int i0=0; descinit_( descB, &n, &s, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info ); }

        if (mloc*sloc > 0)
        {
            B = (double *)malloc(mloc*sloc*sizeof(double)) ;
        }

        init_random_matrix(B,
                           n, s,
                           nb, nb,
                           myrow, mycol,
                           nprow, npcol,
                           mloc,
                           iseed + 1);

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

        if (mloc*sloc > 0)
        {
            X = (double *)malloc(mloc*sloc*sizeof(double)) ;
        }
        { int i1=1; pdlacpy_( "A", &n, &s, B, &i1, &i1, descB, X, &i1, &i1, descB ); }

        { int i1=1; pdgetrs_( "N", &n, &s, A, &i1, &i1, descA, ippiv, X, &i1, &i1, descB, &my_info_solve ); }

        if( my_info_solve < 0 ) {
            fprintf(stderr, "Solve failed on proc (%d, %d): info = %d\n", myrow, mycol, my_info_solve);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Allreduce( &my_info_solve, &info_solve, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if( my_info_solve != 0 ) {
            printf("***Solve failed: info = %d\n", my_info_solve);
            MPI_Finalize();
        }

        { int i1=1; double pone=1.00e+00, mone = -1.00e+00; pdgemm_( "N", "N", &n, &s, &n, &mone, Acpy, &i1, &i1, descA, X, &i1, &i1, descB,
                                                                     &pone, B, &i1, &i1, descB); }
        { int i1=1; AnormI = pdlange_( "I", &n, &n, Acpy, &i1, &i1, descA, work); }
        { int i1=1; XnormI = pdlange_( "I", &n, &s, X, &i1, &i1, descB, work); }
        { int i1=1; RnormI = pdlange_( "I", &n, &s, B, &i1, &i1, descB, work); }
        eps = pdlamch_( &ictxt, "Epsilon" );

        if(iam == 0)
            printf("||A||oo = %e, ||X||oo = %e, ||R||oo = %e\n", AnormI, XnormI, RnormI);

        resid = RnormI / ( ( AnormI * XnormI + BnormI ) * n * eps );

        if( X != NULL ) free( X );
        if( B != NULL ) free( B );

        if ( iam == 0 ){
            printf("**********************     N * S * NB * NP * P * Q *   T     * Gflops * Residual   * Rf * Rs *\n");
            printf("SCAL GETRF            %6d %3d %4d %4d %3d %3d %6.2f    %6.2lf    % 10e %d %d\n", n, s, nb, nprocs, nprow, npcol, elapsed, GFLOPS, resid, info_facto, info_solve);
        }

    } else {

        if ( iam == 0 ){
            printf("********************** N * S * NB * NP * P * Q *    T  * Gflops * R *\n");
            printf("SCAL GETRF            %6d %3d %4d %3d %3d %3d %6.2f %6.2lf %d\n", n, s, nb, nprocs, nprow, npcol, elapsed, GFLOPS, info_facto);
        }
    }

    if(verif == 1) {
        free( Acpy );
        free( work );
    }

    free( A );

    //{ int i0=0; blacs_exit_( &i0 ); } // OK, so that should be done, nevermind ...
    MPI_Finalize();
    return 0;
}
