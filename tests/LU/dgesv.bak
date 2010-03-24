/* ///////////////////////////// P /// L /// A /// S /// M /// A /////////////////////////////// */
/* ///                    PLASMA testing routines (version 2.1.0)                            ///
 * ///                    Author: Bilel Hadri, Hatem Ltaief                                  ///
 * ///                    Release Date: November, 15th 2009                                  ///
 * ///                    PLASMA is a software package provided by Univ. of Tennessee,       ///
 * ///                    Univ. of California Berkeley and Univ. of Colorado Denver          /// */
/* ///////////////////////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////// P /// U /// R /// P /// O /// S /// E /////////////////////////// */
//  testing_dgesv : Test LU routines (factorization and solve) using different scenarios :
//   - single call to PLASMA_dgesv
//   - successive calls to PLASMA_dgetrf and PLASMA_dgetrs
//   - successive calls to PLASMA_dgetrf, PLASMA_dtrsmpl and PLASMA_dtrsm 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cblas.h>
#include <plasma.h>
#include <../src/common.h>
#include <../src/lapack.h>
#include <../src/context.h>
#include <../src/allocate.h>
#include "../src/lapack.h"

#include "dplasma.h"
#include "scheduling.h"
#include "profiling.h"

int check_solution(int, int , double *, int, double *, double *, int, double);

PLASMA_desc descA;
PLASMA_desc descL;
int* _IPIV;

#ifdef DPLASMA_EXECUTE
double *work;
#endif

double time_elapsed, GFLOPS;

static inline double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

int IONE=1;
int ISEED[4] = {0,0,0,1};   /* initial seed for dlarnv() */

/* TODO Remove this ugly stuff */
extern int dgesv_private_memory_initialization(plasma_context_t*);
struct dplasma_memory_pool_t *work_pool = NULL;

int DPLASMA_dgetrf(int ncores, int M, int N, double *A, int LDA, double *L, int *IPIV,
                   int* pargc, char** pargv[])
{
    int NB, MT, NT, nbtasks;
    int status;
    double *Abdl;
    double *Lbdl;
    plasma_context_t *plasma;
#ifdef DPLASMA_EXECUTE
    dplasma_context_t* dplasma;
#endif  /* DPLASMA_EXECUTE */

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dgetrf", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Check input arguments */
    if (M < 0) {
        plasma_error("PLASMA_dgetrf", "illegal value of M");
        return -1;
    }
    if (N < 0) {
        plasma_error("PLASMA_dgetrf", "illegal value of N");
        return -2;
    }
    if (LDA < max(1, M)) {
        plasma_error("PLASMA_dgetrf", "illegal value of LDA");
        return -4;
    }
    /* Quick return */
    if (min(M, N) == 0)
        return PLASMA_SUCCESS;

    /* Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE */
    status = plasma_tune(PLASMA_FUNC_DGESV, M, N, 0);
    if (status != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dgetrf", "plasma_tune() failed");
        return status;
    }

    /* Set NT & NTRHS */
    NB = PLASMA_NB;
    MT = (M%NB==0) ? (M/NB) : (M/NB+1);
    NT = (N%NB==0) ? (N/NB) : (N/NB+1);

    /* Allocate memory for matrices in block layout */
    Abdl = (double *)plasma_shared_alloc(plasma, MT*NT*PLASMA_NBNBSIZE, PlasmaRealDouble);
    Lbdl = (double *)plasma_shared_alloc(plasma, MT*NT*PLASMA_IBNBSIZE, PlasmaRealDouble);
    if (Abdl == NULL || Lbdl == NULL) {
        plasma_error("PLASMA_dgetrf", "plasma_shared_alloc() failed");
        plasma_shared_free(plasma, Abdl);
        plasma_shared_free(plasma, Lbdl);
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }

    /* PLASMA_desc */ descA = plasma_desc_init(
        Abdl, PlasmaRealDouble,
        PLASMA_NB, PLASMA_NB, PLASMA_NBNBSIZE,
        M, N, 0, 0, M, N);

    /* PLASMA_desc */ descL = plasma_desc_init(
        Lbdl, PlasmaRealDouble,
        PLASMA_IB, PLASMA_NB, PLASMA_IBNBSIZE,
        M, N, 0, 0, M, N);

    plasma_parallel_call_3(plasma_lapack_to_tile,
        double*, A,
        int, LDA,
        PLASMA_desc, descA);

#ifdef DPLASMA_EXECUTE
    _IPIV = IPIV;
    dplasma = dplasma_init(ncores, pargc, pargv, PLASMA_NB);
    load_dplasma_objects(dplasma);

    dgesv_private_memory_initialization(plasma);
#if 0
    // TODO: this should be allocated per execution context.
    work = (double *)plasma_private_alloc(plasma, descL.mb*descL.nb, descL.dtyp);
#endif

    time_elapsed = get_cur_time();
    {
        expr_t* constant;

        constant = expr_new_int( PLASMA_NB );
        dplasma_assign_global_symbol( "PLASMA_NB", constant );

        constant = expr_new_int( NT );
        dplasma_assign_global_symbol( "NT", constant );
    }

    load_dplasma_hooks(dplasma);
    nbtasks = enumerate_dplasma_tasks(dplasma);
    time_elapsed = get_cur_time() - time_elapsed;
    printf("DPLASMA initialization %d %d %d %f\n",ncores,N,NB,time_elapsed);
    printf("NBTASKS to run: %d\n", nbtasks);

    {
        dplasma_execution_context_t exec_context;
        int it;

        /* I know what I'm doing ;) */
        exec_context.function = (dplasma_t*)dplasma_find("DGETRF");
        dplasma_set_initial_execution_context(&exec_context);

        time_elapsed = get_cur_time();
        dplasma_schedule(dplasma, &exec_context);
        
        it = dplasma_progress(dplasma);
        printf("main thread did %d tasks\n", it);
        
        time_elapsed = get_cur_time() - time_elapsed;
        printf("DPLASMA DGETRF %d %d %d %f %f\n",1,N,NB,time_elapsed, (2*N/1e3*N/1e3*N/1e3/3.0)/time_elapsed );
    }
#ifdef DPLASMA_PROFILING
    {
        char* filename = NULL;

        asprintf( &filename, "%s.profile", "dgels" );
        dplasma_profiling_dump_xml(filename);
        free(filename);
    }
#endif  /* DPLASMA_PROFILING */
    dplasma_fini(&dplasma);
#else /* DPLASMA_EXECUTE */
    time_elapsed = get_cur_time();
    /* Call the native interface */
    status = PLASMA_dgetrf_Tile(&descA, &descL, IPIV);
    time_elapsed = get_cur_time() - time_elapsed;
    printf("PLASMA DGETRF %d %d %d %f %f\n",1,N,NB,time_elapsed, (2*N/1e3*N/1e3*N/1e3/3.0)/time_elapsed );
#endif  /* DPLASMA_EXECUTE */

    if (status == PLASMA_SUCCESS) {
        /* Return L to the user */
        plasma_memcpy(L, Lbdl, MT*NT*PLASMA_IBNBSIZE, PlasmaRealDouble);

        plasma_parallel_call_3(plasma_tile_to_lapack,
            PLASMA_desc, descA,
            double*, A,
            int, LDA);
    }
    plasma_shared_free(plasma, Abdl);
    plasma_shared_free(plasma, Lbdl);
    return status;
}

/* /////////////////////////// P /// U /// R /// P /// O /// S /// E ///////////////////////////
 */
// PLASMA_dgetrf_Tile - Computes an LU factorization of a general M-by-N matrix A
// using the tile LU algorithm with partial tile pivoting with row interchanges.
// All matrices are passed through descriptors. All dimensions are taken from the descriptors.
/* ///////////////////// A /// R /// G /// U /// M /// E /// N /// T /// S /////////////////////
 */
// A        double* (INOUT)
//          On entry, the M-by-N matrix to be factored.
//          On exit, the tile factors L and U from the factorization.
//
// L        double* (OUT)
//          On exit, auxiliary factorization data, related to the tile L factor,
//          required by PLASMA_dgetrs to solve the system of equations.
//
// IPIV     int* (OUT)
//          The pivot indices that define the permutations (not equivalent to LAPACK).

/* ///////////// R /// E /// T /// U /// R /// N /////// V /// A /// L /// U /// E ///////////// */
//          = 0: successful exit
//          > 0: if i, U(i,i) is exactly zero. The factorization has been completed,
//               but the factor U is exactly singular, and division by zero will occur
//               if it is used to solve a system of equations.

/* //////////////////////////////////// C /// O /// D /// E //////////////////////////////////// */
int PLASMA_dgetrf_Tile(PLASMA_desc *A, PLASMA_desc *L, int *IPIV)
{
    PLASMA_desc descA = *A;
    PLASMA_desc descL = *L;
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dgetrf_Tile", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Check descriptors for correctness */
    if (plasma_desc_check(&descA) != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dgetrf_Tile", "invalid first descriptor");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if (plasma_desc_check(&descL) != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dgetrf_Tile", "invalid second descriptor");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    /* Check input arguments */
    if (descA.nb != descA.mb) {
        plasma_error("PLASMA_dgetrf_Tile", "only square tiles supported");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    /* Quick return */
/*
    if (min(M, N) == 0)
        return PLASMA_SUCCESS;
*/
    /* Clear IPIV and Lbdl */
    plasma_memzero(IPIV, descA.mt*descA.nt*PLASMA_NB, PlasmaInteger);
    plasma_memzero(descL.mat, descL.mt*descL.nt*PLASMA_IBNBSIZE, PlasmaRealDouble);

    /* Set INFO to SUCCESS */
    PLASMA_INFO = PLASMA_SUCCESS;

    plasma_parallel_call_3(plasma_pdgetrf,
        PLASMA_desc, descA,
        PLASMA_desc, descL,
        int*, IPIV);

    return PLASMA_INFO;
}

int main (int argc, char **argv)
{
    /* Check for valid arguments*/
    if (argc < 6){
        printf(" Proper Usage is : ./testing_dgesv ncores N LDA NRHS LDB with \n - ncores : number of cores \n - N : the size of the matrix \n - LDA : leading dimension of the matrix A \n - NRHS : number of RHS \n - LDB : leading dimension of the matrix B \n");
        exit(1);
    }

    int cores = atoi(argv[1]);
    int N     = atoi(argv[2]);
    int LDA   = atoi(argv[3]);
    int NRHS  = atoi(argv[4]);
    int LDB   = atoi(argv[5]);
    double eps;
    int info_solution;
    int i,j;
    int LDAxN = LDA*N;
    int LDBxNRHS = LDB*NRHS;

    double *A1 = (double *)malloc(LDA*N*(sizeof*A1));
    double *A2 = (double *)malloc(LDA*N*(sizeof*A2));
    double *B1 = (double *)malloc(LDB*NRHS*(sizeof*B1));
    double *B2 = (double *)malloc(LDB*NRHS*(sizeof*B2));
    double *L;
    int *IPIV;

    /* Check if unable to allocate memory */
    if ((!A1)||(!A2)||(!B1)||(!B2)){
        printf("Out of Memory \n ");
        exit(0);
    }

    /*----------------------------------------------------------
    *  TESTING DGESV
    */

    /*Plasma Initialize*/
#if defined(DPLASMA_EXECUTE)
    PLASMA_Init(1);
#else
    PLASMA_Init(cores);
#endif  /* defined(DPLASMA_EXECUTE) */

    /*
    PLASMA_Disable(PLASMA_AUTOTUNING);
    PLASMA_Set(PLASMA_TILE_SIZE, 6);
    PLASMA_Set(PLASMA_INNER_BLOCK_SIZE, 3);
    */

    /* Initialize A1 and A2 Matrix */
#if 0
    dlarnv(&IONE, ISEED, &LDAxN, A1);
#endif
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            A2[LDA*j+i] = A1[LDA*j+i] = 0.5 - (double)rand() / RAND_MAX;

    /* Initialize B1 and B2 */
#if 0
    dlarnv(&IONE, ISEED, &LDBxNRHS, B1);
#endif
    for (i = 0; i < N; i++)
        for (j = 0; j < NRHS; j++)
             B2[LDB*j+i] = B1[LDB*j+i] = 0.5 - (double)rand() / RAND_MAX;

    for( i = 0; i < N; i++ )
        A2[LDA*i+i] = A1[LDA*i+i] = A1[LDA*i+i] + 10 * N;

    /* PLASMA DGESV */
    PLASMA_Alloc_Workspace_dgesv(N, &L, &IPIV);

    eps = dlamch("Epsilon");

    /* PLASMA routines */
    DPLASMA_dgetrf(cores, N, N, A2, LDA, L, IPIV, &argc, &argv);
    PLASMA_dtrsmpl(N, NRHS, A2, LDA, L, IPIV, B2, LDB);
    PLASMA_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, N, NRHS, A2,
             LDA, B2, LDB);

    printf("\n");
    printf("------ TESTS FOR PLASMA DGETRF + DTRSMPL + DTRSM  ROUTINE -------  \n");
    printf("            Size of the Matrix %d by %d\n", N, N);
    printf("\n");
    printf(" The matrix A is randomly generated for each test.\n");
    printf("============\n");
    printf(" The relative machine precision (eps) is to be %e \n", eps);
    printf(" Computational tests pass if scaled residuals are less than 60.\n");

    /* Check the solution */
    info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

    if ((info_solution == 0)){
        printf("***************************************************\n");
        printf(" ---- TESTING DGETRF + DTRSMPL + DTRSM ... PASSED !\n");
        printf("***************************************************\n");
    }
    else{
        printf("**************************************************\n");
        printf(" - TESTING DGETRF + DTRSMPL + DTRSM ... FAILED !\n");
        printf("**************************************************\n");
    }

    free(A1); free(A2); free(B1); free(B2); free(IPIV); free(L);

    PLASMA_Finalize();

    exit(0);
}

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution of the linear system
 */

int check_solution(int N, int NRHS, double *A1, int LDA, double *B1, double *B2, int LDB, double eps )
{
    int info_solution;
    double Rnorm, Anorm, Xnorm, Bnorm;
    char norm='I';
    double alpha, beta;
    double *work = (double *)malloc(N*sizeof(double));

    alpha = 1.0;
    beta  = -1.0;

    Xnorm = dlange(&norm, &N, &NRHS, B2, &LDB, work);
    Anorm = dlange(&norm, &N, &N, A1, &LDA, work);
    Bnorm = dlange(&norm, &N, &NRHS, B1, &LDB, work);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, NRHS, N, (alpha), A1, LDA, B2, LDB, (beta), B1, LDB);
    Rnorm=dlange(&norm, &N, &NRHS, B1, &LDB, work);

    printf("============\n");
    printf("Checking the Residual of the solution \n");
    printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps));

    if ( isnan(Rnorm/((Anorm*Xnorm+Bnorm)*N*eps)) || (Rnorm/((Anorm*Xnorm+Bnorm)*N*eps) > 60.0) ){
        printf("-- The solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        printf("-- The solution is CORRECT ! \n");
        info_solution = 0;
    }

    free(work);

    return info_solution;
}
