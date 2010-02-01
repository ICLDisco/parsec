/* ///////////////////////////// P /// L /// A /// S /// M /// A /////////////////////////////// */
/* ///                    PLASMA testing routines (version 2.1.0)                            ///
 * ///                    Author: Bilel Hadri, Hatem Ltaief                                  ///
 * ///                    Release Date: November, 15th 2009                                  ///
 * ///                    PLASMA is a software package provided by Univ. of Tennessee,       ///
 * ///                    Univ. of California Berkeley and Univ. of Colorado Denver          /// */
/* ///////////////////////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////// P /// U /// R /// P /// O /// S /// E /////////////////////////// */
//  testing_dgels : Test least square routines (factorization and solve) using different scenarios :
//   - single call to PLASMA_dgels
//   - successive calls to PLAMSA_zgeqrf and PLASMA_dgeqrs for overdetermined problems
//   - successive calls to PLAMSA_zgeqrf, PLASMA_dormqr and PLASMA_dtrsm for overdetermined problems
//   - successive calls to PLAMSA_zgelqf and PLASMA_dgelqs for underdetermined problems
//   - successive calls to PLAMSA_zgeqrf, PLASMA_dtrsm and PLASMA_dormlq for overdetermined problems

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
int asprintf(char **strp, const char *fmt, ...);


#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <../src/common.h>
#include <../src/lapack.h>
#include <../src/context.h>
#include <../src/allocate.h>
#include <sys/time.h>

#include "dplasma.h"
#include "scheduling.h"
#include "profiling.h"

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

double time_elapsed, GFLOPS;

PLASMA_desc descA;
PLASMA_desc descT;

#ifdef DPLASMA_EXECUTE
    double *work, *tau;
#endif

static inline double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

int check_orthogonality(int, int, int, double*, double);
int check_factorization(int, int, double*, double*, int, double*, double);
int check_solution(int, int, int, double*, int, double*, double*, int, double);

int IONE=1;
int ISEED[4] = {0,0,0,1};   /* initial seed for dlarnv() */

/* TODO Remove this ugly stuff */
extern int dgels_private_memory_initialization(plasma_context_t*);
struct dplasma_memory_pool_t *work_pool = NULL, *tau_pool = NULL;

int DPLASMA_dgeqrf(int ncores, int M, int N, double *A, int LDA, double *T, int* pargc, char** pargv[])
{
    int NB, MT, NT, nbtasks;
    int status;
    double *Abdl;
    double *Tbdl;
    plasma_context_t *plasma;
#ifdef DPLASMA_EXECUTE
    dplasma_context_t* dplasma;
#endif  /* DPLASMA_EXECUTE */

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dgeqrf", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Check input arguments */
    if (M < 0) {
        plasma_error("PLASMA_dgeqrf", "illegal value of M");
        return -1;
    }
    if (N < 0) {
        plasma_error("PLASMA_dgeqrf", "illegal value of N");
        return -2;
    }
    if (LDA < max(1, M)) {
        plasma_error("PLASMA_dgeqrf", "illegal value of LDA");
        return -4;
    }
    /* Quick return */
    if (min(M, N) == 0)
        return PLASMA_SUCCESS;

    /* Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE */
    status = plasma_tune(PLASMA_FUNC_DGELS, M, N, 0);
    if (status != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dgeqrf", "plasma_tune() failed");
        return status;
    }

    /* Set MT & NT */
    NB = PLASMA_NB;
    MT = (M%NB==0) ? (M/NB) : (M/NB+1);
    NT = (N%NB==0) ? (N/NB) : (N/NB+1);

    /* Allocate memory for matrices in block layout */
    Abdl = (double *)plasma_shared_alloc(plasma, MT*NT*PLASMA_NBNBSIZE, PlasmaRealDouble);
    Tbdl = (double *)plasma_shared_alloc(plasma, MT*NT*PLASMA_IBNBSIZE, PlasmaRealDouble);
    if (Abdl == NULL || Tbdl == NULL) {
        plasma_error("PLASMA_dgeqrf", "plasma_shared_alloc() failed");
        plasma_shared_free(plasma, Abdl);
        plasma_shared_free(plasma, Tbdl);
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }

    /* PLASMA_desc */ descA = plasma_desc_init(
        Abdl, PlasmaRealDouble,
        PLASMA_NB, PLASMA_NB, PLASMA_NBNBSIZE,
        M, N, 0, 0, M, N);

    /* PLASMA_desc */ descT = plasma_desc_init(
        Tbdl, PlasmaRealDouble,
        PLASMA_IB, PLASMA_NB, PLASMA_IBNBSIZE,
        M, N, 0, 0, M, N);

    plasma_parallel_call_3(plasma_lapack_to_tile,
        double*, A,
        int, LDA,
        PLASMA_desc, descA);

#ifdef DPLASMA_EXECUTE
    dplasma = dplasma_init(ncores, pargc, pargv);
    load_dplasma_objects(dplasma);

    dgels_private_memory_initialization(plasma);
#if 0
    // TODO: this should be allocated per execution context.
    work = (double *)plasma_private_alloc(plasma, descT.mb*descT.nb, descT.dtyp);
    tau = (double *)plasma_private_alloc(plasma, descA.nb, descA.dtyp);
#endif

    time_elapsed = get_cur_time();
    {
        expr_t* constant;

        constant = expr_new_int( PLASMA_NB );
        dplasma_assign_global_symbol( "NB", constant );

        constant = expr_new_int( NT );
        dplasma_assign_global_symbol( "NT", constant );
        constant = expr_new_int( MT );
        dplasma_assign_global_symbol( "MT", constant );
        /*constant = expr_new_int( ((MT < NT) ? MT : NT) );
          dplasma_assign_global_symbol( "MINMTNT", constant );*/
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
        exec_context.function = (dplasma_t*)dplasma_find("DGEQRT");
        dplasma_set_initial_execution_context(&exec_context);

        time_elapsed = get_cur_time();
        dplasma_schedule(dplasma, &exec_context);
        
        it = dplasma_progress(dplasma);
        printf("main thread did %d tasks\n", it);
        
        time_elapsed = get_cur_time() - time_elapsed;
        printf("DPLASMA DGEQRF %d %d %d %f %f\n",1,N,NB,time_elapsed, (4*N/1e3*N/1e3*N/1e3/3.0)/time_elapsed );
    }
#ifdef DPLASMA_PROFILING
    {
        char* filename = NULL;

        asprintf( &filename, "%s.svg", "dgels" );
        dplasma_profiling_dump_svg(dplasma, filename);
        free(filename);
    }
#endif  /* DPLASMA_PROFILING */
    dplasma_fini(&dplasma);
#else // DPLASMA_EXECUTE
    time_elapsed = get_cur_time();
    plasma_parallel_call_2(plasma_pdgeqrf,
        PLASMA_desc, descA,
        PLASMA_desc, descT);
    time_elapsed = get_cur_time() - time_elapsed;
    printf("PLASMA DGEQRF %d %d %d %f %f\n",ncores,N,NB,time_elapsed, (4*N/1e3*N/1e3*N/1e3/3.0)/time_elapsed );
#endif // DPLASMA_EXECUTE

    if (status == PLASMA_SUCCESS) {
        /* Return T to the user */
        plasma_memcpy(T, Tbdl, MT*NT*PLASMA_IBNBSIZE, PlasmaRealDouble);

        plasma_parallel_call_3(plasma_tile_to_lapack,
            PLASMA_desc, descA,
            double*, A,
            int, LDA);
    }
    plasma_shared_free(plasma, Abdl);
    plasma_shared_free(plasma, Tbdl);

    return status;
}



int main (int argc, char **argv)
{
    /* Check for number of arguments*/
    if ( argc != 7){
        printf(" Proper Usage is : ./testing_dgesv ncores M N LDA NRHS LDB with \n - ncores : number of cores \n - M : number of rows of the matrix A \n - N : number of columns of the matrix A \n - LDA : leading dimension of the matrix A \n - NRHS : number of RHS \n - LDB : leading dimension of the matrix B\n");
        exit(1);
    }

    int cores = atoi(argv[1]);
    int M     = atoi(argv[2]);
    int N     = atoi(argv[3]);
    int LDA   = atoi(argv[4]);
    int NRHS  = atoi(argv[5]);
    int LDB   = atoi(argv[6]);

    int K = min(M, N);
    double eps;
    int info_ortho, info_solution, info_factorization;
    int i,j;
    int LDAxN = LDA*N;
    int LDBxNRHS = LDB*NRHS;

    double *A1 = (double *)malloc(LDA*N*sizeof(double));
    double *A2 = (double *)malloc(LDA*N*sizeof(double));
    double *B1 = (double *)malloc(LDB*NRHS*sizeof(double));
    double *B2 = (double *)malloc(LDB*NRHS*sizeof(double));
    double *Q  = (double *)malloc(LDA*N*sizeof(double));
    double *T;

    /* Check if unable to allocate memory */
    if ((!A1)||(!A2)||(!B1)||(!B2)||(!Q)){
        printf("Out of Memory \n ");
        exit(0);
    }

    /* Plasma Initialization */
#if defined(DPLASMA_EXECUTE)
    PLASMA_Init(1);
#else
    PLASMA_Init(cores);
#endif  /* defined(DPLASMA_EXECUTE) */

    PLASMA_Alloc_Workspace_dgels(M, N, &T);

    /*
    PLASMA_Disable(PLASMA_AUTOTUNING);
    PLASMA_Set(PLASMA_TILE_SIZE, 6);
    PLASMA_Set(PLASMA_INNER_BLOCK_SIZE, 3);
    */

    eps = dlamch("Epsilon");

    /*----------------------------------------------------------
    *  TESTING DGELS
    */

    /* Initialize A1 and A2 */
#if 0
    dlarnv(&IONE, ISEED, &LDAxN, A1);
#endif
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            A2[LDA*j+i] = A1[LDA*j+i] = 0.5 - (double)rand() / RAND_MAX;

    /* Initialize B1 and B2 */
#if 0
    dlarnv(&IONE, ISEED, &LDBxNRHS, B1);
#endif
    for (i = 0; i < M; i++)
        for (j = 0; j < NRHS; j++)
             B2[LDB*j+i] = B1[LDB*j+i] = 0.5 - (double)rand() / RAND_MAX;

    // memset((void*)Q, 0, LDA*N*sizeof(double));
        for (i = 0; i < K; i++)
            Q[LDA*i+i] = 1.0;

    /* PLASMA DGELS */
    if (M >= N) {
        printf("\n");
        printf("------ TESTS FOR PLASMA DGEQRF + DGEQRS ROUTINE -------  \n");
        printf("            Size of the Matrix %d by %d\n", M, N);
        printf("\n");
        printf(" The matrix A is randomly generated for each test.\n");
        printf("============\n");
        printf(" The relative machine precision (eps) is to be %e \n", eps);
        printf(" Computational tests pass if scaled residuals are less than 60.\n");

        /* Plasma routines */
	double time =- get_cur_time();

	// PLASMA_dgels(PlasmaNoTrans, M, N, NRHS, A2, LDA, T, B2, LDB);
	DPLASMA_dgeqrf(cores, M, N, A2, LDA, T, &argc, &argv);
	PLASMA_dorgqr(M, N, K, A2, LDA, T, Q, LDA);
	PLASMA_dgeqrs(M, N, NRHS, A2, LDA, T, B2, LDB);

	time += get_cur_time();
	double flops = (4 * N/1e3*N/1e3*N/1e3/3 ) / (double)(time)  ;
	printf("PLASMALPK DGELS %2d %5d %5d %5d %2d %5d %4.5f %4.5f\n", cores, M, N, LDA, NRHS, LDB, time, flops);

        /* Check the orthogonality, factorization and the solution */
        info_ortho = check_orthogonality(M, N, LDA, Q, eps);
        info_factorization = check_factorization(M, N, A1, A2, LDA, Q, eps);
        info_solution = check_solution(M, N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if ((info_solution == 0)&(info_factorization == 0)&(info_ortho == 0)) {
            printf("***************************************************\n");
            printf(" ---- TESTING DGEQRF + DGEQRS ............ PASSED !\n");
            printf("***************************************************\n");
        }
        else{
            printf("***************************************************\n");
            printf(" - TESTING DGEQRF + DGEQRS ... FAILED !\n");
            printf("***************************************************\n");
        }
    }

    free(A1); free(A2); free(B1); free(B2); free(Q); free(T);

    PLASMA_Finalize();

    exit(0);
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */

int check_orthogonality(int M, int N, int LDQ, double *Q, double eps)
{
    double alpha, beta;
    double normQ;
    char norm='I';
    int info_ortho;
    int i;
    int minMN = min(M, N);

    double *work = (double *)malloc(minMN*sizeof(double));

    alpha = 1.0;
    beta  = -1.0;

    /* Build the idendity matrix USE DLASET?*/
    double *Id = (double *) malloc(minMN*minMN*sizeof(double));
    memset((void*)Id, 0, minMN*minMN*sizeof(double));
    for (i = 0; i < minMN; i++)
        Id[i*minMN+i] = (double)1.0;

    /* Perform Id - Q'Q */
    if (M >= N)
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, N, M, alpha, Q, LDQ, beta, Id, N);
    else
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, M, N, alpha, Q, LDQ, beta, Id, M);

    normQ = dlansy(&norm, lapack_const(PlasmaUpper), &minMN, Id, &minMN, work);

    printf("============\n");
    printf("Checking the orthogonality of Q \n");
    printf("||Id-Q'*Q||_oo / (N*eps) = %e \n",normQ/(minMN*eps));

    if ( isnan(normQ / (minMN * eps)) || (normQ / (minMN * eps) > 60.0) ) {
        printf("-- Orthogonality is suspicious ! \n");
        info_ortho=1;
    }
    else {
        printf("-- Orthogonality is CORRECT ! \n");
        info_ortho=0;
    }

    free(work); free(Id);

    return info_ortho;
}

/*------------------------------------------------------------
 *  Check the factorization QR
 */

int check_factorization(int M, int N, double *A1, double *A2, int LDA, double *Q, double eps )
{
    double Anorm, Rnorm;
    double alpha, beta;
    char norm='I';
    int info_factorization;
    int i,j;

    double *Ql       = (double *)malloc(M*N*sizeof(double));
    double *Residual = (double *)malloc(M*N*sizeof(double));
    double *work              = (double *)malloc(max(M,N)*sizeof(double));

    alpha=1.0;
    beta=0.0;

    if (M >= N) {
        /* Extract the R */
        double *R = (double *)malloc(N*N*sizeof(double));
        memset((void*)R, 0, N*N*sizeof(double));
        dlacpy("U", &M, &N, A2, &LDA, R, &N);

        /* Perform Ql=Q*R */
        memset((void*)Ql, 0, M*N*sizeof(double));
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, N, (alpha), Q, LDA, R, N, (beta), Ql, M);
        free(R);
    }
    else {
        /* Extract the L */
        double *L = (double *)malloc(M*M*sizeof(double));
        memset((void*)L, 0, M*M*sizeof(double));
        dlacpy("L", &M, &N, A2, &LDA, L, &M);

    /* Perform Ql=LQ */
        memset((void*)Ql, 0, M*N*sizeof(double));
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, M, (alpha), L, M, Q, LDA, (beta), Ql, M);
        free(L);
    }

    /* Compute the Residual */
    for (i = 0; i < M; i++)
        for (j = 0 ; j < N; j++)
            Residual[j*M+i] = A1[j*LDA+i]-Ql[j*M+i];

    Rnorm=dlange(&norm, &M, &N, Residual, &M, work);
    Anorm=dlange(&norm, &M, &N, A2, &LDA, work);

    if (M >= N) {
        printf("============\n");
        printf("Checking the QR Factorization \n");
        printf("-- ||A-QR||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));
    }
    else {
        printf("============\n");
        printf("Checking the LQ Factorization \n");
        printf("-- ||A-LQ||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));
    }

    if (isnan(Rnorm / (Anorm * N *eps)) || (Rnorm / (Anorm * N * eps) > 60.0) ) {
        printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else {
        printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    free(work); free(Ql); free(Residual);

    return info_factorization;
}

/*--------------------------------------------------------------
 * Check the solution
 */

int check_solution(int M, int N, int NRHS, double *A1, int LDA, double *B1, double *B2, int LDB, double eps)
{
    int info_solution;
    double Rnorm, Anorm, Xnorm, Bnorm;
    char norm='I';
    double alpha, beta;

    double *work = (double *)malloc(max(M, N)* sizeof(double));

    alpha = 1.0;
    beta  = -1.0;

    Anorm = dlange(&norm, &M, &N, A1, &LDA, work);
    Xnorm = dlange(&norm, &M, &NRHS, B2, &LDB, work);
    Bnorm = dlange(&norm, &N, &NRHS, B1, &LDB, work);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, NRHS, N, (alpha), A1, LDA, B2, LDB, (beta), B1, LDB);

    if (M >= N) {
       double *Residual = (double *)malloc(M*NRHS*sizeof(double));
       memset((void*)Residual, 0, M*NRHS*sizeof(double));
       cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, NRHS, M, (alpha), A1, LDA, B1, LDB, (beta), Residual, M);
       Rnorm = dlange(&norm, &M, &NRHS, Residual, &M, work);
       free(Residual);
    }
    else {
       double *Residual = (double *)malloc(N*NRHS*sizeof(double));
       memset((void*)Residual, 0, N*NRHS*sizeof(double));
       cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, NRHS, M, (alpha), A1, LDA, B1, LDB, (beta), Residual, N);
       Rnorm = dlange(&norm, &N, &NRHS, Residual, &N, work);
       free(Residual);
    }

    printf("============\n");
    printf("Checking the Residual of the solution \n");
    printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||)_oo.N.eps) = %e \n",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps));

    if (isnan(Rnorm / ((Anorm * Xnorm + Bnorm) * N * eps)) || (Rnorm / ((Anorm * Xnorm + Bnorm) * N * eps) > 60.0) ) {
         printf("-- The solution is suspicious ! \n");
         info_solution = 1;
    }
    else {
         printf("-- The solution is CORRECT ! \n");
         info_solution= 0 ;
    }

    free(work);

    return info_solution;
}
