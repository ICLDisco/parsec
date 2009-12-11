/*
-- PLASMA Test Routine
   University of Tennessee
   November 2008

-- Purpose
   TESTING PLASMA_DPOSV
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

extern int load_dplasma_hooks( void );

double time_elapsed, GFLOPS;

PLASMA_desc descA;

int check_factorization(int, double*, double*, int, int , double);
int check_solution(int, int, double*, int, double*, double*, int, double);
double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

int IONE=1;
int ISEED[4] = {0,0,0,1};   /* initial seed for dlarnv() */

int DPLASMA_dpotrf(PLASMA_enum uplo, int N, double *A, int LDA)
{
    int NB, NT;
    int status;
    double *Abdl;
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA_dpotrf", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("PLASMA_dpotrf", "illegal value of uplo");
        return -1;
    }
    if (N < 0) {
        plasma_error("PLASMA_dpotrf", "illegal value of N");
        return -2;
    }
    if (LDA < max(1, N)) {
        plasma_error("PLASMA_dpotrf", "illegal value of LDA");
        return -4;
    }
    /* Quick return */
    if (max(N, 0) == 0)
        return PLASMA_SUCCESS;

    /* Tune NB depending on M, N & NRHS; Set NBNBSIZE */
    status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
    if (status != PLASMA_SUCCESS) {
        plasma_error("PLASMA_dpotrf", "plasma_tune() failed");
        return status;
    }

    /* Set NT */
    NB = PLASMA_NB;
    NT = (N%NB==0) ? (N/NB) : (N/NB+1);

    /* Allocate memory for matrices in block layout */
    Abdl = (double *)plasma_shared_alloc(plasma, NT*NT*PLASMA_NBNBSIZE, PlasmaRealDouble);
    if (Abdl == NULL) {
        plasma_error("PLASMA_dpotrf", "plasma_shared_alloc() failed");
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }

    /*PLASMA_desc*/ descA = plasma_desc_init(
                                             Abdl, PlasmaRealDouble,
                                             PLASMA_NB, PLASMA_NB, PLASMA_NBNBSIZE,
                                             N, N, 0, 0, N, N);

    plasma_parallel_call_3(plasma_lapack_to_tile,
                           double*, A,
                           int, LDA,
                           PLASMA_desc, descA);

    /* Init DPLASMA */
#ifdef DPLASMA_EXECUTE
    load_dplasma_objects();

    time_elapsed = get_cur_time();
    {
        expr_t* constant;

        constant = expr_new_int( PLASMA_NB );
        dplasma_assign_global_symbol( "NB", constant );

        constant = expr_new_int( NT );
        dplasma_assign_global_symbol( "SIZE", constant );
    }

    load_dplasma_hooks();
    time_elapsed = get_cur_time() - time_elapsed;
    printf("DPLASMA initialization %d %d %d %f\n",1,N,NB,time_elapsed);

#ifdef DPLASMA_PROFILING
    dplasma_profiling_init(1024);
#endif  /* DPLASMA_PROFILING */

    {
        dplasma_execution_context_t exec_context;
        /* I know what I'm doing ;) */
        exec_context.function = (dplasma_t*)dplasma_find("POTRF");
        dplasma_set_initial_execution_context(&exec_context);
        time_elapsed = get_cur_time();
        dplasma_schedule(&exec_context);
        dplasma_progress();
        time_elapsed = get_cur_time() - time_elapsed;
        printf("DPLASMA DPOTRF %d %d %d %f %f\n",1,N,NB,time_elapsed, (N/1e3*N/1e3*N/1e3/2.0)/time_elapsed );
    }
#ifdef DPLASMA_PROFILING
    {
        char* filename = NULL;

        asprintf( &filename, "%s.svg", "dposv" );
        dplasma_profiling_dump_svg(filename);
        dplasma_profiling_fini();
        free(filename);
    }
#endif  /* DPLASMA_PROFILING */
#else
    time_elapsed = get_cur_time();
    plasma_parallel_call_2(plasma_pdpotrf,
                           PLASMA_enum, uplo,
                           PLASMA_desc, descA);
    time_elapsed = get_cur_time() - time_elapsed;
    printf("PLASMA DPOTRF %d %d %d %f %f\n",1,N,NB,time_elapsed, (N/1e3*N/1e3*N/1e3/2.0)/time_elapsed );
#endif

    if (PLASMA_INFO == PLASMA_SUCCESS)
    {
        plasma_parallel_call_3(plasma_tile_to_lapack,
                               PLASMA_desc, descA,
                               double*, A,
                               int, LDA);
    }
    plasma_shared_free(plasma, Abdl);

    return PLASMA_INFO;
}

int main (int argc, char **argv)
{
   /* Check for number of arguments*/
   if (argc != 6){
       printf(" Proper Usage is : ./%s ncores N LDA NRHS LDB with \n - ncores : number of cores \n - N : the size of the matrix \n - LDA : leading dimension of the matrix A \n - NRHS : number of RHS \n - LDB : leading dimension of the RHS B \n", (char*)argv[0]);
       exit(1);
   }

   int cores = atoi(argv[1]);
   int N     = atoi(argv[2]);
   int LDA   = atoi(argv[3]);
   int NRHS  = atoi(argv[4]);
   int LDB   = atoi(argv[5]);
   double eps;
   int uplo;
   int info;
   int info_solution, info_factorization;
   int i,j;
   int NminusOne = N-1;
   int LDBxNRHS = LDB*NRHS;

   double *A1   = (double *)malloc(LDA*N*sizeof(double));
   double *A2   = (double *)malloc(LDA*N*sizeof(double));
   double *B1   = (double *)malloc(LDB*NRHS*sizeof(double));
   double *B2   = (double *)malloc(LDB*NRHS*sizeof(double));
   double *WORK = (double *)malloc(2*LDA*sizeof(double));
   double *D                = (double *)malloc(LDA*sizeof(double));

   /* Check if unable to allocate memory */
   if ((!A1)||(!A2)||(!B1)||(!B2)){
       printf("Out of Memory \n ");
       exit(0);
   }

   /* Plasma Initialize */
   PLASMA_Init(cores);


   /*-------------------------------------------------------------
   *  TESTING DPOTRF + DPOTRS
   */
   /* Initialize A1 and A2 for Symmetric Positive Matrix */
   dlarnv(&IONE, ISEED, &LDA, D);
   dlagsy(&N, &NminusOne, D, A1, &LDA, ISEED, WORK, &info);
   for ( i = 0; i < N; i++)
       for (  j = 0; j < N; j++)
           A2[LDA*j+i] = A1[LDA*j+i];

   for ( i = 0; i < N; i++){
       A1[LDA*i+i] = A1[LDA*i+i]+ N ;
       A2[LDA*i+i] = A1[LDA*i+i];
   }

   /* Initialize B1 and B2 */
   dlarnv(&IONE, ISEED, &LDBxNRHS, B1);
   for ( i = 0; i < N; i++)
       for ( j = 0; j < NRHS; j++)
           B2[LDB*j+i] = B1[LDB*j+i];

   /* Plasma routines */
   uplo=PlasmaLower;
   DPLASMA_dpotrf(uplo, N, A2, LDA);
   PLASMA_dpotrs(uplo, N, NRHS, A2, LDA, B2, LDB);
   eps = (double) 1.0e-13;  /* dlamch("Epsilon");*/
   printf("\n");
   printf("------ TESTS FOR PLASMA DPOTRF + DPOTRS ROUTINE -------  \n");
   printf("            Size of the Matrix %d by %d\n", N, N);
   printf("\n");
   printf(" The matrix A is randomly generated for each test.\n");
   printf("============\n");
   printf(" The relative machine precision (eps) is to be %e \n", eps);
   printf(" Computational tests pass if scaled residuals are less than 10.\n");

   /* Check the factorization and the solution */
   info_factorization = check_factorization( N, A1, A2, LDA, uplo, eps);
   info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

   if ((info_solution == 0)&(info_factorization == 0)){
       printf("***************************************************\n");
       printf(" ---- TESTING DPOTRF + DPOTRS ............ PASSED !\n");
       printf("***************************************************\n");
   }
   else{
       printf("****************************************************\n");
       printf(" - TESTING DPOTRF + DPOTRS ... FAILED !\n");
       printf("****************************************************\n");
   }

   free(A1); free(A2); free(B1); free(B2); free(WORK); free(D);

   PLASMA_Finalize();

   exit(0);
}


/*------------------------------------------------------------------------
*  Check the factorization of the matrix A2
*/

int check_factorization(int N, double *A1, double *A2, int LDA, int uplo, double eps)
{
   double Anorm, Rnorm;
   double alpha;
   char norm='I';
   int info_factorization;
   int i,j;

   double *Residual = (double *)malloc(N*N*sizeof(double));
   double *L1       = (double *)malloc(N*N*sizeof(double));
   double *L2       = (double *)malloc(N*N*sizeof(double));
   double *work              = (double *)malloc(N*sizeof(double));

   memset((void*)L1, 0, N*N*sizeof(double));
   memset((void*)L2, 0, N*N*sizeof(double));

   alpha= 1.0;

   dlacpy("ALL", &N, &N, A1, &LDA, Residual, &N);

   /* Dealing with L'L or U'U  */
   if (uplo == PlasmaUpper){
       dlacpy(lapack_const(PlasmaUpper), &N, &N, A2, &LDA, L1, &N);
       dlacpy(lapack_const(PlasmaUpper), &N, &N, A2, &LDA, L2, &N);
       cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
   }
   else{
       dlacpy(lapack_const(PlasmaLower), &N, &N, A2, &LDA, L1, &N);
       dlacpy(lapack_const(PlasmaLower), &N, &N, A2, &LDA, L2, &N);
       cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, (alpha), L1, N, L2, N);
   }

   /* Compute the Residual || A -L'L|| */
   for (i = 0; i < N; i++)
       for (j = 0; j < N; j++)
          Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];

   Rnorm = dlange(&norm, &N, &N, Residual, &N, work);
   Anorm = dlange(&norm, &N, &N, A1, &LDA, work);

   printf("============\n");
   printf("Checking the Cholesky Factorization \n");
   printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));

   if ( isnan(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 10.0) ){
       printf("-- Factorization is suspicious ! \n");
       info_factorization = 1;
   }
   else{
       printf("-- Factorization is CORRECT ! \n");
       info_factorization = 0;
   }

   free(Residual); free(L1); free(L2); free(work);

   return info_factorization;
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
   Rnorm = dlange(&norm, &N, &NRHS, B1, &LDB, work);

   printf("============\n");
   printf("Checking the Residual of the solution \n");
   printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps));

   if (Rnorm/((Anorm*Xnorm+Bnorm)*N*eps) > 10.0){
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
