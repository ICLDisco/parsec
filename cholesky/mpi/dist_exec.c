/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */



#include "mpi.h"
#include <getopt.h>


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cblas.h>
#include <math.h>
#include "plasma.h"
#include <../src/common.h>
#include <../src/lapack.h>
#include <../src/context.h>
#include <../src/allocate.h>
#include <sys/time.h>

#include "dplasma.h"
#include "scheduling.h"
#include "profiling.h"
#include "data_management.h"



double time_elapsed, GFLOPS;

double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

DPLASMA_desc descA;


int check_factorization(int, double*, double*, int, int , double);
int check_solution(int, int, double*, int, double*, double*, int, double);

int main(int argc, char ** argv){
    /* local variables*/

    int cores = 1;
    int nodes;
    int N = 0;
    int LDA = 0;
    int NRHS = 1;
    int LDB = 0;
    double eps;
    PLASMA_enum uplo;
    int info;
    int info_solution, info_factorization;
    int NminusOne; /* = N-1;*/
    int LDBxNRHS; /* = LDB*NRHS;*/
    PLASMA_desc local_desc;
    double *A1;
    double *A2;
    double *B1;
    double *B2;
    double *WORK;
    double *D;
    MPI_Request * requests;
    int req_count;
    dplasma_context_t* dplasma;

    struct option long_options[] =
        {
            {"lda", required_argument,  0, 'a'},
            {"matrix-size", required_argument, 0, 'n'},
            {"nrhs", required_argument,       0, 'r'},
            {"ldb",  required_argument,       0, 'b'},
            {"grid-rows",  required_argument, 0, 'g'},
            {"stile-size",  required_argument, 0, 's'},
            {"help",  no_argument, 0, 'h'},
            {"jdf", required_argument, 0, 'j'},
            {0, 0, 0, 0}
        };

    int option_index = 0;
    int c;
    
    /* mpi init */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &descA.mpi_rank); 

    /* plasma initialization */
    PLASMA_Init(cores);
    
    
    /* parse arguments */
    descA.GRIDrows = 1;
    descA.nrst = 1;
    descA.ncst = 1;
    printf("parsing arguments\n");
    while (1)
    {
        c = getopt_long (argc, argv, "a:n:r:b:g:s:j:h",
                         long_options, &option_index);
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c)
        {
            case 'a':
                LDA = atoi(optarg);
                printf("LDA set to %d\n", LDA);
                break;
                
            case 'n':
                N = atoi(optarg);
                printf("matrix size set to %d\n", N);
                break;
                
            case 'r':
                NRHS  = atoi(optarg);
                printf("number of RHS set to %d\n", NRHS);
                break;
                
            case 'b':
                LDB  = atoi(optarg);
                printf("LDB set to %d\n", LDB);
                break;
                
            case 'g':
                descA.GRIDrows = atoi(optarg);
                printf("%d rows od processes in the process grid\n", descA.GRIDrows);
                break;
            case 's':
                descA.nrst = atoi(optarg);
                descA.ncst = descA.nrst;
                printf("processes receives tiles by blocks of %dx%d\n", descA.nrst, descA.ncst);
                break;
            case '?': /* getopt_long already printed an error message. */
            case 'h':
            default:
                printf("must provide : -n, --matrix-size : the size of the matrix \n Optional arguments are:\n -a --lda : leading dimension of the matrix A (equal matrix size by default) \n -r --nrhs : number of RHS (default: 1) \n -b --ldb : leading dimension of the RHS B (equal matrix size by default)\n -g --grid-rows : number of processes row in the process grid (must divide the total number of processes (default: 1) \n -s --stile-size : number of tile per row (col) in a super tile (default: 1)\n");
                MPI_Abort( MPI_COMM_WORLD, 2);
        }
        
    }
    
    if (N == 0)
    {
        printf("must provide : -n, --matrix-size : the size of the matrix \n Optional arguments are:\n -a --lda : leading dimension of the matrix A (equal matrix size by default) \n -r --nrhs : number of RHS (default: 1) \n -b --ldb : leading dimension of the RHS B (equal matrix size by default)\n -g --grid-rows : number of processes row in the process grid (must divide the total number of processes (default: 1) \n -s --stile-size : number of tile per row (col) in a super tile (default: 1)\n");
        MPI_Abort( MPI_COMM_WORLD, 2 );
        
    }
    if(LDA <= 0)
        LDA = N;
    if (LDB <= 0)
        LDB = N;
    
    if (descA.ncst <= 0)
    {
        printf("select a positive value for super tile size\n");
        MPI_Abort( MPI_COMM_WORLD, 2 );
    }
    
    if ((nodes % descA.GRIDrows) != 0 )
    {
        printf("GRIDrows %d does not devide the total number of nodes %d\n", descA.GRIDrows, nodes);
        MPI_Abort( MPI_COMM_WORLD, 2 );
    }
    
    descA.GRIDcols = nodes / descA.GRIDrows ;
    
    /* rank 0 specific init */
    if (descA.mpi_rank == 0)
    {
        A1   = (double *)malloc(LDA*N*sizeof(double));
        A2   = (double *)malloc(LDA*N*sizeof(double));
        B1   = (double *)malloc(LDB*NRHS*sizeof(double));
        B2   = (double *)malloc(LDB*NRHS*sizeof(double));
        WORK = (double *)malloc(2*LDA*sizeof(double));
        D    = (double *)malloc(LDA*sizeof(double));
        
        NminusOne = N-1;
        LDBxNRHS = LDB*NRHS;
        
        /* generating a random matrix */
        //printf("generating matrix on rank 0\n");
        generate_matrix(N, A1, A2,  B1, B2,  WORK, D, LDA, NRHS, LDB);
        
        // printf("tiling matrix\n");
        tiling(&uplo, N, A2, LDA, &local_desc);
        //printf("structure initialization\n");
        dplasma_desc_init(&local_desc, &descA);
        printf("Data distribution\n");
        distribute_data(&local_desc, &descA, &requests, &req_count);
    }
    else
    { /* prepare data for block reception  */
        /* initialize main tiles description structure (Bcast inside) */
        dplasma_desc_init(NULL, &descA);
	    distribute_data(NULL, &descA, &requests, &req_count);
    }
    /* checking local data ready */
    is_data_distributed(&descA, requests, req_count);



    time_elapsed = get_cur_time();
    dplasma = dplasma_init(cores, NULL, NULL );
    load_dplasma_objects(dplasma);
    {
        expr_t* constant;

        constant = expr_new_int( descA.nb );
        dplasma_assign_global_symbol( "NB", constant );
        constant = expr_new_int( descA.nt );
        dplasma_assign_global_symbol( "SIZE", constant );
        constant = expr_new_int( descA.GRIDrows );
        dplasma_assign_global_symbol( "GRIDrows", constant );
        constant = expr_new_int( descA.GRIDcols );
        dplasma_assign_global_symbol( "GRIDcols", constant );
        constant = expr_new_int( descA.rowRANK );
        dplasma_assign_global_symbol( "rowRANK", constant );
        constant = expr_new_int( descA.colRANK );
        dplasma_assign_global_symbol( "colRANK", constant );
    }
    load_dplasma_hooks(dplasma);
    enumerate_dplasma_tasks();
    time_elapsed = get_cur_time() - time_elapsed;
    printf("DPLASMA initialization %d %d %d %f\n",1,descA.n,descA.nb,time_elapsed);
    
#ifdef DIST_VERIFICATION    
    data_dist_verif(&local_desc, &descA);
    if (descA.mpi_rank == 0)
        plasma_dump(&local_desc);
    data_dump(&descA);
#endif

    /* lets rock! */
    time_elapsed = get_cur_time();
    if(descA.mpi_rank == 0)
    {
        dplasma_execution_context_t exec_context;
        
        /* I know what I'm doing ;) */
#define N descA.n
#define NB descA.nb
        printf("DPLASMA DPOTRF %d %d %d %f %f\n",1,N,NB,time_elapsed, (N/1e3*N/1e3*N/1e3/2.0)/time_elapsed );
        exec_context.function = (dplasma_t*)dplasma_find("POTRF");
        dplasma_set_initial_execution_context(&exec_context);
        dplasma_schedule(dplasma, &exec_context);
    }
    dplasma_progress(dplasma);
    time_elapsed = get_cur_time() - time_elapsed;

    gather_data(&local_desc, &descA);
    
    if(descA.mpi_rank == 0) 
    {
        untiling(&uplo, N, A2, LDA, &local_desc);
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
    }
#undef N
#undef NB
        
    PLASMA_Finalize();
    MPI_Finalize();
    return 0;
}

/*------------------------------------------------------------------------
 * *  Check the factorization of the matrix A2
 * */

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
    double *work     = (double *)malloc(N*sizeof(double));
    
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
 * *  Check the accuracy of the solution of the linear system
 * */

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



