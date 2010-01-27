/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifdef USE_MPI
#include "mpi.h"
#endif  /* defined(USE_MPI) */
#include <getopt.h>


#include <stdlib.h>
#include <stdio.h>
/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

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

/* globals and argv set values */
PLASMA_desc local_desc;
DPLASMA_desc descA;
int cores = 1;
int nodes = 1;
#define N (descA.n)
#define NB (descA.nb)
#define rank (descA.mpi_rank)
int LDA = 0;
int NRHS = 1;
int LDB = 0;


/* int asprintf(char **strp, const char *fmt, ...);*/

static void dague_fini(dplasma_context_t* context);
static void dague_init(int argc, char **argv);
static void cleanup_exit(int ret);
static dplasma_context_t *setup_dplasma(void);

#ifdef DO_THE_NASTY_VALIDATIONS
static int check_factorization(int, double*, double*, int, int , double);
static int check_solution(int, int, double*, int, double*, double*, int, double);
#endif

/* timing profiling etc */
double time_elapsed, GFLOPS;

double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}
#define TIME_START() do { time_elapsed = get_cur_time(); } while(0)
#define TIME_STOP() do { time_elapsed = get_cur_time() - time_elapsed; } while(0)
#define TIME_PRINT(print) do { \
time_elapsed = get_cur_time() - time_elapsed; \
printf("TIMED %f doing\t", time_elapsed); \
printf print; \
} while(0)



int main(int argc, char ** argv){
    /* local variables*/
    double eps;
    double flops, gflops;
    PLASMA_enum uplo;
    double *A2;
#if defined(DO_THE_NASTY_VALIDATIONS)
    int info_solution, info_factorization;
    int NminusOne; /* = N-1;*/
    int LDBxNRHS; /* = LDB*NRHS;*/
    double *A1;
    double *B1;
    double *B2;
    double *WORK;
    double *D;
#endif
#ifdef USE_MPI
    MPI_Request * requests;
    int req_count;
#endif
    dplasma_context_t* dplasma;
    
    dague_init(argc, argv);
    
    /* Matrix creation, tiling and distribution */
    if(rank == rank)
        {
            A2   = (double *)malloc(LDA*N*sizeof(double));
#if defined(DO_THE_NASTY_VALIDATIONS)
            NminusOne = N-1;
            LDBxNRHS = LDB*NRHS;
            A1   = (double *)malloc(LDA*N*sizeof(double));
            B1   = (double *)malloc(LDBxNRHS*sizeof(double));
            B2   = (double *)malloc(LDBxNRHS*sizeof(double));
            WORK = (double *)malloc(2*LDA*sizeof(double));
            D    = (double *)malloc(LDA*sizeof(double));
            
            /* generating a random matrix */
            generate_matrix(N, A1, A2,  B1, B2,  WORK, D, LDA, NRHS, LDB);
#else        
            /* generating a random matrix */
            int i, j;
            for ( i = 0; i < N; i++)
                for ( j = i; j < N; j++) {
                    A2[LDA*j+i] = A2[LDA*i+j] = (double)rand() / RAND_MAX;
                }
            for ( i = 0; i < N; i++){
                A2[LDA*i+i] = A2[LDA*i+i] + 10*N;
            }
#endif
            tiling(&uplo, N, A2, LDA, &local_desc);
#ifdef trickUSE_MPI
            dplasma_desc_bcast(&local_desc, &descA);
            TIME_START();
            distribute_data(&local_desc, &descA, &requests, &req_count);
        }
    else
        { /* prepare data for block reception  */
            TIME_START();
            dplasma_desc_bcast(NULL, &descA);
            distribute_data(NULL, &descA, &requests, &req_count);
        }
    /* wait for data distribution to finish before continuing */
    is_data_distributed(&descA, requests, req_count);
    TIME_PRINT(("data distribution on rank %d\n", rank));    
# if defined(DO_THE_NASTY_VALIDATIONS)
    data_dist_verif(&local_desc, &descA);
    if(rank == 0)
        plasma_dump(&local_desc);
    data_dump(&descA);
# endif
#else
            dplasma_desc_init(&local_desc, &descA);
        }
#endif
    
    TIME_START();
    dplasma = setup_dplasma();
    
    if(rank == 0)
        {
            dplasma_execution_context_t exec_context;
            
            
            
        /* I know what I'm doing ;) */
            exec_context.function = (dplasma_t*)dplasma_find("POTRF");
            dplasma_set_initial_execution_context(&exec_context);
            
#ifdef DPLASMA_WARM_UP
            dplasma_schedule(dplasma, &exec_context);
            
            /* Now that everything is created start the timer */
            time_elapsed = get_cur_time();
            
            dplasma_progress(dplasma);
            time_elapsed = get_cur_time() - time_elapsed;
            printf("Warming up: DPOTRF %d %d %d %f %f\n", cores,N,NB,time_elapsed, (N/1e3*N/1e3*N/1e3/3.0)/time_elapsed );
#endif  /* DPLASMA_WARM_UP */
            
            
            dplasma_schedule(dplasma, &exec_context);
            
            /* warm the cache for the first tile */
            {
                int i, j;
                double useless = 0.0;
                
                for( i = 0; i < descA.nb; i++ ) {
                    for( j = 0; j < descA.nb; j++ ) {
                        useless += ((double*)descA.mat)[i*descA.nb+j];
                    }
                }
                /*printf( "Useless value %f\n", useless );*/
            }
            
        }

#ifdef DPLASMA_WARM_UP
    dplasma_progress(dplasma);
#endif

    TIME_PRINT(("dplasma initialization %d %d %d\n", 1, descA.n, descA.nb));

    TIME_START();
/* lets rock! */
    dplasma_progress(dplasma);
    TIME_PRINT(("executing kernels on rank %d:\t%d %d %f Gflops\n", rank, N, NB, gflops = flops = (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed * nodes)));

#ifdef trickUSE_MPI    
    TIME_START();
    gather_data(&local_desc, &descA);
    TIME_PRINT(("data reduction on rank %d (to rank 0)\n", rank));
#endif
#ifdef USE_MPI
    MPI_Reduce(&flops, &gflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

    if(rank == 0) 
    {
        eps = (double) 1.0e-13;  /* dlamch("Epsilon");*/
        printf("\n");
        printf("------ TESTS FOR PLASMA DPOTRF + DPOTRS ROUTINE -------  \n");
        printf("            Size of the Matrix %d by %d\n", N, N);
        printf("\n");
        printf(" The matrix A is randomly generated for each test.\n");
        printf("============\n");
        printf(" The relative machine precision (eps) is to be %e \n", eps);
        printf(" Computational tests pass if scaled residuals are less than 10.\n");        
#if defined(DO_THE_NASTY_VALIDATIONS)
        untiling(&uplo, N, A2, LDA, &local_desc);
        PLASMA_dpotrs(uplo, N, NRHS, A2, LDA, B2, LDB);
        
        /* Check the factorization and the solution */
        info_factorization = check_factorization(N, A1, A2, LDA, uplo, eps);
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);
        
        if((info_solution == 0) && (info_factorization == 0)) {
            printf("***************************************************\n");
            printf(" ---- TESTING DPOTRF + DPOTRS ............ PASSED !\n");
            printf("***************************************************\n");
            printf(" ---- GFLOPS ............................. %.4f\n", gflops);
            printf("***************************************************\n");
        }
        else{
            printf("****************************************************\n");
            printf(" - TESTING DPOTRF + DPOTRS ... FAILED !\n");
            printf("****************************************************\n");
        }
        
        free(A1); free(B1); free(B2); free(WORK); free(D);
#else   
        printf("***************************************************\n");
        printf(" ---- TESTING DPOTRF + DPOTRS ............ NOTEST !\n");
        printf("***************************************************\n");
        printf(" ---- GFLOPS .............................. %.4f\n", gflops);
        printf("***************************************************\n");
#endif
        free(A2);
    }

#ifdef DPLASMA_PROFILING
{
    char* filename = NULL;
    
    asprintf( &filename, "%s-%d.svg", "dposv-mpi", rank );
    dplasma_profiling_dump_svg(dplasma, filename);
    free(filename);
}
#endif  /* DPLASMA_PROFILING */

    dague_fini(dplasma);
    return 0;
}

static void dague_fini(dplasma_context_t* dplasma)
{
    dplasma_fini(&dplasma);
    PLASMA_Finalize();
#ifdef USE_MPI
    MPI_Finalize();
#endif
}

static void dague_init(int argc, char **argv)
{
    struct option long_options[] =
    {
        {"lda", required_argument,  0, 'a'},
        {"matrix-size", required_argument, 0, 'n'},
        {"nrhs", required_argument,       0, 'r'},
        {"ldb",  required_argument,       0, 'b'},
        {"grid-rows",  required_argument, 0, 'g'},
        {"stile-size",  required_argument, 0, 's'},
        {"help",  no_argument, 0, 'h'},
        {"nb-cores", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

#ifdef USE_MPI
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
#else
    nodes = 1;
    rank = 0;
#endif
    
    /* parse arguments */
    descA.GRIDrows = 1;
    descA.nrst = descA.ncst = 1;
    printf("parsing arguments\n");
    do
    {
        int c;
        int option_index = 0;
        
        c = getopt_long (argc, argv, "a:n:r:b:g:s:c:h",
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
                break;
            case 's':
                descA.ncst = descA.nrst = atoi(optarg);
                if(descA.ncst <= 0)
                {
                    printf("select a positive value for super tile size\n");
                    cleanup_exit(2);
                }                
                printf("processes receives tiles by blocks of %dx%d\n", descA.nrst, descA.ncst);
                break;
            case 'c':
                cores = atoi(optarg);
                if(cores<= 0)
                    cores=1;
                printf("Number of cores (computing threads) set to %d\n", cores);
                break;
            case '?': /* getopt_long already printed an error message. */
            case 'h':
            default:
                printf("\
Mandatory argument:\n\
   -n, --matrix-size : the size of the matrix\n\
Optional arguments:\n\
   -a --lda : leading dimension of the matrix A (equal matrix size by default)\n\
   -r --nrhs : number of RHS (default: 1)\n\
   -b --ldb : leading dimension of the RHS B (equal matrix size by default)\n\
   -g --grid-rows : number of processes row in the process grid (must divide the total number of processes (default: 1)\n\
   -s --stile-size : number of tile per row (col) in a super tile (default: 1)\n\
   -c --nb-cores : number of computing threads to use\n");
                cleanup_exit(0);
        }
    } while(1);
    
    if(N == 0)
    {
        printf("must provide : -n, --matrix-size : the size of the matrix \n Optional arguments are:\n -a --lda : leading dimension of the matrix A (equal matrix size by default) \n -r --nrhs : number of RHS (default: 1) \n -b --ldb : leading dimension of the RHS B (equal matrix size by default)\n -g --grid-rows : number of processes row in the process grid (must divide the total number of processes (default: 1) \n -s --stile-size : number of tile per row (col) in a super tile (default: 1)\n");
        cleanup_exit(2);
    } 
    descA.GRIDcols = nodes / descA.GRIDrows ;
    if((nodes % descA.GRIDrows) != 0)
    {
        printf("GRIDrows %d does not divide the total number of nodes %d\n", descA.GRIDrows, nodes);
        cleanup_exit(2);
    }
    printf("Grid is %dx%d\n", descA.GRIDrows, descA.GRIDcols);
    if(LDA <= 0) 
    {
        LDA = N;
    }
    if(LDB <= 0) 
    {
        LDB = N;        
    }
    
    PLASMA_Init(cores);
}

static void cleanup_exit(int ret)
{
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, ret);
#else
    exit(ret);
#endif
}

static dplasma_context_t *setup_dplasma(void)
{
    dplasma_context_t *dplasma;
    
    dplasma = dplasma_init(cores, NULL, NULL);
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
    enumerate_dplasma_tasks(dplasma);
    
    return dplasma;
}

#undef N
#undef NB
#undef rank



#ifdef DO_THE_NASTY_VALIDATIONS
/*------------------------------------------------------------------------
 * *  Check the factorization of the matrix A2
 * */
static int check_factorization(int N, double *A1, double *A2, int LDA, int uplo, double eps)
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
static int check_solution(int N, int NRHS, double *A1, int LDA, double *B1, double *B2, int LDB, double eps )
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

#endif
