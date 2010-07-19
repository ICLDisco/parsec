/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "dague.h"
#ifdef USE_MPI
#include "remote_dep.h"
#include <mpi.h>
#endif  /* defined(USE_MPI) */

#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <cblas.h>
#include <math.h>
#include "plasma.h"
#include <../src/common.h>
#include <../src/lapack.h>
#include <../src/context.h>
#include <../src/allocate.h>

#include "scheduling.h"
#include "profiling.h"
#include "two_dim_rectangle_cyclic.h"
#include "cholesky.h"


/*******************************
 * globals and argv set values *
 *******************************/
/* timing profiling etc */
typedef enum {
    DO_PLASMA,
    DO_DAGUE
} backend_argv_t;

double time_elapsed;
double sync_time_elapsed;
int dposv_force_nb = 120;
#define NB dposv_force_nb
int pri_change = 0;
int do_warmup = 0;
int do_nasty_validations = 0;
int do_distributed_generation = 1;
backend_argv_t backend = DO_DAGUE;
int cores = 1;
int nodes = 1;
int N = 0;

int rank = 0;
int LDA = 0;
int NRHS = 1;
int LDB = 0;
int GRIDrows = 1;
int nrst = 1;
int ncst = 1;
PLASMA_enum uplo = PlasmaLower;

PLASMA_desc descA;
two_dim_block_cyclic_t ddescA;

static dague_object_t *dague_cholesky = NULL;

#if defined(USE_MPI)
MPI_Datatype SYNCHRO = MPI_BYTE;
#endif  /* USE_MPI */

/**********************************
 * static functions
 **********************************/

static void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            "   number           : the size of the matrix\n"
            "Optional arguments:\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -c --nb-cores    : number of computing threads to use\n"
            "   -d --dague     : use DAGUE backend (default)\n"
            "   -e --stile-col   : number of tile per col in a super tile (default: 1)\n"
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -p --plasma      : use PLASMA backend\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -s --stile-row   : number of tile per row in a super tile (default: 1)\n"
            "   -x --xcheck      : do extra nasty result validations\n"
            "   -w --warmup      : do some warmup, if > 1 also preload cache\n"
            "   -P --pri_change  : the position on the diagonal from the end where we switch the priority (default: 0)\n"
            "   -B --block-size  : change the block size from the size tuned by PLASMA\n");
}

static void runtime_init(int argc, char **argv)
{
#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
    {
        {"nb-cores",    required_argument,  0, 'c'},
        {"matrix-size", required_argument,  0, 'n'},
        {"lda",         required_argument,  0, 'a'},
        {"nrhs",        required_argument,  0, 'r'},
        {"ldb",         required_argument,  0, 'b'},
        {"grid-rows",   required_argument,  0, 'g'},
        {"stile-row",   required_argument,  0, 's'},
        {"stile-col",   required_argument,  0, 'e'},
        {"xcheck",      no_argument,        0, 'x'},
        {"warmup",      optional_argument,  0, 'w'},
        {"dague",     no_argument,        0, 'd'},
        {"plasma",      no_argument,        0, 'p'},
        {"block-size",  required_argument,  0, 'B'},
        {"pri_change",  required_argument,  0, 'P'},
        {"help",        no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };
#endif  /* defined(HAVE_GETOPT_LONG) */

    
    /* parse arguments */

    do
    {
        int c;
#if defined(HAVE_GETOPT_LONG)
        int option_index = 0;
        
        c = getopt_long (argc, argv, "dpxc:n:a:r:b:g:e:s:w::B:P:h",
                         long_options, &option_index);
#else
        c = getopt (argc, argv, "dpxc:n:a:r:b:g:e:s:w::B:P:h");
#endif  /* defined(HAVE_GETOPT_LONG) */
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c)
        {
            case 'p': 
                backend = DO_PLASMA;
                do_distributed_generation = 0;
                break; 
            case 'd':
                backend = DO_DAGUE;
                break;

            case 'c':
                cores = atoi(optarg);
                if(cores<= 0)
                    cores=1;
                //printf("Number of cores (computing threads) set to %d\n", cores);
                break;

            case 'n':
                N = atoi(optarg);
                //printf("matrix size set to %d\n", N);
                break;

            case 'g':
                GRIDrows = atoi(optarg);
                break;
            case 's':
                nrst = atoi(optarg);
                if(nrst <= 0)
                {
                    fprintf(stderr, "select a positive value for the row super tile size\n");
                    exit(2);
                }                
                /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
                break;
            case 'e':
                ncst = atoi(optarg);
                if(ncst <= 0)
                {
                    fprintf(stderr, "select a positive value for the col super tile size\n");
                    exit(2);
                }                
                /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
                break;
                
            case 'r':
                NRHS  = atoi(optarg);
                printf("number of RHS set to %d\n", NRHS);
                break;
            case 'a':
                LDA = atoi(optarg);
                printf("LDA set to %d\n", LDA);
                break;                
            case 'b':
                LDB  = atoi(optarg);
                printf("LDB set to %d\n", LDB);
                break;
                
            case 'x':
                do_nasty_validations = 1;
                do_distributed_generation = 0;
                fprintf(stderr, "Results are checked on rank 0, distributed matrix generation is disabled.\n");
                if(do_warmup)
                {
                    fprintf(stderr, "Results cannot be correct with warmup! Validations and warmup are exclusive; please select only one.\n");
                    exit(2);
                }
                break; 
            case 'w':
                if(optarg)
                    do_warmup = atoi(optarg);
                else
                    do_warmup = 1;
                if(do_nasty_validations)
                {
                    fprintf(stderr, "Results cannot be correct with warmup! Validations and warmup are exclusive; please select only one.\n");
                    exit(2);
                }
                break;
                
        case 'B':
                if(optarg)
                {
                    dposv_force_nb = atoi(optarg);
                }
                else
                {
                    fprintf(stderr, "Argument is mandatory for -B (--block-size) flag.\n");
                    exit(2);
                }
                break;

        case 'P':
                pri_change = atoi(optarg);
                break;
        case 'h':
            print_usage();
            exit(0);
        case '?': /* getopt_long already printed an error message. */
        default:
            break; /* Assume anything else is dague/mpi stuff */
        }
    } while(1);
    
    if((DO_PLASMA == backend) && (nodes > 1))
        {
            fprintf(stderr, "using the PLASMA backend for distributed runs is meaningless. Either use DAGUE (-d, --dague), or run in single node mode.\n");
            exit(2);
        }
    
    while(N == 0)
        {
            if(optind < argc)
                {
                    N = atoi(argv[optind++]);
                    continue;
                }
            print_usage(); 
            exit(2);
        }
    if((nodes % GRIDrows) != 0)
        {
            fprintf(stderr, "GRIDrows %d does not divide the total number of nodes %d\n", ddescA.GRIDrows, nodes);
            exit(2);
        }
    //printf("Grid is %dx%d\n", ddescA.GRIDrows, ddescA.GRIDcols);
    
    if(LDA <= 0) 
        {
            LDA = N;
        }
    if(LDB <= 0) 
        {
            LDB = N;        
        }
    
    switch(backend)
        {
        case DO_PLASMA:
            PLASMA_Init(cores);
            break;
        case DO_DAGUE:
            PLASMA_Init(1);
            break;
        }
}


static void runtime_fini(void);

static dague_context_t *setup_dague(int* pargc, char** pargv[]);
static void cleanup_dague(dague_context_t* context);
static void warmup_dague(dague_context_t* dague);

static void create_matrix(int N, PLASMA_enum* uplo, 
                          double** pA1, double** pA2, 
                          double** pB1, double** pB2, 
                          int LDA, int NRHS, int LDB, PLASMA_desc* local);
//static void scatter_matrix(PLASMA_desc* local, DAGUE_desc* dist);
//static void gather_matrix(PLASMA_desc* local, DAGUE_desc* dist);
static void check_matrix(int N, PLASMA_enum* uplo, 
                         double* A1, double* A2, 
                         double* B1, double* B2,
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops);

static int check_factorization(int, double*, double*, int, int , double);
static int check_solution(int, int, double*, int, double*, double*, int, double);


static inline double get_cur_time(){
    double t;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    t=tv.tv_sec+tv.tv_usec/1e6;
    return t;
}

#define TIME_START() do { time_elapsed = get_cur_time(); } while(0)
#define TIME_STOP() do { time_elapsed = get_cur_time() - time_elapsed; } while(0)
#define TIME_PRINT(print) do { \
  TIME_STOP(); \
  printf("[%d] TIMED %f s :\t", rank, time_elapsed); \
  printf print; \
} while(0)


#ifdef USE_MPI
# define SYNC_TIME_START() do {                 \
        MPI_Barrier(MPI_COMM_WORLD);            \
        sync_time_elapsed = get_cur_time();     \
    } while(0)
# define SYNC_TIME_STOP() do {                                  \
        MPI_Barrier(MPI_COMM_WORLD);                            \
        sync_time_elapsed = get_cur_time() - sync_time_elapsed; \
    } while(0)
# define SYNC_TIME_PRINT(print) do {                                \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("### TIMED %f s :\t", sync_time_elapsed);    \
            printf print;                                           \
        }                                                           \
  } while(0)

/* overload exit in MPI mode */
#   define exit(ret) MPI_Abort(MPI_COMM_WORLD, ret)

#else 
# define SYNC_TIME_START() do { sync_time_elapsed = get_cur_time(); } while(0)
# define SYNC_TIME_STOP() do { sync_time_elapsed = get_cur_time() - sync_time_elapsed; } while(0)
# define SYNC_TIME_PRINT(print) do {                                \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("### TIMED %f doing\t", sync_time_elapsed);  \
            printf print;                                           \
        }                                                           \
    } while(0)
#endif



int main(int argc, char ** argv)
{
    double gflops;
    double *A1;
    double *A2;
    double *B1;
    double *B2;
    dague_context_t* dague;

    
    
#ifdef USE_MPI
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
#else
    nodes = 1;
    rank = 0;
#endif
    /* parsing arguments */
    runtime_init(argc, argv);
    if(do_distributed_generation)
        {
            /* initializing matrix structure */
            two_dim_block_cyclic_init(&ddescA, matrix_RealDouble, nodes, cores, rank, dposv_force_nb, dposv_force_nb, 0, N, N, 0, 0, LDA, LDA, nrst, ncst, GRIDrows);
            /* matrix generation */
            generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA);
            printf("matrix generated\n");
        }
    else
        if(0 == rank)
            create_matrix(N, &uplo, &A1, &A2, &B1, &B2, LDA, NRHS, LDB, &descA);

    switch(backend)
    {
        case DO_PLASMA: {
            plasma_context_t* plasma = plasma_context_self();

            if(do_warmup)
            {
                TIME_START();
                plasma_parallel_call_2(plasma_pdpotrf, 
                                       PLASMA_enum, uplo, 
                                       PLASMA_desc, descA);
                TIME_PRINT(("_plasma warmup:\t\t%d %d %f Gflops\n", N, PLASMA_NB,
                            (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed)));
            }
            TIME_START();
            plasma_parallel_call_2(plasma_pdpotrf,
                                   PLASMA_enum, uplo,
                                   PLASMA_desc, descA);
            TIME_PRINT(("_plasma computation:\t%d %d %f Gflops\n", N, PLASMA_NB, 
                        gflops = (N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed)));
            break;
        }
        case DO_DAGUE: {

    
            /*** THIS IS THE DAGUE COMPUTATION ***/
            TIME_START();
            dague = setup_dague(&argc, &argv);
            if(0 == rank)
            {
                dague_execution_context_t exec_context;

                /* I know what I'm doing ;) */
                exec_context.function = dague_find(dague_cholesky, "POTRF");
                exec_context.dague_object = dague_cholesky;
                exec_context.priority = 0;
                exec_context.locals[0].value = 0;

                dague_schedule(dague, &exec_context);
            }
            TIME_PRINT(("Dague initialization:\t%d %d\n", N, dposv_force_nb));

            if(do_warmup)
                warmup_dague(dague);
    
            /* lets rock! */
            SYNC_TIME_START();
            TIME_START();
            dague_progress(dague);
            TIME_PRINT(("Dague proc %d:\ttasks: %d\t%f task/s\n", rank, dague_cholesky->nb_local_tasks, 
                        dague_cholesky->nb_local_tasks/time_elapsed));
            SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, dposv_force_nb, 
                             gflops = (((N/1e3)*(N/1e3)*(N/1e3)/3.0))/(sync_time_elapsed)));

            TIME_PRINT(("Dague priority change at position \t%d\n", ddescA.super.nt - pri_change));
            cleanup_dague(dague);
            /*** END OF DAGUE COMPUTATION ***/

            /* if(!do_distributed_generation) */
            /*     gather_matrix(&descA, &ddescA); */
            break;
        }
    }

    if(0 == rank)
        check_matrix(N, &uplo, A1, A2, B1, B2, LDA, NRHS, LDB, &descA, gflops);

    runtime_fini();
    return 0;
}


static void runtime_fini(void)
{
    PLASMA_Finalize();
#ifdef USE_MPI
    MPI_Finalize();
#endif    
}



static dague_context_t *setup_dague(int* pargc, char** pargv[])
{
    dague_context_t *dague;
   
    dague = dague_init(cores, pargc, pargv, dposv_force_nb);

#ifdef USE_MPI
    /**
     * Redefine the default type after dague_init.
     */
    {
        char type_name[MPI_MAX_OBJECT_NAME];
    
        snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE*%d*%d", dposv_force_nb, dposv_force_nb);
    
        MPI_Type_contiguous(NB * NB, MPI_DOUBLE, &DAGUE_DEFAULT_DATA_TYPE);
        MPI_Type_set_name(DAGUE_DEFAULT_DATA_TYPE, type_name);
        MPI_Type_commit(&DAGUE_DEFAULT_DATA_TYPE);
    }
#endif  /* USE_MPI */

    dague_cholesky = (dague_object_t*)dague_cholesky_new( (dague_ddesc_t*)&ddescA, 
                                                          ddescA.super.nb, ddescA.super.nt, pri_change );
    dague->taskstodo += dague_cholesky->nb_local_tasks;

    printf("Cholesky %dx%d has %d tasks to run. Total nb tasks to run: %d\n", 
           ddescA.super.nb, ddescA.super.nt, dague_cholesky->nb_local_tasks, dague->taskstodo);

    printf("GRIDrows = %d, GRIDcols = %d, rrank = %d, crank = %d\n", ddescA.GRIDrows, ddescA.GRIDcols, ddescA.rowRANK, ddescA.colRANK );
    return dague;
}

static void cleanup_dague(dague_context_t* dague)
{
#ifdef DAGUE_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "dposv", rank );
    dague_profiling_dump_xml(filename);
    free(filename);
#endif  /* DAGUE_PROFILING */
    
    dague_fini(&dague);
}

static void warmup_dague(dague_context_t* dague)
{
    TIME_START();
    dague_progress(dague);
    TIME_PRINT(("Warmup on rank %d:\t%d %d\n", rank, N, NB));
        
    if(0 == rank)    
    {
        /* warm the cache for the first tile */
        dague_execution_context_t exec_context;
        if(do_warmup > 1)
        {
            int i, j;
            double useless = 0.0;
            for( i = 0; i < ddescA.super.nb; i++ ) {
                for( j = 0; j < ddescA.super.nb; j++ ) {
                    useless += ((double*)ddescA.mat)[i*ddescA.super.nb+j];
                }
            }
        }

        /* Ok, now get ready for the same thing again. */
        exec_context.function = dague_find(dague_cholesky, "POTRF");
        exec_context.dague_object = dague_cholesky;
        exec_context.priority = 0;
        exec_context.locals[0].value = 0;

        dague_schedule(dague, &exec_context);
    }
# ifdef USE_MPI
    /* Make sure everybody is done with warmup before proceeding */
    MPI_Barrier(MPI_COMM_WORLD);
# endif    
}

#undef N
#undef NB


static void create_matrix(int N, PLASMA_enum* uplo, 
                          double** pA1, double** pA2, 
                          double** pB1, double** pB2, 
                          int LDA, int NRHS, int LDB, PLASMA_desc* local)
{
#define A1      (*pA1)
#define A2      (*pA2)
#define B1      (*pB1)
#define B2      (*pB2)
    int i, j;
    
    if(do_nasty_validations)
    {
        A1   = (double *)malloc(LDA*N*sizeof(double));
        A2   = (double *)malloc(LDA*N*sizeof(double));
        B1   = (double *)malloc(LDB*NRHS*sizeof(double));
        B2   = (double *)malloc(LDB*NRHS*sizeof(double));
        /* Check if unable to allocate memory */
        if((!pA1) || (!pA2) || (!pB1) || (!pB2))
        {
            printf("Out of Memory \n ");
            exit(1);
        }

        /* generating a random matrix */
        for ( i = 0; i < N; i++)
            for ( j = i; j < N; j++) {
                A2[LDA*j+i] = A1[LDA*j+i] = (double)rand() / RAND_MAX;
                A2[LDA*i+j] = A1[LDA*i+j] = A1[LDA*j+i];
            }
        for ( i = 0; i < N; i++) {
            A2[LDA*i+i] = A1[LDA*i+i] += 10*N;
        }
        /* Initialize B1 and B2 */
        for ( i = 0; i < N; i++)
            for ( j = 0; j < NRHS; j++)
                B2[LDB*j+i] = B1[LDB*j+i] = (double)rand() / RAND_MAX;
    }
    else
    {        
        /* Only need A2 */
        A1 = B1 = B2 = NULL;
        A2   = (double *)malloc(LDA*N*sizeof(double));
        /* Check if unable to allocate memory */
        if (!A2){
            printf("Out of Memory \n ");
            exit(1);
        }

        /* generating a random matrix */
        for ( i = 0; i < N; i++)
            for ( j = i; j < N; j++) {
                A2[LDA*j+i] = A2[LDA*i+j] = (double)rand() / RAND_MAX;
            }
        for ( i = 0; i < N; i++) {
            A2[LDA*i+i] = A2[LDA*i+i] + 10 * N;
        }
    }
    
    //    tiling(uplo, N, A2, LDA, NRHS, local);
#undef A1
#undef A2 
#undef B1 
#undef B2 
}

/* static void scatter_matrix(PLASMA_desc* local, DAGUE_desc* dist) */
/* { */
/*     if(do_distributed_generation) */
/*     { */
/*         TIME_START(); */
/*         dague_description_init(dist, LDA, LDB, NRHS, uplo); */
/*         rand_dist_matrix(dist); */
/*         /\*TIME_PRINT(("distributed matrix generation on rank %d\n", dist->mpi_rank));*\/ */
/*         return; */
/*     } */
    
/*     TIME_START(); */
/*     if(0 == rank) */
/*     { */
/*         dague_desc_init(local, dist); */
/*     } */

/* #ifdef USE_MPI */
/*     dague_desc_bcast(local, dist); */
/*     distribute_data(local, dist); */
/*     /\*TIME_PRINT(("data distribution on rank %d\n", dist->mpi_rank));*\/ */
    
/* #if defined(DATA_VERIFICATIONS) */
/*     if(do_nasty_validations) */
/*     { */
/*         data_dist_verif(local, dist); */
/* #if defined(PRINT_ALL_BLOCKS) */
/*         if(rank == 0) */
/*             plasma_dump(local); */
/*         data_dump(dist); */
/* #endif /\* PRINT_ALL_BLOCKS *\/ */
/*     } */
/* #endif /\* DATA_VERIFICATIONS *\/ */
/* #endif  /\* USE_MPI *\/ */
/* } */

/* static void gather_matrix(PLASMA_desc* local, DAGUE_desc* dist) */
/* { */
/* # ifdef USE_MPI */
/*     if(do_nasty_validations) */
/*     { */
/*         TIME_START(); */
/*         gather_data(local, dist); */
/*         TIME_PRINT(("data reduction on rank %d (to rank 0)\n", dist->mpi_rank)); */
/*     } */
/* # endif */
/* } */

static void check_matrix(int N, PLASMA_enum* uplo, 
                         double* A1, double* A2, 
                         double* B1, double* B2,  
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops)
{    
    int info_solution, info_factorization;
    double eps = dlamch("Epsilon");

    printf("\n");
    printf("------ TESTS FOR PLASMA DPOTRF + DPOTRS ROUTINE -------  \n");
    printf("            Size of the Matrix %d by %d\n", N, N);
    printf("\n");
    printf(" The matrix A is randomly generated for each test.\n");
    printf("============\n");
    printf(" The relative machine precision (eps) is to be %e \n", eps);
    printf(" Computational tests pass if scaled residuals are less than 10.\n");        
    if(do_nasty_validations)
    {
        //      untiling(uplo, N, A2, LDA, local);
        PLASMA_dpotrs(*uplo, N, NRHS, A2, LDA, B2, LDB);

        /* Check the factorization and the solution */
        info_factorization = check_factorization(N, A1, A2, LDA, *uplo, eps);
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);

        if((info_solution == 0) && (info_factorization == 0)) 
        {
            printf("****************************************************\n");
            printf(" ---- TESTING DPOTRF + DPOTRS ............ PASSED ! \n");
            printf("****************************************************\n");
            printf(" ---- GFLOPS ............................. %.4f\n", gflops);
            printf("****************************************************\n");
        }
        else 
        {
            printf("*****************************************************\n");
            printf(" ---- TESTING DPOTRF + DPOTRS ............ FAILED !  \n");
            printf("*****************************************************\n");
        }
        free(A1); free(B1); free(B2);
    }
    else
    {
        printf("****************************************************\n");
        printf(" ---- TESTING DPOTRF + DPOTRS ............ SKIPPED !\n");
        printf("****************************************************\n");
        printf(" ---- n= %d np= %d nc= %d g= %dx%d (%dx%d)\t %.4f GFLOPS\n", N, nodes, cores, ddescA.GRIDrows, ddescA.GRIDcols, ddescA.nrst, ddescA.ncst, gflops);
        printf("****************************************************\n");
    }
    //    free(A2);
}



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
    
    if ( isnan(Rnorm/((Anorm*Xnorm+Bnorm)*N*eps)) || Rnorm/((Anorm*Xnorm+Bnorm)*N*eps) > 10.0) {
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
