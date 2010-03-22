/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifdef USE_MPI
#include <mpi.h>
#endif  /* defined(USE_MPI) */

#include <getopt.h>
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

#include "dplasma.h"
#include "scheduling.h"
#include "profiling.h"
#include "data_management.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dplasma_context_t *setup_dplasma(int* pargc, char** pargv[]);
static void cleanup_dplasma(dplasma_context_t* context);
static void warmup_dplasma(dplasma_context_t* dplasma);

static void create_matrix(int N, PLASMA_enum* uplo, 
                          double** pA1, double** pA2, 
                          double** pB1, double** pB2, 
                          int LDA, int NRHS, int LDB, PLASMA_desc* local);
static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist);


/* timing profiling etc */
double time_elapsed;
double sync_time_elapsed;

int syn_force_nb = 0;
static int task_duration = 10000;
static int tree_depth = 1;

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
# define SYNC_TIME_START() do { \
    MPI_Barrier(MPI_COMM_WORLD); \
    sync_time_elapsed = get_cur_time(); \
  } while(0)
# define SYNC_TIME_STOP() do { \
    MPI_Barrier(MPI_COMM_WORLD); \
    sync_time_elapsed = get_cur_time() - sync_time_elapsed; \
  } while(0)
# define SYNC_TIME_PRINT(print) do { \
    SYNC_TIME_STOP(); \
    if(0 == rank) { \
      printf("### TIMED %f s :\t", sync_time_elapsed); \
      printf print; \
    } \
  } while(0)

/* overload exit in MPI mode */
#   define exit(ret) MPI_Abort(MPI_COMM_WORLD, ret)

#else 
# define SYNC_TIME_START() do { sync_time_elapsed = get_cur_time(); } while(0)
# define SYNC_TIME_STOP() do { sync_time_elapsed = get_cur_time() - sync_time_elapsed; } while(0)
# define SYNC_TIME_PRINT(print) do { \
    SYNC_TIME_STOP(); \
    if(0 == rank) { \
      printf("### TIMED %f doing\t", sync_time_elapsed); \
      printf print; \
    } \
  } while(0)
#endif

typedef enum {
    DO_PLASMA,
    DO_DPLASMA
} backend_argv_t;

/* globals and argv set values */
int do_warmup = 0;
int do_nasty_validations = 0;
int do_distributed_generation = 0;
backend_argv_t backend = DO_DPLASMA;
int cores = 1;
int nodes = 1;
int nbtasks = -1;
#define N (ddescA.n)
#define NB (ddescA.nb)
#define rank (ddescA.mpi_rank)
int LDA = 0;
int NRHS = 1;
int LDB = 0;
PLASMA_enum uplo = PlasmaLower;

PLASMA_desc descA;
DPLASMA_desc ddescA;

int main(int argc, char ** argv)
{
    double gflops;
    double *A1;
    double *A2;
    double *B1;
    double *B2;
    dplasma_context_t* dplasma;

    //#ifdef VTRACE
      // VT_OFF();
    //#endif

    runtime_init(argc, argv);

    if(0 == rank)
        create_matrix(N, &uplo, &A1, &A2, &B1, &B2, LDA, NRHS, LDB, &descA);

    scatter_matrix(&descA, &ddescA);

    //#ifdef VTRACE 
    //    VT_ON();
    //#endif
    
    /*** THIS IS THE DPLASMA COMPUTATION ***/
    TIME_START();
    dplasma = setup_dplasma(&argc, &argv);
    if(0 == rank)
        {
            dplasma_execution_context_t exec_context;
            
            /* I know what I'm doing ;) */
            exec_context.function = (dplasma_t*)dplasma_find("DOWN");
            dplasma_set_initial_execution_context(&exec_context);
            dplasma_schedule(dplasma, &exec_context);
        }
    TIME_PRINT(("Dplasma initialization:\t%d %d\n", N, NB));

    if(do_warmup)
        warmup_dplasma(dplasma);
    
    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dplasma_progress(dplasma);
    TIME_PRINT(("Dplasma proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));
    SYNC_TIME_PRINT(("Dplasma computation:\t%d %d %f gflops\n", N, NB, gflops = (N/1e3*N/1e3*N/1e3/3.0+N/1e3*N/1e3/2.0)/(sync_time_elapsed)));

    cleanup_dplasma(dplasma);
    /*** END OF DPLASMA COMPUTATION ***/
    runtime_fini();
    return 0;
}

static void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            "   number           : the size of the matrix\n"
            "Optional arguments:\n"
            "   -c --nb-cores    : number of computing threads to use\n"
            "   -d --dplasma     : use DPLASMA backend (default)\n"
            "   -p --plasma      : use PLASMA backend\n"
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -s --stile-size  : number of tile per row (col) in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -w --warmup      : do some warmup, if > 1 also preload cache\n"
            "   -m --dist-matrix : generate tiled matrix in a distributed way\n"
            "   -B --block-size  : change the block size from the size tuned by PLASMA\n"
            "   -t --task-duration: change the duration of the tasks (default: %d us)\n"
            "   -D --tree-depth  : change the depth of the tree (number of tasks = 2*2^D, default: %d)\n", task_duration, tree_depth);
}

static void runtime_init(int argc, char **argv)
{
    struct option long_options[] =
    {
        {"nb-cores",    required_argument,  0, 'c'},
        {"matrix-size", required_argument,  0, 'n'},
        {"lda",         required_argument,  0, 'a'},
        {"nrhs",        required_argument,  0, 'r'},
        {"ldb",         required_argument,  0, 'b'},
        {"grid-rows",   required_argument,  0, 'g'},
        {"stile-size",  required_argument,  0, 's'},
        {"warmup",      optional_argument,  0, 'w'},
        {"dplasma",     no_argument,        0, 'd'},
        {"plasma",      no_argument,        0, 'p'},
        {"dist-matrix", no_argument,        0, 'm'},
        {"block-size",  required_argument,  0, 'B'},
        {"task-duration", required_argument, 0, "t"},
        {"tree-depth",  required_argument,  0, "D"},
        {"help",        no_argument,        0, 'h'},
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
    ddescA.GRIDrows = 1;
    ddescA.nrst = ddescA.ncst = 1;
    do
    {
        int c;
        int option_index = 0;
        
        c = getopt_long (argc, argv, "dpmc:n:a:r:b:g:s:w::B:hD:t:",
                         long_options, &option_index);
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c)
        {
            case 'p': 
                backend = DO_PLASMA;
                break; 
            case 'd':
                backend = DO_DPLASMA;
                break;

            case 'c':
                cores = atoi(optarg);
                if(cores<= 0)
                    cores=1;
                ddescA.cores = cores;
                //printf("Number of cores (computing threads) set to %d\n", cores);
                break;

            case 'n':
                N = atoi(optarg);
                //printf("matrix size set to %d\n", N);
                break;

            case 'g':
                ddescA.GRIDrows = atoi(optarg);
                break;
            case 's':
                ddescA.ncst = ddescA.nrst = atoi(optarg);
                if(ddescA.ncst <= 0)
                {
                    fprintf(stderr, "select a positive value for super tile size\n");
                    exit(2);
                }                
                //printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);
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
            case 'm':
                do_distributed_generation = 1;
                if(do_nasty_validations)
                {
                    fprintf(stderr, "Results cannot be checked with distributed matrix generation! Validations and distributed generation are exclusive; please select only one.\n");
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
                    syn_force_nb = atoi(optarg);
                }
                else
                {
                    fprintf(stderr, "Argument is mandatory for -B (--block-size) flag.\n");
                    exit(2);
                }
                break;

        case 'D':
            tree_depth = atoi(optarg);
            break;

        case 't':
            task_duration = atoi(optarg);
            break;

            case 'h':
                print_usage();
                exit(0);
            case '?': /* getopt_long already printed an error message. */
            default:
                break; /* Assume anything else is dplasma/mpi stuff */
        }
    } while(1);
    
    if((DO_PLASMA == backend) && (nodes > 1))
    {
        fprintf(stderr, "using the PLASMA backend for distributed runs is meaningless. Either use DPLASMA (-d, --dplasma), or run in single node mode.\n");
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

    ddescA.GRIDcols = nodes / ddescA.GRIDrows ;
    if((nodes % ddescA.GRIDrows) != 0)
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
        case DO_DPLASMA:
            PLASMA_Init(1);
            break;
    }
}

static void runtime_fini(void)
{
    PLASMA_Finalize();
#ifdef USE_MPI
    MPI_Finalize();
#endif    
}



static dplasma_context_t *setup_dplasma(int* pargc, char** pargv[])
{
    dplasma_context_t *dplasma;
   
    dplasma = dplasma_init(cores, pargc, pargv, ddescA.nb);
    load_dplasma_objects(dplasma);
    {
        expr_t* constant;
        
        constant = expr_new_int( tree_depth );
        dplasma_assign_global_symbol( "DEPTH", constant );
        printf("DEPTH = %d\n", tree_depth);
        constant = expr_new_int( task_duration );
        dplasma_assign_global_symbol( "TASKDURATION", constant );
        constant = expr_new_int( ddescA.GRIDrows );
        dplasma_assign_global_symbol( "GRIDrows", constant );
        constant = expr_new_int( ddescA.GRIDcols );
        dplasma_assign_global_symbol( "GRIDcols", constant );
        constant = expr_new_int( ddescA.rowRANK );
        dplasma_assign_global_symbol( "rowRANK", constant );
        constant = expr_new_int( ddescA.colRANK );
        dplasma_assign_global_symbol( "colRANK", constant );
        constant = expr_new_int( ddescA.nrst );
        dplasma_assign_global_symbol( "stileSIZE", constant );
    }
    load_dplasma_hooks(dplasma);
    nbtasks = enumerate_dplasma_tasks(dplasma);
    
    printf("Number of tasks to do: %d\n", nbtasks);

    return dplasma;
}

static void cleanup_dplasma(dplasma_context_t* dplasma)
{
#ifdef DPLASMA_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "syn", rank );
    dplasma_profiling_dump_xml(filename);
    free(filename);
#endif  /* DPLASMA_PROFILING */
    
    dplasma_fini(&dplasma);
}

static void warmup_dplasma(dplasma_context_t* dplasma)
{
    TIME_START();
    dplasma_progress(dplasma);
    TIME_PRINT(("Warmup on rank %d:\t%d %d\n", rank, N, NB));
    
    enumerate_dplasma_tasks(dplasma);
    
    if(0 == rank)    
    {
        /* warm the cache for the first tile */
        dplasma_execution_context_t exec_context;
        if(do_warmup > 1)
        {
            int i, j;
            double useless = 0.0;
            for( i = 0; i < ddescA.nb; i++ ) {
                for( j = 0; j < ddescA.nb; j++ ) {
                    useless += ((double*)ddescA.mat)[i*ddescA.nb+j];
                }
            }
        }

        /* Ok, now get ready for the same thing again. */
        exec_context.function = (dplasma_t*)dplasma_find("DOWN");
        dplasma_set_initial_execution_context(&exec_context);
        dplasma_schedule(dplasma, &exec_context);
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

    if(do_distributed_generation) 
    {
        A1 = A2 = B1 = B2 = NULL;
        return;
    }
    
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
    
    tiling(uplo, N, A2, LDA, NRHS, local);
#undef A1
#undef A2 
#undef B1 
#undef B2 
}

static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
#ifdef USE_MPI
    MPI_Request * requests;
    int req_count;

    if(do_distributed_generation)
    {
        TIME_START();
        dplasma_description_init(dist, LDA, LDB, NRHS, uplo);
        rand_dist_matrix(dist);
        TIME_PRINT(("distributed matrix generation on rank %d\n", dist->mpi_rank));
        return;
    }
    
    TIME_START();
    if(0 == rank)
    {
        dplasma_desc_init(local, dist);
    }
    dplasma_desc_bcast(local, dist);
    distribute_data(local, dist);
    TIME_PRINT(("data distribution on rank %d\n", dist->mpi_rank));
    
#if defined(DATA_VERIFICATIONS)
    if(do_nasty_validations)
    {
        data_dist_verif(local, dist);
#if defined(PRINT_ALL_BLOCKS)
        if(rank == 0)
            plasma_dump(local);
        data_dump(dist);
#endif /* PRINT_ALL_BLOCKS */
    }
#endif /* DATA_VERIFICATIONS */
    
#else /* NO MPI */
    dplasma_desc_init(local, dist);
#endif
}

#undef rank

