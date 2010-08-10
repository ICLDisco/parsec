/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma.h"
#include "remote_dep.h"
#ifdef USE_MPI
#include <mpi.h>
#endif  /* defined(USE_MPI) */

#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <lapack.h>
#include <control/common.h>
#include <control/context.h>
#include <control/allocate.h>
#include <sys/time.h>

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

static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist);
static void check_matrix(int N, PLASMA_enum* uplo, 
                         double* A1, double* A2, 
                         double* B1, double* B2,
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops);

/* timing profiling etc */
double time_elapsed;
double sync_time_elapsed;

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
PLASMA_desc descA, descB, descC;
DPLASMA_desc ddescA, ddescB, ddescC;
int do_warmup = 0;
int do_nasty_validations = 0;
int cores = 1;
int nodes = 1;
int nbtasks = -1;
int N;
#define NB (ddescA.nb)
#define rank (ddescA.mpi_rank)
int LDA = 0;
int NRHS = 1;
int LDB = 0;
PLASMA_enum uplo = PlasmaLower;
backend_argv_t backend = DO_DPLASMA;

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

    switch(backend)
    {
        case DO_PLASMA: {
            plasma_context_t* plasma = plasma_context_self();
#if 0
            if(do_warmup)
            {
                TIME_START();
                plasma_parallel_call_4(plasma_pdgemm, 
                                       PLASMA_enum, uplo, 
                                       PLASMA_desc, descA,
                                       PLASMA_desc, descB,
                                       PLASMA_desc, descC);
                TIME_PRINT(("_plasma warmup:\t\t%d %d %f Gflops\n", N, PLASMA_NB,
                            2*(N/1e3*N/1e3*N/1e3)/(time_elapsed)));
            }
            TIME_START();
            plasma_parallel_call_4(plasma_pdgemm,
                                   PLASMA_enum, uplo,
                                   PLASMA_desc, descA,
                                   PLASMA_desc, descB,
                                   PLASMA_desc, descC);
            TIME_PRINT(("_plasma computation:\t%d %d %f Gflops\n", N, PLASMA_NB, 
                        gflops = 2*(N/1e3*N/1e3*N/1e3)/(time_elapsed)));
#else
            printf( "No support for PLASMA dgemm yet." );
#endif
            break;
        }
        case DO_DPLASMA: {
            scatter_matrix(&descA, &ddescA);
            scatter_matrix(&descB, &ddescB);
            scatter_matrix(&descC, &ddescC);

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
                exec_context.function = (dplasma_t*)dplasma_find("STARTUP");
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
            SYNC_TIME_PRINT(("Dplasma computation:\t%d %d %f gflops\n", N, NB,
                             gflops = 2*(N/1e3*N/1e3*N/1e3)/(sync_time_elapsed)));

            cleanup_dplasma(dplasma);
            /*** END OF DPLASMA COMPUTATION ***/

            break;
        }
    }

    if(0 == rank)
        check_matrix(N, &uplo, A1, A2, B1, B2, LDA, NRHS, LDB, &descA, gflops);

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
            "   -N --matrix-size : the size of the matrix\n"
            "   -d --dplasma     : use DPLASMA backend (default)\n"
            "   -p --plasma      : use PLASMA backend\n"
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -s --stile-size  : number of tile per row (col) in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -x --xcheck      : do extra nasty result validations\n"
            "   -w --warmup      : do some warmup, if > 1 also preload cache\n"
            "   -B --block-size  : change the block size from the size tuned by PLASMA\n");
}

static void runtime_init(int argc, char **argv)
{
#if defined(HAVE_GETOPT_LONG)
    struct option long_options[] =
    {
        {"nb-cores",    required_argument,  0, 'c'},
        {"matrix-size", required_argument,  0, 'N'},
        {"lda",         required_argument,  0, 'a'},
        {"nrhs",        required_argument,  0, 'r'},
        {"ldb",         required_argument,  0, 'b'},
        {"grid-rows",   required_argument,  0, 'g'},
        {"stile-size",  required_argument,  0, 's'},
        {"xcheck",      no_argument,        0, 'x'},
        {"warmup",      optional_argument,  0, 'w'},
        {"dplasma",     no_argument,        0, 'd'},
        {"plasma",      no_argument,        0, 'p'},
        {"dist-matrix", no_argument,        0, 'm'},
        {"block-size",  required_argument,  0, 'B'},
        {"help",        no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };
#endif  /* defined(HAVE_GETOPT_LONG) */
    int block_forced = 0;
    int internal_block_forced = 0;

#ifdef USE_MPI
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
#else
    nodes = 1;
#endif
    
    /* parse arguments */
    ddescA.GRIDrows = 1;
    ddescA.nrst = ddescA.ncst = 1;
    do
    {
        int c;
#if defined(HAVE_GETOPT_LONG)
        int option_index = 0;
        c = getopt_long (argc, argv, "dpxmc:N:a:r:b:g:s:w:B:h",
                         long_options, &option_index);
#else
        c = getopt (argc, argv, "dpxmc:N:a:r:b:g:s:w:B:h");
#endif  /* defined(HAVE_GETOPT_LONG) */
      
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
                //printf("Number of cores (computing threads) set to %d\n", cores);
                break;

            case 'N':
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
                
            case 'x':
                do_nasty_validations = 1;
                if(do_warmup)
                {
                    fprintf(stderr, "Results cannot be correct with warmup! Validations and warmup are exclusive; please select only one.\n");
                    exit(2);
                }
                break; 
            case 'm':
                fprintf(stderr, "This argument is not useful for GEMM\n");
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
                    block_forced = atoi(optarg);
                    ddescA.nb = block_forced;
                }
                else
                {
                    fprintf(stderr, "Argument is mandatory for -B (--block-size) flag.\n");
                    exit(2);
                }
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
    ddescA.n = ddescA.m = N;

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
    plasma_tune(PLASMA_FUNC_DGEMM, N, N, NRHS);
    if( 0 != block_forced ) {
        plasma_context_t* plasma = plasma_context_self();

        PLASMA_NB = block_forced;
        PLASMA_NBNBSIZE = PLASMA_NB * PLASMA_NB;

        plasma->autotuning_enabled = 0;
    }
    ddescA.cores = cores;

    ddescB = ddescC = ddescA;
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

#ifdef USE_MPI
    /**
     * Redefine the default type after dplasma_init.
     */
    {
        char type_name[MPI_MAX_OBJECT_NAME];
    
        snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE*%d*%d", NB, NB);
    
        MPI_Type_contiguous(NB * NB, MPI_DOUBLE, &DPLASMA_DEFAULT_DATA_TYPE);
        MPI_Type_set_name(DPLASMA_DEFAULT_DATA_TYPE, type_name);
        MPI_Type_commit(&DPLASMA_DEFAULT_DATA_TYPE);
    }
#endif  /* USE_MPI */

    load_dplasma_objects(dplasma);
    {
        expr_t* constant;
        
        constant = expr_new_int( ddescA.nb );
        dplasma_assign_global_symbol( "NB", constant );
        constant = expr_new_int( ddescA.nt );
        dplasma_assign_global_symbol( "SIZE", constant );
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
    return dplasma;
}

static void cleanup_dplasma(dplasma_context_t* dplasma)
{
#ifdef DPLASMA_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "dgemm", rank );
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
        exec_context.function = (dplasma_t*)dplasma_find("POTRF");
        dplasma_set_initial_execution_context(&exec_context);
        dplasma_schedule(dplasma, &exec_context);
    }
# ifdef USE_MPI
    /* Make sure everybody is done with warmup before proceeding */
    MPI_Barrier(MPI_COMM_WORLD);
# endif    
}

static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
    dplasma_description_init(dist, LDA, LDB, NRHS, uplo);
    rand_dist_matrix(dist);
    /*TIME_PRINT(("distributed matrix generation on rank %d\n", dist->mpi_rank));*/
    *local = *(PLASMA_desc*)dist;
    return;
}

static void check_matrix(int N, PLASMA_enum* uplo, 
                         double* A1, double* A2, 
                         double* B1, double* B2,  
                         int LDA, int NRHS, int LDB, PLASMA_desc* local, 
                         double gflops)
{
    double eps = dlamch("Epsilon");

    printf("\n");
    printf("------ TESTS FOR PLASMA DPOTRF + DPOTRS ROUTINE -------  \n");
    printf("            Size of the Matrix %d by %d\n", N, N);
    printf("\n");
    printf(" The matrix A is randomly generated for each test.\n");
    printf("============\n");
    printf(" The relative machine precision (eps) is to be %e \n", eps);
    printf(" Computational tests pass if scaled residuals are less than 10.\n");        
    
    printf("****************************************************\n");
    printf(" ---- TESTING DPOTRF + DPOTRS ............ SKIPPED !\n");
    printf("****************************************************\n");
    printf(" ---- n= %d np= %d nc= %d g= %dx%d\t %.4f GFLOPS\n", N, nodes, cores, ddescA.GRIDrows, ddescA.GRIDcols, gflops);
    printf("****************************************************\n");
}
