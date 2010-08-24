/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
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
#include <control/common.h>
#include <control/context.h>
#include <control/allocate.h>
#include <sys/time.h>

#include "scheduling.h"
#include "profiling.h"
#include "two_dim_rectangle_cyclic.h"
#include "gemm.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dague_context_t *setup_dague(int* pargc, char** pargv[]);
static void cleanup_dague(dague_context_t* context);

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

/* globals and argv set values */
PLASMA_desc descA, descB, descC;
two_dim_block_cyclic_t ddescA, ddescB, ddescC;
static dague_object_t* dague_gemm = NULL;
int do_nasty_validations = 0;
int cores = 1;
int nodes = 1;
int nbtasks = -1;
int N;
int LDA = 0;
int NRHS = 1;
int LDB = 0;
int rank = 0;

int main(int argc, char ** argv)
{
    double gflops;
    dague_context_t* dague;

    //#ifdef VTRACE
      // VT_OFF();
    //#endif

    runtime_init(argc, argv);

    //#ifdef VTRACE 
    //    VT_ON();
    //#endif
    
    /*** THIS IS THE DAGUE COMPUTATION ***/
    TIME_START();
    dague = setup_dague(&argc, &argv);
    if(0 == rank) {
        dague_execution_context_t exec_context;

        /* I know what I'm doing ;) */
        exec_context.function = (dague_t*)dague_find(dague_gemm, "STARTUP");
        exec_context.dague_object = dague_gemm;
        exec_context.priority = 0;
        exec_context.locals[0].value = 0;

        dague_schedule(dague, &exec_context);
    }
    TIME_PRINT(("Dague initialization:\t%d %d\n", N, ddescA.super.nb));

    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_progress(dague);
    TIME_PRINT(("Dague proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));
    SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, ddescA.super.nb,
                     gflops = 2*(N/1e3*N/1e3*N/1e3)/(sync_time_elapsed)));

    cleanup_dague(dague);
    /*** END OF DAGUE COMPUTATION ***/

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
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -s --stile-size  : number of tile per row (col) in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -x --xcheck      : do extra nasty result validations\n"
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
        c = getopt_long (argc, argv, "xmc:N:a:r:b:g:s:B:h",
                         long_options, &option_index);
#else
        c = getopt (argc, argv, "xmc:N:a:r:b:g:s:B:h");
#endif  /* defined(HAVE_GETOPT_LONG) */
      
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c)
        {
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
                break; 
            case 'm':
                fprintf(stderr, "This argument is not useful for GEMM\n");
                break;
            case 'B':
                if(optarg)
                {
                    block_forced = atoi(optarg);
                    ddescA.super.nb = block_forced;
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
                break; /* Assume anything else is dague/mpi stuff */
        }
    } while(1);
    
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
    ddescA.super.n = ddescA.super.m = N;

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
    
    PLASMA_Init(1);

    plasma_tune(PLASMA_FUNC_DGEMM, N, N, NRHS);
    if( 0 != block_forced ) {
        plasma_context_t* plasma = plasma_context_self();

        PLASMA_NB = block_forced;
        PLASMA_NBNBSIZE = PLASMA_NB * PLASMA_NB;

        plasma->autotuning_enabled = 0;
    }

    ddescB = ddescC = ddescA;
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

    dague = dague_init(cores, pargc, pargv, ddescA.super.nb);

#ifdef USE_MPI
    /**
     * Redefine the default type after dague_init.
     */
    {
        char type_name[MPI_MAX_OBJECT_NAME];
    
        snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE*%d*%d", ddescA.super.nb, ddescA.super.nb);
    
        MPI_Type_contiguous(ddescA.super.nb * ddescA.super.nb, MPI_DOUBLE, &DAGUE_DEFAULT_DATA_TYPE);
        MPI_Type_set_name(DAGUE_DEFAULT_DATA_TYPE, type_name);
        MPI_Type_commit(&DAGUE_DEFAULT_DATA_TYPE);
    }
#endif  /* USE_MPI */

    dague_gemm = (dague_object_t*)dague_gemm_new( (dague_ddesc_t*)&ddescB, (dague_ddesc_t*)&ddescA, (dague_ddesc_t*)&ddescC,
                                                  ddescA.super.nb, ddescA.super.nt );
    dague->taskstodo += dague_gemm->nb_local_tasks;

    return dague;
}

static void cleanup_dague(dague_context_t* dague)
{
#ifdef DAGUE_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "dgemm", rank );
    dague_profiling_dump_xml(filename);
    free(filename);
#endif  /* DAGUE_PROFILING */
    
    dague_fini(&dague);
}
