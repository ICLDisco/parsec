/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#ifdef USE_MPI
#include "remote_dep.h"
extern dague_arena_t DAGUE_DEFAULT_DATA_TYPE;
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
#include <dplasma.h>
#include <sys/time.h>

#include "scheduling.h"
#include "profiling.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

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
int nbtasks = 0;
int N;
int NB = 200;
int IB = 120;
int LDA = 0;
int NRHS = 1;
int LDB = 0;
int rank = 0;
int nrst = 1;
int ncst = 1;
int GRIDrows = 1;

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    //#ifdef VTRACE
    // VT_OFF();
    //#endif

    runtime_init(argc, argv);

    two_dim_block_cyclic_init(&ddescA, matrix_RealFloat, nodes, cores, rank,
                              NB, NB, IB, N, N, 0, 0,
                              N, N, nrst, ncst, GRIDrows);
    two_dim_block_cyclic_init(&ddescB, matrix_RealFloat, nodes, cores, rank,
                              NB, NB, IB, N, N, 0, 0,
                              N, N, nrst, ncst, GRIDrows);
    two_dim_block_cyclic_init(&ddescC, matrix_RealFloat, nodes, cores, rank,
                              NB, NB, IB, N, N, 0, 0,
                              N, N, nrst, ncst, GRIDrows);
    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescC);    //#ifdef VTRACE 
    //    VT_ON();
    //#endif
    
    /*** THIS IS THE DAGUE COMPUTATION ***/
    TIME_START();
    dague = setup_dague(&argc, &argv);
    TIME_PRINT(("Dague initialization:\t%d %u\n", N, ddescA.super.nb));

    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_progress(dague);
    TIME_PRINT(("Dague proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, ((double)nbtasks)/time_elapsed));
    SYNC_TIME_PRINT(("Dague computation:\t%d %u %f gflops\n", N, ddescA.super.nb,
                     2*(N/1e3*N/1e3*N/1e3)/(sync_time_elapsed)));

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

#ifdef USE_MPI
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    /*sleep(20);*/
#else
    nodes = 1;
#endif
    
    /* parse arguments */
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
                GRIDrows = atoi(optarg);
                break;
            case 's':
                ncst = nrst = atoi(optarg);
                if(ncst <= 0)
                {
                    fprintf(stderr, "select a positive value for super tile size\n");
                    exit(2);
                }                
                //printf("processes receives tiles by blocks of %dx%d\n", nrst, ncst);
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
                
            case 'B':
                if(optarg)
                {
                    block_forced = atoi(optarg);
                    NB = block_forced;
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

    if(LDA < 0) 
    {
        LDA = N;
    }
    if(LDB < 0) 
    {
        LDB = N;        
    }
    
    PLASMA_Init(1);
    if ( block_forced != 0 )
    {
	PLASMA_Disable(PLASMA_AUTOTUNING);
	PLASMA_Set(PLASMA_TILE_SIZE, block_forced);
    }    
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
        MPI_Datatype default_ddt;

        snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_FLOAT*%u*%u", ddescA.super.nb, ddescA.super.nb);
    
        MPI_Type_contiguous(ddescA.super.nb * ddescA.super.nb, MPI_FLOAT, &default_ddt);
        MPI_Type_set_name(default_ddt, type_name);
        MPI_Type_commit(&default_ddt);
        dague_arena_construct(&DAGUE_DEFAULT_DATA_TYPE, NB*NB*sizeof(float), 
                              DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
    }
#endif  /* USE_MPI */

    dague_gemm = DAGUE_sgemm_New( PlasmaNoTrans, PlasmaNoTrans,
				  ddescC.super.mt, ddescC.super.nt, ddescA.super.nt,
				  (double)-1.0, (tiled_matrix_desc_t*)&ddescA, 
				  (tiled_matrix_desc_t*)&ddescB, 
				  (double)1.0, (tiled_matrix_desc_t*)&ddescC);

    /* TODO : Check if it's the correct parameters */
    /* (dague_ddesc_t*)&ddescB, (dague_ddesc_t*)&ddescA, (dague_ddesc_t*)&ddescC, */
    /*   ddescA.super.nb, */
    /*   ddescA.super.mt, ddescB.super.nt, ddescA.super.nt, */
    /*   -1.0,  /\* alpha *\/ */
    /*   1.0 ); /\* beta *\/ */
    dague_enqueue( dague, (dague_object_t*)dague_gemm);

    nbtasks = dague_gemm->nb_local_tasks;
    printf("SGEMM %ux%ux%u has %d tasks to run.\n",
           ddescA.super.nb, ddescA.super.nt, ddescA.super.nt, nbtasks);
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
