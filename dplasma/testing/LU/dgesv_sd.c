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
#include <plasma.h>
#include <lapack.h>

#include "scheduling.h"
#include "profiling.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"
#include "dplasma.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dague_context_t *setup_dague(int* pargc, char** pargv[]);
static void cleanup_dague(dague_context_t* context);

#if defined(DEBUG_MATRICES)
static void debug_matrices(void);
#else
#define debug_matrices()
#endif

static dague_object_t* dague_LU = NULL;

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
int cores = 1;
int nodes = 1;
int nbtasks = -1;
int N = 0;
int NB = 120;
int IB = 40;
int rank;
int LDA = 0;
int NRHS = 1;
int LDB = 0;
int nrst = 1;
int ncst = 1;
PLASMA_enum uplo = PlasmaLower;
int GRIDrows = 1;
int INFO;

two_dim_block_cyclic_t ddescA;
two_dim_block_cyclic_t ddescdLIPIV;
#if defined(DISTRIBUTED)
MPI_Datatype LOWER_TILE, UPPER_TILE, PIVOT_VECT, LITTLE_L_PIVOT_VECT;
#endif

FILE* matout = NULL;

int main(int argc, char ** argv)
{
    double gflops;
    dague_context_t* dague;
    
    //#ifdef VTRACE
    // VT_OFF();
    //#endif
    matout = stderr;

#if defined(DISTRIBUTED)
    /* mpi init */
    MPI_Init(&argc, &argv);
    /*sleep(20);*/
    MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nodes = 1;
    rank = 0;
#endif
    {
        char matout_file[128];
        snprintf(matout_file, 128, "%s-%d.matout", argv[0], rank);
        matout = fopen(matout_file, "w");
    }
    runtime_init(argc, argv);

    two_dim_block_cyclic_init(&ddescA, matrix_RealDouble, nodes, cores, rank, 
                              NB, NB, N, N, 0, 0, 
                              N, N, nrst, ncst, GRIDrows);
    two_dim_block_cyclic_init(&ddescdLIPIV,
                              matrix_RealDouble,
                              nodes,
                              cores,
                              rank,
                              NB,
                              IB,
                              (IB+1)*ddescA.super.mt, N,
                              0, 0,
                              (IB+1)*ddescA.super.mt, N,
                              nrst, ncst,
                              GRIDrows);


    /* matrix generation */
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles * (size_t)ddescA.super.bsiz * (size_t)ddescA.super.mtype);
    ddescdLIPIV.mat = dague_data_allocate((size_t)ddescdLIPIV.super.nb_local_tiles * (size_t)ddescdLIPIV.super.bsiz * (size_t)ddescdLIPIV.super.mtype);
    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescdLIPIV);
    memset(ddescdLIPIV.mat, NB * sizeof(int), 0);

    /*** THIS IS THE DAGUE COMPUTATION ***/
    TIME_START();
    dague = setup_dague(&argc, &argv);
    TIME_PRINT(("Dague initialization:\t%d %d\n", N, NB));
            
    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_progress(dague);
    TIME_PRINT(("Dague proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));
    gflops = (2*N/1e3*N/1e3*N/1e3/3.0);
    SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, NB,
                     gflops/(sync_time_elapsed)));
#if 0
    {
        char ddescA_rank_name[128];
        snprintf( ddescA_rank_name, 128, "A-sd-%d.dat", rank );
        data_write( (tiled_matrix_desc_t*)&ddescA, ddescA_rank_name);
    }
#endif
    cleanup_dague(dague);
    /*** END OF DAGUE COMPUTATION ***/

    runtime_fini();
    {
        fclose(matout);
    }
    return 0;
}

static void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            "   number           : the size of the matrix\n"
            "Optional arguments:\n"
            "   -c --nb-cores    : number of computing threads to use\n"
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -s --stile-row   : number of tile per row in a super tile (default: 1)\n"
            "   -e --stile-col   : number of tile per col in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -t --matrix-type : 0 for LU-like matrix, !=0 for cholesky-like matrix\n"
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
            {"block-size",  required_argument,  0, 'B'},
            {"internal-block-size", required_argument, 0, 'I'},
            {"matrix-type", required_argument,  0, 't'},
            {"help",        no_argument,        0, 'h'},
            {0, 0, 0, 0}
        };
#endif  /* defined(HAVE_GETOPT_LONG) */

    do
        {
            int c;
#if defined(HAVE_GETOPT_LONG)
            int option_index = 0;
            c = getopt_long (argc, argv, "c:n:a:r:b:g:e:s:B:t:I:h",
                             long_options, &option_index);
#else
            c = getopt (argc, argv, "c:n:a:r:b:g:e:s:B:t:I:h");
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
                
                case 'B':
                    if(optarg)
                        {
                            NB = atoi(optarg);
                        }
                    else
                        {
                            fprintf(stderr, "Argument is mandatory for -B (--block-size) flag.\n");
                            exit(2);
                        }
                    break;
                
                case 'I':
                    if(optarg)
                        {
                            IB = atoi(optarg);
                        }
                    else
                        {
                            fprintf(stderr, "Argument is mandatory for -I (--internal-block-size) flag.\n");
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

   
    if((nodes % GRIDrows) != 0)
        {
            fprintf(stderr, "GRIDrows %d does not divide the total number of nodes %d\n", GRIDrows, nodes);
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

    {
	PLASMA_Disable(PLASMA_AUTOTUNING);
	PLASMA_Set(PLASMA_TILE_SIZE, NB);
	PLASMA_Set(PLASMA_INNER_BLOCK_SIZE, IB);
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
    
    dague = dague_init(cores, pargc, pargv, NB);
    
    dague_LU = (dague_object_t*)DAGUE_dgetrf_sd_New( (tiled_matrix_desc_t*)&ddescdLIPIV,
                                                     (tiled_matrix_desc_t*)&ddescA, IB, &INFO );
    dague_enqueue( dague, (dague_object_t*)dague_LU);

    nbtasks = dague_LU->nb_local_tasks;
    printf("LU %ux%u has %u tasks to run. Total nb tasks to run: %u\n", 
           ddescA.super.nb, ddescA.super.nt, dague_LU->nb_local_tasks, dague->taskstodo);
    printf("GRIDrows = %u, GRIDcols = %u, rrank = %u, crank = %u\n", 
           ddescA.GRIDrows, ddescA.GRIDcols, ddescA.rowRANK, ddescA.colRANK );

    return dague;
}

static void cleanup_dague(dague_context_t* dague)
{
#ifdef DAGUE_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "dgesv", rank );
    dague_profiling_dump_xml(filename);
    free(filename);
#endif  /* DAGUE_PROFILING */
    
    dague_fini(&dague);
}

