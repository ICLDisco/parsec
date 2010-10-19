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
#include <inttypes.h>
#include <string.h>
#include <sys/time.h>

#include <cblas.h>
#include <math.h>
#include <plasma.h>
#include <lapack.h>

#include "scheduling.h"
#include "profiling.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"
#include "dpotrf.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

/*******************************
 * globals and argv set values *
 *******************************/
/* timing profiling etc */
double time_elapsed;
double sync_time_elapsed;
int dposv_force_nb = 120;
#define NB dposv_force_nb
int pri_change = 0;
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
sym_two_dim_block_cyclic_t ddescA;

static dague_object_t *dague_dpotrf = NULL;

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
            "   -e --stile-col   : number of tile per col in a super tile (default: 1)\n"
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -s --stile-row   : number of tile per row in a super tile (default: 1)\n"
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
        
            c = getopt_long (argc, argv, "c:n:a:r:b:g:e:s:B:P:h",
                             long_options, &option_index);
#else
            c = getopt (argc, argv, "c:n:a:r:b:g:e:s:B:P:h");
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
                    /*printf("processes receives tiles by blocks of %dx%d\n", nrst, ncst);*/
                    break;
                case 'e':
                    ncst = atoi(optarg);
                    if(ncst <= 0)
                        {
                            fprintf(stderr, "select a positive value for the col super tile size\n");
                            exit(2);
                        }                
                    /*printf("processes receives tiles by blocks of %dx%d\n", nrst, ncst);*/
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
    
    while(N == 0) {
        if(optind < argc) {
            N = atoi(argv[optind++]);
            continue;
        }
        print_usage(); 
        exit(2);
    }
    if((nodes % GRIDrows) != 0) {
        fprintf(stderr, "GRIDrows %d does not divide the total number of nodes %d\n", GRIDrows, nodes);
        exit(2);
    }
    //printf("Grid is %dx%d\n", ddescA.GRIDrows, ddescA.GRIDcols);
    
    if(LDA <= 0) {
        LDA = N;
    }
    if(LDB <= 0) {
        LDB = N;        
    }
    
    PLASMA_Init(1);
}


static void runtime_fini(void);

static dague_context_t *setup_dague(int* pargc, char** pargv[]);
static void cleanup_dague(dague_context_t* context);

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
    /* initializing matrix structure */
    sym_two_dim_block_cyclic_init(&ddescA, matrix_RealDouble,
                              nodes,
                              cores,
                              rank,
                              dposv_force_nb, dposv_force_nb,
                              N, N,
                              0, 0,
                              LDA, LDA,
                              GRIDrows);
    /* matrix generation */
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles * (size_t)ddescA.super.bsiz * (size_t)ddescA.super.mtype);
    generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);

    /*** THIS IS THE DAGUE COMPUTATION ***/
    SYNC_TIME_START();
    dague = setup_dague(&argc, &argv);

    SYNC_TIME_PRINT(("Dague initialization:\t%d %d\n", N, dposv_force_nb));

    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_progress(dague);
    TIME_PRINT(("priority for %d/%u:\ttasks: %u\t%f task/s\n", pri_change, ddescA.super.nt, dague_dpotrf->nb_local_tasks, 
                dague_dpotrf->nb_local_tasks/time_elapsed));
    SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, dposv_force_nb, gflops = (((N/1e3)*(N/1e3)*(N/1e3)/3.0))/(sync_time_elapsed)));
    (void)gflops;

    cleanup_dague(dague);
    /*** END OF DAGUE COMPUTATION ***/

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
        MPI_Datatype default_ddt;

        snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE*%d*%d", dposv_force_nb, dposv_force_nb);
    
        MPI_Type_contiguous(NB * NB, MPI_DOUBLE, &default_ddt);
        MPI_Type_set_name(default_ddt, type_name);
        MPI_Type_commit(&default_ddt);
        dague_arena_construct(&DAGUE_DEFAULT_DATA_TYPE, NB*NB*sizeof(double), 
                              DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
    }
#endif  /* USE_MPI */

    dague_dpotrf = (dague_object_t*)dague_dpotrf_new( (dague_ddesc_t*)&ddescA, 
                                                          ddescA.super.nb, ddescA.super.nt, pri_change );
    dague_enqueue( dague, (dague_object_t*)dague_dpotrf);

    printf("Cholesky %ux%u has %u tasks to run. Total nb tasks to run: %u\n", 
           ddescA.super.nb, ddescA.super.nt, dague_dpotrf->nb_local_tasks, dague->taskstodo);

    printf("GRIDrows = %u, GRIDcols = %u, rrank = %u, crank = %u\n", ddescA.GRIDrows, ddescA.GRIDcols, ddescA.rowRANK, ddescA.colRANK );
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
