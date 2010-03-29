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

#include <math.h>

#include "dplasma.h"
#include "scheduling.h"
#include "profiling.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dplasma_context_t *setup_dplasma(int* pargc, char** pargv[]);
static void cleanup_dplasma(dplasma_context_t* context);
static void create_data(void);

/* timing profiling etc */
double time_elapsed;
double sync_time_elapsed;

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

/* globals and argv set values */
int cores = 1;
int nodes = 1;
int rank  = 0;
int nbtasks = -1;

char *DataIn;
int   N = 0;

int main(int argc, char ** argv)
{
    dplasma_context_t* dplasma;

    //#ifdef VTRACE
      // VT_OFF();
    //#endif

    runtime_init(argc, argv);

    create_data();

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
    TIME_PRINT(("Dplasma initialization:\t%d %d\n", N, nodes));
    
    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dplasma_progress(dplasma);
    TIME_PRINT(("Dplasma proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));

    if( rank == 0 ) {
        nbtasks = (1<<(N+1)) + (1 << N) - 2;
        printf(" %d tasks done in %f s: %f tasks/s, overhead is %f s\n", 
               nbtasks, time_elapsed, (float)nbtasks/time_elapsed, time_elapsed - (nbtasks*(float)task_duration/1000000.0) / (nodes * cores));
    }

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
            "   -g --grid-rows   : number of processes row in the process grid (must divide the total number of processes (default: 1)\n"
            "   -t --task-duration: change the duration of the tasks (default: %d us)\n"
            "   -D --tree-depth  : change the depth of the tree (number of tasks = 2*2^D, default: %d)\n", task_duration, tree_depth);
}

static void runtime_init(int argc, char **argv)
{
    struct option long_options[] =
    {
        {"nb-cores",      required_argument,  0, 'c'},
        {"data-size",     required_argument,  0, 'n'},
        {"task-duration", required_argument,  0, 't'},
        {"tree-depth",    required_argument,  0, 'D'},
        {"help",          no_argument,        0, 'h'},
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
    do
    {
        int c;
        int option_index = 0;
        
        c = getopt_long (argc, argv, "c:n:g:t:D:h",
                         long_options, &option_index);
        
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
}

static void runtime_fini(void)
{
#ifdef USE_MPI
    MPI_Finalize();
#endif    
}

static dplasma_context_t *setup_dplasma(int* pargc, char** pargv[])
{
    dplasma_context_t *dplasma;

    dplasma = dplasma_init(cores, pargc, pargv, N);
    load_dplasma_objects(dplasma);
    {
        expr_t* constant;
        
        constant = expr_new_int( tree_depth );
        dplasma_assign_global_symbol( "DEPTH", constant );
        printf("DEPTH = %d\n", tree_depth);
        constant = expr_new_int( task_duration );
        dplasma_assign_global_symbol( "TASKDURATION", constant );
        constant = expr_new_int( nodes );
        dplasma_assign_global_symbol( "GRIDcols", constant );
        constant = expr_new_int( rank );
        dplasma_assign_global_symbol( "colRANK", constant );
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

static void create_data(void)
{
    int i;

    DataIn = (char *)malloc(N * N * sizeof(double) * sizeof(char));
    for(i = 0; i < N; i++) {
        DataIn[i] = (char)(rank + i);
    }
}
