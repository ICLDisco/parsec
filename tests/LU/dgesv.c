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
#include "LU.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

static void runtime_init(int argc, char **argv);
static void runtime_fini(void);

static dague_context_t *setup_dague(int* pargc, char** pargv[]);
static void cleanup_dague(dague_context_t* context);

static void create_datatypes(void);

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
/*printf("[%d] TIMED %f s :\t", rank, time_elapsed);*/ \
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
/*printf("### TIMED %f s :\t", sync_time_elapsed);*/ \
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
/*printf("### TIMED %f doing\t", sync_time_elapsed);*/ \
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

two_dim_block_cyclic_t ddescA;
two_dim_block_cyclic_t ddescL;
two_dim_block_cyclic_t ddescIPIV;
#if defined(DISTRIBUTED)
MPI_Datatype LOWER_TILE, UPPER_TILE, PIVOT_VECT, LITTLE_L;
#endif

/* TODO Remove this ugly stuff */
extern int dgesv_private_memory_initialization(int mb, int nb);
struct dague_memory_pool_t *work_pool = NULL;

int main(int argc, char ** argv)
{
    double gflops;
    dague_context_t* dague;
    
    //#ifdef VTRACE
    // VT_OFF();
    //#endif
    
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

    runtime_init(argc, argv);

    two_dim_block_cyclic_init(&ddescA, matrix_RealDouble, nodes, cores, rank, 
                              NB, NB, IB, N, N, 0, 0, 
                              N, N, nrst, ncst, GRIDrows);
    two_dim_block_cyclic_init(&ddescL, matrix_RealDouble, nodes, cores, rank, 
                              IB, NB, IB, N, N, 0, 0, 
                              N, N, nrst, ncst, GRIDrows);
    two_dim_block_cyclic_init(&ddescIPIV, matrix_Integer, nodes, cores, rank, 
                              1, NB, IB, ddescA.super.lnt, N, 0, 0, 
                              ddescA.super.lnt, N, nrst, ncst, GRIDrows);

    /* matrix generation */
    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescL);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescIPIV);

    /*** THIS IS THE DAGUE COMPUTATION ***/
    TIME_START();
    dague = setup_dague(&argc, &argv);
    if(0 == rank)
        {
            dague_execution_context_t exec_context;
                
            /* I know what I'm doing ;) */
            exec_context.function = (dague_t*)dague_find(dague_LU, "DGETRF");
            exec_context.dague_object = dague_LU;
            exec_context.priority = 0;
            exec_context.locals[0].value = 0;

            dague_schedule(dague, &exec_context);
        }
    TIME_PRINT(("Dague initialization:\t%d %d\n", N, NB));
            
    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_progress(dague);
    TIME_PRINT(("Dague proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));
    SYNC_TIME_PRINT(("Dague computation:\t%d %d %f gflops\n", N, NB,
                     gflops = (2*N/1e3*N/1e3*N/1e3/3.0)/(sync_time_elapsed)));
            
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
    
    create_datatypes();

    dague_LU = (dague_object_t*)dague_LU_new( (dague_ddesc_t*)&ddescL,(dague_ddesc_t*)&ddescIPIV,
                                              (dague_ddesc_t*)&ddescA,
                                              ddescA.super.n, ddescA.super.nb, ddescA.super.lnt, ddescA.super.ib );
    dague->taskstodo += dague_LU->nb_local_tasks;
    nbtasks = dague_LU->nb_local_tasks;
    printf("LU %dx%d has %d tasks to run. Total nb tasks to run: %d\n", 
           ddescA.super.nb, ddescA.super.nt, dague_LU->nb_local_tasks, dague->taskstodo);
    printf("GRIDrows = %d, GRIDcols = %d, rrank = %d, crank = %d\n", 
           ddescA.GRIDrows, ddescA.GRIDcols, ddescA.rowRANK, ddescA.colRANK );

    dgesv_private_memory_initialization(IB, NB);
    
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

/*
 * These datatype creation function works only when the matrix
 * is COLUMN major. In case the matrix storage is ROW major
 * these functions have to be changed.
 */
static void create_datatypes(void)
{
#if defined(USE_MPI)
    int *blocklens, *indices, count, i;
    MPI_Datatype tmp;
    MPI_Aint lb, ub;

    count = NB; 
    blocklens = (int*)malloc( count * sizeof(int) );
    indices = (int*)malloc( count * sizeof(int) );

    /* UPPER_TILE with the diagonal */
    for( i = 0; i < count; i++ ) {
        blocklens[i] = i + 1;
        indices[i] = i * NB;
    }

    MPI_Type_indexed(count, blocklens, indices, MPI_DOUBLE, &UPPER_TILE);
    MPI_Type_set_name(UPPER_TILE, "Upper");
    MPI_Type_commit(&UPPER_TILE);
    
    MPI_Type_get_extent(UPPER_TILE, &lb, &ub);
    
    /* LOWER_TILE without the diagonal */
    for( i = 0; i < count-1; i++ ) {
        blocklens[i] = NB - i - 1;
        indices[i] = i * NB + i + 1;
    }

    MPI_Type_indexed(count-1, blocklens, indices, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, NB*NB*sizeof(double), &LOWER_TILE);
    MPI_Type_set_name(LOWER_TILE, "Lower");
    MPI_Type_commit(&LOWER_TILE);
    
    /* LITTLE_L is a NB*IB rectangle (containing IB*IB Lower tiles) */
    MPI_Type_contiguous(NB*IB, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, NB*NB*sizeof(double), &LITTLE_L);
    MPI_Type_set_name(LITTLE_L, "L");
    MPI_Type_commit(&LITTLE_L);
    
    /* IPIV is a contiguous of size 1*NB */
    MPI_Type_contiguous(NB, MPI_INT, &tmp);
    MPI_Type_create_resized(tmp, 0, NB*NB*sizeof(double), &PIVOT_VECT);
    MPI_Type_set_name(PIVOT_VECT, "IPIV");
    MPI_Type_commit(&PIVOT_VECT);
    
    free(blocklens);
    free(indices);
#endif
}

#if 0
static void check_matrix(int N, PLASMA_enum* uplo, 
                         double* A1, double* A2, 
                         double* B1, double* B2,
                         double* L, int* IPIV,
                         int LDA, int NRHS, int LDB, 
                         PLASMA_desc* dA, PLASMA_desc* dL,
                         double gflops)
{    
    int info_solution;
    double eps = dlamch("Epsilon");
    
    printf("\n");
    printf("------ TESTS FOR PLASMA DGETRF + DTRSMPL + DTRSM  ROUTINE -------  \n");
    printf("            Size of the Matrix %d by %d\n", N, N);
    printf("\n");
    printf(" The matrix A is randomly generated for each test.\n");
    printf("============\n");
    printf(" The relative machine precision (eps) is to be %e \n", eps);
    printf(" Computational tests pass if scaled residuals are less than 60.\n");        
    if(do_nasty_validations)
    {
	PLASMA_Tile_to_Lapack(&dA, A2, LDA);

        plasma_memcpy(L, dL->mat, dL->mt*dL->nt*IB*NB, PlasmaRealDouble);
        
#if defined(USE_MPI)
        // We should have done something in the like of 
        //        memcpy(IPIV, ddescIPIV.mat, sizeof(int)*ddescIPIV.mb*ddescIPIV.lmt*ddescIPIV.lnt);
        // in the gather.
#endif

        PLASMA_dtrsmpl(N, NRHS, A2, LDA, L, IPIV, B2, LDB);
        PLASMA_dtrsm(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, N, NRHS, A2,
                     LDA, B2, LDB);
        
        info_solution = check_solution(N, NRHS, A1, LDA, B1, B2, LDB, eps);
        
        if((info_solution == 0)) 
        {
            printf("****************************************************\n");
            printf(" ---- TESTING DGETRF + DTRSMPL + DTRSM ... PASSED ! \n");
            printf("****************************************************\n");
            printf(" ---- GFLOPS ............................. %.4f\n", gflops);
            printf("****************************************************\n");
        }
        else 
        {
            printf("*****************************************************\n");
            printf(" ---- TESTING DGETRF + DTRSMPL + DTRSM ... FAILED !  \n");
            printf("*****************************************************\n");
        }
        free(A1); free(B1); free(B2);
    }
    else
    {
        printf("****************************************************\n");
        printf(" ---- TESTING DGETRF + DTRSMPL + DTRSM ... SKIPPED !\n");
        printf("****************************************************\n");
        printf(" ---- n= %d np= %d nc= %d g= %dx%d (%dx%d)  %.4f GFLOPS\n", N, nodes, cores, ddescA.GRIDrows, ddescA.GRIDcols, ddescA.nrst, ddescA.ncst, gflops);
        printf("****************************************************\n");
    }
}

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution of the linear system
 */

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
    Rnorm=dlange(&norm, &N, &NRHS, B1, &LDB, work);

    printf("============\n");
    printf("Checking the Residual of the solution \n");
    printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps));

    if ( isnan(Rnorm/((Anorm*Xnorm+Bnorm)*N*eps)) || (Rnorm/((Anorm*Xnorm+Bnorm)*N*eps) > 60.0) ){
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
