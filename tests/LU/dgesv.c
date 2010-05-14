/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "dplasma.h"
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
#include <../src/common.h>
#include <../src/lapack.h>
#include <../src/context.h>
#include <../src/allocate.h>

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
                          double** pL, int** pIPIV,
                          int LDA, int NRHS, int LDB, 
                          PLASMA_desc* dA, PLASMA_desc* dL);
static void create_dl_IPIV();
static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist);
static void gather_matrix(PLASMA_desc* local, DPLASMA_desc* dist);
static void gather_ipiv(void);
static void check_matrix(int N, PLASMA_enum* uplo, 
                         double* A1, double* A2, 
                         double* B1, double* B2,
                         double* L, int* IPIV,
                         int LDA, int NRHS, int LDB, 
                         PLASMA_desc* dA, PLASMA_desc* dL,
                         double gflops);
static void create_datatypes(void);

#if defined(DEBUG_MATRICES)
static void debug_matrices(void);
#else
#define debug_matrices()
#endif

static int check_solution(int, int, double*, int, double*, double*, int, double);


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

typedef enum {
    DO_PLASMA,
    DO_DPLASMA
} backend_argv_t;

/* globals and argv set values */
int do_warmup = 0;
int do_nasty_validations = 0;
int do_distributed_generation = 1;
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
PLASMA_desc descL;
DPLASMA_desc ddescL;
int* _IPIV;
DPLASMA_desc ddescIPIV;
#if defined(USE_MPI)
MPI_Datatype LOWER_TILE, UPPER_TILE, PIVOT_VECT, LITTLE_L;
#endif
static int matgen = 0;

/* TODO Remove this ugly stuff */
extern int dgesv_private_memory_initialization(plasma_context_t*);
struct dplasma_memory_pool_t *work_pool = NULL;

static void display_ddesc(const char *name, DPLASMA_desc *d)
{
    DEBUG(("%s->mat = %p          // pointer to the beginning of the matrix\n", name, d->mat));
    DEBUG(("%s->dtyp = %d         // precision of the matrix\n", name, d->dtyp));
    DEBUG(("%s->mb = %d           // number of rows in a tile\n", name, d->mb));
    DEBUG(("%s->nb = %d           // number of columns in a tile\n", name, d->nb));
    DEBUG(("%s->ib = %d           // number of columns in an inner block\n", name, d->ib));
    DEBUG(("%s->bsiz = %d         // size in elements including padding\n", name, d->bsiz));
    DEBUG(("%s->lm = %d           // number of rows of the entire matrix\n", name, d->lm));
    DEBUG(("%s->ln = %d           // number of columns of the entire matrix\n", name, d->ln));
    DEBUG(("%s->lmt = %d          // number of tile rows of the entire matrix - derived parameter\n", name, d->lmt));
    DEBUG(("%s->lnt = %d          // number of tile columns of the entire matrix - derived parameter\n", name, d->lnt));
    DEBUG(("%s->i = %d            // row index to the beginning of the submatrix\n", name, d->i));
    DEBUG(("%s->j = %d            // column indes to the beginning of the submatrix\n", name, d->j));
    DEBUG(("%s->m = %d            // number of rows of the submatrix\n", name, d->m));
    DEBUG(("%s->n = %d            // number of columns of the submatrix\n", name, d->n));
    DEBUG(("%s->mt = %d           // number of tile rows of the submatrix - derived parameter\n", name, d->mt));
    DEBUG(("%s->nt = %d           // number of tile columns of the submatrix - derived parameter\n", name, d->nt));
    DEBUG(("%s->nrst = %d         // max number of tile rows in a super-tile\n", name, d->nrst));
    DEBUG(("%s->ncst = %d         // max number of tile columns in a super tiles\n", name, d->ncst));
    DEBUG(("%s->mpi_rank = %d     // well... mpi rank...\n", name, d->mpi_rank));
    DEBUG(("%s->GRIDrows = %d     // number of processes rows in the process grid\n", name, d->GRIDrows));
    DEBUG(("%s->GRIDcols = %d     // number of processes cols in the process grid\n", name, d->GRIDcols));
    DEBUG(("%s->cores = %d        // number of cores used for computation per node\n", name, d->cores));
    DEBUG(("%s->nodes = %d        // number of nodes involved in the computation\n", name, d->nodes));
    DEBUG(("%s->colRANK = %d      // process column rank in the process grid - derived parameter\n", name, d->colRANK));
    DEBUG(("%s->rowRANK = %d      // process row rank in the process grid - derived parameter\n", name, d->rowRANK));
    DEBUG(("%s->nb_elem_r = %d    // number of row tiles  handled by this process\n", name, d->nb_elem_r));
    DEBUG(("%s->nb_elem_c = %d    // number of column tiles handled by this process\n", name, d->nb_elem_c));
}

static void display_desc(const char *name, PLASMA_desc *d)
{
    DEBUG(("%s->mat = %p          // pointer to the beginning of the matrix\n", name, d->mat));
    DEBUG(("%s->dtyp = %d         // precision of the matrix\n", name, d->dtyp));
    DEBUG(("%s->mb = %d           // number of rows in a tile\n", name, d->mb));
    DEBUG(("%s->nb = %d           // number of columns in a tile\n", name, d->nb));
    DEBUG(("%s->bsiz = %d         // size in elements including padding\n", name, d->bsiz));
    DEBUG(("%s->lm = %d           // number of rows of the entire matrix\n", name, d->lm));
    DEBUG(("%s->ln = %d           // number of columns of the entire matrix\n", name, d->ln));
    DEBUG(("%s->lmt = %d          // number of tile rows of the entire matrix - derived parameter\n", name, d->lmt));
    DEBUG(("%s->lnt = %d          // number of tile columns of the entire matrix - derived parameter\n", name, d->lnt));
    DEBUG(("%s->i = %d            // row index to the beginning of the submatrix\n", name, d->i));
    DEBUG(("%s->j = %d            // column indes to the beginning of the submatrix\n", name, d->j));
    DEBUG(("%s->m = %d            // number of rows of the submatrix\n", name, d->m));
    DEBUG(("%s->n = %d            // number of columns of the submatrix\n", name, d->n));
    DEBUG(("%s->mt = %d           // number of tile rows of the submatrix - derived parameter\n", name, d->mt));
    DEBUG(("%s->nt = %d           // number of tile columns of the submatrix - derived parameter\n", name, d->nt));
}

int main(int argc, char ** argv)
{
    double gflops;
    double *A1;
    double *A2;
    double *B1;
    double *B2;
    double *L;
    dplasma_context_t* dplasma;
    
    //#ifdef VTRACE
    // VT_OFF();
    //#endif
    
    runtime_init(argc, argv);
    if(0 == rank)
        create_matrix(N, &uplo, &A1, &A2, &B1, &B2, &L, &_IPIV, LDA, NRHS, LDB, &descA, &descL);

    switch(backend)
    {
        case DO_PLASMA: {
            plasma_context_t* plasma = plasma_context_self();

            if(do_warmup)
            {
                TIME_START();                
                plasma_parallel_call_3(plasma_pdgetrf,
                                       PLASMA_desc, descA,
                                       PLASMA_desc, descL,
                                       int*, _IPIV);
                TIME_PRINT(("_plasma warmup:\t\t%d %d %f Gflops\n", N, PLASMA_NB,
                            (2*N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed)));
            }
            TIME_START();
            plasma_parallel_call_3(plasma_pdgetrf,
                                   PLASMA_desc, descA,
                                   PLASMA_desc, descL,
                                   int*, _IPIV);
            TIME_PRINT(("_plasma computation:\t%d %d %f Gflops\n", N, PLASMA_NB, 
                        gflops = (2*N/1e3*N/1e3*N/1e3/3.0)/(time_elapsed)));
            break;
        }
        case DO_DPLASMA: {
            
            scatter_matrix(&descA, &ddescA);/*  create/distribute  matrix A */
            create_dl_IPIV();
            create_datatypes();

            /*** THIS IS THE DPLASMA COMPUTATION ***/
            TIME_START();
            dplasma = setup_dplasma(&argc, &argv);
            if(0 == rank)
            {
                dplasma_execution_context_t exec_context;
                
                /* I know what I'm doing ;) */
                exec_context.function = (dplasma_t*)dplasma_find("DGETRF");
                dplasma_set_initial_execution_context(&exec_context);
                dplasma_schedule(dplasma, &exec_context);
            }
            TIME_PRINT(("Dplasma initialization:\t%d %d\n", N, NB));
            
            if(do_warmup)
                warmup_dplasma(dplasma);
            
            plasma_context_t *plasma = plasma_context_self();

            /* lets rock! */
            SYNC_TIME_START();
            TIME_START();
            dplasma_progress(dplasma);
            TIME_PRINT(("Dplasma proc %d:\ttasks: %d\t%f task/s\n", rank, nbtasks, nbtasks/time_elapsed));
            SYNC_TIME_PRINT(("Dplasma computation:\t%d %d %f gflops\n", N, NB,
                             gflops = (2*N/1e3*N/1e3*N/1e3/3.0)/(sync_time_elapsed)));
            
            cleanup_dplasma(dplasma);
            /*** END OF DPLASMA COMPUTATION ***/

            if(do_nasty_validations) {
                gather_matrix(&descA, &ddescA);
                gather_matrix(&descL, &ddescL);
                gather_ipiv();
            }
            break;
        }
    }
    
    debug_matrices();

    if(0 == rank)
        check_matrix(N, &uplo, A1, A2, B1, B2, L, _IPIV, LDA, NRHS, LDB, &descA, &descL, gflops);
    
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
            "   -s --stile-row   : number of tile per row in a super tile (default: 1)\n"
            "   -e --stile-col   : number of tile per col in a super tile (default: 1)\n"
            "   -a --lda         : leading dimension of the matrix A (equal matrix size by default)\n"
            "   -b --ldb         : leading dimension of the RHS B (equal matrix size by default)\n"
            "   -r --nrhs        : Number of Right Hand Side (default: 1)\n"
            "   -x --xcheck      : do extra nasty result validations\n"
            "   -w --warmup      : do some warmup, if > 1 also preload cache\n"
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
        {"xcheck",      no_argument,        0, 'x'},
        {"warmup",      optional_argument,  0, 'w'},
        {"dplasma",     no_argument,        0, 'd'},
        {"plasma",      no_argument,        0, 'p'},
        {"block-size",  required_argument,  0, 'B'},
        {"internal-block-size", required_argument, 0, 'I'},
        {"matrix-type", required_argument,  0, 't'},
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
    rank = 0;
#endif
    
    /* parse arguments */
    ddescA.GRIDrows = 1;
    ddescA.nrst = ddescA.ncst = 1;
    do
    {
        int c;
#if defined(HAVE_GETOPT_LONG)
        int option_index = 0;
        c = getopt_long (argc, argv, "dpxc:n:a:r:b:g:e:s:w::B:t:I:h",
                         long_options, &option_index);
#else
        c = getopt (argc, argv, "dpxc:n:a:r:b:g:e:s:w::B:t:I:h");
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
                backend = DO_DPLASMA;
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
                ddescA.GRIDrows = atoi(optarg);
                break;
            case 's':
                ddescA.nrst = atoi(optarg);
                if(ddescA.nrst <= 0)
                    {
                        fprintf(stderr, "select a positive value for the row super tile size\n");
                        exit(2);
                    }                
                /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
                break;
            case 'e':
                ddescA.ncst = atoi(optarg);
                if(ddescA.ncst <= 0)
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
                        block_forced = atoi(optarg);
                        ddescA.nb = block_forced;
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
                        internal_block_forced = atoi(optarg);
                        ddescA.ib = internal_block_forced;
                    }
                else
                    {
                        fprintf(stderr, "Argument is mandatory for -I (--internal-block-size) flag.\n");
                        exit(2);
                    }
                break;
            case 't':
                matgen = atoi(optarg);
                if (matgen)
                    printf("cholesky like matrix generation \n");
                else
                    printf("LU like matrix generation \n");
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
    ddescA.cores = cores;
    switch(backend)
    {
        case DO_PLASMA:
            PLASMA_Init(cores);
            break;
        case DO_DPLASMA:
            PLASMA_Init(1);
            break;
    }

    plasma_tune(PLASMA_FUNC_DGESV, N, N, NRHS);
    if( 0 != block_forced ) {
        plasma_context_t* plasma = plasma_context_self();

        PLASMA_NB = block_forced;
        PLASMA_NBNBSIZE = PLASMA_NB * PLASMA_NB;

        if( 0 != internal_block_forced ) {
            if( PLASMA_NB % internal_block_forced != 0 ) {
                fprintf(stderr, "Invalid IB flag: %d (internal block size) does not divide %d (block size)\n",
                        internal_block_forced, block_forced);
                exit(1);
            }
            PLASMA_IB = internal_block_forced;
        } else {
            PLASMA_IB = (PLASMA_NB / 5);
            if( (PLASMA_NB % PLASMA_IB) != 0 ) {
                fprintf(stderr, "Invalid -B flag: heuristic is to take Inner block of %d, which is not a divisor of the Block size %d\n",
                        PLASMA_IB, PLASMA_NB);
                exit(0);
            }
            ddescA.ib = PLASMA_IB;
            /*printf("Using an internal block size of %d\n",PLASMA_IB);*/
        }

        PLASMA_IBNBSIZE = PLASMA_IB * PLASMA_NB;

        plasma->autotuning_enabled = 0;
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
    plasma_context_t* plasma = plasma_context_self();
    
    dplasma = dplasma_init(cores, pargc, pargv, ddescA.nb);
    load_dplasma_objects(dplasma);
    
    dgesv_private_memory_initialization(plasma);
#if 0
    // TODO: this should be allocated per execution context.
    work = (double *)plasma_private_alloc(plasma, descL.mb*descL.nb, descL.dtyp);
#endif
    
    {
        expr_t* constant;
        
        constant = expr_new_int( ddescA.nt );
        dplasma_assign_global_symbol( "NT", constant );
        constant = expr_new_int( ddescA.GRIDrows );
        dplasma_assign_global_symbol( "GRIDrows", constant );
        constant = expr_new_int( ddescA.GRIDcols );
        dplasma_assign_global_symbol( "GRIDcols", constant );
        constant = expr_new_int( ddescA.rowRANK );
        dplasma_assign_global_symbol( "rowRANK", constant );
        constant = expr_new_int( ddescA.colRANK );
        dplasma_assign_global_symbol( "colRANK", constant );
        constant = expr_new_int( ddescA.nrst );
        dplasma_assign_global_symbol( "rtileSIZE", constant );
        constant = expr_new_int( ddescA.ncst );
        dplasma_assign_global_symbol( "ctileSIZE", constant );
    }
    load_dplasma_hooks(dplasma);
    nbtasks = enumerate_dplasma_tasks(dplasma);
    
    return dplasma;
}

static void cleanup_dplasma(dplasma_context_t* dplasma)
{
#ifdef DPLASMA_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", "dgesv", rank );
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
        exec_context.function = (dplasma_t*)dplasma_find("DGETRF");
        dplasma_set_initial_execution_context(&exec_context);
        dplasma_schedule(dplasma, &exec_context);
    }
# ifdef USE_MPI
    /* Make sure everybody is done with warmup before proceeding */
    MPI_Barrier(MPI_COMM_WORLD);
# endif    
}

/*
 * These datatype creation function works only when the matrix
 * is COLUMN major. In case the matrix storage is ROW major
 * these functions have to be changed.
 */
static void create_datatypes(void)
{
#if defined(USE_MPI)
    plasma_context_t* plasma = plasma_context_self();
    int *blocklens, *indices, count, i;
    MPI_Datatype tmp;
    MPI_Aint lb, ub;
    int IB = PLASMA_IB;

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

#undef N
#undef NB


static void create_matrix(int N, PLASMA_enum* uplo, 
                          double** pA1, double** pA2, 
                          double** pB1, double** pB2, 
                          double** pL, int** pIPIV,
                          int LDA, int NRHS, int LDB, PLASMA_desc* dA, PLASMA_desc* dL)
{
#define A1      (*pA1)
#define A2      (*pA2)
#define B1      (*pB1)
#define B2      (*pB2)
#define L       (*pL)
#define IPIV    (*pIPIV)
    int i, j;
    
    if(do_distributed_generation) 
    {
        A1 = A2 = B1 = B2 = L = NULL;
        IPIV = NULL;
        return;
    }
    
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
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                A2[LDA*j+i] = A1[LDA*j+i] = 0.5 - (double)rand() / RAND_MAX;
        for (i = 0; i < N; i++)
            for (j = 0; j < NRHS; j++)
                B2[LDB*j+i] = B1[LDB*j+i] = 0.5 - (double)rand() / RAND_MAX;        
        /*
        for( i = 0; i < N; i++ )
            A2[LDA*i+i] = A1[LDA*i+i] = A1[LDA*i+i] + 10 * N;
        */
    }
    else
    {
        /* Only need A2 */
        A1 = B1 = B2 = NULL;
        A2 = (double *)malloc(LDA*N*sizeof(double));
        /* Check if unable to allocate memory */
        if (!A2){
            printf("Out of Memory \n ");
            exit(1);
        }
        
        /* generating a random matrix */
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                A2[LDA*j+i] = 0.5 - (double)rand() / RAND_MAX;
        /*
        for( i = 0; i < N; i++ )
            A2[LDA*i+i] = A2[LDA*i+i] + 10 * N;        
        */
    }
    
    
    plasma_context_t* plasma = plasma_context_self();
    double* Abdl;
    double* Lbdl;
    int NB, NT;
    
    NB = PLASMA_NB;
    NT = (N%NB==0) ? (N/NB) : (N/NB+1);

    PLASMA_Alloc_Workspace_dgesv(N, &L, &IPIV);
    Abdl = (double*) plasma_shared_alloc(plasma, NT*NT*PLASMA_NBNBSIZE, PlasmaRealDouble);
    Lbdl = (double*) plasma_shared_alloc(plasma, NT*NT*PLASMA_IBNBSIZE, PlasmaRealDouble);
    *dA = plasma_desc_init(Abdl, PlasmaRealDouble,
                           PLASMA_NB, PLASMA_NB, PLASMA_NBNBSIZE,
                           N, N, 0, 0, N, N);
    *dL = plasma_desc_init(Lbdl, PlasmaRealDouble,
                           PLASMA_IB, PLASMA_NB, PLASMA_IBNBSIZE,
                           N, N, 0, 0, N, N);    
    plasma_parallel_call_3(plasma_lapack_to_tile,
                           double*, A2,
                           int, LDA,
                           PLASMA_desc, *dA);
    
    plasma_memzero(IPIV, dA->mt*dA->nt*PLASMA_NB, PlasmaInteger);
    plasma_memzero(dL->mat, dL->mt*dL->nt*PLASMA_IBNBSIZE, PlasmaRealDouble);
    
#undef A1
#undef A2 
#undef B1 
#undef B2
#undef L
#undef IPIV
}

/* create ipiv and dl.
 * must be call after scatter/creation of matrix A 
 */
static void create_dl_IPIV()
{
    /* assign same values for both matrix description */
    ddescL = ddescA; 
    ddescIPIV = ddescA;

    /* now change L*/
    ddescL.mb =  ddescA.ib;
    ddescL.bsiz = ddescA.nb * ddescA.ib;
    ddescL.lm = ddescA.lmt * ddescA.ib;
    ddescL.m = ddescL.lm;
    ddescL.mat = calloc(ddescA.nb_elem_r * ddescA.nb_elem_c * ddescL.bsiz, sizeof(double));

    /* and change IPIV */
    ddescIPIV.nb = 1;
    ddescIPIV.bsiz = ddescA.mb;
    ddescIPIV.ln = ddescA.lnt;
    ddescIPIV.n = ddescIPIV.ln;
#ifdef USE_MPI
    ddescIPIV.mat = calloc(ddescA.nb_elem_r * ddescA.nb_elem_c * ddescIPIV.bsiz, sizeof(int));
#else
    ddescIPIV.mat = _IPIV;
#endif
    return;
}


static void scatter_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
    if(do_distributed_generation)
    {
        TIME_START();
        dplasma_description_init(dist, LDA, LDB, NRHS, uplo);
        rand_dist_matrix(dist, matgen);
        /*TIME_PRINT(("distributed matrix generation on rank %d\n", dist->mpi_rank));*/
        return;
    }
    
    TIME_START();
    if(0 == rank)
    {
        dplasma_desc_init(local, dist);
    }
#ifdef USE_MPI
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
    
#endif /* NO MPI */
}

static void gather_matrix(PLASMA_desc* local, DPLASMA_desc* dist)
{
    if(do_distributed_generation) 
    {
        return;
    }
# ifdef USE_MPI
    if(do_nasty_validations)
    {
        TIME_START();
        gather_data(local, dist);
        TIME_PRINT(("data reduction on rank %d (to rank 0)\n", dist->mpi_rank));
    }
# endif
}

static void gather_ipiv(void)
{
#if USE_MPI
    int i, j, k, _rank;

    if( do_distributed_generation )
        return;

    if( do_nasty_validations ) {
        k = 0;
        if ( ddescIPIV.mpi_rank == 0 )
            {
                for(j = 0; j < ddescIPIV.lnt ; j++)
                    for (i = 0 ; i < ddescIPIV.lmt ; i++ )
                        {
                            _rank = dplasma_get_rank_for_tile(&ddescIPIV, i, j);
                            if (_rank == 0)
                                memcpy(&_IPIV[descA.nb*i+descA.nb*descA.lmt*j], 
                                       dplasma_get_local_IPIV(&ddescIPIV, i, j), 
                                       ddescIPIV.bsiz * sizeof(int));
                            else
                                {
                                    MPI_Recv(&_IPIV[descA.nb*i+descA.nb*descA.lmt*j], 
                                             ddescIPIV.bsiz, MPI_INT, _rank, 1, MPI_COMM_WORLD, 
                                             MPI_STATUS_IGNORE);
                                }
                        }
            }
        else
            {
                for(j = 0; j < ddescIPIV.lnt ; j++)
                    for (i = 0 ; i < ddescIPIV.lmt ; i++ )
                        {
                            _rank = dplasma_get_rank_for_tile(&ddescIPIV, i, j);
                            if (_rank == ddescIPIV.mpi_rank)
                                {
                                    MPI_Send( dplasma_get_local_IPIV(&ddescIPIV, i, j), ddescIPIV.bsiz, MPI_INT, 0, 1, MPI_COMM_WORLD );
                                }
                        }
            }
    }
#endif
}

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
        plasma_context_t* plasma = plasma_context_self();
        plasma_parallel_call_3(plasma_tile_to_lapack,
                               PLASMA_desc, *dA,
                               double*, A2,
                               int, LDA);
        plasma_memcpy(L, dL->mat, dL->mt*dL->nt*PLASMA_IBNBSIZE, PlasmaRealDouble);
        
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

#undef rank


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

#if defined(DEBUG_MATRICES)
#if defined(USE_MPI)
#define A(m,n) dplasma_get_local_tile_s(&ddescA, m, n)
#define L(m,n) dplasma_get_local_tile_s(&ddescL, m, n)
#define descA ddescA
#define descL ddescL
#else
#define A(m,n) &(((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)])
#define L(m,n) &(((double*)descL.mat)[descL.bsiz*(m)+descL.bsiz*descL.lmt*(n)])
#endif
#define MAXDBLSTRLEN 16

static void debug_matrices(void)
{
    int tilem, tilen;
    int m, n, len, pos;
    double *a;
    char *line;
#if defined(USE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    len = 32 + (MAXDBLSTRLEN + 1) * descA.nb;
    line = (char *)malloc( len );

    for(tilem = 0; tilem < descA.mt; tilem++) {
        for(tilen = 0; tilen < descA.nt; tilen++) {
#if defined(USE_MPI)
            if( dplasma_get_rank_for_tile(&ddescA, tilem, tilen) == rank ) {
#endif
                a = A(tilem, tilen);
                fprintf(stderr, "[%d] A(%d, %d) = \n", rank, tilem, tilen);
                pos = 0;
                for(m = 0; m < descA.mb; m++) {
                    for(n = 0; n < descA.nb; n++) {
                        pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + descA.mb * n]);
                    }
                    fprintf(stderr, "[%d]   %s\n", rank, line);
                    pos = 0;
                }
#if defined(USE_MPI)
            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
    }

    for(tilem = 0; tilem < descL.mt; tilem++) {
        for(tilen = 0; tilen < descL.nt; tilen++) {
#if defined(USE_MPI)
            if( dplasma_get_rank_for_tile(&ddescL, tilem, tilen) == rank ) {
#endif
                a = L(tilem, tilen);
                fprintf(stderr, "[%d] dL(%d, %d) = \n", rank, tilem, tilen);
                pos = 0;
                for(m = 0; m < descL.mb; m++) {
                    for(n = 0; n < descL.nb; n++) {
                        pos += snprintf(line + pos, len-pos, "%9.5f ", a[m + descL.mb * n]);
                    }
                    fprintf(stderr, "[%d]   %s\n", rank, line);
                    pos = 0;
                }
#if defined(USE_MPI)
            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
    }

    free(line);
}
#undef descA
#undef descL
#undef A
#undef L
#endif /* defined(DEBUG_MATRICES) */
