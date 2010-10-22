#include "dague.h"
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#include <stdlib.h>
#include <stdio.h>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <plasma.h>
#include "testscommon.h"

/*******************************
 * globals and argv set values *
 *******************************/
#if defined(USE_MPI)
MPI_Datatype SYNCHRO = MPI_BYTE;
#endif  /* USE_MPI */

int   side[2]  = { PlasmaLeft,    PlasmaRight };
int   uplo[2]  = { PlasmaUpper,   PlasmaLower };
int   diag[2]  = { PlasmaNonUnit, PlasmaUnit  };
int   trans[3] = { PlasmaNoTrans, PlasmaTrans, PlasmaConjTrans };

char *sidestr[2]  = { "Left ", "Right" };
char *uplostr[2]  = { "Upper", "Lower" };
char *diagstr[2]  = { "NonUnit", "Unit   " };
char *transstr[3] = { "N", "T", "H" };

/**********************************
 *  Print Usage
 **********************************/
void print_usage(void)
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

void runtime_init(int argc, char **argv, int *iparam)
{
    /* Just in case someone forget to add the initialization :) */
    memset(iparam, 0, IPARAM_INBPARAM*sizeof(int)); 

    /* Initialize iparam */
    iparam[IPARAM_RANK]     = 0;     /* Rank                              */
    iparam[IPARAM_NNODES]   = 1;     /* Number of nodes                   */
    iparam[IPARAM_NCORES]   = 1;     /* Number of cores                   */
    iparam[IPARAM_NGPUS]    = 0;     /* Number of GPUs                    */
    iparam[IPARAM_M]        = 0;     /* Number of rows of the matrix      */
    iparam[IPARAM_N]        = 0;     /* Number of columns of the matrix   */
    iparam[IPARAM_LDA]      = 0;     /* Leading dimension of the matrix   */
    iparam[IPARAM_NRHS]     = 1;     /* Number of right hand side         */
    iparam[IPARAM_LDB]      = 0;     /* Leading dimension of rhs          */
    iparam[IPARAM_MB]       = 120;   /* Number of rows in a tile          */
    iparam[IPARAM_NB]       = 120;   /* Number of columns in a tile       */
    iparam[IPARAM_IB]       = 40;    /* Inner-blocking size               */
    iparam[IPARAM_CHECK]    = 0;     /* Checking activated or not         */
    iparam[IPARAM_GDROW]    = 1;     /* Number of rows in the grid        */
    iparam[IPARAM_STM]      = 1;     /* Number of rows in a super-tile    */
    iparam[IPARAM_STN]      = 1;     /* Number of columns in a super-tile */
    iparam[IPARAM_PRIORITY] = 0;
    
#ifdef USE_MPI
    /* mpi init */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &iparam[IPARAM_NNODES]);
    MPI_Comm_rank(MPI_COMM_WORLD, &iparam[IPARAM_RANK]); 
#endif

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
        {"checking",    no_argument,        0, 'C'},
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
        
        c = getopt_long (argc, argv, "c:n:a:r:b:g:e:s:B:P:Ch",
                         long_options, &option_index);
#else
        c = getopt (argc, argv, "c:n:a:r:b:g:e:s:B:P:Ch");
#endif  /* defined(HAVE_GETOPT_LONG) */
        
        /* Detect the end of the options. */
        if (c == -1)
            break;
        
        switch (c)
        {
	case 'c':
	    iparam[IPARAM_NCORES] = atoi(optarg);
	    if( iparam[IPARAM_NCORES] <= 0 )
		iparam[IPARAM_NCORES] = 1;
	    //printf("Number of cores (computing threads) set to %d\n", cores);
	    break;
	    
	case 'n':
	    iparam[IPARAM_N] = atoi(optarg);
	    //printf("matrix size set to %d\n", N);
	    break;
	    
	case 'g':
	    iparam[IPARAM_GDROW] = atoi(optarg);
	    break;
	case 's':
	    iparam[IPARAM_STM] = atoi(optarg);
	    if( iparam[IPARAM_STM] <= 0 )
                {
                    fprintf(stderr, "select a positive value for the row super tile size\n");
                    exit(2);
                }                
	    /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
	    break;
	case 'e':
	    iparam[IPARAM_STN] = atoi(optarg);
	    if( iparam[IPARAM_STN] <= 0 )
                {
                    fprintf(stderr, "select a positive value for the col super tile size\n");
                    exit(2);
                }                
	    /*printf("processes receives tiles by blocks of %dx%d\n", ddescA.nrst, ddescA.ncst);*/
	    break;
            
	case 'r':
	    iparam[IPARAM_NRHS] = atoi(optarg);
	    printf("number of RHS set to %d\n", iparam[IPARAM_NRHS]);
	    break;
	case 'a':
	    iparam[IPARAM_LDA] = atoi(optarg);
	    printf("LDA set to %d\n", iparam[IPARAM_LDA]);
	    break;                
	case 'b':
	    iparam[IPARAM_LDB] = atoi(optarg);
	    printf("LDB set to %d\n",iparam[IPARAM_LDB]);
	    break;
            
        case 'B':
	    iparam[IPARAM_NB] = atoi(optarg);
	    if( iparam[IPARAM_NB] <= 0 )
                {
                    fprintf(stderr, "select a positive value for the block size\n");
                    exit(2);
                }
	    break;
	    
        case 'P':
	    iparam[IPARAM_PRIORITY] = atoi(optarg);
	    break;
        case 'C':
	    iparam[IPARAM_CHECK] = 1;
	    break;
        case 'h':
            print_usage();
            exit(0);
        case '?': /* getopt_long already printed an error message. */
        default:
            break; /* Assume anything else is dague/mpi stuff */
        }
    } while(1);
    
    while(iparam[IPARAM_N] == 0)
        {
            if(optind < argc)
                {
                    iparam[IPARAM_N] = atoi(argv[optind++]);
                    continue;
                }
            print_usage(); 
            exit(2);
        }

    /* For now, we only have square matrices */
    iparam[IPARAM_M]  = iparam[IPARAM_N];
    iparam[IPARAM_MB] = iparam[IPARAM_NB];
    if((iparam[IPARAM_NNODES] % iparam[IPARAM_GDROW]) != 0)
        {
            fprintf(stderr, "GRIDrows %d does not divide the total number of nodes %d\n", iparam[IPARAM_GDROW], iparam[IPARAM_NNODES]);
            exit(2);
        }
    //printf("Grid is %dx%d\n", ddescA.GRIDrows, ddescA.GRIDcols);
    
    if(iparam[IPARAM_LDA] <= 0) 
        {
            iparam[IPARAM_LDA] = iparam[IPARAM_M];
        }
    if(iparam[IPARAM_LDB] <= 0) 
        {
	    iparam[IPARAM_LDB] = iparam[IPARAM_M];
        }
    
    /* PLASMA_Init(1); */
}

void runtime_fini(void)
{
    /* PLASMA_Finalize(); */
#ifdef USE_MPI
    MPI_Finalize();
#endif    
}

/*
 *  DaGUE Setup
 */
dague_context_t *setup_dague(int* pargc, char** pargv[], int *iparam, int type)
{
    dague_context_t *dague;
    
    dague = dague_init(iparam[IPARAM_NCORES], pargc, pargv, iparam[IPARAM_MB]);

#ifdef USE_MPI
    /**
     * Redefine the default type after dague_init.
     */
    {
        MPI_Datatype default_ddt;
        char type_name[MPI_MAX_OBJECT_NAME];
    
	switch( type ) {
	case PlasmaRealFloat:
	    snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_FLOAT*%d*%d", 
		     iparam[IPARAM_MB], iparam[IPARAM_NB]);
    
	    MPI_Type_contiguous(iparam[IPARAM_MB]*iparam[IPARAM_NB], MPI_FLOAT, &default_ddt);
	    MPI_Type_set_name(default_ddt, type_name);
	    MPI_Type_commit(&default_ddt);
	    dague_arena_construct(&DAGUE_DEFAULT_DATA_TYPE, iparam[IPARAM_MB]*iparam[IPARAM_NB]*sizeof(float), 
				  DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
	    break;
	case PlasmaRealDouble:
	    snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE*%d*%d", 
		     iparam[IPARAM_MB], iparam[IPARAM_NB]);
    
	    MPI_Type_contiguous(iparam[IPARAM_MB]*iparam[IPARAM_NB], MPI_DOUBLE, &default_ddt);
	    MPI_Type_set_name(default_ddt, type_name);
	    MPI_Type_commit(&default_ddt);
	    dague_arena_construct(&DAGUE_DEFAULT_DATA_TYPE, iparam[IPARAM_MB]*iparam[IPARAM_NB]*sizeof(double), 
				  DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
	    break;
	case PlasmaComplexFloat:
	    snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_COMPLEX*%d*%d", 
		     iparam[IPARAM_MB], iparam[IPARAM_NB]);
    
	    MPI_Type_contiguous(iparam[IPARAM_MB]*iparam[IPARAM_NB], MPI_COMPLEX, &default_ddt);
	    MPI_Type_set_name(default_ddt, type_name);
	    MPI_Type_commit(&default_ddt);
	    dague_arena_construct(&DAGUE_DEFAULT_DATA_TYPE, iparam[IPARAM_MB]*iparam[IPARAM_NB]*sizeof(PLASMA_Complex32_t), 
				  DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
	    break;
	case PlasmaComplexDouble:
	    snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE_COMPLEX*%d*%d", 
		     iparam[IPARAM_MB], iparam[IPARAM_NB]);
    
	    MPI_Type_contiguous(iparam[IPARAM_MB]*iparam[IPARAM_NB], MPI_DOUBLE_COMPLEX, &default_ddt);
	    MPI_Type_set_name(default_ddt, type_name);
	    MPI_Type_commit(&default_ddt);
	    dague_arena_construct(&DAGUE_DEFAULT_DATA_TYPE, iparam[IPARAM_MB]*iparam[IPARAM_NB]*sizeof(PLASMA_Complex64_t), 
				  DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
	    break;
	default:
	    fprintf(stderr, "Type Inconnu\n");
	    exit(2);
	}
    }
#endif  /* USE_MPI */

    return dague;
}

void cleanup_dague(dague_context_t* dague, char *name)
{
#ifdef DAGUE_PROFILING
    char* filename = NULL;
    
    asprintf( &filename, "%s.%d.profile", name, rank );
    dague_profiling_dump_xml(filename);
    free(filename);
#else
    (void) name;
#endif  /* DAGUE_PROFILING */
    dague_fini(&dague);
}
