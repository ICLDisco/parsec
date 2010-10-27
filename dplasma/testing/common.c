#include "common.h"
#include "dague.h"
#include <plasma.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#ifdef USE_MPI
#include <mpi.h>
#endif


/*******************************
 * globals and argv set values *
 *******************************/
#if defined(USE_MPI)
MPI_Datatype SYNCHRO = MPI_BYTE;
#endif  /* USE_MPI */

const int   side[2]  = { PlasmaLeft,    PlasmaRight };
const int   uplo[2]  = { PlasmaUpper,   PlasmaLower };
const int   diag[2]  = { PlasmaNonUnit, PlasmaUnit  };
const int   trans[3] = { PlasmaNoTrans, PlasmaTrans, PlasmaConjTrans };

const char *sidestr[2]  = { "Left ", "Right" };
const char *uplostr[2]  = { "Upper", "Lower" };
const char *diagstr[2]  = { "NonUnit", "Unit   " };
const char *transstr[3] = { "N", "T", "H" };


/**********************************
 * Command line arguments 
 **********************************/
void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            " number            : dimension (N) of the matrices (required)\n"
            "Optional arguments:\n"
            " -c --nb-cores     : number of concurent threads (default: number of physical hyper-threads)\n"
            " -g --nb-gpus      : number of GPU (default: 0)\n"
            " -p -P --grid-rows : rows (P) in the PxQ process grid   (default: NP)\n"
            " -q -Q --grid-cols : columns (Q) in the PxQ process grid (default: NP/P)\n"
            " -k --pri-change   : activate prioritized DAG k steps before the end (default: 0)\n"
            "                   : with no argument, prioritized DAG from the start\n"
            "\n"
            " -N                : dimension (N) of the matrices (required)\n"
            " -M                : dimension (M) of the matrices (default: N)\n"
            " -K --RHS  C<-A*B+C: dimension (K) of the matrices (default: N)\n"
            "           AX=B    : columns in the right hand side (default: 1)\n"
            " -A --LDA          : leading dimension of the matrix A (default: full)\n"
            " -B --LDB          : leading dimension of the matrix B (default: full)\n"
            " -C --LDC          : leading dimension of the matrix C (default: full)\n"
            " -t --NB           : columns in a tile    (default: autotuned)\n"
            " -T --MB           : rows in a tile  (default: NB)\n"
            " -s --SNB          : columns of tiles in a supertile (default: 1)\n"
            " -S --SMB          : rows of tiles in a supertile (default: 1)\n"
            " -x --check        : verify the results\n"
            "\n"
            " -v --verbose      : extra verbose output\n"
            " -h --help         : this message\n"
           );
}

#define GETOPT_STRING "c:g;p:P:q:Q:k;N:M:A:B:C:i:t:T:s:S:xv;h"

#if defined(HAVE_GETOPT_LONG)
static struct option long_options[] =
{
    {"nb-cores",    required_argument,  0, 'c'},
    {"nb-gpus",     required_argument,  0, 'g'},
    {"grid-rows",   required_argument,  0, 'p'},
    {"grid-cols",   required_argument,  0, 'q'},
    {"pri-change",  optional_argument,  0, 'k'},

    {"N",           required_argument,  0, 'N'},
    {"M",           required_argument,  0, 'M'},
    {"K",           required_argument,  0, 'K'},
    {"RHS",         required_argument,  0, 'K'},
    {"LDA",         required_argument,  0, 'A'},
    {"LDB",         required_argument,  0, 'B'},
    {"LDC",         required_argument,  0, 'C'},
    {"IB",          required_argument,  0, 'i'},
    {"NB",          required_argument,  0, 't'},
    {"MB",          required_argument,  0, 'T'},
    {"SNB",         required_argument,  0, 's'},
    {"SMB",         required_argument,  0, 'S'},
    {"check",       no_argument,        0, 'x'},

    {"verbose",     optional_argument,  0, 'v'},
    {"help",        no_argument,        0, 'h'},
    {0, 0, 0, 0}
};
#endif  /* defined(HAVE_GETOPT_LONG) */

static void parse_arguments(int argc, char** argv, int* iparam) 
{
    int optind = 0;
    int verbose;
    int c;

    do
    {
#if defined(HAVE_GETOPT_LONG)
        c = getopt_long(argc, argv, GETOPT_STRING,
                        long_options, &optind);
#else
        c = getopt(argc, argv, GETOPT_STRING);
        (void) optind;
#endif  /* defined(HAVE_GETOPT_LONG) */
        
        switch(c)
        {
            case 'c': iparam[IPARAM_NCORES] = atoi(optarg); break;
            case 'g':
                if(optarg)  iparam[IPARAM_NGPUS] = atoi(optarg);
                else        iparam[IPARAM_NGPUS] = INT_MAX;
                break;
            case 'p': iparam[IPARAM_P] = atoi(optarg); break;
            case 'q': iparam[IPARAM_Q] = atoi(optarg);
            case 'k':
                if(optarg)  iparam[IPARAM_PRIO] = atoi(optarg);
                else        iparam[IPARAM_PRIO] = INT_MAX;
            
            case 'N': iparam[IPARAM_N] = atoi(optarg); break;
            case 'M': iparam[IPARAM_M] = atoi(optarg); break;
            case 'K': iparam[IPARAM_K] = atoi(optarg); break;
            case 'A': iparam[IPARAM_LDA] = atoi(optarg); break;
            case 'B': iparam[IPARAM_LDB] = atoi(optarg); break;
            case 'C': iparam[IPARAM_LDC] = atoi(optarg); break;
            case 'i': iparam[IPARAM_IB] = atoi(optarg); break;
            case 't': iparam[IPARAM_NB] = atoi(optarg); break;
            case 'T': iparam[IPARAM_MB] = atoi(optarg); break;
            case 's': iparam[IPARAM_SNB] = atoi(optarg); break;
            case 'S': iparam[IPARAM_SMB] = atoi(optarg); break;
            case 'x': iparam[IPARAM_CHECK] = 1; break; 
            
            case 'v': 
                if(optarg)  iparam[IPARAM_VERBOSE] = atoi(optarg);
                else        iparam[IPARAM_VERBOSE] = 2;
                break;
            case 'h': print_usage(); exit(0);
            
            case '?': /* getopt_long already printed an error message. */
                exit(1);
            default:
                break; /* Assume anything else is dague/mpi stuff */
        }
    } while(-1 != c);
    verbose = iparam[IPARAM_VERBOSE];
    
    /* Set some sensible default to the number of cores */
    if(iparam[IPARAM_NCORES] <= 0)
    {
        iparam[IPARAM_NCORES] = sysconf(_SC_NPROCESSORS_ONLN);
        if(iparam[IPARAM_NCORES] == -1)
        {
            perror("sysconf(_SC_NPROCESSORS_ONLN)\n");
            iparam[IPARAM_NCORES] = 1;
        }
        if(verbose) 
            fprintf(stderr, "++ cores detected: %d\n", iparam[IPARAM_NCORES]);
    }
    
    /* Check the process grid */
    if(0 == iparam[IPARAM_P])
        iparam[IPARAM_P] = iparam[IPARAM_NNODES];
    if(0 == iparam[IPARAM_Q])
        iparam[IPARAM_Q] = iparam[IPARAM_NNODES] / iparam[IPARAM_P];
    int pqnp = iparam[IPARAM_Q] * iparam[IPARAM_P];
    if(pqnp > iparam[IPARAM_NNODES])
    {
        fprintf(stderr, "xx the process grid PxQ (%dx%d) is larger than the number of nodes (%d)!\n", iparam[IPARAM_P], iparam[IPARAM_Q], iparam[IPARAM_NNODES]);
        exit(2);
    }
    if(verbose && (pqnp < iparam[IPARAM_NNODES])) 
    {
        fprintf(stderr, "!! the process grid PxQ (%dx%d) is smaller than the number of nodes (%d). Some nodes are idling!\n", iparam[IPARAM_P], iparam[IPARAM_Q], iparam[IPARAM_NNODES]);
    }
    if(verbose > 1) fprintf(stderr, "++ nodes:\t%d\n"
                                    "++ cores:\t%d\n"
                                    "++ (PxQ):\t(%dx%d)"
                                    "++ PUs:\t%d", 
                                    iparam[IPARAM_NNODES],
                                    iparam[IPARAM_NCORES],
                                    iparam[IPARAM_P], iparam[IPARAM_Q],
                                    pqnp * iparam[IPARAM_NCORES]); 

    /* Set matrices dimensions to default values if not provided */
    /* Search for N as a bare number if not provided by -N */
    while(0 == iparam[IPARAM_N])
    {
        if(optind < argc)
        {
            iparam[IPARAM_N] = atoi(argv[optind++]);
            continue;
        }
        fprintf(stderr, "xx the matrix size (N) is not set!\n");
        exit(2);
    }
    if(0 == iparam[IPARAM_M]) iparam[IPARAM_M] = iparam[IPARAM_N];
    if(0 == iparam[IPARAM_K]) iparam[IPARAM_K] = iparam[IPARAM_N];
     
    /* Set some sensible defaults for the leading dimensions */
    if(-'m' == iparam[IPARAM_LDA]) iparam[IPARAM_LDA] = iparam[IPARAM_M];
    if(-'n' == iparam[IPARAM_LDA]) iparam[IPARAM_LDA] = iparam[IPARAM_N];
    if(-'k' == iparam[IPARAM_LDA]) iparam[IPARAM_LDA] = iparam[IPARAM_K];
    if(-'m' == iparam[IPARAM_LDB]) iparam[IPARAM_LDB] = iparam[IPARAM_M];
    if(-'n' == iparam[IPARAM_LDB]) iparam[IPARAM_LDB] = iparam[IPARAM_N];
    if(-'k' == iparam[IPARAM_LDB]) iparam[IPARAM_LDB] = iparam[IPARAM_K];
    if(-'m' == iparam[IPARAM_LDC]) iparam[IPARAM_LDC] = iparam[IPARAM_M];
    if(-'n' == iparam[IPARAM_LDC]) iparam[IPARAM_LDC] = iparam[IPARAM_N];
    if(-'k' == iparam[IPARAM_LDC]) iparam[IPARAM_LDC] = iparam[IPARAM_K];

    /* Set no defaults for IB, NB, MB, the algorithm have to do it */
    
    /* No supertiling by default */    
    if(0 == iparam[IPARAM_SNB]) iparam[IPARAM_SNB] = 1;
    if(0 == iparam[IPARAM_SMB]) iparam[IPARAM_SMB] = 1;

    if(verbose > 1) 
    {
        fprintf(stderr, "++ NxM:\t%dx%d\n-- K/RHS:\t%d\n",
                        iparam[IPARAM_N], iparam[IPARAM_M], iparam[IPARAM_K]);
    }
    if(verbose > 2)
    {
        fprintf(stderr, "++ LDA:\t%d\n", iparam[IPARAM_LDA]);
        if(iparam[IPARAM_LDB]) fprintf(stderr, "++ LDB:\t%d\n", iparam[IPARAM_LDB]);
        if(iparam[IPARAM_LDC]) fprintf(stderr, "++ LDC:\t%d\n", iparam[IPARAM_LDB]);
    }
    if(verbose > 1)
    {
        fprintf(stderr, "++ NBxMB:\t%dx%d\n", 
                        iparam[IPARAM_NB], iparam[IPARAM_MB]);
        if(iparam[IPARAM_IB])
            fprintf(stderr, "++ IB:\t%d\n", iparam[IPARAM_IB]);
        if(iparam[IPARAM_SNB] * iparam[IPARAM_SMB] != 1)
            fprintf(stderr, "++ SNBxSMB:\t%dx%d\n", iparam[IPARAM_SNB], iparam[IPARAM_SMB]);
    }
}

static void iparam_default(int* iparam)
{
    /* Just in case someone forget to add the initialization :) */
    memset(iparam, 0, IPARAM_SIZEOF * sizeof(int)); 
    iparam[IPARAM_NNODES] = 1;
}

void iparam_default_facto(int* iparam)
{
    iparam_default(iparam);
    iparam[IPARAM_K] = 1;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = 0;
    iparam[IPARAM_LDC] = 0;
}

void iparam_default_solve(int* iparam)
{
    iparam_default(iparam);
    iparam[IPARAM_K] = 1;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'n';
    iparam[IPARAM_LDC] = 0;
    iparam[IPARAM_M] = -'n';
}

void iparam_default_gemm(int* iparam)
{
    iparam_default(iparam);
    iparam[IPARAM_K] = 0;
    /* no support for transpose yet */
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'k';
    iparam[IPARAM_LDC] = -'m';
}

#ifdef DAGUE_PROFILING
static char* argvzero;
#endif

dague_context_t* setup_dague(int argc, char **argv, int *iparam)
{
#ifdef DAGUE_PROFILING
    argvzero = argv[0];
#endif
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &iparam[IPARAM_NNODES]);
    MPI_Comm_rank(MPI_COMM_WORLD, &iparam[IPARAM_RANK]); 
#endif
    parse_arguments(argc, argv, iparam);
    return dague_init(iparam[IPARAM_NCORES], &argc, &argv);
}

void cleanup_dague(dague_context_t* dague)
{
#ifdef DAGUE_PROFILING
    char* filename = NULL;
#if defined(USE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    asprintf(&filename, "%s.%d.profile", argvzero, rank);
#else
    asprintf(&filename, "%s.profile", argvzero);
#endif
    dague_profiling_dump_xml(filename);
    free(filename);
#endif  /* DAGUE_PROFILING */

    dague_fini(&dague);
#ifdef USE_MPI
    MPI_Finalize();
#endif    
}

