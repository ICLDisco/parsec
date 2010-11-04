/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#include "common.h"
#include "common_timing.h"
#include "dague.h"
#include <plasma.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#if defined(DAGUE_CUDA_SUPPORT)
#include "gpu_data.h"
#endif

/*******************************
 * globals and argv set values *
 *******************************/
#if defined(HAVE_MPI)
MPI_Datatype SYNCHRO = MPI_BYTE;
#endif  /* HAVE_MPI */

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
            " -c --cores        : number of concurent threads (default: number of physical hyper-threads)\n"
            " -g --gpus         : number of GPU (default: 0)\n"
            " -p -P --grid-rows : rows (P) in the PxQ process grid   (default: NP)\n"
            " -q -Q --grid-cols : columns (Q) in the PxQ process grid (default: NP/P)\n"
            " -k --prio-switch  : activate prioritized DAG k steps before the end (default: 0)\n"
            "                   : with no argument, prioritized DAG from the start\n"
            "\n"
            " -N                : dimension (N) of the matrices (required)\n"
            " -M                : dimension (M) of the matrices (default: N)\n"
            " -K --NRHS C<-A*B+C: dimension (K) of the matrices (default: N)\n"
            "           AX=B    : columns in the right hand side (default: 1)\n"
            " -A --LDA          : leading dimension of the matrix A (default: full)\n"
            " -B --LDB          : leading dimension of the matrix B (default: full)\n"
            " -C --LDC          : leading dimension of the matrix C (default: full)\n"
            " -i --IB           : inner blocking     (default: autotuned)\n"
            " -t --NB           : columns in a tile  (default: autotuned)\n"
            " -T --MB           : rows in a tile     (default: autotuned)\n"
            " -s --SNB          : columns of tiles in a supertile (default: 1)\n"
            " -S --SMB          : rows of tiles in a supertile (default: 1)\n"
            " -x --check        : verify the results\n"
            "\n"
            " -v --verbose      : extra verbose output\n"
            " -h --help         : this message\n"
           );
}

#define GETOPT_STRING "c:g::p:P:q:Q:k::N:M:K:A:B:C:i:t:T:s:S:xv::h"

#if defined(HAVE_GETOPT_LONG)
static struct option long_options[] =
{
    {"cores",       required_argument,  0, 'c'},
    {"c",           required_argument,  0, 'c'},
    {"gpus",        required_argument,  0, 'g'},
    {"g",           required_argument,  0, 'g'},
    {"grid-rows",   required_argument,  0, 'p'},
    {"p",           required_argument,  0, 'p'},
    {"P",           required_argument,  0, 'p'},
    {"grid-cols",   required_argument,  0, 'q'},
    {"q",           required_argument,  0, 'q'},
    {"Q",           required_argument,  0, 'q'},
    {"prio-switch", optional_argument,  0, 'k'},
    {"k",           optional_argument,  0, 'k'},

    {"N",           required_argument,  0, 'N'},
    {"M",           required_argument,  0, 'M'},
    {"K",           required_argument,  0, 'K'},
    {"NRHS",        required_argument,  0, 'K'},
    {"LDA",         required_argument,  0, 'A'},
    {"A",           required_argument,  0, 'A'},
    {"LDB",         required_argument,  0, 'B'},
    {"B",           required_argument,  0, 'B'},
    {"LDC",         required_argument,  0, 'C'},
    {"C",           required_argument,  0, 'C'},
    {"IB",          required_argument,  0, 'i'},
    {"i",           required_argument,  0, 'i'},
    {"NB",          required_argument,  0, 't'},
    {"t",           required_argument,  0, 't'},
    {"MB",          required_argument,  0, 'T'},
    {"T",           required_argument,  0, 'T'},
    {"SNB",         required_argument,  0, 's'},
    {"s",           required_argument,  0, 's'},
    {"SMB",         required_argument,  0, 'S'},
    {"S",           required_argument,  0, 'S'},
    {"check",       no_argument,        0, 'x'},
    {"x",           required_argument,  0, 'x'},

    {"verbose",     optional_argument,  0, 'v'},
    {"v",           optional_argument,  0, 'v'},
    {"help",        no_argument,        0, 'h'},
    {"h",           no_argument,        0, 'h'},
    {0, 0, 0, 0}
};
#endif  /* defined(HAVE_GETOPT_LONG) */

static void parse_arguments(int argc, char** argv, int* iparam) 
{
    int opt = 0;
    int c;

    do
    {
#if defined(HAVE_GETOPT_LONG)
        c = getopt_long_only(argc, argv, "",
                        long_options, &opt);
#else
        c = getopt(argc, argv, GETOPT_STRING);
        (void) opt;
#endif  /* defined(HAVE_GETOPT_LONG) */
    
 //       printf("%c: %s = %s\n", c, long_options[opt].name, optarg);
        switch(c)
        {
            case 'c': iparam[IPARAM_NCORES] = atoi(optarg); break;
            case 'g':
                if(iparam[IPARAM_NGPUS] == -1)
                {
                    fprintf(stderr, "!!! This test does not have GPU support. GPU disabled.\n");
                    break;
                }
                if(optarg)  iparam[IPARAM_NGPUS] = atoi(optarg);
                else        iparam[IPARAM_NGPUS] = INT_MAX;
                break;
            case 'p': case 'P': iparam[IPARAM_P] = atoi(optarg); break;
            case 'q': case 'Q': iparam[IPARAM_Q] = atoi(optarg); break;
            case 'k':
                if(optarg)  iparam[IPARAM_PRIO] = atoi(optarg);
                else        iparam[IPARAM_PRIO] = INT_MAX;
                break;
            
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
    int verbose = iparam[IPARAM_RANK] ? 0 : iparam[IPARAM_VERBOSE];
    
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
            fprintf(stderr, "+++ cores detected      : %d\n", iparam[IPARAM_NCORES]);
    }
    if(iparam[IPARAM_NGPUS] < 0) iparam[IPARAM_NGPUS] = 0;
    
    /* Check the process grid */
    if(0 == iparam[IPARAM_P])
        iparam[IPARAM_P] = iparam[IPARAM_NNODES];
    if(0 == iparam[IPARAM_Q])
        iparam[IPARAM_Q] = iparam[IPARAM_NNODES] / iparam[IPARAM_P];
    int pqnp = iparam[IPARAM_Q] * iparam[IPARAM_P];
    if(pqnp > iparam[IPARAM_NNODES])
    {
        fprintf(stderr, "xxx the process grid PxQ (%dx%d) is larger than the number of nodes (%d)!\n", iparam[IPARAM_P], iparam[IPARAM_Q], iparam[IPARAM_NNODES]);
        exit(2);
    }
    if(verbose && (pqnp < iparam[IPARAM_NNODES])) 
    {
        fprintf(stderr, "!!! the process grid PxQ (%dx%d) is smaller than the number of nodes (%d). Some nodes are idling!\n", iparam[IPARAM_P], iparam[IPARAM_Q], iparam[IPARAM_NNODES]);
    }
    if(verbose > 1) fprintf(stderr, "+++ nodes x cores + gpu : %d x %d + %d (%d+%d)\n"
                                    "+++ P x Q               : %d x %d (%d/%d)\n",
                                    iparam[IPARAM_NNODES],
                                    iparam[IPARAM_NCORES],
                                    iparam[IPARAM_NGPUS],
                                    iparam[IPARAM_NNODES] * iparam[IPARAM_NCORES],
                                    iparam[IPARAM_NNODES] * iparam[IPARAM_NGPUS],
                                    iparam[IPARAM_P], iparam[IPARAM_Q],
                                    pqnp, iparam[IPARAM_NNODES]); 

    /* Set matrices dimensions to default values if not provided */
    /* Search for N as a bare number if not provided by -N */
    while(0 == iparam[IPARAM_N])
    {
        if(optind < argc)
        {
            iparam[IPARAM_N] = atoi(argv[optind++]);
            continue;
        }
        fprintf(stderr, "xxx the matrix size (N) is not set!\n");
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
    assert(iparam[IPARAM_IB]); /* check that defaults have been set */
    if(iparam[IPARAM_MB] <= 0 && iparam[IPARAM_NB] > 0)
        iparam[IPARAM_MB] = iparam[IPARAM_NB];
    if(iparam[IPARAM_NB] < 0) iparam[IPARAM_NB] = -iparam[IPARAM_NB];
    if(iparam[IPARAM_MB] == 0) iparam[IPARAM_MB] = iparam[IPARAM_NB];
    if(iparam[IPARAM_MB] < 0) iparam[IPARAM_MB] = -iparam[IPARAM_MB];

    /* No supertiling by default */    
    if(0 == iparam[IPARAM_SNB]) iparam[IPARAM_SNB] = 1;
    if(0 == iparam[IPARAM_SMB]) iparam[IPARAM_SMB] = 1;

    if(verbose) 
    {
        fprintf(stderr, "+++ N x M x K|NRHS      : %d x %d x %d\n",
                        iparam[IPARAM_N], iparam[IPARAM_M], iparam[IPARAM_K]);
    }
    if(verbose > 1)
    {
        if(iparam[IPARAM_LDB] && iparam[IPARAM_LDC])
            fprintf(stderr, "+++ LDA , LDB , LDC     : %d , %d , %d\n", iparam[IPARAM_LDA], iparam[IPARAM_LDB], iparam[IPARAM_LDC]);
        else if(iparam[IPARAM_LDB])
            fprintf(stderr, "+++ LDA , LDB           : %d , %d\n", iparam[IPARAM_LDA], iparam[IPARAM_LDB]);
        else
            fprintf(stderr, "+++ LDA                 : %d\n", iparam[IPARAM_LDA]);
    }
    if(verbose)
    {
        if(iparam[IPARAM_IB] > 0)
            fprintf(stderr, "+++ NB x MB , IB        : %d x %d , %d\n", 
                            iparam[IPARAM_NB], iparam[IPARAM_MB], iparam[IPARAM_IB]);
        else
            fprintf(stderr, "+++ NB x MB             : %d x %d\n", 
                            iparam[IPARAM_NB], iparam[IPARAM_MB]);

        if(iparam[IPARAM_SNB] * iparam[IPARAM_SMB] != 1)
            fprintf(stderr, "+++ SNB x SMB           : %d x %d\n", iparam[IPARAM_SNB], iparam[IPARAM_SMB]);
    }
}

static void iparam_default(int* iparam)
{
    /* Just in case someone forget to add the initialization :) */
    memset(iparam, 0, IPARAM_SIZEOF * sizeof(int)); 
    iparam[IPARAM_NNODES] = 1;
    iparam[IPARAM_NGPUS] = -1;
}

void iparam_default_ibnbmb(int* iparam, int ib, int nb, int mb)
{
    iparam[IPARAM_IB] = ib ? ib : -1;
    iparam[IPARAM_NB] = -nb;
    iparam[IPARAM_MB] = -mb;
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
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &iparam[IPARAM_NNODES]);
    MPI_Comm_rank(MPI_COMM_WORLD, &iparam[IPARAM_RANK]); 
#endif
    parse_arguments(argc, argv, iparam);
    int verbose = iparam[IPARAM_VERBOSE];
    if(iparam[IPARAM_RANK] > 0 && verbose < 4) verbose = 0;
    
    TIME_START();
    dague_context_t* ctx = dague_init(iparam[IPARAM_NCORES], &argc, &argv);
#if defined(DAGUE_CUDA_SUPPORT)
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(0 != dague_gpu_init(&iparam[IPARAM_NGPUS], 0))
        {
            fprintf(stderr, "xxx DAGuE is unable to initialize the CUDA environment.\n");
            exit(3);
        }
    }
#endif
    if(verbose > 2) TIME_PRINT(iparam[IPARAM_RANK], ("DAGuE initialized\n"));
    return ctx;
}

void cleanup_dague(dague_context_t* dague)
{
#ifdef DAGUE_PROFILING
    char* filename = NULL;
#if defined(HAVE_MPI)
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
#ifdef HAVE_MPI
    MPI_Finalize();
#endif    
}

