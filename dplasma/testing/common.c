/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#include "dague_config.h"
#include "dague.h"
#include "dague_hwloc.h"
#include "execution_unit.h"

#include "common.h"
#include "common_timing.h"
#include <plasma.h>

#include <stdlib.h>
#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#if defined(HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(HAVE_GETOPT_H) */
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#if defined(HAVE_CUDA)
#include "gpu_data.h"
#endif

#include "dague_prof_grapher.h"
#include "schedulers.h"
#include "vpmap.h"

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
const int   norms[4] = { PlasmaMaxNorm, PlasmaOneNorm, PlasmaInfNorm, PlasmaFrobeniusNorm };

const char *sidestr[2]  = { "Left ", "Right" };
const char *uplostr[2]  = { "Upper", "Lower" };
const char *diagstr[2]  = { "NonUnit", "Unit   " };
const char *transstr[3] = { "N", "T", "H" };
const char *normsstr[4] = { "Max", "One", "Inf", "Fro" };

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

/**********************************
 * Command line arguments
 **********************************/
void print_usage(void)
{
    fprintf(stderr,
            "Mandatory argument:\n"
            " number            : dimension (N) of the matrices (required)\n"
            "Optional arguments:\n"
            " -p -P --grid-rows : rows (P) in the PxQ process grid   (default: NP)\n"
            " -q -Q --grid-cols : columns (Q) in the PxQ process grid (default: NP/P)\n"
            "\n"
            " -N                : dimension (N) of the matrices (required)\n"
            " -M                : dimension (M) of the matrices (default: N)\n"
            " -K --NRHS C<-A*B+C: dimension (K) of the matrices (default: N)\n"
            "           AX=B    : columns in the right hand side (default: 1)\n"
            " -A --LDA          : leading dimension of the matrix A (default: full)\n"
            " -B --LDB          : leading dimension of the matrix B (default: full)\n"
            " -C --LDC          : leading dimension of the matrix C (default: full)\n"
            " -i --IB           : inner blocking     (default: autotuned)\n"
            " -t --MB           : rows in a tile     (default: autotuned)\n"
            " -T --NB           : columns in a tile  (default: autotuned)\n"
            " -s --SMB          : rows of tiles in a supertile (default: 1)\n"
            " -S --SNB          : columns of tiles in a supertile (default: 1)\n"
            " -x --check        : verify the results\n"
            " -X --check_inv    : verify the results against the inverse\n"
            "\n"
            "    --qr_a         : Size of TS domain. (specific to xgeqrf_param)\n"
            "    --qr_p         : Size of the high level tree. (specific to xgeqrf_param)\n"
            " -d --domino       : Enable/Disable the domino between upper and lower trees. (specific to xgeqrf_param) (default: 1)\n"
            " -r --tsrr         : Enable/Disable the round-robin on TS domain. (specific to xgeqrf_param) (default: 1)\n"
            "    --treel        : Tree used for low level reduction inside nodes. (specific to xgeqrf_param)\n"
            "    --treeh        : Tree used for high level reduction between nodes, only if qr_p > 1. (specific to xgeqrf_param)\n"
            "                      (0: Flat, 1: Greedy, 2: Fibonacci, 3: Binary)\n"

            " -y --butlvl       : Level of the Butterfly (starting from 0).\n"
            "\n"
            " -v --verbose      : extra verbose output\n"
            " -h --help         : this message\n"
            "\n"
            );
    fprintf(stderr,
            "\n"
            " -c --cores        : number of concurent threads (default: number of physical hyper-threads)\n"
            " -g --gpus         : number of GPU (default: 0)\n"
            " -o --scheduler    : select the scheduler (default: LFQ)\n"
            "                     Accepted values:\n"
            "                       LFQ -- Local Flat Queues\n"
            "                       GD  -- Global Dequeue\n"
            "                       LHQ -- Local Hierarchical Queues\n"
            "                       AP  -- Absolute Priorities\n"
            "                       PBQ -- Priority Based Local Flat Queues\n"
            "                       LTQ -- Local Tree Queues\n"
            "\n"
            "    --dot          : create a dot output file (default: don't)\n"
            "\n"
            "    --ht nbth      : enable a SMT/HyperThreadind binding using nbth hyper-thread per core.\n"
            "                     This parameter must be declared before the virtual process distribution parameter\n"
            " -V --vpmap        : select the virtual process map (default: flat map)\n"
            "                     Accepted values:\n"
            "                       flat  -- Flat Map: all cores defined with -c are under the same virtual process\n"
            "                       hwloc -- Hardware Locality based: threads up to -c are created and threads\n"
            "                                bound on cores that are under the same socket are also under the same\n"
            "                                virtual process\n"
            "                       rr:n:p:c -- create n virtual processes per real process, each virtual process with p threads\n"
            "                                   bound in a round-robin fashion on the number of cores c (overloads the -c flag)\n"
            "                       file:filename -- uses filename to load the virtual process map. Each entry details a virtual\n"
            "                                        process mapping using the semantic  [mpi_rank]:nb_thread:binding  with:\n"
            "                                        - mpi_rank : the mpi process rank (empty if not relevant)\n"
            "                                        - nb_thread : the number of threads under the virtual process\n"
            "                                                      (overloads the -c flag)\n"
            "                                        - binding : a set of cores for the thread binding. Accepted values are:\n"
            "                                          -- a core list          (exp: 1,3,5-6)\n"
            "                                          -- a hexadecimal mask   (exp: 0xff012)\n"
            "                                          -- a binding range expression: [start];[end];[step] \n"
            "                                             wich defines a round-robin one thread per core distribution from start\n"
            "                                             (default 0) to end (default physical core number) by step (default 1)\n"
            "\n"
            "\n"
            "ENVIRONMENT\n"
            "  [SDCZ]<FUNCTION> : defines the priority limit of a given function for a given precision\n"
            "\n");
            dague_usage();
}

#define GETOPT_STRING "c:o:g::p:P:q:Q:N:M:K:A:B:C:i:t:T:s:S:xXv::hd:r:y:V:"

#if defined(HAVE_GETOPT_LONG)
static struct option long_options[] =
{
    /* PaRSEC specific options */
    {"cores",       required_argument,  0, 'c'},
    {"c",           required_argument,  0, 'c'},
    {"o",           required_argument,  0, 'o'},
    {"scheduler",   required_argument,  0, 'o'},
    {"gpus",        required_argument,  0, 'g'},
    {"g",           required_argument,  0, 'g'},
    {"V",           required_argument,  0, 'V'},
    {"vpmap",       required_argument,  0, 'V'},
    {"ht",          required_argument,  0, 'H'},
    {"dot",         required_argument,  0, '.'},

    /* Generic Options */
    {"grid-rows",   required_argument,  0, 'p'},
    {"p",           required_argument,  0, 'p'},
    {"P",           required_argument,  0, 'p'},
    {"grid-cols",   required_argument,  0, 'q'},
    {"q",           required_argument,  0, 'q'},
    {"Q",           required_argument,  0, 'q'},

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
    {"MB",          required_argument,  0, 't'},
    {"t",           required_argument,  0, 't'},
    {"NB",          required_argument,  0, 'T'},
    {"T",           required_argument,  0, 'T'},
    {"SMB",         required_argument,  0, 's'},
    {"s",           required_argument,  0, 's'},
    {"SNB",         required_argument,  0, 'S'},
    {"S",           required_argument,  0, 'S'},
    {"check",       no_argument,        0, 'x'},
    {"x",           no_argument,        0, 'x'},
    {"check_inv",   no_argument,        0, 'X'},
    {"X",           no_argument,        0, 'X'},

    /* HQR options */
    {"qr_a",        required_argument,  0, '0'},
    {"qr_p",        required_argument,  0, '1'},
    {"d",           required_argument,  0, 'd'},
    {"domino",      required_argument,  0, 'd'},
    {"r",           required_argument,  0, 'r'},
    {"tsrr",        required_argument,  0, 'r'},
    {"treel",       required_argument,  0, 'l'},
    {"treeh",       required_argument,  0, 'L'},

    /* HERBT options */
    {"butlvl",      required_argument,  0, 'y'},
    {"y",           required_argument,  0, 'y'},

    /* Auxiliary options */
    {"verbose",     optional_argument,  0, 'v'},
    {"v",           optional_argument,  0, 'v'},
    {"help",        no_argument,        0, 'h'},
    {"h",           no_argument,        0, 'h'},
    {0, 0, 0, 0}
};
#endif  /* defined(HAVE_GETOPT_LONG) */

static void parse_arguments(int *_argc, char*** _argv, int* iparam)
{
    int opt = 0;
    int c;
    int argc = *_argc;
    char **argv = *_argv;
    char *add_dot = NULL;

    do {
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
            case 'o':
                fprintf(stderr, "The usage of this option as an argument to the testing is deprecated.\n"
                        "It should be passed directly to PaRSEC instead\n");
                if( !strcmp(optarg, "LFQ") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_LFQ;
                else if( !strcmp(optarg, "LTQ") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_LTQ;
                else if( !strcmp(optarg, "AP") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_AP;
                else if( !strcmp(optarg, "LHQ") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_LHQ;
                else if( !strcmp(optarg, "GD") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_GD;
                else if( !strcmp(optarg, "PBQ") )
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_PBQ;
                else {
                    fprintf(stderr, "#!!!!! malformed scheduler value %s (accepted: LFQ AP LHQ GD PBQ LTQ). Reverting to default LFQ\n",
                            optarg);
                    iparam[IPARAM_SCHEDULER] = DAGUE_SCHEDULER_LFQ;
                }
                break;

            case 'g':
                fprintf(stderr, "The usage of this option as an argument to the testing is deprecated.\n"
                        "It should be passed directly to PaRSEC instead\n");
                if(iparam[IPARAM_NGPUS] == -1) {
                    fprintf(stderr, "#!!!!! This test does not have GPU support. GPU disabled.\n");
                    break;
                }
                if(optarg)  iparam[IPARAM_NGPUS] = atoi(optarg);
                else        iparam[IPARAM_NGPUS] = INT_MAX;
                break;

            case 'p': case 'P': iparam[IPARAM_P] = atoi(optarg); break;
            case 'q': case 'Q': iparam[IPARAM_Q] = atoi(optarg); break;
            case 'N': iparam[IPARAM_N] = atoi(optarg); break;
            case 'M': iparam[IPARAM_M] = atoi(optarg); break;
            case 'K': iparam[IPARAM_K] = atoi(optarg); break;
            case 'A': iparam[IPARAM_LDA] = atoi(optarg); break;
            case 'B': iparam[IPARAM_LDB] = atoi(optarg); break;
            case 'C': iparam[IPARAM_LDC] = atoi(optarg); break;
            case 'i': iparam[IPARAM_IB] = atoi(optarg); break;

            case 't': iparam[IPARAM_MB] = atoi(optarg); break;
            case 'T': iparam[IPARAM_NB] = atoi(optarg); break;
            case 's': iparam[IPARAM_SMB] = atoi(optarg); break;
            case 'S': iparam[IPARAM_SNB] = atoi(optarg); break;

            case 'X': iparam[IPARAM_CHECKINV] = 1;
            case 'x': iparam[IPARAM_CHECK] = 1; iparam[IPARAM_VERBOSE] = max(2, iparam[IPARAM_VERBOSE]); break;

                /* HQR parameters */
            case '0': iparam[IPARAM_QR_TS_SZE]    = atoi(optarg); break;
            case '1': iparam[IPARAM_QR_HLVL_SZE]  = atoi(optarg); break;

            case 'd': iparam[IPARAM_QR_DOMINO]    = atoi(optarg) ? 1 : 0; break;
            case 'r': iparam[IPARAM_QR_TSRR]      = atoi(optarg) ? 1 : 0; break;

            case 'l': iparam[IPARAM_LOWLVL_TREE]  = atoi(optarg); break;
            case 'L': iparam[IPARAM_HIGHLVL_TREE] = atoi(optarg); break;

                /* Butterfly parameters */
            case 'y': iparam[IPARAM_BUT_LEVEL] = atoi(optarg); break;

            case 'v':
                if(optarg)  iparam[IPARAM_VERBOSE] = atoi(optarg);
                else        iparam[IPARAM_VERBOSE] = 2;
                break;

            case 'H':
                fprintf(stderr, "The usage of this option as an argument to the testing is deprecated.\n"
                        "It should be passed directly to PaRSEC instead\n");
                dague_hwloc_allow_ht(strtol(optarg, (char **) NULL, 10)); break;

            case 'V':
                fprintf(stderr, "The usage of this option as an argument to the testing is deprecated.\n"
                        "It should be passed directly to PaRSEC instead\n");
                if( !strncmp(optarg, "display", 7 )) {
                    vpmap_display_map(stderr);
                } else {
                    /* Change the vpmap choice: first cancel the previous one */
                    vpmap_fini();
                    if( !strncmp(optarg, "flat", 4) ) {
                        /* default case (handled in dague_init) */
                    } else if( !strncmp(optarg, "hwloc", 5) ) {
                        vpmap_init_from_hardware_affinity();
                    } else if( !strncmp(optarg, "file:", 5) ) {
                        vpmap_init_from_file(optarg + 5);
                    } else if( !strncmp(optarg, "rr:", 3) ) {
                        int n, p, co;
                        sscanf(optarg, "rr:%d:%d:%d", &n, &p, &co);
                        vpmap_init_from_parameters(n, p, co);
                        iparam[IPARAM_NCORES] = co;
                    } else {
                        fprintf(stderr, "#XXXXX invalid VPMAP choice (-V argument): %s\n", optarg);
                        print_usage();
                        exit(1);
                    }
                }
                break;

            case '.':
                add_dot = optarg;
                break;

            case 'h': print_usage(); exit(0);

            case '?': /* getopt_long already printed an error message. */
                exit(1);
            default:
                break; /* Assume anything else is dague/mpi stuff */
        }
    } while(-1 != c);

    if( NULL != add_dot ) {
        int i, has_dashdash = 0, has_daguedot = 0;
        for(i = 1; i < argc; i++) {
            if( !strcmp( argv[i], "--") ) {
                has_dashdash = 1;
            }
            if( has_dashdash && !strncmp( argv[i], "--dague_dot", 11 ) ) {
                has_daguedot = 1;
                break;
            }
        }
        if( !has_daguedot ) {
            char **tmp;
            int  tmpc;
            if( !has_dashdash ) {
                tmpc = *(_argc) + 2;
                tmp = (char **)malloc((tmpc+1) * sizeof(char*));
                tmp[ tmpc - 2 ] = strdup("--");
            } else {
                tmpc = *(_argc) + 1;
                tmp = (char **)malloc((tmpc+1) * sizeof(char*));
            }
            for(i = 0; i < (*_argc);i++)
                tmp[i] = (*_argv)[i];

            asprintf( &tmp[ tmpc - 1 ], "--dague_dot=%s", add_dot );
            tmp[ tmpc     ] = NULL;

            *_argc = tmpc;
            *_argv = tmp;
        }
    }

    int verbose = iparam[IPARAM_RANK] ? 0 : iparam[IPARAM_VERBOSE];

    if(iparam[IPARAM_NGPUS] < 0) iparam[IPARAM_NGPUS] = 0;

    /* Check the process grid */
    if(0 == iparam[IPARAM_P])
        iparam[IPARAM_P] = iparam[IPARAM_NNODES];
    if(0 == iparam[IPARAM_Q])
        iparam[IPARAM_Q] = iparam[IPARAM_NNODES] / iparam[IPARAM_P];
    int pqnp = iparam[IPARAM_Q] * iparam[IPARAM_P];
    if(pqnp > iparam[IPARAM_NNODES])
    {
        fprintf(stderr, "#XXXXX the process grid PxQ (%dx%d) is larger than the number of nodes (%d)!\n", iparam[IPARAM_P], iparam[IPARAM_Q], iparam[IPARAM_NNODES]);
        exit(2);
    }
    if(verbose && (pqnp < iparam[IPARAM_NNODES]))
    {
        fprintf(stderr, "#!!!!! the process grid PxQ (%dx%d) is smaller than the number of nodes (%d). Some nodes are idling!\n", iparam[IPARAM_P], iparam[IPARAM_Q], iparam[IPARAM_NNODES]);
    }

    /* Set matrices dimensions to default values if not provided */
    /* Search for N as a bare number if not provided by -N */
    while(0 == iparam[IPARAM_N])
    {
        if(optind < argc)
        {
            iparam[IPARAM_N] = atoi(argv[optind++]);
            continue;
        }
        fprintf(stderr, "#XXXXX the matrix size (N) is not set!\n");
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
    if(iparam[IPARAM_NB] <= 0 && iparam[IPARAM_MB] > 0) iparam[IPARAM_NB] = iparam[IPARAM_MB];
    if(iparam[IPARAM_MB] <= 0 && iparam[IPARAM_NB] > 0) iparam[IPARAM_MB] = iparam[IPARAM_NB];
    if(iparam[IPARAM_MB] < 0) iparam[IPARAM_MB] = -iparam[IPARAM_MB];
    if(iparam[IPARAM_NB] < 0) iparam[IPARAM_NB] = -iparam[IPARAM_NB];

    /* No supertiling by default */
    if(-'p' == iparam[IPARAM_SMB]) iparam[IPARAM_SMB] = (iparam[IPARAM_M]/iparam[IPARAM_MB])/iparam[IPARAM_P];
    if(-'q' == iparam[IPARAM_SNB]) iparam[IPARAM_SNB] = (iparam[IPARAM_N]/iparam[IPARAM_NB])/iparam[IPARAM_Q];
    if(0 == iparam[IPARAM_SMB]) iparam[IPARAM_SMB] = 1;
    if(0 == iparam[IPARAM_SNB]) iparam[IPARAM_SNB] = 1;
}

static void print_arguments(int* iparam)
{
    int verbose = iparam[IPARAM_RANK] ? 0 : iparam[IPARAM_VERBOSE];

    if(verbose)
        fprintf(stderr, "#+++++ cores detected       : %d\n", iparam[IPARAM_NCORES]);

    if(verbose > 1) fprintf(stderr, "#+++++ nodes x cores + gpu  : %d x %d + %d (%d+%d)\n"
                                    "#+++++ P x Q                : %d x %d (%d/%d)\n",
                            iparam[IPARAM_NNODES],
                            iparam[IPARAM_NCORES],
                            iparam[IPARAM_NGPUS],
                            iparam[IPARAM_NNODES] * iparam[IPARAM_NCORES],
                            iparam[IPARAM_NNODES] * iparam[IPARAM_NGPUS],
                            iparam[IPARAM_P], iparam[IPARAM_Q],
                            iparam[IPARAM_Q] * iparam[IPARAM_P], iparam[IPARAM_NNODES]);

    if(verbose)
    {
        fprintf(stderr, "#+++++ M x N x K|NRHS       : %d x %d x %d\n",
                iparam[IPARAM_M], iparam[IPARAM_N], iparam[IPARAM_K]);
    }

    if(verbose > 2)
    {
        if(iparam[IPARAM_LDB] && iparam[IPARAM_LDC])
            fprintf(stderr, "#+++++ LDA , LDB , LDC      : %d , %d , %d\n", iparam[IPARAM_LDA], iparam[IPARAM_LDB], iparam[IPARAM_LDC]);
        else if(iparam[IPARAM_LDB])
            fprintf(stderr, "#+++++ LDA , LDB            : %d , %d\n", iparam[IPARAM_LDA], iparam[IPARAM_LDB]);
        else
            fprintf(stderr, "#+++++ LDA                  : %d\n", iparam[IPARAM_LDA]);
    }

    if(verbose)
    {
        if(iparam[IPARAM_IB] > 0)
            fprintf(stderr, "#+++++ MB x NB , IB         : %d x %d , %d\n",
                            iparam[IPARAM_MB], iparam[IPARAM_NB], iparam[IPARAM_IB]);
        else
            fprintf(stderr, "#+++++ MB x NB              : %d x %d\n",
                    iparam[IPARAM_MB], iparam[IPARAM_NB]);
        if(iparam[IPARAM_SNB] * iparam[IPARAM_SMB] != 1)
            fprintf(stderr, "#+++++ SMB x SNB            : %d x %d\n", iparam[IPARAM_SMB], iparam[IPARAM_SNB]);
    }
}




static void iparam_default(int* iparam)
{
    /* Just in case someone forget to add the initialization :) */
    memset(iparam, 0, IPARAM_SIZEOF * sizeof(int));
    iparam[IPARAM_NNODES] = 1;
    iparam[IPARAM_NGPUS] = -1;
    iparam[IPARAM_QR_DOMINO] = 1;
    iparam[IPARAM_QR_TSRR] = 1;
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
    iparam[IPARAM_SMB] = -'p';
    iparam[IPARAM_SNB] = -'q';
}

#ifdef DAGUE_PROF_TRACE
static char* argvzero;
#endif

dague_context_t* setup_dague(int argc, char **argv, int *iparam)
{
#ifdef DAGUE_PROF_TRACE
    argvzero = argv[0];
#endif
#ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &iparam[IPARAM_NNODES]);
    MPI_Comm_rank(MPI_COMM_WORLD, &iparam[IPARAM_RANK]);
#else
    iparam[IPARAM_NNODES] = 1;
    iparam[IPARAM_RANK] = 0;
#endif
    parse_arguments(&argc, &argv, iparam);
    int verbose = iparam[IPARAM_VERBOSE];
    if(iparam[IPARAM_RANK] > 0 && verbose < 4) verbose = 0;

    TIME_START();
    dague_context_t* ctx = dague_init(iparam[IPARAM_NCORES], &argc, &argv);
    /* If the number of cores has not been defined as a parameter earlier
     update it with the default parameter computed in dague_init. */
    if(iparam[IPARAM_NCORES] <= 0)
    {
        int p, nb_total_comp_threads = 0;
        for(p = 0; p < ctx->nb_vp; p++) {
            nb_total_comp_threads += ctx->virtual_processes[p]->nb_cores;
        }
        iparam[IPARAM_NCORES] = nb_total_comp_threads;
    }
    print_arguments(iparam);

#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(0 != dague_gpu_init(ctx, &iparam[IPARAM_NGPUS], verbose>3))
        {
            fprintf(stderr, "#XXXXX DAGuE is unable to initialize the CUDA environment.\n");
            exit(3);
        }
    }
#endif

    dague_set_scheduler( ctx, dague_schedulers_array[ iparam[IPARAM_SCHEDULER] ] );

    if(verbose > 2) TIME_PRINT(iparam[IPARAM_RANK], ("DAGuE initialized\n"));
    return ctx;
}

void cleanup_dague(dague_context_t* dague, int *iparam)
{
#ifdef DAGUE_PROF_TRACE
    char* filename = NULL;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    asprintf(&filename, "%s.%d.profile", argvzero, rank);
#else
    asprintf(&filename, "%s.profile", argvzero);
#endif
    dague_profiling_dump_dbp(filename);
    free(filename);
#endif  /* DAGUE_PROF_TRACE */
#if defined(HAVE_CUDA)
    if( iparam[IPARAM_NGPUS] > 0 ) {
        if( 0 != dague_gpu_fini() ) {
            fprintf(stderr, "#XXXXX DAGuE is unable to finalize the CUDA environment.\n");
        }
    }
#endif  /* defined(HAVE_CUDA) */
    dague_fini(&dague);

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    (void)iparam;
}
