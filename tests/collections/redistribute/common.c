/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/execution_stream.h"
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>

#if defined(PARSEC_HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(PARSEC_HAVE_GETOPT_H) */

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

const char *redistribute_distribution_name(int distribution)
{
    switch(distribution) {
    case REDISTRIBUTE_DIST_2DBC:
        return "2dbc";
    case REDISTRIBUTE_DIST_SBC:
        return "sbc";
    default:
        return "unknown";
    }
}

const char *redistribute_memory_location_name(int memory_location)
{
    switch(memory_location) {
    case REDISTRIBUTE_MEMORY_HOST:
        return "cpu";
    case REDISTRIBUTE_MEMORY_MANAGED:
        return "managed";
    case REDISTRIBUTE_MEMORY_DEVICE:
        return "cuda";
    default:
        return "unknown";
    }
}

static int parse_distribution_arg(const char *arg)
{
    if( 0 == strcasecmp(arg, "2dbc") ) {
        return REDISTRIBUTE_DIST_2DBC;
    }
    if( (0 == strcasecmp(arg, "sbc")) ||
        (0 == strcasecmp(arg, "sdb")) ) {
        return REDISTRIBUTE_DIST_SBC;
    }

    fprintf(stderr, "#XXXXX unsupported distribution '%s' (expected 2dbc, sbc, or sdb)\n", arg);
    exit(2);
    return REDISTRIBUTE_DIST_2DBC;
}

static int parse_memory_location_arg(const char *arg)
{
    if( (0 == strcasecmp(arg, "cpu")) ||
        (0 == strcasecmp(arg, "host")) ) {
        return REDISTRIBUTE_MEMORY_HOST;
    }
    if( 0 == strcasecmp(arg, "managed") ) {
        return REDISTRIBUTE_MEMORY_MANAGED;
    }
    if( (0 == strcasecmp(arg, "cuda")) ||
        (0 == strcasecmp(arg, "device")) ) {
        return REDISTRIBUTE_MEMORY_DEVICE;
    }

    fprintf(stderr, "#XXXXX unsupported memory location '%s' (expected cpu, managed, or cuda)\n", arg);
    exit(2);
    return REDISTRIBUTE_MEMORY_HOST;
}

/**********************************
 * Command line arguments
 **********************************/
void print_usage(void)
{
    fprintf(stderr,
            "Optional arguments:\n"
            "\n Source Matrix:\n" 
            " -P --source-grid-rows : rows (P) in the PxQ process grid (default: NP)\n"
            " -Q --source-grid-cols : columns (Q) in the PxQ process grid (default: NP/P)\n"
            " -g --source-distribution : source distribution, 2dbc, sbc, or sdb (default: 2dbc)\n"
            " -l --source-memory    : source memory location, cpu, managed, or cuda (default: cpu)\n"
            " -M                    : dimension (M) of the matrices (default: N)\n"
            " -N                    : dimension (N) of the matrices (required)\n"
            " -t --MB               : rows in a tile     (default: autotuned)\n"
            " -T --NB               : columns in a tile  (default: autotuned)\n"
            " -s --SMB              : rows of tiles in a supertile (default: 1)\n"
            " -S --SNB              : columns of tiles in a supertile (default: 1)\n"
            " -I                    : set row displacement\n"
            " -J                    : set column displacement\n"
            "\n Target/Redistributed Matrix:\n" 
            " -p --target-grid-rows : rows (p) in the pxq process grid (default: NP)\n"
            " -q --target-grid-cols : columns (q) in the pxq process grid (default: NP/p)\n"
            " -G --target-distribution : target distribution, 2dbc, sbc, or sdb (default: 2dbc)\n"
            " -L --target-memory    : target memory location, cpu, managed, or cuda (default: cpu)\n"
            " -a --MR               : set redistributed M size\n"
            " -A --NR               : set redistributed N size\n"
            " -b --MBR              : set redistributed MB size\n"
            " -B --NBR              : set redistributed NB size\n"
            " -d --SMBR             : rows of tiles in a supertile (default: 1)\n"
            " -D --SNBR             : columns of tiles in a supertile (default: 1)\n"
            " -i                    : set redistributed row displacement\n"
            " -j                    : set redistributed column displacement\n"
            "\n Matrix Common:\n" 
            " -R --radius           : set radius of ghost region\n"
            " -m --submatrix-rows   : set row size of submatrix to be redistributed\n"
            " -n --submatrix-cols   : set column size of submatrix to be redistributed\n"
            "\n Others:\n" 
            " -x --check            : verify the results\n"
            " -u --network-bandwidth: bandwidth of network, bits\n"
            " -U --memcpy-bandwidth : bandwidth of memcpy, bits\n"
            " -v --verbose          : extra verbose output\n"
            " -h --help             : this message\n"
            " -z --time             : get run time\n"
            " -e --num-runs         : number of runs\n"
            " -f --thread_multiple  : 0/default, serialized test runtime; others, multiple-thread test runtime\n"
            " -y --no-optimization  : no_optimization version, send the whole tile to target; default 0, not no_optimization version\n"
            " -c --cores            : number of concurrent threads (default: number of physical hyper-threads)\n"
            "\n Notes:\n"
            "    SBC stores a triangular tile set; the redistributed rectangle must fit entirely\n"
            "    within the stored upper or lower tile triangle of each SBC descriptor.\n"
            " -- -flag              : use parsec 'flag', details -- --help\n"
            "\n");
}

#define GETOPT_STRING "c:P:Q:g:l:M:N:t:T:s:S:I:J:p:q:G:L:a:A:b:B:d:D:i:j:R:m:n:x:v:h:z:u:U:y:e:f:"

#if defined(PARSEC_HAVE_GETOPT_LONG)
static struct option long_options[] =
{

    /* Source Matrix */
    {"source-grid-rows",   required_argument,  0, 'P'},
    {"P",                  required_argument,  0, 'P'},
    {"source-grid-cols",   required_argument,  0, 'Q'},
    {"Q",                  required_argument,  0, 'Q'},
    {"source-distribution",required_argument,  0, 'g'},
    {"source-dist",        required_argument,  0, 'g'},
    {"g",                  required_argument,  0, 'g'},
    {"source-memory",      required_argument,  0, 'l'},
    {"source-memory-location", required_argument,  0, 'l'},
    {"l",                  required_argument,  0, 'l'},
    {"N",                  required_argument,  0, 'N'},
    {"M",                  required_argument,  0, 'M'},
    {"MB",                 required_argument,  0, 't'},
    {"t",                  required_argument,  0, 't'},
    {"NB",                 required_argument,  0, 'T'},
    {"T",                  required_argument,  0, 'T'},
    {"SMB",                required_argument,  0, 's'},
    {"s",                  required_argument,  0, 's'},
    {"SNB",                required_argument,  0, 'S'},
    {"S",                  required_argument,  0, 'S'},
    {"I",                  required_argument,  0, 'I'},
    {"J",                  required_argument,  0, 'J'},

    /* Target/Redistributed Matrix */
    {"target-grid-rows",   required_argument,  0, 'p'},
    {"p",                  required_argument,  0, 'p'},
    {"target-grid-cols",   required_argument,  0, 'q'},
    {"q",                  required_argument,  0, 'q'},
    {"target-distribution",required_argument,  0, 'G'},
    {"target-dist",        required_argument,  0, 'G'},
    {"G",                  required_argument,  0, 'G'},
    {"target-memory",      required_argument,  0, 'L'},
    {"target-memory-location", required_argument,  0, 'L'},
    {"L",                  required_argument,  0, 'L'},
    {"MR",                 required_argument,  0, 'a'},
    {"a",                  required_argument,  0, 'a'},
    {"NR",                 required_argument,  0, 'A'},
    {"A",                  required_argument,  0, 'A'},
    {"MBR",                required_argument,  0, 'b'},
    {"b",                  required_argument,  0, 'b'},
    {"NBR",                required_argument,  0, 'B'},
    {"B",                  required_argument,  0, 'B'},
    {"SMBR",               required_argument,  0, 'd'},
    {"d",                  required_argument,  0, 'd'},
    {"SNBR",               required_argument,  0, 'D'},
    {"D",                  required_argument,  0, 'D'},
    {"i",                  required_argument,  0, 'i'},
    {"j",                  required_argument,  0, 'j'},

    /* Radius */
    {"R",                  required_argument,  0, 'R'},
    {"radius",             required_argument,  0, 'R'},

    /* Submatrix to be redistributed */
    {"submatrix-rows",     required_argument,  0, 'm'},
    {"m",                  required_argument,  0, 'm'},
    {"submatrix-cols",     required_argument,  0, 'n'},
    {"n",                  required_argument,  0, 'n'},

    /* Check result */
    {"check",              no_argument,        0, 'x'},
    {"x",                  no_argument,        0, 'x'},

    /* Bandwidth of network */
    {"network-bandwidth",  required_argument,  0, 'u'},
    {"u",                  required_argument,  0, 'u'},

    /* Bandwidth of memcpy */
    {"memcpy-bandwidth",   required_argument,  0, 'U'},
    {"U",                  required_argument,  0, 'U'},

    /* Auxiliary options */
    {"verbose",            optional_argument,  0, 'v'},
    {"v",                  optional_argument,  0, 'v'},
    {"help",               no_argument,        0, 'h'},
    {"h",                  no_argument,        0, 'h'},

    /* Number of runs */ 
    {"num-runs",           required_argument,  0, 'e'},
    {"e",                  required_argument,  0, 'e'},

    /* MPI thread init type */
    {"thread-multiple",    required_argument,  0, 'f'},
    {"f",                  required_argument,  0, 'f'},

    /* no_optimization version */ 
    {"no-optimization",    required_argument,  0, 'y'},
    {"y",                  required_argument,  0, 'y'},
     
    /* Time */
    {"time",               no_argument,        0, 'z'},
    {"z",                  no_argument,        0, 'z'},

    /* PaRSEC specific options */
    {"cores",              required_argument,  0, 'c'},
    {"c",                  required_argument,  0, 'c'},

    {0, 0, 0, 0}
};
#endif  /* defined(PARSEC_HAVE_GETOPT_LONG) */

static void parse_arguments(int *_argc, char*** _argv, int* iparam, double *dparam)
{
    int opt = 0;
    int c;
    int argc = *_argc;
    char **argv = *_argv;

    /* Default */
    iparam[IPARAM_CHECK] = 0; 
    iparam[IPARAM_GETTIME] = 0; 
    iparam[IPARAM_VERBOSE] = 0; 
    iparam[IPARAM_NCORES] = -1;
    iparam[IPARAM_RADIUS] = 0;
    iparam[IPARAM_SOURCE_DIST] = REDISTRIBUTE_DIST_2DBC;
    iparam[IPARAM_TARGET_DIST] = REDISTRIBUTE_DIST_2DBC;
    iparam[IPARAM_SOURCE_MEMORY] = REDISTRIBUTE_MEMORY_HOST;
    iparam[IPARAM_TARGET_MEMORY] = REDISTRIBUTE_MEMORY_HOST;

    /* No supertiling by default */
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_SMB_R] = 1;
    iparam[IPARAM_SNB_R] = 1;

    /* Default number of runs: 1 */
    iparam[IPARAM_NUM_RUNS] = 1;

    /* Default to serialized access to the selected test runtime. */
    iparam[IPARAM_THREAD_MULTIPLE] = 0;

    /* Default Not no_optimization version */ 
    iparam[IPARAM_NO_OPTIMIZATION_VERSION] = 0;

    /* Bandwidth 0 by default */
    dparam[DPARAM_NETWORK_BANDWIDTH] = 0.0;
    dparam[DPARAM_MEMCPY_BANDWIDTH] = 0.0;

    do {
#if defined(PARSEC_HAVE_GETOPT_LONG)
        c = getopt_long_only(argc, argv, "",
                        long_options, &opt);
#else
        c = getopt(argc, argv, GETOPT_STRING);
        (void) opt;
#endif  /* defined(PARSEC_HAVE_GETOPT_LONG) */

        switch(c)
        {
            /* Source */
            case 'P': iparam[IPARAM_P] = atoi(optarg); break;
            case 'Q': iparam[IPARAM_Q] = atoi(optarg); break;
            case 'g': iparam[IPARAM_SOURCE_DIST] = parse_distribution_arg(optarg); break;
            case 'l': iparam[IPARAM_SOURCE_MEMORY] = parse_memory_location_arg(optarg); break;
            case 'M': iparam[IPARAM_M] = atoi(optarg); break;
            case 'N': iparam[IPARAM_N] = atoi(optarg); break;
            case 't': iparam[IPARAM_MB] = atoi(optarg); break;
            case 'T': iparam[IPARAM_NB] = atoi(optarg); break;
            case 's': iparam[IPARAM_SMB] = atoi(optarg); break;
            case 'S': iparam[IPARAM_SNB] = atoi(optarg); break;
            case 'I': iparam[IPARAM_DISI] = atoi(optarg); break;
            case 'J': iparam[IPARAM_DISJ] = atoi(optarg); break;

            /* Target/redistribute */
            case 'p': iparam[IPARAM_P_R] = atoi(optarg); break;
            case 'q': iparam[IPARAM_Q_R] = atoi(optarg); break;
            case 'G': iparam[IPARAM_TARGET_DIST] = parse_distribution_arg(optarg); break;
            case 'L': iparam[IPARAM_TARGET_MEMORY] = parse_memory_location_arg(optarg); break;
            case 'a': iparam[IPARAM_M_R] = atoi(optarg); break;
            case 'A': iparam[IPARAM_N_R] = atoi(optarg); break;
            case 'b': iparam[IPARAM_MB_R] = atoi(optarg); break;
            case 'B': iparam[IPARAM_NB_R] = atoi(optarg); break;
            case 'd': iparam[IPARAM_SMB_R] = atoi(optarg); break;
            case 'D': iparam[IPARAM_SNB_R] = atoi(optarg); break;
            case 'i': iparam[IPARAM_DISI_R] = atoi(optarg); break;
            case 'j': iparam[IPARAM_DISJ_R] = atoi(optarg); break;

            /* Common */
            case 'R': iparam[IPARAM_RADIUS] = atoi(optarg); break;
            case 'm': iparam[IPARAM_M_SUB] = atoi(optarg); break;
            case 'n': iparam[IPARAM_N_SUB] = atoi(optarg); break;

            /* Others */
            case 'x': iparam[IPARAM_CHECK] = 1; break;
            case 'u': dparam[DPARAM_NETWORK_BANDWIDTH] = atof(optarg); break;
            case 'U': dparam[DPARAM_MEMCPY_BANDWIDTH] = atof(optarg); break;
            case 'v':
                if(optarg)  iparam[IPARAM_VERBOSE] = atoi(optarg);
                else        iparam[IPARAM_VERBOSE] = 2;
                break;
            case 'h': print_usage(); exit(0);
                break;
            case 'z': iparam[IPARAM_GETTIME] = 1; break;
            case 'e': iparam[IPARAM_NUM_RUNS] = atoi(optarg); break;
            case 'f': iparam[IPARAM_THREAD_MULTIPLE] = atoi(optarg); break;
            case 'y': iparam[IPARAM_NO_OPTIMIZATION_VERSION] = atoi(optarg); break;
            case 'c': iparam[IPARAM_NCORES] = atoi(optarg); break;

            case '?': /* getopt_long already printed an error message. */
                exit(1);
                break;

            default:
                break; /* Assume anything else is parsec/mpi stuff */
        }
    } while(-1 != c);

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

    /* Search for NR as a bare number if not provided by -NR */
    while(0 == iparam[IPARAM_N_R])
    {
        if(optind < argc)
        {
            iparam[IPARAM_N_R] = atoi(argv[optind++]);
            continue;
        }
        fprintf(stderr, "#XXXXX the redistributed matrix size (NR) is not set!\n");
        exit(2);
    }

    /* Set no defaults for NB, MB, the algorithm have to do it */
    if(iparam[IPARAM_NB] <= 0 && iparam[IPARAM_MB] > 0) iparam[IPARAM_NB] = iparam[IPARAM_MB];
    if(iparam[IPARAM_MB] <= 0 && iparam[IPARAM_NB] > 0) iparam[IPARAM_MB] = iparam[IPARAM_NB];
    if(iparam[IPARAM_MB] < 0) iparam[IPARAM_MB] = -iparam[IPARAM_MB];
    if(iparam[IPARAM_NB] < 0) iparam[IPARAM_NB] = -iparam[IPARAM_NB];

    /* Set no defaults for NBR, MBR, the algorithm have to do it */
    if(iparam[IPARAM_NB_R] <= 0 && iparam[IPARAM_MB_R] > 0) iparam[IPARAM_NB_R] = iparam[IPARAM_MB_R];
    if(iparam[IPARAM_MB_R] <= 0 && iparam[IPARAM_NB_R] > 0) iparam[IPARAM_MB_R] = iparam[IPARAM_NB_R];
    if(iparam[IPARAM_MB_R] < 0) iparam[IPARAM_MB_R] = -iparam[IPARAM_MB_R];
    if(iparam[IPARAM_NB_R] < 0) iparam[IPARAM_NB_R] = -iparam[IPARAM_NB_R];

    if( iparam[IPARAM_MB] <= 0 || iparam[IPARAM_NB] <= 0 ){ 
        fprintf(stderr, "#XXXXX Source tile size (NB) is negative!\n");
        exit(2);
    }

    if( iparam[IPARAM_MB_R] <= 0 || iparam[IPARAM_NB_R] <= 0 ){   
        fprintf(stderr, "#XXXXX Target/Redistributed tile size (NBR) is negative!\n");
        exit(2);
    }

    if( iparam[IPARAM_M_SUB] <= 0 || iparam[IPARAM_N_SUB] <= 0 ){
        fprintf(stderr, "#XXXXX Redistributed submatrix size is not positive!\n");
        exit(2);
    }
}

static void print_arguments(int* iparam)
{
    int verbose = iparam[IPARAM_RANK] ? 0 : iparam[IPARAM_VERBOSE];

    if(verbose)
    {
        fprintf(stderr, "#++++++++++++++++++++ Source +++++++++++++++++++++\n");

        fprintf(stderr, "#+++++ P x Q :                   %d x %d\n",
                        iparam[IPARAM_P], iparam[IPARAM_Q]);

        fprintf(stderr, "#+++++ distribution :            %s\n",
                        redistribute_distribution_name(iparam[IPARAM_SOURCE_DIST]));

        fprintf(stderr, "#+++++ memory location :         %s\n",
                        redistribute_memory_location_name(iparam[IPARAM_SOURCE_MEMORY]));

        if(iparam[IPARAM_SNB] * iparam[IPARAM_SMB] != 1)
            fprintf(stderr, "#+++++ SMB x SNB :               %d x %d\n", 
                             iparam[IPARAM_SMB], iparam[IPARAM_SNB]);

        fprintf(stderr, "#+++++ M x N :                   %d x %d\n",
                iparam[IPARAM_M], iparam[IPARAM_N]);

        fprintf(stderr, "#+++++ MB x NB :                 %d x %d\n",
                         iparam[IPARAM_MB], iparam[IPARAM_NB]);

        fprintf(stderr, "#+++++ disI, disJ :              %d , %d\n",
                         iparam[IPARAM_DISI], iparam[IPARAM_DISJ]);

        fprintf(stderr, "#++++++++++++++++++++ Target +++++++++++++++++++++\n");

        fprintf(stderr, "#+++++ PR x QR :                 %d x %d\n",
                        iparam[IPARAM_P_R], iparam[IPARAM_Q_R]);

        fprintf(stderr, "#+++++ distribution :            %s\n",
                        redistribute_distribution_name(iparam[IPARAM_TARGET_DIST]));

        fprintf(stderr, "#+++++ memory location :         %s\n",
                        redistribute_memory_location_name(iparam[IPARAM_TARGET_MEMORY]));

        fprintf(stderr, "#+++++ MR x NR :                 %d x %d\n", 
                         iparam[IPARAM_M_R], iparam[IPARAM_N_R]);

        if(iparam[IPARAM_SNB_R] * iparam[IPARAM_SMB_R] != 1)
            fprintf(stderr, "#+++++ SMBR x SNBR :             %d x %d\n",
                             iparam[IPARAM_SMB_R], iparam[IPARAM_SNB_R]);

        fprintf(stderr, "#+++++ MBR x NBR:                %d x %d\n", 
                         iparam[IPARAM_MB_R], iparam[IPARAM_NB_R]);

        fprintf(stderr, "#+++++ disi , disj :             %d , %d \n",
                         iparam[IPARAM_DISI_R], iparam[IPARAM_DISJ_R]);

        fprintf(stderr, "#++++++++++++++++++++ Common +++++++++++++++++++++\n");

        fprintf(stderr, "#+++++ nodes x cores :           %d x %d\n",
                        iparam[IPARAM_NNODES], iparam[IPARAM_NCORES]);

        fprintf(stderr, "#+++++ sub_m x sub_n :           %d x %d\n",
                        iparam[IPARAM_M_SUB], iparam[IPARAM_N_SUB]);

        fprintf(stderr, "#+++++ Radius :                  %d \n",
                         iparam[IPARAM_RADIUS]);
    }
}

parsec_context_t* setup_parsec(int argc, char **argv, int *iparam, double *dparam)
{
    parsec_context_t* ctx = NULL;
    int rc;

    parse_arguments(&argc, &argv, iparam, dparam);

    /* Once we got out arguments, we should pass whatever is left down */
    int parsec_argc = argc - optind;
    char** parsec_argv = argv + optind;
    rc = parsec_tests_context_init(iparam[IPARAM_NCORES],
                                   iparam[IPARAM_THREAD_MULTIPLE] ?
                                       PARSEC_TEST_THREAD_MULTIPLE :
                                       PARSEC_TEST_THREAD_SERIALIZED,
                                   &parsec_argc, &parsec_argv,
                                   &ctx,
                                   &iparam[IPARAM_RANK],
                                   &iparam[IPARAM_NNODES]);
    if( PARSEC_SUCCESS != rc ) {
        fprintf(stderr, "parsec_tests_context_init failed: %d\n", rc);
        exit(-1);
    }

    int verbose = iparam[IPARAM_VERBOSE];
    if(iparam[IPARAM_RANK] > 0 && verbose < 4) verbose = 0;

    SYNC_TIME_START(ctx);

    /* If the number of cores has not been defined as a parameter earlier
     update it with the default parameter computed in parsec_init. */
    if(iparam[IPARAM_NCORES] <= 0)
    {
        int p, nb_total_comp_threads = 0;
        for(p = 0; p < ctx->nb_vp; p++) {
            nb_total_comp_threads += ctx->virtual_processes[p]->nb_cores;
        }
        iparam[IPARAM_NCORES] = nb_total_comp_threads;
    }
    print_arguments(iparam);

    if(verbose > 2) SYNC_TIME_PRINT(ctx, iparam[IPARAM_RANK], ("PaRSEC initialized\n"));
    return ctx;
}

void cleanup_parsec(parsec_context_t* parsec, int *iparam, double *dparam)
{
    int rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
    (void)iparam;
    (void)dparam;
}
