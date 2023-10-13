/*
 * Copyright (c) 2019-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "stencil_internal.h"
#include "tests/tests_timing.h"

/* Timming */
double sync_time_elapsed = 0.0;

/* Global array of weight */
DTYPE * weight_1D;

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, nodes, ch;
    int pargc = 0;
    char **pargv;
    double gflops, flops;
    int i, jj;

    /* Default */
    int m = 0;
    int M = 8;
    int N = 8;
    int MB = 4;
    int NB = 4;
    int P = 1;
    int KP = 1;
    int KQ = 1;
    int cores = -1;
    int iter = 10;
    int R = 1;

    while ((ch = getopt(argc, argv, "m:M:N:t:T:s:S:P:Q:c:I:R:h:")) != -1) {
        switch (ch) {
            case 'm': m = atoi(optarg); break;
            case 'M': M = atoi(optarg); break;
            case 'N': N = atoi(optarg); break;
            case 't': MB = atoi(optarg); break;
            case 'T': NB = atoi(optarg); break;
            case 's': KP = atoi(optarg); break;
            case 'S': KQ = atoi(optarg); break;
            case 'P': P = atoi(optarg); break;
            case 'c': cores = atoi(optarg); break;
            case 'I': iter = atoi(optarg); break;
            case 'R': R = atoi(optarg); break;
            case '?': case 'h': default:
                fprintf(stderr,
                        "-m : initialize MPI_THREAD_MULTIPLE (default: 0/no)\n"
                        "-M : row dimension (M) of the matrices (default: 8)\n"
                        "-N : column dimension (N) of the matrices (default: 8)\n"
                        "-t : row dimension (MB) of the tiles (default: 4)\n"
                        "-T : column dimension (NB) of the tiles (default: 4)\n"
                        "-s : rows of tiles in a k-cyclic distribution (default: 1)\n"
                        "-S : columns of tiles in a k-cyclic distribution (default: 1)\n"
                        "-P : rows (P) in the PxQ process grid (default: 1)\n"
                        "-c : number of cores used (default: -1/all cores)\n"
                        "-I : iterations (default: 10)\n"
                        "-R : radius (default: 1)\n"
                        "\n");
                 exit(1);
        }
    }

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        int requested = m? MPI_THREAD_MULTIPLE: MPI_THREAD_SERIALIZED;
        MPI_Init_thread(&argc, &argv, requested, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nodes = 1;
    rank = 0;
#endif

    pargc = 0; pargv = NULL;
    for(i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }

    if(0) {
        volatile int loop = 1;
        fprintf(stderr, "gdb -p %d\n", getpid());
        while(loop)
            sleep(1);
    }

    /* Initialize PaRSEC */
    parsec = parsec_init(cores, &pargc, &pargv);

    if( NULL == parsec ) {
        /* Failed to correctly initialize. In a correct scenario report
         * upstream, but in this particular case bail out.
         */
        exit(-1);
    }

    /* If the number of cores has not been defined as a parameter earlier
     * update it with the default parameter computed in parsec_init. */
    if(cores <= 0)
    {
        int p, nb_total_comp_threads = 0;
        for(p = 0; p < parsec->nb_vp; p++) {
            nb_total_comp_threads += parsec->virtual_processes[p]->nb_cores;
        }
        cores = nb_total_comp_threads;
    }

    /* Make sure valid parameters are passed */
    if( M < 1 || N < 1 || MB < 1 || NB < 1 || P < 1 || KP < 1 || KQ < 1 || iter < 1 || R < 1 ) {
        if( 0 == rank ) {
            fprintf(stderr, "Wrong value is passed !!! -h for help\n");
            fprintf(stderr, "M %d N %d MB %d NB %d P %d KP %d KQ %d cores %d iteration %d R %d m %d\n",
                    M, N, MB, NB, P, KP, KQ, cores, iter, R, m);
        }
        exit(1); 
    }

    /* Used for ghost region */
    int NNB = (int)(ceil((double)N/NB));

    /* No. of buffers */
    int MMB = (int)(ceil((double)M/MB));

    /* Make sure at least two buffers are used, otherwise result is not correct */
    if( MMB < 2 ) {
        if( 0 == rank )
            fprintf(stderr, "At least two buffers is needed, which is ceil(M/MB)= %d M= %d MB= %d\n", MMB, M, MB);
        exit(1);
    }

    /* Flops */
    flops = FLOPS_STENCIL_1D(N*MB);

    /* initializing matrix structure */
    /* Y */
    parsec_matrix_block_cyclic_t dcA;
    parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                rank, MB, NB+2*R, M, N+2*R*NNB, 0, 0,
                                M, N+2*R*NNB,
                                P, nodes/P, KP, KQ, 0, 0);
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                   (size_t)dcA.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "dcA");

    /*
     * Init dcA (not including ghost region) to i*1.0+j*1.0
     * Init ghost region to 0.0
     */
    int *op_args = (int *)malloc(sizeof(int));
    *op_args = R;
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)stencil_1D_init_ops, op_args);

    /* intialize weight_1D */
    weight_1D = (DTYPE *)malloc(sizeof(DTYPE) * (2*R+1));

    for(jj = 1; jj <= R; jj++) {
        WEIGHT_1D(jj) = (DTYPE)(1.0/(2.0*jj*R));
        WEIGHT_1D(-jj) = -(DTYPE)(1.0/(2.0*jj*R));
    }
    WEIGHT_1D(0) = (DTYPE)1.0;

    /* Generete LOOPGEN Kernel */
#if LOOPGEN
    if( 0 == rank ){
        char command[50];
        snprintf(command, sizeof(command), "./loop_gen_1D %d", R);
        int err = system(command);

        if( err ){
            fprintf(stderr, "loog_gen_1D failed: %s\n", command);
            return(PARSEC_ERROR);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Stencil_1D */
    SYNC_TIME_START();
    parsec_stencil_1D(parsec, (parsec_tiled_matrix_t *)&dcA, iter, R);
    SYNC_TIME_PRINT(rank, ("Stencil" "\tN= %d NB= %d M= %d MB= %d "
                           "PxQ= %d %d KPxKQ= %d %d "
                           "Iteration= %d Radius= %d Kernel_type= %d "
                           "Number_of_buffers= %d cores= %d : %lf gflops\n",
                           N, NB, M, MB, P, nodes/P, KP, KQ, iter, R, LOOPGEN,
                           MMB, cores, gflops=(flops/1e9)/sync_time_elapsed));

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA);

    /* Clean up parsec*/
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif


    return 0;
}
