/*
 * Copyright (c) 2019-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "cuda_test_internal.h"

/* Timing */
double sync_time_elapsed = 0.0;

/**
 * @brief init operator
 *
 * @param [in] es: execution stream
 * @param [in] descA: tiled matrix date descriptor
 * @param [inout] A:  inout data
 * @param [in] uplo: matrix shape
 * @param [in] m: tile row index
 * @param [in] n: tile column index
 * @param [in] args: NULL 
 */
static int matrix_init_ops(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_t *descA,
                        void *_A, parsec_matrix_uplo_t uplo,
                        int m, int n, void *args)
{
    double *A = (double *)_A;

    for(int j = 0; j < descA->nb; j++)
        for(int i = j; i <= descA->mb ; i++)
            A[j*descA->mb+i] = 1.0;

    /* Address warning when compile */
    if( 0 ) printf("%p %d %d %d %p\n", es, uplo, m, n, args);

    return 0;
}

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, nodes, ch, rc;
    int pargc = 0;
    char **pargv;

    /* Default */
    int m = 0;
    int N = 8;
    int NB = 4;
    int P = 1;
    int KP = 1;
    int KQ = 1;
    int cores = -1;
    int nb_gpus = 0;
    int info = 0;

    while ((ch = getopt(argc, argv, "m:N:t:s:S:P:c:g:h")) != -1) {
        switch (ch) {
            case 'm': m = atoi(optarg); break;
            case 'N': N = atoi(optarg); break;
            case 't': NB = atoi(optarg); break;
            case 's': KP = atoi(optarg); break;
            case 'S': KQ = atoi(optarg); break;
            case 'P': P = atoi(optarg); break;
            case 'c': cores = atoi(optarg); break;
            case 'g': nb_gpus = atoi(optarg); break;
            case '?': case 'h': default:
                fprintf(stderr,
                        "-m : request multiple-thread support from the test runtime (default: 0/no)\n"
                        "-N : column dimension (N) of the matrices (default: 8)\n"
                        "-t : row dimension (MB) of the tiles (default: 4)\n"
                        "-s : rows of tiles in a k-cyclic distribution (default: 1)\n"
                        "-S : columns of tiles in a k-cyclic distribution (default: 1)\n"
                        "-P : rows (P) in the PxQ process grid (default: 1)\n"
                        "-c : number of cores used (default: -1)\n"
                        "-g : number of GPUs used (default: 0)\n"
                        "-h : print this help message\n"
                        "\n");
                 exit(1);
        }
    }

    pargc = 0; pargv = NULL;
    for(int i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            pargc = argc - i;
            pargv = argv + i;
            break;
        }
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    extern char **environ;
    char *value;
    asprintf(&value, "%d", nb_gpus);
    parsec_setenv_mca_param( "device_cuda_enabled", value, &environ );
    free(value);
#endif

    rc = parsec_tests_context_init(cores,
                                   m ? PARSEC_TEST_THREAD_MULTIPLE : PARSEC_TEST_THREAD_SERIALIZED,
                                   &pargc, &pargv, &parsec, &rank, &nodes);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if( nb_gpus < 1 && 0 == rank ) {
        fprintf(stderr, "Warning: if run on GPUs, please set --gpus=value bigger than 0\n");
    }
#endif

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

    /* initializing matrix structure */
    parsec_matrix_block_cyclic_t dcA;
    parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                rank, NB, NB, N, N, 0, 0,
                                N, N, P, nodes/P, KP, KQ, 0, 0); 
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                   (size_t)dcA.super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "dcA");

    /* Init dcA to symmetric positive definite */ 
    parsec_apply( parsec, PARSEC_MATRIX_FULL,
                  (parsec_tiled_matrix_t *)&dcA,
                  (parsec_tiled_matrix_unary_op_t)matrix_init_ops, NULL);

    /* Main routines */
    SYNC_TIME_START(parsec);
    info = parsec_get_best_device_check(parsec, (parsec_tiled_matrix_t *)&dcA);
    SYNC_TIME_PRINT(parsec, rank, ("Get_best_device" "\tN= %d NB= %d "
                           "PxQ= %d %d KPxKQ= %d %d cores= %d nb_gpus= %d\n",
                           N, NB, P, nodes/P, KP, KQ, cores, parsec_nb_devices-2)); 

    /* Check result */
    if( 0 == rank && info != 0 ) {
        fprintf(stderr, "Result is Wrong (info %d) !!!\n", info);
    }

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA);

    /* Clean up parsec*/
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return (0 == info)? EXIT_SUCCESS: EXIT_FAILURE;
}
