/**
 * Copyright (c) 2019-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/include/parsec/execution_stream.h"
#include "parsec/utils/mca_param.h"

#include "rtt.h"

#if defined(DISTRIBUTED)
#include <mpi.h>
#endif
 #include <math.h>

static int nb_gpus = 1, gpu_mask = 0xff;
static int cuda_device_index_len = 0, *cuda_device_index = NULL;

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
    memset(_A, 1, m*n);

    /* Address warning when compile */
#if 1
    parsec_data_key_t key = descA->super.data_key((parsec_data_collection_t*)descA, m, n);
    parsec_data_t* data = descA->super.data_of_key((parsec_data_collection_t*)descA, key);
    parsec_advise_data_on_device(data,
                                 cuda_device_index[m % cuda_device_index_len],
                                 PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);
#endif
    (void)es; (void)uplo;(void)n;(void)m;(void)args;
    return 0;
}

static void
__parsec_rtt_destructor(parsec_rtt_taskpool_t *rtt_tp)
{
    parsec_type_free(&(rtt_tp->arenas_datatypes[PARSEC_rtt_DEFAULT_ADT_IDX].opaque_dtt));
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_rtt_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_rtt_destructor);

parsec_taskpool_t *rtt_New(parsec_context_t *ctx, size_t size, int roundtrips)
{
    parsec_rtt_taskpool_t *tp = NULL;
    parsec_datatype_t block;
    size_t mb = sqrt(size), nb = size / mb;

    if (mb <= 0) {
        fprintf(stderr, "To work, RTT must do at least one round time trip of at least one byte\n");
        return (parsec_taskpool_t *)tp;
    }

    parsec_matrix_block_cyclic_t* dcA = (parsec_matrix_block_cyclic_t *)calloc(1, sizeof(parsec_matrix_block_cyclic_t));
    parsec_matrix_block_cyclic_init(dcA, PARSEC_MATRIX_BYTE, PARSEC_MATRIX_TILE,
                                    ctx->my_rank,
                                    mb, nb,
                                    mb, ctx->nb_nodes * nb,
                                    0, 0,
                                    mb, ctx->nb_nodes * nb,
                                    1, ctx->nb_nodes, 1, 1,
                                    0, 0);
    dcA->mat = parsec_data_allocate((size_t)dcA->super.nb_local_tiles *
                                    (size_t)dcA->super.bsiz *
                                    (size_t)parsec_datadist_getsizeoftype(dcA->super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    /* Initialize and place the dcA */
    parsec_apply(ctx, PARSEC_MATRIX_FULL,
                 (parsec_tiled_matrix_t *)dcA,
                 (parsec_tiled_matrix_unary_op_t)matrix_init_ops, NULL);

    tp = parsec_rtt_new((parsec_data_collection_t*)dcA, roundtrips, ctx->nb_nodes);

    ptrdiff_t lb, extent;
    parsec_type_create_contiguous(mb*nb, parsec_datatype_uint8_t, &block);
    parsec_type_extent(block, &lb, &extent);

    parsec_arena_datatype_construct(&tp->arenas_datatypes[PARSEC_rtt_DEFAULT_ADT_IDX],
                                    extent, PARSEC_ARENA_ALIGNMENT_SSE,
                                    block);
    return (parsec_taskpool_t *)tp;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int size = 1, rank = 0, loops = 100, frags = 1, nb_runs = 1, cores = 2, do_sleep = 0, ch, use_opt = 1;
    struct timeval tstart, tend;
    size_t msg_size = 8*1024;
    double t, bw;

    while ((ch = getopt(argc, argv, "c:g:G:l:f:m:n:s:")) != -1) {
        switch (ch) {
            case 'c': cores = atoi(optarg); use_opt += 2; break;
            case 'g': nb_gpus = atoi(optarg); use_opt += 2; break;
            case 'G': gpu_mask = atoi(optarg); use_opt += 2; break;
            case 'l': loops = atoi(optarg); use_opt += 2; break;
            case 'f': frags = atoi(optarg); use_opt += 2; break;
            case 'm': msg_size = (size_t)atoi(optarg); use_opt += 2; break;
            case 'n': nb_runs = atoi(optarg); use_opt += 2; break;
            case 's': do_sleep = atoi(optarg); use_opt += 2; break;
            default:
                fprintf(stderr,
                        "-c : number of cores to use (default 2)\n"
                        "-g : number of GPU to use (default 1)\n"
                        "-G : GPU mask to use (-1 to modulo rank per node)\n"
                        "-l : loops of bandwidth(default: 100)\n"
                        "-f : frags, number of fragments (default: 1)\n"
                        "-m : size, size of message (default: 1024 * 8)\n"
                        "-n : number of runs (default: 1)\n"
                        "-s : number of seconds to sleep before running the tests\n"
                        "\n");
                 exit(1);
        }
    }
    /* Remove all options already acknowledged */
    if( NULL == argv[optind] ) {
        argc = 1;
    } else {
        memcpy(&argv[1], &argv[use_opt+1], (argc - use_opt) * sizeof(char*));
        argc -= use_opt;
    }
    argv[argc] = NULL;
#if defined(DISTRIBUTED)
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    extern char **environ;
    char *value;
    asprintf(&value, "%d", nb_gpus);
    parsec_setenv_mca_param("device_cuda_enabled", value, &environ);
    free(value);
    value = NULL;
    if (0xFF != gpu_mask) {
        asprintf(&value, "%d", gpu_mask);
        parsec_setenv_mca_param("device_cuda_mask", value, &environ);
        free(value);
        value = NULL;
    }
#endif
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* DISTRIBUTED */
    if( 0 == rank ) {
        printf("Running %d tests of %d steps RTT with a data of size %zu\n",
               nb_runs, loops, msg_size);
    }
    parsec = parsec_init(cores, &argc, &argv);

    /* can the test run? */
    nb_gpus = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_CUDA);
    assert(nb_gpus >= 0);
    if(nb_gpus == 0) {
        parsec_warning("This test can only run if at least one GPU device is present");
        exit(-PARSEC_ERR_DEVICE);
    }
    if( do_sleep ) {
        sleep(do_sleep);
    }
    cuda_device_index = (int *)malloc(parsec_nb_devices * sizeof(int));
    cuda_device_index_len = 0;
    for (int dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if (PARSEC_DEV_CUDA & device->type) {
            cuda_device_index[cuda_device_index_len++] = device->device_index;
        }
    }

#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif  /* defined(PARSEC_HAVE_MPI) */
    gettimeofday(&tstart, NULL);
    for( int test_id = 0; test_id < nb_runs; test_id++ ) {
        tp = rtt_New(parsec, msg_size, loops);
        if( NULL != tp ) {
            parsec_context_add_taskpool(parsec, tp);
            parsec_context_start(parsec);
            parsec_context_wait(parsec);
            parsec_taskpool_free(tp);
        }
    }
#if defined(PARSEC_HAVE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif  /* defined(PARSEC_HAVE_MPI) */
    gettimeofday(&tend, NULL);

    if( 0 == rank ) {
        t = ((tend.tv_sec - tstart.tv_sec) * 1000000.0 + (tend.tv_usec - tstart.tv_usec)) / 1000000.0;  /* in seconds */
        double total_payload = (double)nb_runs * (double)loops * (double)msg_size / 1024.0 / 1024.0 / 1024.0;
        bw = total_payload / t;
        printf("%d\t%d\t%d\t%zu\t%08.4g s\t%4.8g GB/s\n", nb_runs, frags, loops, msg_size*sizeof(uint8_t), t, bw);
    }

    free(cuda_device_index); cuda_device_index = NULL;
    cuda_device_index_len = 0;
    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif /* DISTRIBUTED */
    return 0;
}
