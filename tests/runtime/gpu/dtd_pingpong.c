/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_data.h"
#include "tests/tests_timing.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

static int TILE_FULL;
static int32_t nb_errors = 0;
static int verbose=0;

#define NLOOP 8

int cpu_ping( parsec_execution_stream_t *es,
                  parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int rank;
    int nb;
    parsec_dtd_unpack_args(this_task, &rank, &data, &nb);

    if(verbose)
      fprintf(stderr, "cpu_ping(): on CPU of MPI rank %d data of size %d\n", es->virtual_process->parsec_context->my_rank, nb);

    for(int idx = 0; idx < nb; idx ++)
        data[idx] = idx;

    return PARSEC_HOOK_RETURN_DONE;
}

int cpu_pong( parsec_execution_stream_t *es,
                  parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int rank, idx;
    parsec_dtd_unpack_args(this_task, &rank, &data, &idx);

    if(verbose)
      fprintf(stderr, "cpu_pong(%d): on CPU of MPI rank %d\n", idx, es->virtual_process->parsec_context->my_rank);

    data[idx] += idx;

    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
extern void cuda_pong_kernel(int *dev_data, int idx);

int cuda_pong(parsec_device_gpu_module_t *gpu_device,
              parsec_gpu_task_t *gpu_task,
              parsec_gpu_exec_stream_t *gpu_stream)
{
    (void)gpu_device;
    (void)gpu_stream;

    int *data;
    void *dev_data;
    int rank, idx;

    parsec_task_t *this_task = gpu_task->ec;
    parsec_dtd_unpack_args(this_task, &rank, &data, &idx);

    if(verbose)
      fprintf(stderr, "gpu_pong(%d): on GPU %s of MPI rank %d\n", idx, gpu_device->super.name, this_task->taskpool->context->my_rank);

    dev_data = parsec_dtd_get_dev_ptr(this_task, 0);

    cuda_pong_kernel(dev_data, idx);

    return PARSEC_HOOK_RETURN_DONE;
}
#endif

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
extern void hip_pong_kernel(int *dev_data, int idx);

int hip_pong(parsec_device_gpu_module_t *gpu_device,
              parsec_gpu_task_t *gpu_task,
              parsec_gpu_exec_stream_t *gpu_stream)
{
    (void)gpu_device;
    (void)gpu_stream;

    int *data;
    void *dev_data;
    int rank, idx;

    parsec_task_t *this_task = gpu_task->ec;
    parsec_dtd_unpack_args(this_task, &rank, &data, &idx);

    if(verbose)
      fprintf(stderr, "hip_pong(%d): on GPU %s of MPI rank %d\n", idx, gpu_device->super.name, this_task->taskpool->context->my_rank);

    dev_data = parsec_dtd_get_dev_ptr(this_task, 0);

    hip_pong_kernel(dev_data, idx);

    return PARSEC_HOOK_RETURN_DONE;
}
#endif

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nb, rc, nb_gpus = 0;
    parsec_arena_datatype_t *adt;
    parsec_device_module_t **gpu_devices = NULL;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if(MPI_THREAD_MULTIPLE > provided) {
            parsec_fatal( "This benchmark requires MPI_THREAD_MULTIPLE because it uses simultaneously MPI within the PaRSEC runtime, and in the main program loop");
        }
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    parsec = parsec_init( cores, &argc, &argv );
#if defined(PARSEC_PROF_TRACE)
    parsec_profiling_start();
#endif

    for(unsigned int i = 0; i < parsec_nb_devices; i++) {
        parsec_device_module_t *dev = parsec_mca_device_get(i);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        if( dev->type == PARSEC_DEV_CUDA )
            nb_gpus++;
#endif
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        if( dev->type == PARSEC_DEV_HIP )
            nb_gpus++;
#endif
    }
    if(nb_gpus > 0) {
        gpu_devices = (parsec_device_module_t **)malloc(sizeof(parsec_device_module_t*)*nb_gpus);
        nb_gpus = 0;
        for(unsigned int i = 0; i < parsec_nb_devices; i++) {
            parsec_device_module_t *dev = parsec_mca_device_get(i);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
            if( dev->type == PARSEC_DEV_CUDA) {
                gpu_devices[nb_gpus] = dev;
                nb_gpus++;
            }
#endif
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
            if( dev->type == PARSEC_DEV_HIP) {
                gpu_devices[nb_gpus] = dev;
                nb_gpus++;
            }
#endif
        }
    } else {
        if(0 == rank) {
            fprintf(stderr, "Warning: test disabled because there is no GPU detected with this run\n");
        }
        parsec_fini(&parsec);
        MPI_Finalize();
        return EXIT_SUCCESS; /* So that useless tests don't make the CI fail */
    }

    nb = 3 * NLOOP * world * nb_gpus;

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_add2arena( adt, parsec_datatype_int32_t, PARSEC_MATRIX_FULL, 0,
                      nb, 1, nb, PARSEC_ARENA_ALIGNMENT_SSE, -1);

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_task_class_t *ping_tc = parsec_dtd_create_task_class(dtd_tp, "ping",
                                                                sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY | PARSEC_PROFILE_INFO, "rank",
                                                                PASSED_BY_REF, PARSEC_OUTPUT | TILE_FULL,
                                                                sizeof(int), PARSEC_VALUE | PARSEC_PROFILE_INFO, "nb",
                                                                PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, ping_tc, PARSEC_DEV_CPU, cpu_ping);
    parsec_task_class_t *pong_tc = parsec_dtd_create_task_class(dtd_tp, "pong",
                                                                sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY | PARSEC_PROFILE_INFO, "rank",
                                                                PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_PUSHOUT,
                                                                sizeof(int), PARSEC_VALUE | PARSEC_PROFILE_INFO, "idx",
                                                                PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, pong_tc, PARSEC_DEV_CPU, cpu_pong);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    parsec_dtd_task_class_add_chore(dtd_tp, pong_tc, PARSEC_DEV_CUDA, cuda_pong);
#endif
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    parsec_dtd_task_class_add_chore(dtd_tp, pong_tc, PARSEC_DEV_HIP, hip_pong);
#endif

    parsec_dtd_tile_t *tile = parsec_dtd_tile_new(dtd_tp, 0);
    parsec_dtd_insert_task_with_task_class(dtd_tp, ping_tc, 0,
                                            PARSEC_DEV_CPU,
                                            PARSEC_DTD_EMPTY_FLAG, &rank,
                                            PARSEC_DTD_EMPTY_FLAG, tile,
                                            PARSEC_DTD_EMPTY_FLAG, &nb,
                                            PARSEC_DTD_ARG_END);

    int idx = 0;
    for(int loop = 0; loop < NLOOP; loop++) {
        for(int rank = 0; rank < world; rank++) {
            for(int dev = 0; dev < nb_gpus; dev++) {
                parsec_dtd_insert_task_with_task_class(dtd_tp, pong_tc, 0,
                                               PARSEC_DEV_CPU,
                                               PARSEC_DTD_EMPTY_FLAG, &rank,
                                               PARSEC_DTD_EMPTY_FLAG, tile,
                                               PARSEC_DTD_EMPTY_FLAG, &idx,
                                               PARSEC_DTD_ARG_END);
                idx+=1;
                parsec_advise_data_on_device(tile->data_copy->original, gpu_devices[dev]->device_index, PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);
                parsec_dtd_insert_task_with_task_class(dtd_tp, pong_tc, 0,
                                               gpu_devices[dev]->type,
                                               PARSEC_DTD_EMPTY_FLAG, &rank,
                                               PARSEC_DTD_EMPTY_FLAG, tile,
                                               PARSEC_DTD_EMPTY_FLAG, &idx,
                                               PARSEC_DTD_ARG_END);
                idx+=1;
                parsec_dtd_insert_task_with_task_class(dtd_tp, pong_tc, 0,
                                               gpu_devices[dev]->type,
                                               PARSEC_DTD_EMPTY_FLAG, &rank,
                                               PARSEC_DTD_EMPTY_FLAG, tile,
                                               PARSEC_DTD_EMPTY_FLAG, &idx,
                                               PARSEC_DTD_ARG_END);
                idx+=1;
            }
        }
    }

    /* Rank 0 keeps the tile alive for checking, the others can flush it out */
    if(0 == rank) {
        PARSEC_OBJ_RETAIN(tile);
    }
    parsec_dtd_data_flush(dtd_tp, tile);

    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if(0 == rank) {
        int32_t *data = parsec_data_copy_get_ptr(tile->data_copy);
        for(int n = 0; n < nb; n++) {
            if(2*n != data[n]) {
                printf("Final value at index %d is %d, expected %d\n", n, data[n], 2*n);
                nb_errors++;
            }
        }
        PARSEC_OBJ_RELEASE(tile);
    }

    parsec_dtd_task_class_release(dtd_tp, ping_tc);

    parsec_del2arena(adt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);

    parsec_taskpool_free( dtd_tp );
    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    if(nb_errors > 0)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
