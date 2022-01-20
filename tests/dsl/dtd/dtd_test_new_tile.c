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

/* IDs for the Arena Datatypes */
static int TILE_FULL;
static int32_t nb_errors = 0;
static int verbose=0;

#if defined(PARSEC_HAVE_CUDA)
extern void dtd_test_new_tile_init(int *dev_data, int nb, int idx);
extern void dtd_test_new_tile_sum_add(int *dev_data, int nb, int idx, int *acc, int verbose);
extern void dtd_test_new_tile_multiply_by_two(int *dev_data, int nb, int idx);
#endif

#define NCASE 8
#define NB    8

int cpu_set_to_i( parsec_execution_stream_t *es,
                  parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int rank, idx, nb;
    parsec_dtd_unpack_args(this_task, &rank, &data, &nb, &idx);

    if(verbose)
      fprintf(stderr, "cpu_set_to_i(%d): on CPU of MPI rank %d\n", idx, es->virtual_process->parsec_context->my_rank);

    for(int i = 0; i < nb; i++)
        data[i] = i;

    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(PARSEC_HAVE_CUDA)
int cuda_set_to_i(parsec_device_gpu_module_t *gpu_device,
                  parsec_gpu_task_t *gpu_task,
                  parsec_gpu_exec_stream_t *gpu_stream)
{
    (void)gpu_device;
    (void)gpu_stream;

    int *data;
    void *dev_data;
    int rank, nb, idx;

    parsec_task_t *this_task = gpu_task->ec;
    parsec_dtd_unpack_args(this_task, &rank, &data, &nb, &idx);

    if(verbose)
      fprintf(stderr, "cuda_set_to_i(%d): on GPU %s of MPI rank %d\n", idx, gpu_device->super.name, this_task->taskpool->context->my_rank);

    dev_data = parsec_dtd_get_dev_ptr(this_task, 0);

    int devid;
    cudaError_t err = cudaGetDevice(&devid);
    assert(cudaSuccess == err);

    dtd_test_new_tile_init(dev_data, nb, idx);

    return PARSEC_HOOK_RETURN_DONE;
}
#endif

int cpu_multiply_by_2( parsec_execution_stream_t *es,
                       parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int nb, idx;
    parsec_dtd_unpack_args(this_task, &data, &nb, &idx);

    if(verbose)
      fprintf(stderr, "multiply_by_2(%d): on CPU of MPI rank %d\n", idx, es->virtual_process->parsec_context->my_rank);

    for(int i = 0; i < nb; i++) {
        if(i != data[i]) {
            fprintf(stderr, "Error in multiply_by_2(%d) on rank %d at index %d: expected %d, got %d\n",
                    idx, this_task->taskpool->context->my_rank, i, i, data[i]);
            parsec_atomic_fetch_inc_int32(&nb_errors);
        }
        data[i] *= 2;
    }

    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(PARSEC_HAVE_CUDA)
int cuda_multiply_by_2(parsec_device_gpu_module_t *gpu_device,
                       parsec_gpu_task_t *gpu_task,
                       parsec_gpu_exec_stream_t *gpu_stream)
{
    (void)gpu_device;
    (void)gpu_stream;

    int *data;
    void *dev_data;
    int nb, idx;

    parsec_task_t *this_task = gpu_task->ec;
    parsec_dtd_unpack_args(this_task, &data, &nb, &idx);

    if(verbose)
      fprintf(stderr, "cuda_multiply_by_2(%d): on GPU %s of MPI rank %d\n", idx, gpu_device->super.name, this_task->taskpool->context->my_rank);

    dev_data = parsec_dtd_get_dev_ptr(this_task, 0);

    // Call the asynchronous kernel (written in CUDA):
    dtd_test_new_tile_multiply_by_two(dev_data, nb, idx);

    return PARSEC_HOOK_RETURN_DONE;
}
#endif

int cpu_accumulate( parsec_execution_stream_t *es,
                    parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int nb, idx;
    int32_t *acc, lacc, **gpu_accs;
    parsec_dtd_unpack_args(this_task, &data, &nb, &idx, &acc, &gpu_accs);

    if(verbose)
      fprintf(stderr, "cpu_accumulate(%d): on CPU of MPI rank %d\n", idx, es->virtual_process->parsec_context->my_rank);

    lacc = 0;
    for(int i = 0; i < nb; i++) {
        if(2*i != data[i]) {
            fprintf(stderr, "Error in cpu_accumulate(%d) on rank %d at index %d: expected %d, got %d\n",
                    idx, this_task->taskpool->context->my_rank, i, 2*i, data[i]);
            parsec_atomic_fetch_inc_int32(&nb_errors);
        }
        lacc += data[i];
    }
    if(verbose)
      printf("cpu_accumulate(%d) contributes with %d\n", idx, lacc);
    parsec_atomic_fetch_add_int32(acc, lacc);

    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(PARSEC_HAVE_CUDA)
int cuda_accumulate(parsec_device_gpu_module_t *gpu_device,
                    parsec_gpu_task_t *gpu_task,
                    parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)gpu_device;

    int *data;
    void *dev_data;
    int nb, idx;
    int32_t *acc, **gpu_accs;

    (void)gpu_stream;

    parsec_task_t *this_task = gpu_task->ec;
    parsec_dtd_unpack_args(this_task, &data, &nb, &idx, &acc, &gpu_accs);

    if(verbose)
      fprintf(stderr, "cuda_accumulate(%d): on GPU %s of MPI rank %d\n", idx, gpu_device->super.name, this_task->taskpool->context->my_rank);

    dev_data = parsec_dtd_get_dev_ptr(this_task, 0);

    // Call the asynchronous kernel (written in CUDA):
    dtd_test_new_tile_sum_add(dev_data, nb, idx, gpu_accs[cuda_device->cuda_index], verbose);

    return PARSEC_HOOK_RETURN_DONE;
}
#endif

int cpu_reduce( parsec_execution_stream_t *es,
                parsec_task_t *this_task )
{
    (void)es;
    int *dst_data;
    int *src_data;
    int nb, r, t, lacc;
    int32_t *acc;
    parsec_dtd_unpack_args(this_task, &dst_data, &src_data, &nb, &r, &t, &acc);

    if(verbose)
      fprintf(stderr, "cpu_reduce(%d): on CPU of MPI rank %d\n", t, es->virtual_process->parsec_context->my_rank);

    lacc = 0;
    assert(dst_data != src_data);
    for(int i =0; i < nb; i++) {
       dst_data[i] += src_data[i];
       lacc += src_data[i];
    }
    if(r != 0) {
       fprintf(stderr, "cpu_reduce(%d) contributes with %d\n", t, lacc);
       parsec_atomic_fetch_add_int32(acc, lacc);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nb, rc, nb_gpus = 0;
    int32_t acc, expected = 0, *pacc = &acc, **gpu_accs = NULL;
    parsec_arena_datatype_t *adt;
#if defined(PARSEC_HAVE_CUDA)
    parsec_device_cuda_module_t **gpu_devices = NULL;
#endif

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if(MPI_THREAD_MULTIPLE > provided) {
            parsec_fatal( "This benchmark requires MPI_THREAD_MULTIPLE because it uses simultaneously MPI within the PaRSEC runtime, and in the main program loop (in SYNC_TIME_START)");
        }
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    nb = NB; /* tile_size */

    parsec = parsec_init( cores, &argc, &argv );

#if defined(PARSEC_HAVE_CUDA)
    for(unsigned int i = 0; i < parsec_nb_devices; i++) {
        parsec_device_module_t *dev = parsec_mca_device_get(i);
        if( dev->type == PARSEC_DEV_CUDA )
            nb_gpus++;
    }
    if(nb_gpus > 0) {
        gpu_accs = (int **)malloc(sizeof(int *) * nb_gpus);
        gpu_devices = (parsec_device_cuda_module_t **)malloc(sizeof(parsec_device_cuda_module_t*)*nb_gpus);
        nb_gpus = 0;
        for(unsigned int i = 0; i < parsec_nb_devices; i++) {
            parsec_device_module_t *dev = parsec_mca_device_get(i);
            if( dev->type == PARSEC_DEV_CUDA) {
                cudaError_t status;
                parsec_device_cuda_module_t *gpu_device = (parsec_device_cuda_module_t *)dev;
                status = cudaSetDevice( gpu_device->cuda_index );
                if( cudaSuccess != status ) {
                    fprintf(stderr, "Unable to select CUDA device %d: %s -- device ignored\n", gpu_device->cuda_index,
                            cudaGetErrorString(status));
                } else {
                    status = cudaMalloc((void**)&gpu_accs[nb_gpus], sizeof(int));
                    if( cudaSuccess != status ) {
                        fprintf(stderr, "Unable to allocate memory on CUDA device %d: %s -- device ignored\n",
                                gpu_device->cuda_index, cudaGetErrorString(status));
                    } else {
                        status = cudaMemset((void*)gpu_accs[nb_gpus], 0, sizeof(int));
                        if( cudaSuccess != status ) {
                            fprintf(stderr, "Unable to initialize memory on CUDA device %d: %s -- device ignored\n",
                                    gpu_device->cuda_index, cudaGetErrorString(status));
                        } else {
                            gpu_devices[nb_gpus] = gpu_device;
                            nb_gpus++;
                        }
                    }
                }
            }
        }
    } else {
        gpu_accs = &pacc;
    }
#else
    gpu_accs = &pacc;
    nb_gpus = 0;
#endif

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_add2arena( adt, parsec_datatype_int32_t,PARSEC_MATRIX_FULL, 0,
                      nb, 1, nb, PARSEC_ARENA_ALIGNMENT_SSE, -1);

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_task_class_t *first_tc = parsec_dtd_create_task_class(dtd_tp, "set_to_i",
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 PASSED_BY_REF, PARSEC_OUTPUT | TILE_FULL,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 PARSEC_DTD_ARG_END);
#if defined(PARSEC_HAVE_CUDA)
    parsec_dtd_task_class_add_chore(dtd_tp, first_tc, PARSEC_DEV_CUDA, cuda_set_to_i);
#endif
    parsec_dtd_task_class_add_chore(dtd_tp, first_tc, PARSEC_DEV_CPU, cpu_set_to_i);

    parsec_task_class_t *second_tc = parsec_dtd_create_task_class(dtd_tp, "multiply_by_2",
                                                                  PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                                                  sizeof(int), PARSEC_VALUE,
                                                                  sizeof(int), PARSEC_VALUE,
                                                                  PARSEC_DTD_ARG_END);
#if defined(PARSEC_HAVE_CUDA)
    parsec_dtd_task_class_add_chore(dtd_tp, second_tc, PARSEC_DEV_CUDA, cuda_multiply_by_2);
#endif
    parsec_dtd_task_class_add_chore(dtd_tp, second_tc, PARSEC_DEV_CPU, cpu_multiply_by_2);

    parsec_task_class_t *third_tc = parsec_dtd_create_task_class(dtd_tp, "accumulate",
                                                                  PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                                                  sizeof(int), PARSEC_VALUE,
                                                                  sizeof(int), PARSEC_VALUE,
                                                                  sizeof(int), PARSEC_REF,
                                                                  sizeof(int*), PARSEC_REF,
                                                                  PARSEC_DTD_ARG_END);
#if defined(PARSEC_HAVE_CUDA)
    parsec_dtd_task_class_add_chore(dtd_tp, third_tc, PARSEC_DEV_CUDA, cuda_accumulate);
#endif
    parsec_dtd_task_class_add_chore(dtd_tp, third_tc, PARSEC_DEV_CPU, cpu_accumulate);

    parsec_task_class_t *fourth_tc = parsec_dtd_create_task_class(dtd_tp, "reduce",
                                                                 PASSED_BY_REF, PARSEC_OUTPUT | TILE_FULL | PARSEC_AFFINITY,
                                                                 PASSED_BY_REF, PARSEC_INPUT | TILE_FULL,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 sizeof(int), PARSEC_REF,
                                                                 PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, fourth_tc, PARSEC_DEV_CPU, cpu_reduce);

    parsec_dtd_tile_t **new_tiles;
    new_tiles = (parsec_dtd_tile_t**)calloc(sizeof(parsec_dtd_tile_t *), NCASE*world);

    acc = 0;

    for(int t = 0; t < NCASE*world; t++) {
        int r = t % world;
        int tcase = t / world;

        if(0 == rank || r == rank)
            expected += 2 * (NB * (NB-1))/2;

        // tile is a new unique tile, without a backend data until it is
        // written upon by the first task
        parsec_dtd_tile_t *tile = parsec_dtd_tile_new(dtd_tp, r);
        new_tiles[t] = tile;

        int first_on_gpu = (nb_gpus > 0) && !(tcase & 1);
        int second_on_gpu = (nb_gpus > 0) && !(tcase & 2);
        int third_on_gpu = (nb_gpus > 0) && !(tcase & 4);
        int first_pushout = PARSEC_DTD_EMPTY_FLAG;
        int second_pushout = PARSEC_DTD_EMPTY_FLAG;
        int third_pushout = PARSEC_DTD_EMPTY_FLAG;
#if defined(PARSEC_HAVE_CUDA)
        if(first_on_gpu && !second_on_gpu) first_pushout = PARSEC_PUSHOUT;
        if(second_on_gpu && !third_on_gpu) second_pushout = PARSEC_PUSHOUT;
        if(third_on_gpu) third_pushout = PARSEC_PUSHOUT;
#endif

        parsec_dtd_insert_task_with_task_class(dtd_tp, first_tc, 0,
                                               first_on_gpu ? PARSEC_DEV_CUDA : PARSEC_DEV_CPU,
                                               PARSEC_AFFINITY, &r,
                                               first_pushout, tile,
                                               PARSEC_DTD_EMPTY_FLAG, &nb,
                                               PARSEC_DTD_EMPTY_FLAG, &t,
                                               PARSEC_DTD_ARG_END);
        parsec_dtd_insert_task_with_task_class(dtd_tp, second_tc, 0,
                                               second_on_gpu ?  PARSEC_DEV_CUDA : PARSEC_DEV_CPU,
                                               second_pushout, tile,
                                               PARSEC_DTD_EMPTY_FLAG, &nb,
                                               PARSEC_DTD_EMPTY_FLAG, &t,
                                               PARSEC_DTD_ARG_END);
        parsec_dtd_insert_task_with_task_class(dtd_tp, third_tc, 0,
                                               third_on_gpu ? PARSEC_DEV_CUDA : PARSEC_DEV_CPU,
                                               third_pushout, tile,
                                               PARSEC_DTD_EMPTY_FLAG, &nb,
                                               PARSEC_DTD_EMPTY_FLAG, &t,
                                               PARSEC_DTD_EMPTY_FLAG, &acc,
                                               PARSEC_DTD_EMPTY_FLAG, gpu_accs,
                                               PARSEC_DTD_ARG_END);
    }
    for(int t = 1; t < NCASE*world; t++) {
        /* We now reduce everything on tile 0 */
        int r = t % world;
        parsec_dtd_insert_task_with_task_class(dtd_tp, fourth_tc, 0, PARSEC_DEV_CPU,
                                               PARSEC_DTD_EMPTY_FLAG, new_tiles[0],
                                               PARSEC_DTD_EMPTY_FLAG, new_tiles[t],
                                               PARSEC_DTD_EMPTY_FLAG, &nb,
                                               PARSEC_DTD_EMPTY_FLAG, &r,
                                               PARSEC_DTD_EMPTY_FLAG, &t,
                                               PARSEC_DTD_EMPTY_FLAG, &acc,
                                               PARSEC_DTD_ARG_END);
    }

    parsec_dtd_tile_t *tile0 = new_tiles[0];
    if(0 == rank) {
        PARSEC_OBJ_RETAIN(tile0);
    }

    for(int r = 0; r < NCASE*world; r++) {
        parsec_dtd_data_flush(dtd_tp, new_tiles[r]);
    }

    rc = parsec_dtd_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    free(new_tiles);

#if defined(PARSEC_HAVE_CUDA)
    for(int i = 0; i < nb_gpus; i++) {
        cudaError_t status;
        parsec_device_cuda_module_t *gpu_device = gpu_devices[i];
        int gpu_acc = 0;
        status = cudaSetDevice( gpu_device->cuda_index );
        if(cudaSuccess != status) {
            fprintf(stderr, "Unable to re-select CUDA device %d: %s -- fatal error\n", gpu_device->cuda_index,
                    cudaGetErrorString(status));
            nb_errors++;
        }
        status = cudaMemcpy(&gpu_acc, gpu_accs[i], sizeof(int), cudaMemcpyDeviceToHost);
        if( cudaSuccess != status ) {
            fprintf(stderr, "Unable to copy back accumulator from CUDA device %d: %s -- fatal error\n",
                    gpu_device->cuda_index, cudaGetErrorString(status));
            nb_errors++;
        }
        acc += gpu_acc;
        status = cudaFree(gpu_accs[i]);
        if( cudaSuccess != status ) {
            fprintf(stderr, "Unable to free memory on CUDA device %d: %s -- fatal error\n",
                    gpu_device->cuda_index, cudaGetErrorString(status));
            nb_errors++;
        }
    }
#endif

    if(0 == rank) {
        int32_t *data = parsec_data_copy_get_ptr(tile0->data_copy);
        for(int n = 0; n < nb; n++) {
            if(2*NCASE*world*n != data[n]) {
                printf("Rank 0: reduced value at index %d is %d, expected %d\n", n, data[n], 2*NCASE*world*n);
                nb_errors++;
            }
        }
        PARSEC_OBJ_RELEASE(tile0);
    }

    if(acc != expected) {
        fprintf(stderr, "Rank %d failure: acc = %d, expected %d\n", rank, acc, expected);
        nb_errors++;
    } else {
        printf("Rank %d success: acc = %d, which is the expected value\n", rank, acc);
    }

    parsec_dtd_task_class_release(dtd_tp, first_tc);
    parsec_dtd_task_class_release(dtd_tp, second_tc);
    parsec_dtd_task_class_release(dtd_tp, third_tc);

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
