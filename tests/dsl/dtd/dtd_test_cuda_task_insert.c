#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "tests/tests_data.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#include "parsec/mca/device/cuda/device_cuda.h"

static int TILE_FULL;

static volatile int32_t error_count;

#define WITH_GPU_TASK ((1<<0))
#define WITH_CPU_TASK ((1<<1))

static char *mode_to_string(int mode)
{
    if( (mode & WITH_CPU_TASK) && (mode & WITH_GPU_TASK) )
        return "GPU|CPU";
    if( mode & WITH_CPU_TASK )
        return "CPU";
    if( mode & WITH_GPU_TASK )
        return "GPU";
    return "?!? Neither CPU or GPU !?!";
}

static unsigned int unique_id(int myrank, int i, int mt, int j, int nb)
{
    return myrank + (j + i * nb) * mt + 1;
}

int print_cuda_info_task(parsec_device_cuda_module_t *cuda_device,
                         parsec_gpu_task_t *gpu_task,
                         parsec_gpu_exec_stream_t *gpu_stream)
{
    int i;
    parsec_task_t *this_task = gpu_task->ec;
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;

    parsec_dtd_unpack_args(this_task, &i);

    printf("  Task number: %d on rank %d\n", i, this_task->taskpool->context->my_rank);
    if(this_task->taskpool->context->my_rank != i) {
        parsec_output( 0, "**** Error: task %d is supposed to run on rank %d only, it's running on rank %d\n",
                       i, i, this_task->taskpool->context->my_rank);
        parsec_atomic_fetch_inc_int32(&error_count);
    }

    printf("  CUDA device idx: %d\n", cuda_device->cuda_index);
    printf( "  CUDA Stream name: %s\n", gpu_stream->name);
    printf("  CUDA device compute capability: %d.%d\n",
           cuda_device->major, cuda_device->minor);
    printf( "  CUDA Stream: %p\n", (void*)(uintptr_t)cuda_stream->cuda_stream);
    printf("  CUDA device num exec stream: %d\n",
           cuda_device->super.num_exec_streams);

    return PARSEC_HOOK_RETURN_DONE;
}

int test_cuda_print_info(int world, int rank, parsec_context_t *parsec_context)
{
    int ret;
    (void)rank;
    printf("[dtd_test_cuda][print_info]\n");

    error_count = 0;

    // Create new DTD taskpool
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_task_class_t *print_info_tc;

    ret = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(ret, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    ret = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(ret, "parsec_context_add_taskpool");

    print_info_tc = parsec_dtd_create_task_class(tp, "PrintCudaInfo",
                                                 sizeof(int), PARSEC_VALUE,
                                                 PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, print_info_tc, PARSEC_DEV_CUDA, print_cuda_info_task);

    for( int i = 0; i < world; ++i ) {
        int prio = 1;
        parsec_dtd_insert_task_with_task_class(tp, print_info_tc, prio, PARSEC_DEV_ALL,
                                               PARSEC_AFFINITY, &i,
                                               PARSEC_DTD_ARG_END);
    }

    // Wait for task completion
    ret = parsec_dtd_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(ret, "parsec_dtd_taskpool_wait");

    ret = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(ret, "parsec_context_wait");

    parsec_dtd_task_class_release(tp, print_info_tc);

    parsec_taskpool_free(tp);

    return error_count;
}

int cuda_memset_task_fn(parsec_device_cuda_module_t *cuda_device,
                        parsec_gpu_task_t *gpu_task,
                        parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;
    (void)cuda_device;

    int *data;
    void *dev_data;
    int rank;
    int nb;

    parsec_task_t *this_task = gpu_task->ec;

    parsec_dtd_unpack_args(this_task, &data, &nb, &rank);

    dev_data = parsec_dtd_get_dev_ptr(this_task, 0);

    int devid;
    cudaError_t err = cudaGetDevice(&devid);
    assert(cudaSuccess == err);
    (void)err;

    cudaMemsetAsync((void *)dev_data, 0xFF, nb * sizeof(int), cuda_stream->cuda_stream);
    cudaStreamSynchronize(cuda_stream->cuda_stream);

    return PARSEC_HOOK_RETURN_DONE;
}

int memset_task_fn(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int *data;
    int rank;
    int nb;

    parsec_dtd_unpack_args(this_task, &data, &nb, &rank);

    memset((void *)data, 0xFF, nb * sizeof(int));

    return PARSEC_HOOK_RETURN_DONE;
}

int test_cuda_memset(int world, int myrank, parsec_context_t *parsec_context, int mode)
{
    // Error code return by parsec routines
    int perr;

    // Tile size
    int nb = 10;

    // Total number of tiles
    int mt = 16;
    int nt = world;

    error_count = 0;

    printf("[dtd_test_cuda][memset]\n");

    // Create new DTD taskpool
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_arena_datatype_t *adt = parsec_dtd_create_arena_datatype(parsec_context, &TILE_FULL);
    // unless `parsec_dtd_taskpool_new()` is called first.
    parsec_add2arena_rect(adt, parsec_datatype_int32_t, nb, 1, nb);

    parsec_tiled_matrix_t *dcA;
    dcA = create_and_distribute_data(myrank, world, nb, nt*mt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_key_t key;
    parsec_data_copy_t *parsec_data_copy;
    parsec_data_t *parsec_data;
    unsigned int *data_ptr;

    parsec_task_class_t *memset_tc;

    for( int i = 0; i < mt; i++ ) {
        key = A->data_key(A, i*nt + myrank, 0);
        parsec_data = A->data_of_key(A, key);
        parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
        data_ptr = (unsigned int *)parsec_data_copy_get_ptr(parsec_data_copy);
        for( int j = 0; j < nb; j++ )
            data_ptr[j] = unique_id(myrank, i, mt, j, nb);
    }

    printf("[dtd_test_cuda][memset][%s] A(%d) = %p, A(%d)[0] = %d .. A(%d)[nb-1] = %d\n",
           mode_to_string(mode), (mt-1)*nt + myrank, data_ptr,
           (mt-1)*nt + myrank, data_ptr[0],
           (mt-1)*nt + myrank, data_ptr[nb - 1]);

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    memset_tc = parsec_dtd_create_task_class(tp, "memset",
                                             PASSED_BY_REF, PARSEC_INOUT | TILE_FULL,
                                             sizeof(int), PARSEC_VALUE,
                                             sizeof(int), PARSEC_VALUE,
                                             PARSEC_DTD_ARG_END);
    /* First added is tested first */
    if( mode & WITH_GPU_TASK )
        parsec_dtd_task_class_add_chore(tp, memset_tc, PARSEC_DEV_CUDA, cuda_memset_task_fn);
    if( mode & WITH_CPU_TASK )
        parsec_dtd_task_class_add_chore(tp, memset_tc, PARSEC_DEV_CPU, memset_task_fn);

    for( int rank = 0; rank < world; ++rank ) {
        int prio = 1;
        int device;

        for(int i = 0; i < mt; i++) {
            if( (mode & WITH_CPU_TASK) && (mode & WITH_GPU_TASK) ) {
                device = i % 2 == 0 ? PARSEC_DEV_CPU : PARSEC_DEV_CUDA;
            } else if( mode & WITH_CPU_TASK ) {
                device = PARSEC_DEV_CPU;
            } else {
                assert( mode & WITH_GPU_TASK );
                device = PARSEC_DEV_CUDA;
            }
            key = A->data_key(A, i*nt + myrank, 0);
            parsec_dtd_insert_task_with_task_class(tp, memset_tc, prio, device,
                                                   PARSEC_PUSHOUT, PARSEC_DTD_TILE_OF_KEY(A, key),
                                                   PARSEC_DTD_EMPTY_FLAG, &nb,
                                                   PARSEC_AFFINITY, &rank,
                                                   PARSEC_DTD_ARG_END);
        }
    }

    parsec_dtd_data_flush_all(tp, A);

    // Wait for task completion
    perr = parsec_dtd_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    perr = 0;
    for( int i = 0; i < mt; i++ ) {
        key = A->data_key(A, i*nt + myrank, 0);
        parsec_data = A->data_of_key(A, key);
        parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
        data_ptr = (unsigned int *)parsec_data_copy_get_ptr(parsec_data_copy);
        for( int j = 0; j < nb; j++ ) {
            if( data_ptr[j] != 0xFFFFFFFF ) {
                parsec_output(0,
                              "*** Error: on rank %d, A(%d)[%d] = %x, was set to %x before task, expected "
                              "0xFFFFFFFF after task\n",
                              myrank, i*nt+myrank, j, data_ptr[j], unique_id(myrank, i, mt, j, nb));
                error_count++;
                perr++;
            }
        }
    }
    if( 0 == perr )
        printf("[dtd_test_cuda][memset][%s] data_ptr = %p, data_ptr[0] = %x .. data_ptr[nb-1] = %x\n",
               mode_to_string(mode), data_ptr, data_ptr[0], data_ptr[nb - 1]);

    // Cleanup data and parsec data structures
    parsec_type_free(&adt->opaque_dtt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec_context, TILE_FULL);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);

    parsec_dtd_task_class_release(tp, memset_tc );

    parsec_taskpool_free(tp);

    return error_count;
}

int read_task_fn(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    unsigned int *data;
    int rank, i, mt, nb, expect_memset;

    parsec_dtd_unpack_args(this_task, &data, &rank, &i, &mt, &nb, &expect_memset);

    for(int j = 0; j < nb; j++) {
        if( expect_memset && (data[j] != 0xFFFFFFFF) ) {
            parsec_output(0,
                          "*** Error: element %d of task %d, %d is 0x%x. Was set to 0x%x, expecting 0xFFFFFF after "
                          "cudaMemset\n",
                          j, i, rank, data[j], unique_id(rank, i, mt, j, nb));
            parsec_atomic_fetch_inc_int32(&error_count);
        }
        if( !expect_memset && (data[j] != unique_id(rank, i, mt, j, nb)) ) {
            parsec_output(0,
                          "*** Error: element %d of task %d, %d is 0x%x expecting 0x%x\n",
                          j, i, rank, data[j], unique_id(rank, i, mt, j, nb));
            parsec_atomic_fetch_inc_int32(&error_count);
        }
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int test_cuda_memset_and_read(int world, int myrank, parsec_context_t *parsec_context, int mode)
{
    // Error code return by parsec routines
    int perr;

    // Tile size
    int nb = 10;
    // Total number of tiles
    int mt = 16;
    int nt = world;

    parsec_task_class_t *memset_tc;

    error_count = 0;

    printf("[dtd_test_cuda][memset_and_read]\n");

    // Create new DTD taskpool
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_arena_datatype_t *adt = parsec_dtd_create_arena_datatype(parsec_context, &TILE_FULL);
    parsec_add2arena_rect(adt, parsec_datatype_int32_t, nb, 1, nb);

    parsec_tiled_matrix_t *dcA;
    dcA = create_and_distribute_data(myrank, world, nb, nt*mt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_key_t key;
    parsec_data_copy_t *parsec_data_copy;
    parsec_data_t *parsec_data;
    unsigned int *data_ptr;

    for(int i = 0; i < mt; i++) {
        key = A->data_key(A, i*nt+myrank, 0);
        parsec_data = A->data_of_key(A, key);
        parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
        data_ptr = (unsigned int *)parsec_data_copy_get_ptr(parsec_data_copy);
        for( int j = 0; j < nb; j++ )
            data_ptr[j] = unique_id(myrank, i, mt, j, nb);
    }

    printf("[dtd_test_cuda][memset][%s] A(%d) = %p, A(%d)[0] = %d .. A(%d)[nb-1] = %d\n",
           mode_to_string(mode),
           (mt-1)*nt+myrank, data_ptr, (mt-1)*nt+myrank, data_ptr[0], (mt-1)*nt+myrank, data_ptr[nb-1]);

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    memset_tc = parsec_dtd_create_task_class(tp, "memset",
                                             PASSED_BY_REF, PARSEC_INOUT | TILE_FULL,
                                             sizeof(int), PARSEC_VALUE,
                                             sizeof(int), PARSEC_VALUE,
                                             PARSEC_DTD_ARG_END);
    if( mode & WITH_GPU_TASK )
        parsec_dtd_task_class_add_chore(tp, memset_tc, PARSEC_DEV_CUDA, cuda_memset_task_fn);
    if( mode & WITH_CPU_TASK )
        parsec_dtd_task_class_add_chore(tp, memset_tc, PARSEC_DEV_CPU, memset_task_fn);

    for( int rank = 0; rank < world; ++rank ) {
        int prio = 1;
        int expect_memset = 1;
        for(int i = 0; i < mt; i++) {
            key = A->data_key(A, i*nt+myrank, 0);
            parsec_dtd_insert_task_with_task_class(tp, memset_tc, prio, PARSEC_DEV_ALL,
                                                   PARSEC_PUSHOUT, PARSEC_DTD_TILE_OF_KEY(A, key),
                                                   PARSEC_DTD_EMPTY_FLAG, &nb,
                                                   PARSEC_AFFINITY, &rank,
                                                   PARSEC_DTD_ARG_END);
        }
        for(int i = 0; i < mt; i++) {
            key = A->data_key(A, i*nt+myrank, 0);
            parsec_dtd_insert_task(tp, read_task_fn, prio, PARSEC_DEV_CPU, "Read",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key), PARSEC_INPUT | TILE_FULL,
                                   sizeof(int), &rank, PARSEC_VALUE | PARSEC_AFFINITY,
                                   sizeof(int), &i, PARSEC_VALUE,
                                   sizeof(int), &mt, PARSEC_VALUE,
                                   sizeof(int), &nb, PARSEC_VALUE,
                                   sizeof(int), &expect_memset, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
        }
    }

    parsec_dtd_data_flush_all(tp, A);

    // Wait for task completion
    perr = parsec_dtd_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    printf("[dtd_test_cuda][memset] data_ptr = %p, data_ptr[0] = %d, data_ptr[1] = %d\n",
           data_ptr, data_ptr[0], data_ptr[1]);

    // Cleanup data and parsec data structures
    parsec_type_free(&adt->opaque_dtt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec_context, TILE_FULL);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);

    parsec_dtd_task_class_release(tp, memset_tc);

    parsec_taskpool_free(tp);

    return error_count;

}

int write_task_fn(parsec_execution_stream_t *es,
                  parsec_task_t *this_task)
{
    (void)es;
    unsigned int *data;
    int nb, rank, i, mt;

    parsec_dtd_unpack_args(this_task, &data, &nb, &rank, &i, &mt);

    for(int j = 0; j < nb; j++)
        data[j] = unique_id(rank, i, mt, j, nb);

    return PARSEC_HOOK_RETURN_DONE;
}

int cuda_read_task_fn(parsec_device_cuda_module_t *cuda_device,
                      parsec_gpu_task_t *gpu_task,
                      parsec_gpu_exec_stream_t *gpu_stream)
{
    int *data;
    void *dev_data;
    int rank;
    int nb;
    parsec_cuda_exec_stream_t* cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;

    parsec_task_t *this_task = gpu_task->ec;

    (void)cuda_device;

    parsec_dtd_unpack_args(this_task, &data, &nb, &rank);

    dev_data = parsec_dtd_get_dev_ptr(this_task, 0);

    int devid;
    cudaError_t err = cudaGetDevice(&devid);
    assert(cudaSuccess == err);
    (void)err;

    int *data_cpy = (int *)malloc(nb * sizeof(int));

    cudaMemcpyAsync(data_cpy, dev_data, nb * sizeof(int),
                    cudaMemcpyDeviceToHost, cuda_stream->cuda_stream);
    cudaStreamSynchronize(cuda_stream->cuda_stream);

    free(data_cpy);

    return PARSEC_HOOK_RETURN_DONE;
}


int test_cuda_memset_write_read(int world, int myrank, parsec_context_t *parsec_context, int mode)
{
    // Error code return by parsec routines
    int perr;

    // Tile size
    int nb = 10;
    // Total number of tiles
    int mt = 16;
    int nt = world;

    error_count = 0;

    printf("[dtd_test_cuda][memset_write_read]\n");

    // Create new DTD taskpool
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_arena_datatype_t *adt = parsec_dtd_create_arena_datatype(parsec_context, &TILE_FULL);
    parsec_add2arena_rect(adt, parsec_datatype_int32_t, nb, 1, nb);

    parsec_tiled_matrix_t *dcA;
    dcA = create_and_distribute_data(myrank, world, nb, nt*mt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_key_t key;
    parsec_data_copy_t *parsec_data_copy;
    parsec_data_t *parsec_data;
    unsigned int *data_ptr;

    parsec_task_class_t *memset_tc, *cudaread_tc;

    for(int i = 0; i < mt; i++) {
        key = A->data_key(A, i*nt+myrank, 0);
        parsec_data = A->data_of_key(A, key);
        parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
        data_ptr = (unsigned int *)parsec_data_copy_get_ptr(parsec_data_copy);
        for( int j = 0; j < nb; j++ ) {
            data_ptr[j] = unique_id(myrank, i, mt, j, nb);
        }
    }

    printf("[dtd_test_cuda][memset] A(%d) = %p, A(%d)[0] = %d .. A(%d)[nb-1] = %d\n",
           (mt-1)*nt+myrank, data_ptr,
           (mt-1)*nt+myrank, data_ptr[0],
           (mt-1)*nt+myrank, data_ptr[nb-1]);

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    memset_tc = parsec_dtd_create_task_class(tp, "memset",
                                             PASSED_BY_REF, PARSEC_INOUT | TILE_FULL,
                                             sizeof(int), PARSEC_VALUE,
                                             sizeof(int), PARSEC_VALUE,
                                             PARSEC_DTD_ARG_END);
    if( mode & WITH_GPU_TASK )
        parsec_dtd_task_class_add_chore(tp, memset_tc, PARSEC_DEV_CUDA, cuda_memset_task_fn);
    if( mode & WITH_CPU_TASK )
        parsec_dtd_task_class_add_chore(tp, memset_tc, PARSEC_DEV_CPU, memset_task_fn);

    cudaread_tc = parsec_dtd_create_task_class(tp, "cudaread",
                                               PASSED_BY_REF, PARSEC_INPUT | TILE_FULL,
                                               sizeof(int), PARSEC_VALUE,
                                               sizeof(int), PARSEC_VALUE,
                                               PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, cudaread_tc, PARSEC_DEV_CUDA, cuda_read_task_fn);

    for( int rank = 0; rank < world; ++rank ) {
        int prio = 1;
        int expect_memset = 0;

        for(int i = 0; i < mt; i++) {
            key = A->data_key(A, i*nt+myrank, 0);

            parsec_dtd_insert_task(tp, read_task_fn, prio, PARSEC_DEV_CPU, "Read",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key), PARSEC_INPUT | TILE_FULL,
                                   sizeof(int), &rank, PARSEC_VALUE | PARSEC_AFFINITY,
                                   sizeof(int), &i, PARSEC_VALUE,
                                   sizeof(int), &mt, PARSEC_VALUE,
                                   sizeof(int), &nb, PARSEC_VALUE,
                                   sizeof(int), &expect_memset, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);

            parsec_dtd_insert_task_with_task_class(tp, memset_tc, prio, PARSEC_DEV_ALL,
                                                   PARSEC_PUSHOUT, PARSEC_DTD_TILE_OF_KEY(A, key),
                                                   PARSEC_DTD_EMPTY_FLAG, &nb,
                                                   PARSEC_AFFINITY, &rank,
                                                   PARSEC_DTD_ARG_END);

            parsec_dtd_insert_task(tp, write_task_fn, prio, PARSEC_DEV_CPU, "Write",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key), /* INPUT */
                                   PARSEC_INOUT | TILE_FULL | PARSEC_PULLIN,
                                   sizeof(int), &nb, PARSEC_VALUE,
                                   sizeof(int), &rank, PARSEC_VALUE | PARSEC_AFFINITY,
                                   sizeof(int), &i, PARSEC_VALUE,
                                   sizeof(int), &mt, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);

            parsec_dtd_insert_task_with_task_class(tp, cudaread_tc, prio, PARSEC_DEV_ALL,
                                                   PARSEC_INPUT, PARSEC_DTD_TILE_OF_KEY(A, key),
                                                   PARSEC_DTD_EMPTY_FLAG, &nb,
                                                   PARSEC_AFFINITY, &rank,
                                                   PARSEC_DTD_ARG_END);
        }
    }

    parsec_dtd_data_flush_all(tp, A);

    // Wait for task completion
    perr = parsec_dtd_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    printf("[dtd_test_cuda][memset] data_ptr = %p, data_ptr[0] = %d, data_ptr[1] = %d\n",
           data_ptr, data_ptr[0], data_ptr[1]);

    parsec_dtd_task_class_release(tp, memset_tc);
    parsec_dtd_task_class_release(tp, cudaread_tc);

    // Cleanup data and parsec data structures
    parsec_type_free(&adt->opaque_dtt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec_context, TILE_FULL);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);

    parsec_taskpool_free(tp);

    return error_count;
}

int get_nb_cuda_devices()
{
    int nb = 0;

    for( int dev = 0; dev < (int)parsec_nb_devices; dev++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }

    return nb;
}

int *get_cuda_device_index()
{
    int *dev_index = NULL;

    dev_index = (int *)malloc(parsec_nb_devices * sizeof(int));
    int i = 0;
    for( int dev = 0; dev < (int)parsec_nb_devices; dev++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[i++] = device->device_index;
        }
    }

    return dev_index;
}

int cuda_cpy_task_fn(parsec_device_cuda_module_t *cuda_device,
                     parsec_gpu_task_t *gpu_task,
                     parsec_gpu_exec_stream_t *gpu_stream)
{
    int *data_0, *data_1;
    void *dev_data_0, *dev_data_1;
    int rank;
    int nb;
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;

    (void)cuda_device;

    parsec_task_t *this_task = gpu_task->ec;

    parsec_dtd_unpack_args(this_task, &data_0, &data_1, &nb, &rank);

    dev_data_0 = parsec_dtd_get_dev_ptr(this_task, 0);
    dev_data_1 = parsec_dtd_get_dev_ptr(this_task, 1);

    int devid;
    cudaError_t err = cudaGetDevice(&devid);
    assert(cudaSuccess == err);
    (void)err;

    printf("[cuda_cpy_task_fn] devid = %d, data_0_cpu = %p, data_0_gpu = %p\n", devid, data_0, (void *)dev_data_0);

    cudaMemcpyAsync(dev_data_1, dev_data_0, nb * sizeof(int),
                    cudaMemcpyDeviceToDevice, cuda_stream->cuda_stream);

    return PARSEC_HOOK_RETURN_DONE;
}


int test_cuda_multiple_devices(int world, int myrank, parsec_context_t *parsec_context)
{
    // Error code return by parsec routines
    int perr;

    // Tile size
    int nb = 1000;
    // Total number of tiles
    int mt = 1;
    int nt = world;

    int num_devices = get_nb_cuda_devices();
    printf("[dtd_test_cuda][multiple_devices] num_devices = %d\n", num_devices);

    // Make sure we have multiple devices
    if(num_devices < 2) return -1;

    int *cuda_dev_index = get_cuda_device_index();
    assert(NULL != cuda_dev_index);

    // Create new DTD taskpool
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_task_class_t *cudaread_tc, *cudacpy_tc;

    parsec_arena_datatype_t *adt = parsec_dtd_create_arena_datatype(parsec_context, &TILE_FULL);
    parsec_add2arena_rect(adt, parsec_datatype_int32_t, nb, 1, nb);

    parsec_tiled_matrix_t *dcA;
    dcA = create_and_distribute_data(myrank, world, nb, 3*nt*mt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_key_t key;
    parsec_data_copy_t *parsec_data_copy;
    parsec_data_t *parsec_data;
    int *data_dev_0_ptr, *data_dev_1_ptr, *data_dev_2_ptr;

    key = A->data_key(A, 0, 0);
    parsec_data = A->data_of_key(A, key);
    parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
    data_dev_0_ptr = (int *)parsec_data_copy_get_ptr(parsec_data_copy);
    data_dev_0_ptr[0] = 1;

    key = A->data_key(A, 1, 0);
    parsec_data = A->data_of_key(A, key);
    parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
    data_dev_1_ptr = (int *)parsec_data_copy_get_ptr(parsec_data_copy);
    data_dev_1_ptr[0] = 1;

    key = A->data_key(A, 2, 0);
    parsec_data = A->data_of_key(A, key);
    parsec_data_copy = parsec_data_get_copy(parsec_data, 0);
    data_dev_2_ptr = (int *)parsec_data_copy_get_ptr(parsec_data_copy);
    data_dev_2_ptr[0] = 1;

    printf("[dtd_test_cuda][multiple_devices] data_dev_0_ptr[0] = %d, data_dev_1_ptr[0] = %d\n",
           data_dev_0_ptr[0], data_dev_1_ptr[0]);

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    cudaread_tc = parsec_dtd_create_task_class(tp, "cudaread",
                                               PASSED_BY_REF, PARSEC_INPUT | TILE_FULL,
                                               sizeof(int), PARSEC_VALUE,
                                               sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                               PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, cudaread_tc, PARSEC_DEV_CUDA, cuda_read_task_fn);

    cudacpy_tc = parsec_dtd_create_task_class(tp, "cudacpy",
                                              PASSED_BY_REF, PARSEC_INPUT | TILE_FULL,
                                              PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_PUSHOUT,
                                              sizeof(int), PARSEC_VALUE,
                                              sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                              PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, cudacpy_tc, PARSEC_DEV_CUDA, cuda_cpy_task_fn);

    for( int rank = 0; rank < world; ++rank ) {
        int prio = 1;

        for(int i = 0; i < mt; i+=3) {
            parsec_data_key_t key_0 = A->data_key(A, i*nt+myrank, 0);
            uint32_t rank_of_key_0 = A->rank_of_key(A, key_0);
            printf("[dtd_test_cuda][multiple_devices] key_0 = %lu, rank_of_key_0 = %d\n", key_0, rank_of_key_0);
            parsec_dtd_tile_t *tile_0 = parsec_dtd_tile_of(A, key_0);

            parsec_advise_data_on_device(
                                         tile_0->data_copy->original, cuda_dev_index[0],
                                         PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);

            parsec_data_key_t key_1 = A->data_key(A, (i + 1)*nt+myrank, 0);
            uint32_t rank_of_key_1 = A->rank_of_key(A, key_1);
            printf("[dtd_test_cuda][multiple_devices] key_1 = %lu, rank_of_key_1 = %d\n", key_1, rank_of_key_1);
            parsec_dtd_tile_t *tile_1 = parsec_dtd_tile_of(A, key_1);

            parsec_advise_data_on_device(
                                         tile_1->data_copy->original, cuda_dev_index[1],
                                         PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);

            parsec_data_key_t key_2 = A->data_key(A, (i + 2)*nt+myrank, 0);
            uint32_t rank_of_key_2 = A->rank_of_key(A, key_2);
            printf("[dtd_test_cuda][multiple_devices] key_2 = %lu, rank_of_key_2 = %d\n", key_2, rank_of_key_2);
            parsec_dtd_tile_t *tile_2 = parsec_dtd_tile_of(A, key_2);

            parsec_advise_data_on_device(
                                         tile_2->data_copy->original, cuda_dev_index[0],
                                         PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);

            parsec_dtd_insert_task(
                                   tp, write_task_fn, prio, PARSEC_DEV_CPU, "Write",
                                   PASSED_BY_REF, tile_0, /* INPUT */ PARSEC_INOUT | TILE_FULL /*| PULLIN*/,
                                   sizeof(int), &nb, PARSEC_VALUE,
                                   sizeof(int), &rank, PARSEC_VALUE | PARSEC_AFFINITY,
                                   sizeof(int), &i, PARSEC_VALUE,
                                   sizeof(int), &mt, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);

            parsec_dtd_insert_task_with_task_class(
                                                   tp, cudaread_tc, prio, PARSEC_DEV_ALL,
                                                   PARSEC_INPUT, tile_0,
                                                   PARSEC_DTD_EMPTY_FLAG, &nb,
                                                   PARSEC_DTD_EMPTY_FLAG, &rank,
                                                   PARSEC_DTD_ARG_END);

            parsec_dtd_insert_task_with_task_class(
                                                   tp, cudacpy_tc, prio, PARSEC_DEV_ALL,
                                                   PARSEC_INPUT, tile_0,
                                                   PARSEC_PUSHOUT, tile_2,
                                                   PARSEC_DTD_EMPTY_FLAG, &nb,
                                                   PARSEC_DTD_EMPTY_FLAG, &rank,
                                                   PARSEC_DTD_ARG_END);
        }
    }

    /* cudaDeviceSynchronize(); */

    parsec_dtd_data_flush_all(tp, A);

    // Wait for task completion
    perr = parsec_dtd_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_dtd_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    printf("[dtd_test_cuda][multiple_devices] data_dev_0_ptr[0] = %d, data_dev_1_ptr[0] = %d, data_dev_2_ptr[0] = %d\n",
           data_dev_0_ptr[0], data_dev_1_ptr[0], data_dev_2_ptr[0]);

    parsec_dtd_task_class_release(tp, cudaread_tc);
    parsec_dtd_task_class_release(tp, cudacpy_tc);

    // Cleanup data and parsec data structures
    parsec_type_free(&adt->opaque_dtt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec_context, TILE_FULL);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);

    parsec_taskpool_free(tp);

    return 0;
}

static int print_test_result(const char *testname, int rc)
{
    const char *green;
    const char *blue;
    const char *red;
    const char *normal;

    if(isatty(1)) {
        green = "\e[32m";
        blue = "\e[36m";
        red = "\e[31m";
        normal = "\e[0m";
    } else {
        green = blue = red = normal = "";
    }
    if(rc < 0) {
        printf("%sTest %s info not run%s\n", blue, testname, normal);
        return 1;
    } else if (rc == 0) {
        printf("%sTest %s succeeded%s\n", green, testname, normal);
        return 0;
    } else {
        printf("%sTest %s failed with return value %d%s\n", red, testname, rc, normal);
        return 1;
    }
}

int main(int argc, char **argv)
{
    int ret = 0, rc;
    parsec_context_t *parsec_context = NULL;
    int rank, world;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    // Number of CPU cores involved
    int ncores = -1; // Use all available cores
    parsec_context = parsec_init(ncores, &argc, &argv);

    rc = !(get_nb_cuda_devices() >= 1);
    print_test_result("Have CUDA accelerators", rc);
    if(rc != 0) {
        parsec_fini(&parsec_context);
        return -1;
    }

    rc = test_cuda_print_info(world, rank, parsec_context);
    ret += print_test_result("cuda info", rc);


    rc = test_cuda_memset(world, rank, parsec_context, WITH_CPU_TASK);
    ret += print_test_result("memset (CPU only)", rc);

    rc = test_cuda_memset(world, rank, parsec_context, WITH_GPU_TASK);
    ret += print_test_result("memset (GPU only)", rc);

    rc = test_cuda_memset(world, rank, parsec_context, WITH_CPU_TASK|WITH_GPU_TASK);
    ret += print_test_result("memset (alternating CPU and GPU)", rc);


    rc = test_cuda_memset_and_read(world, rank, parsec_context, WITH_GPU_TASK);
    ret += print_test_result("cuda memset and read (GPU Only)", rc);

    rc = test_cuda_memset_and_read(world, rank, parsec_context, WITH_CPU_TASK);
    ret += print_test_result("cuda memset and read (CPU Only)", rc);

    rc = test_cuda_memset_and_read(world, rank, parsec_context, WITH_CPU_TASK | WITH_GPU_TASK);
    ret += print_test_result("cuda memset and read (both CPU and GPU)", rc);


    rc = test_cuda_memset_write_read(world, rank, parsec_context, WITH_GPU_TASK);
    ret += print_test_result("cuda write and read (GPU Only)", rc);

    rc = test_cuda_memset_write_read(world, rank, parsec_context, WITH_CPU_TASK);
    ret += print_test_result("cuda write and read (CPU Only)", rc);

    rc = test_cuda_memset_write_read(world, rank, parsec_context, WITH_CPU_TASK | WITH_GPU_TASK);
    ret += print_test_result("cuda write and read (both CPU and GPU)", rc);


    rc = test_cuda_multiple_devices(world, rank, parsec_context);
    ret += print_test_result("cuda multiple devices", rc);

    parsec_fini(&parsec_context);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    return ret;
}
