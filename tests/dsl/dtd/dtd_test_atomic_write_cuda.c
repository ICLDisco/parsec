/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/*
 * PARSEC_ATOMIC_WRITE on GPU: the tasks of a run write the same data concurrently, so they must all
 * execute on the same device. Each scenario has a run of tasks with only a CUDA chore, that write
 * their own slot and update a counter atomically (they read the previous content):
 *
 *   host:  the data is on the host and no device owns it: the run brings it to its device. Without
 *          the pinning, the load balancing would spread the run over the GPUs.
 *   owner: a GPU task initializes the data first, so that GPU owns it and the run follows it.
 *
 * In both, a GPU task that writes the data follows the run: it must see every write, and pushes
 * the data out so that the host can check it too.
 *
 * With more than one rank, other scenarios run, where the run is not on the owner of the data:
 *
 *   remote: rank 0 owns the data, the run executes on the GPU of the last rank, and a GPU task that
 *           pushes the data out verifies the result before the flush brings it home.
 *   remote_switch: half of the run is on the last rank and half on rank 0, both on a GPU. The second
 *           half reads what the first wrote, so the data must leave the GPU of the first rank (only
 *           its epilogue can do that), cross the network, and enter the GPU of rank 0.
 */

#include "parsec/runtime.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime.h>

#include "tests/tests_data.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#define NB_SLOTS 28
#define NB_TASKS 24
#define TILE_SIZE_IN_INT32 32

typedef struct {
    int32_t counter;
    int32_t spare[3];
    int32_t slot[NB_SLOTS];
} shared_t;

extern void dtd_test_atomic_write_cuda_init(void *dev_data, void *stream);
extern void dtd_test_atomic_write_cuda_write(void *dev_data, int slot, int count, void *stream);
extern void dtd_test_atomic_write_cuda_verify(void *dev_data, int expected_counter, int nb_slots, void *stream);

static int TILE_FULL;
static volatile int32_t nb_errors = 0;
static volatile int32_t init_device = -1;
static volatile int32_t verify_device = -1;
static volatile int32_t used_device[NB_SLOTS];

static void
report_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    parsec_output(0, "*** Error: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    (void)parsec_atomic_fetch_inc_int32((int32_t *)&nb_errors);
}

/* INOUT on the host: start from a known state */
static int
task_host_init(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;

    parsec_dtd_unpack_args(this_task, &shared);
    memset(shared, 0, sizeof(*shared));
    return PARSEC_HOOK_RETURN_DONE;
}

/* INOUT on a GPU: that device owns the data from now on */
static int
task_gpu_init(parsec_device_gpu_module_t *gpu_device, parsec_gpu_task_t *gpu_task,
              parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)gpu_stream;
    void *dev_data;

    parsec_dtd_unpack_args(gpu_task->ec, &dev_data);
    init_device = gpu_device->super.device_index;
    dtd_test_atomic_write_cuda_init(dev_data, cuda_stream->cuda_stream);
    cudaStreamSynchronize(cuda_stream->cuda_stream);
    return PARSEC_HOOK_RETURN_DONE;
}

/* ATOMIC_WRITE on a GPU */
static int
task_gpu_atomic_write(parsec_device_gpu_module_t *gpu_device, parsec_gpu_task_t *gpu_task,
                      parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)gpu_stream;
    void *dev_data;
    int slot, count;

    parsec_dtd_unpack_args(gpu_task->ec, &dev_data, &slot, &count);
    used_device[slot] = gpu_device->super.device_index;
    dtd_test_atomic_write_cuda_write(dev_data, slot, count, cuda_stream->cuda_stream);
    cudaStreamSynchronize(cuda_stream->cuda_stream);
    return PARSEC_HOOK_RETURN_DONE;
}

static void
check_content(const shared_t *shared, int expected_counter, const char *where)
{
    if( expected_counter >= 0 && expected_counter != shared->counter )
        report_error("%s: the counter should be %d, it is %d", where, expected_counter, shared->counter);
    for( int slot = 0; slot < NB_TASKS; slot++ ) {
        if( slot + 1 != shared->slot[slot] )
            report_error("%s: slot %d should be %d, it is %d", where, slot, slot + 1, shared->slot[slot]);
    }
}

/* INOUT on a GPU, once the run is over: it has to see every write of the run, which only
 * holds if they all landed on the copy of this device */
static int
task_gpu_verify(parsec_device_gpu_module_t *gpu_device, parsec_gpu_task_t *gpu_task,
                parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)gpu_stream;
    void *dev_data;
    int expected_counter;

    parsec_dtd_unpack_args(gpu_task->ec, &dev_data, &expected_counter);
    verify_device = gpu_device->super.device_index;
    dtd_test_atomic_write_cuda_verify(dev_data, expected_counter, NB_TASKS, cuda_stream->cuda_stream);
    cudaStreamSynchronize(cuda_stream->cuda_stream);
    return PARSEC_HOOK_RETURN_DONE;
}

static void
check_one_device(const char *scenario)
{
    int first = used_device[0];
    int distinct = 0;
    int32_t seen[NB_TASKS];

    for( int slot = 0; slot < NB_TASKS; slot++ ) {
        int known = 0;
        for( int i = 0; i < distinct; i++ ) known |= (seen[i] == used_device[slot]);
        if( !known ) seen[distinct++] = used_device[slot];
    }
    if( first < 0 ) {
        report_error("%s: the run did not execute", scenario);
    } else if( distinct > 1 ) {
        report_error("%s: the tasks of the run executed on %d different devices", scenario, distinct);
    }
    parsec_output(0, "  [%s] the %d tasks of the run executed on device %d\n", scenario, NB_TASKS, first);
}

static void
run_scenario(parsec_context_t *parsec, int rank, int world, int owner_scenario)
{
    const char *scenario = owner_scenario ? "owner" : "host";
    int rc;

    for( int slot = 0; slot < NB_SLOTS; slot++ ) used_device[slot] = -1;
    init_device = -1;
    verify_device = -1;

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    parsec_arena_datatype_t *adt = parsec_matrix_adt_new_rect(parsec_datatype_int32_t,
                                                              TILE_SIZE_IN_INT32, 1, TILE_SIZE_IN_INT32);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    parsec_tiled_matrix_t *dcA = create_and_distribute_data(rank, world, 8, 1);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");
    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    rc = parsec_context_add_taskpool(parsec, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_task_class_t *init_tc = parsec_dtd_create_task_class(dtd_tp, "gpu_init",
                                                                PASSED_BY_REF, PARSEC_INOUT | TILE_FULL,
                                                                PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, init_tc, PARSEC_DEV_CUDA, task_gpu_init);
    parsec_task_class_t *write_tc = parsec_dtd_create_task_class(dtd_tp, "gpu_atomic_write",
                                                                 PASSED_BY_REF, PARSEC_ATOMIC_WRITE | TILE_FULL,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, write_tc, PARSEC_DEV_CUDA, task_gpu_atomic_write);
    parsec_task_class_t *verify_tc = parsec_dtd_create_task_class(dtd_tp, "gpu_verify",
                                                                  PASSED_BY_REF, PARSEC_INOUT | TILE_FULL,
                                                                  sizeof(int), PARSEC_VALUE,
                                                                  PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, verify_tc, PARSEC_DEV_CUDA, task_gpu_verify);

    if( owner_scenario ) {
        parsec_dtd_insert_task_with_task_class(dtd_tp, init_tc, 0, PARSEC_DEV_CUDA,
                                               PARSEC_AFFINITY, PARSEC_DTD_TILE_OF_KEY(A, 0),
                                               PARSEC_DTD_ARG_END);
    } else {
        parsec_dtd_insert_task(dtd_tp, task_host_init, 0, PARSEC_DEV_CPU, "host_init",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);
    }

    for( int slot = 0; slot < NB_TASKS; slot++ ) {
        int count = 1;
        parsec_dtd_insert_task_with_task_class(dtd_tp, write_tc, 0, PARSEC_DEV_CUDA,
                                               PARSEC_AFFINITY, PARSEC_DTD_TILE_OF_KEY(A, 0),
                                               PARSEC_DTD_EMPTY_FLAG, &slot,
                                               PARSEC_DTD_EMPTY_FLAG, &count,
                                               PARSEC_DTD_ARG_END);
    }

    {
        /* The host only gets the result of the run when the exclusive access that follows it
         * pushes it out, like for any other GPU write. */
        int expected_counter = NB_TASKS;
        parsec_dtd_insert_task_with_task_class(dtd_tp, verify_tc, 0, PARSEC_DEV_CUDA,
                                               PARSEC_PUSHOUT | PARSEC_AFFINITY, PARSEC_DTD_TILE_OF_KEY(A, 0),
                                               PARSEC_DTD_EMPTY_FLAG, &expected_counter,
                                               PARSEC_DTD_ARG_END);
    }

    parsec_dtd_data_flush_all(dtd_tp, A);

    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_data_copy_t *copy = parsec_data_get_copy(A->data_of_key(A, 0), 0);
    shared_t *shared = (shared_t *)parsec_data_copy_get_ptr(copy);
    check_content(shared, NB_TASKS, "host after the flush");
    if( 0 != shared->spare[0] )
        report_error("%s: the GPU task that follows the run found %d mismatches (counter %d)",
                     scenario, shared->spare[0], shared->spare[1]);
    check_one_device(scenario);
    if( owner_scenario && verify_device != used_device[0] ) {
        report_error("owner: the run executed on device %d, the exclusive task that follows on device %d",
                     used_device[0], verify_device);
    }
    if( owner_scenario && init_device != used_device[0] ) {
        report_error("owner: the run executed on device %d, the data was owned by device %d",
                     used_device[0], init_device);
    }

    parsec_dtd_task_class_release(dtd_tp, init_tc);
    parsec_dtd_task_class_release(dtd_tp, write_tc);
    parsec_dtd_task_class_release(dtd_tp, verify_tc);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);
    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_taskpool_free(dtd_tp);
}

/* The placement of a task is its last parameter, it is passed by value to run it there */
static void
run_remote_scenario(parsec_context_t *parsec, int rank, int world, int switch_ranks)
{
    const int owner = 0;
    const int far = world - 1;
    const int nb_runs = switch_ranks ? 2 : 1;
    const int half = switch_ranks ? NB_TASKS / 2 : NB_TASKS;
    const char *scenario = switch_ranks ? "remote_switch" : "remote";
    int rc;

    for( int slot = 0; slot < NB_SLOTS; slot++ ) used_device[slot] = -1;
    verify_device = -1;

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    parsec_arena_datatype_t *adt = parsec_matrix_adt_new_rect(parsec_datatype_int32_t,
                                                              TILE_SIZE_IN_INT32, 1, TILE_SIZE_IN_INT32);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    parsec_tiled_matrix_t *dcA = create_and_distribute_data(rank, world, 8, 1);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");
    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    if( rank == owner && A->rank_of_key(A, 0) != owner )
        report_error("%s: the data is not owned by rank %d", scenario, owner);

    rc = parsec_context_add_taskpool(parsec, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_task_class_t *write_tc = parsec_dtd_create_task_class(dtd_tp, "gpu_atomic_write_at",
                                                                 PASSED_BY_REF, PARSEC_ATOMIC_WRITE | TILE_FULL,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 sizeof(int), PARSEC_VALUE,
                                                                 sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                                                 PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, write_tc, PARSEC_DEV_CUDA, task_gpu_atomic_write);
    parsec_task_class_t *verify_tc = parsec_dtd_create_task_class(dtd_tp, "gpu_verify_at",
                                                                  PASSED_BY_REF, PARSEC_INOUT | TILE_FULL,
                                                                  sizeof(int), PARSEC_VALUE,
                                                                  sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                                                  PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(dtd_tp, verify_tc, PARSEC_DEV_CUDA, task_gpu_verify);

    /* The data starts from a known state on the last rank, where the run executes: from the host, so the
     * run has to bring it to its GPU. */
    parsec_dtd_insert_task(dtd_tp, task_host_init, 0, PARSEC_DEV_CPU, "host_init",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL,
                           sizeof(int), &far, PARSEC_VALUE | PARSEC_AFFINITY,
                           PARSEC_DTD_ARG_END);

    /* A run away from the owner, and when the ranks switch a run on the owner after it: the counter only
     * reaches NB_TASKS if the second run saw what the first one left on the GPU of another rank. */
    for( int run = 0; run < nb_runs; run++ ) {
        int where = run ? owner : far;
        for( int i = 0; i < half; i++ ) {
            int slot = run * half + i;
            int count = 1;
            parsec_dtd_insert_task_with_task_class(dtd_tp, write_tc, 0, PARSEC_DEV_CUDA,
                                                   PARSEC_DTD_EMPTY_FLAG, PARSEC_DTD_TILE_OF_KEY(A, 0),
                                                   PARSEC_DTD_EMPTY_FLAG, &slot,
                                                   PARSEC_DTD_EMPTY_FLAG, &count,
                                                   PARSEC_DTD_EMPTY_FLAG, &where,
                                                   PARSEC_DTD_ARG_END);
        }
    }

    {
        int expected_counter = NB_TASKS;
        parsec_dtd_insert_task_with_task_class(dtd_tp, verify_tc, 0, PARSEC_DEV_CUDA,
                                               PARSEC_PUSHOUT, PARSEC_DTD_TILE_OF_KEY(A, 0),
                                               PARSEC_DTD_EMPTY_FLAG, &expected_counter,
                                               PARSEC_DTD_EMPTY_FLAG, &far,
                                               PARSEC_DTD_ARG_END);
    }

    parsec_dtd_data_flush_all(dtd_tp, A);

    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if( rank == owner ) {
        parsec_data_copy_t *copy = parsec_data_get_copy(A->data_of_key(A, 0), 0);
        shared_t *shared = (shared_t *)parsec_data_copy_get_ptr(copy);
        check_content(shared, NB_TASKS, "owner after the flush");
        if( 0 != shared->spare[0] )
            report_error("%s: the GPU task on rank %d found %d mismatches (counter %d)",
                         scenario, far, shared->spare[0], shared->spare[1]);
    }

    /* Each rank only ran its own run, and a run stays on one device */
    {
        int expected_first = 0, expected_end = 0;  /* the slots this rank is expected to have run */
        if( rank == far ) { expected_first = 0; expected_end = half; }
        else if( rank == owner && switch_ranks ) { expected_first = half; expected_end = NB_TASKS; }
        int device = -1;
        for( int slot = 0; slot < NB_TASKS; slot++ ) {
            int mine = (slot >= expected_first && slot < expected_end);
            if( mine != (used_device[slot] >= 0) )
                report_error("%s: rank %d %s slot %d", scenario, rank, mine ? "did not run" : "ran", slot);
            if( used_device[slot] < 0 ) continue;
            if( device < 0 ) device = used_device[slot];
            else if( device != used_device[slot] )
                report_error("%s: rank %d: the run executed on devices %d and %d", scenario, rank, device, used_device[slot]);
        }
        parsec_output(0, "  [%s] rank %d executed slots [%d, %d) on device %d\n",
                      scenario, rank, expected_first, expected_end, device);
        if( (rank == far) != (verify_device >= 0) )
            report_error("%s: the verification %s on rank %d", scenario, rank == far ? "did not run" : "ran", rank);
        if( rank == far && 0 == far )
            report_error("%s: the test needs more than one rank", scenario);
    }

    parsec_dtd_task_class_release(dtd_tp, write_tc);
    parsec_dtd_task_class_release(dtd_tp, verify_tc);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);
    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_taskpool_free(dtd_tp);
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int rank = 0, world = 1, cores = -1, require_gpu = 0;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
#endif

    for( int i = 1; i < argc; i++ ) {
        if( 0 != strcmp(argv[i], "--require-gpu") ) continue;
        require_gpu = 1;
        for( int j = i; j < argc; j++ ) argv[j] = argv[j+1];
        argc--;
        break;
    }

    parsec = parsec_init(cores, &argc, &argv);

    int nb_gpus = 0;
    for( unsigned int i = 0; i < parsec_nb_devices; i++ ) {
        if( parsec_mca_device_get(i)->type & PARSEC_DEV_CUDA ) nb_gpus++;
    }
    if( 0 == nb_gpus ) {
        if( require_gpu ) parsec_warning("This test requires a GPU but none is present");
        parsec_fini(&parsec);
#if defined(PARSEC_HAVE_MPI)
        MPI_Finalize();
#endif
        return require_gpu ? -PARSEC_ERR_DEVICE : EXIT_SUCCESS;
    }
    parsec_output(0, "\nChecking PARSEC_ATOMIC_WRITE on %d GPU(s)\n", nb_gpus);

    if( 1 == world ) {
        run_scenario(parsec, rank, world, 0);
        run_scenario(parsec, rank, world, 1);
    } else {
        run_remote_scenario(parsec, rank, world, 0);
        run_remote_scenario(parsec, rank, world, 1);
    }

    parsec_fini(&parsec);
#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    if( nb_errors > 0 ) {
        fprintf(stderr, "dtd_test_atomic_write_cuda: %d error(s)\n", (int)nb_errors);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
