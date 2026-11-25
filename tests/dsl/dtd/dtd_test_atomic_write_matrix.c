/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/*
 * PARSEC_ATOMIC_WRITE runs followed by every kind of access, wherever the run and the access execute
 * (owner of the data or not, CPU or GPU), with operations the test controls and validates. The tasks
 * of a run write their own slot and count in a shared counter (read first: an update in place), and
 * the tasks that follow check everything the run did. Each case is:
 *
 *   init (CPU task on the init rank) -> run of NB_RUN ATOMIC_WRITE tasks -> follower -> flush
 *
 * with the init and the run on the owner (rank 0) or on the other rank, the run on the CPU or a GPU,
 * and the follower being
 *
 *   - only the flush: the owner gets the data back wherever the run was (the one case where the run
 *     needs no closing task: a CPU run on the owner after a writer on the owner);
 *   - a task that reads (INPUT) or updates (INOUT) the data, on the rank of the run or the other one,
 *     on the CPU or a GPU: it must see the whole run, and the owner the update;
 *   - a second run on the other rank, on the CPU or a GPU, that continues the counter.
 *
 * What the owner holds after the flush is checked as well.
 */

#include "parsec/runtime.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "tests/tests_data.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

#define NB_SLOTS 28
#define NB_RUN 8
#define MARKER 0x5a5a
#define TILE_SIZE_IN_INT32 32

typedef struct {
    int32_t counter;
    int32_t spare[3];  /* spare[0]: the marker the tasks that update the data leave */
    int32_t slot[NB_SLOTS];
} shared_t;

extern void dtd_test_atomic_write_cuda_write(void *dev_data, int slot, int count, void *stream);
extern void dtd_test_atomic_write_matrix_check(void *dev_data, int expected_counter, int expected_slots,
                                               int set_marker, int marker, int *errors, void *stream);

typedef enum {
    FOLLOW_FLUSH = 0,
    FOLLOW_INPUT,
    FOLLOW_INOUT,
    FOLLOW_ATOMIC
} follower_t;

typedef struct {
    int init_rank;
    int run_rank;
    int run_gpu;
    follower_t follower;
    int follower_rank;
    int follower_gpu;
} test_case_t;

static int TILE_FULL;
static volatile int32_t nb_errors = 0;
static volatile int32_t executed[2 * NB_RUN];
static volatile int32_t used_device[2 * NB_RUN];
static int *gpu_errors = NULL;  /* managed memory: where the GPU checks count what they find wrong */
static char current_case[160];

static const char *follower_name[] = { "flush", "input", "inout", "atomic" };

static void
report_error(const char *format, ...)
{
    char message[256];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    fprintf(stderr, "*** Error: [%s] %s\n", current_case, message);
    (void)parsec_atomic_fetch_inc_int32((int32_t *)&nb_errors);
}

static int
count_mismatches(const shared_t *shared, int expected_counter, int expected_slots)
{
    int mismatches = (shared->counter != expected_counter);
    for( int slot = 0; slot < NB_SLOTS; slot++ ) {
        int expected = (slot < expected_slots) ? slot + 1 : 0;
        mismatches += (shared->slot[slot] != expected);
    }
    return mismatches;
}

/* INOUT on the CPU: start from a known state */
static int
task_init(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;

    parsec_dtd_unpack_args(this_task, &shared);
    memset(shared, 0, sizeof(*shared));
    return PARSEC_HOOK_RETURN_DONE;
}

/* ATOMIC_WRITE on the CPU */
static int
task_cpu_atomic_write(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int slot;

    parsec_dtd_unpack_args(this_task, &shared, &slot);
    (void)parsec_atomic_fetch_inc_int32(&shared->counter);
    shared->slot[slot] = slot + 1;
    used_device[slot] = 0;
    (void)parsec_atomic_fetch_inc_int32((int32_t *)&executed[slot]);
    return PARSEC_HOOK_RETURN_DONE;
}

/* ATOMIC_WRITE on a GPU */
static int
task_gpu_atomic_write(parsec_device_gpu_module_t *gpu_device, parsec_gpu_task_t *gpu_task,
                      parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)gpu_stream;
    void *dev_data;
    int slot;

    parsec_dtd_unpack_args(gpu_task->ec, &dev_data, &slot);
    used_device[slot] = gpu_device->super.device_index;
    (void)parsec_atomic_fetch_inc_int32((int32_t *)&executed[slot]);
    dtd_test_atomic_write_cuda_write(dev_data, slot, 1, cuda_stream->cuda_stream);
    cudaStreamSynchronize(cuda_stream->cuda_stream);
    return PARSEC_HOOK_RETURN_DONE;
}

/* INPUT or INOUT on the CPU: check everything the run did, and the INOUT leaves its marker */
static int
task_cpu_check(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int expected_counter, expected_slots, set_marker;

    parsec_dtd_unpack_args(this_task, &shared, &expected_counter, &expected_slots, &set_marker);
    int mismatches = count_mismatches(shared, expected_counter, expected_slots);
    if( mismatches )
        report_error("the check on the CPU found %d mismatches (counter %d, expected %d)",
                     mismatches, shared->counter, expected_counter);
    if( set_marker )
        shared->spare[0] = MARKER;
    return PARSEC_HOOK_RETURN_DONE;
}

/* The same on a GPU */
static int
task_gpu_check(parsec_device_gpu_module_t *gpu_device, parsec_gpu_task_t *gpu_task,
               parsec_gpu_exec_stream_t *gpu_stream)
{
    (void)gpu_device;
    parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)gpu_stream;
    void *dev_data;
    int expected_counter, expected_slots, set_marker;

    parsec_dtd_unpack_args(gpu_task->ec, &dev_data, &expected_counter, &expected_slots, &set_marker);
    dtd_test_atomic_write_matrix_check(dev_data, expected_counter, expected_slots, set_marker, MARKER,
                                       gpu_errors, cuda_stream->cuda_stream);
    cudaStreamSynchronize(cuda_stream->cuda_stream);
    return PARSEC_HOOK_RETURN_DONE;
}

static parsec_task_class_t *
create_atomic_write_class(parsec_taskpool_t *dtd_tp, int gpu)
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(dtd_tp, gpu ? "gpu_atomic_write" : "cpu_atomic_write",
                                                           PASSED_BY_REF, PARSEC_ATOMIC_WRITE | TILE_FULL,
                                                           sizeof(int), PARSEC_VALUE,
                                                           sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                                           PARSEC_DTD_ARG_END);
    if( gpu ) parsec_dtd_task_class_add_chore(dtd_tp, tc, PARSEC_DEV_CUDA, task_gpu_atomic_write);
    else      parsec_dtd_task_class_add_chore(dtd_tp, tc, PARSEC_DEV_CPU, task_cpu_atomic_write);
    return tc;
}

static parsec_task_class_t *
create_check_class(parsec_taskpool_t *dtd_tp, int access, int gpu)
{
    parsec_task_class_t *tc = parsec_dtd_create_task_class(dtd_tp, gpu ? "gpu_check" : "cpu_check",
                                                           PASSED_BY_REF, access | TILE_FULL,
                                                           sizeof(int), PARSEC_VALUE,
                                                           sizeof(int), PARSEC_VALUE,
                                                           sizeof(int), PARSEC_VALUE,
                                                           sizeof(int), PARSEC_VALUE | PARSEC_AFFINITY,
                                                           PARSEC_DTD_ARG_END);
    if( gpu ) parsec_dtd_task_class_add_chore(dtd_tp, tc, PARSEC_DEV_CUDA, task_gpu_check);
    else      parsec_dtd_task_class_add_chore(dtd_tp, tc, PARSEC_DEV_CPU, task_cpu_check);
    return tc;
}

static void
insert_run(parsec_taskpool_t *dtd_tp, parsec_task_class_t *tc, int gpu, parsec_data_collection_t *A,
           int first_slot, int where)
{
    for( int i = 0; i < NB_RUN; i++ ) {
        int slot = first_slot + i;
        parsec_dtd_insert_task_with_task_class(dtd_tp, tc, 0, gpu ? PARSEC_DEV_CUDA : PARSEC_DEV_CPU,
                                               PARSEC_DTD_EMPTY_FLAG, PARSEC_DTD_TILE_OF_KEY(A, 0),
                                               PARSEC_DTD_EMPTY_FLAG, &slot,
                                               PARSEC_DTD_EMPTY_FLAG, &where,
                                               PARSEC_DTD_ARG_END);
    }
}

static void
run_case(parsec_context_t *parsec, int rank, int world, const test_case_t *c)
{
    const int owner = 0;
    const int nb_runs = (FOLLOW_ATOMIC == c->follower) ? 2 : 1;
    const int nb_updates = (FOLLOW_INOUT == c->follower);
    int rc;

    if( FOLLOW_FLUSH == c->follower ) {
        snprintf(current_case, sizeof(current_case), "init on %d, run on %d (%s), then the flush",
                 c->init_rank, c->run_rank, c->run_gpu ? "GPU" : "CPU");
    } else {
        snprintf(current_case, sizeof(current_case), "init on %d, run on %d (%s), then %s on %d (%s)",
                 c->init_rank, c->run_rank, c->run_gpu ? "GPU" : "CPU", follower_name[c->follower],
                 c->follower_rank, c->follower_gpu ? "GPU" : "CPU");
    }
    for( int slot = 0; slot < 2 * NB_RUN; slot++ ) {
        executed[slot] = 0;
        used_device[slot] = -1;
    }

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

    parsec_task_class_t *write_tc = create_atomic_write_class(dtd_tp, c->run_gpu);
    parsec_task_class_t *second_write_tc = NULL;
    parsec_task_class_t *check_tc = NULL;
    if( FOLLOW_ATOMIC == c->follower ) {
        second_write_tc = create_atomic_write_class(dtd_tp, c->follower_gpu);
        if( second_write_tc == write_tc ) report_error("the same task class");
    } else if( FOLLOW_FLUSH != c->follower ) {
        check_tc = create_check_class(dtd_tp, FOLLOW_INOUT == c->follower ? PARSEC_INOUT : PARSEC_INPUT,
                                      c->follower_gpu);
    }

    parsec_dtd_insert_task(dtd_tp, task_init, 0, PARSEC_DEV_CPU, "init",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL,
                           sizeof(int), (void *)&c->init_rank, PARSEC_VALUE | PARSEC_AFFINITY,
                           PARSEC_DTD_ARG_END);

    insert_run(dtd_tp, write_tc, c->run_gpu, A, 0, c->run_rank);

    if( FOLLOW_ATOMIC == c->follower ) {
        insert_run(dtd_tp, second_write_tc, c->follower_gpu, A, NB_RUN, c->follower_rank);
    } else if( FOLLOW_FLUSH != c->follower ) {
        int expected_counter = NB_RUN, expected_slots = NB_RUN, set_marker = nb_updates;
        /* Like after any other GPU write, the owner only gets the update of a GPU task if it pushes it out */
        parsec_dtd_insert_task_with_task_class(dtd_tp, check_tc, 0,
                                               c->follower_gpu ? PARSEC_DEV_CUDA : PARSEC_DEV_CPU,
                                               (nb_updates && c->follower_gpu) ? PARSEC_PUSHOUT : PARSEC_DTD_EMPTY_FLAG,
                                               PARSEC_DTD_TILE_OF_KEY(A, 0),
                                               PARSEC_DTD_EMPTY_FLAG, &expected_counter,
                                               PARSEC_DTD_EMPTY_FLAG, &expected_slots,
                                               PARSEC_DTD_EMPTY_FLAG, &set_marker,
                                               PARSEC_DTD_EMPTY_FLAG, (void *)&c->follower_rank,
                                               PARSEC_DTD_ARG_END);
    }

    parsec_dtd_data_flush_all(dtd_tp, A);

    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    /* What the GPU checks found wrong on this rank */
    if( NULL != gpu_errors && 0 != *gpu_errors ) {
        report_error("the check on a GPU found %d mismatches", *gpu_errors);
        *gpu_errors = 0;
    }

    /* The owner has everything back */
    if( rank == owner ) {
        parsec_data_copy_t *copy = parsec_data_get_copy(A->data_of_key(A, 0), 0);
        shared_t *shared = (shared_t *)parsec_data_copy_get_ptr(copy);
        int mismatches = count_mismatches(shared, nb_runs * NB_RUN, nb_runs * NB_RUN);
        if( mismatches )
            report_error("the owner has %d mismatches once flushed (counter %d, expected %d)",
                         mismatches, shared->counter, nb_runs * NB_RUN);
        if( (nb_updates ? MARKER : 0) != shared->spare[0] )
            report_error("the owner has marker %#x once flushed, expected %#x", shared->spare[0],
                         nb_updates ? MARKER : 0);
    }

    /* Each task of a run executed once, where it had to, on one device */
    for( int run = 0; run < nb_runs; run++ ) {
        int where = run ? c->follower_rank : c->run_rank;
        int gpu = run ? c->follower_gpu : c->run_gpu;
        int device = -1;
        for( int i = 0; i < NB_RUN; i++ ) {
            int slot = run * NB_RUN + i;
            int expected = (rank == where);
            if( executed[slot] != expected )
                report_error("slot %d executed %d times on rank %d, expected %d", slot, (int)executed[slot],
                             rank, expected);
            if( !expected ) continue;
            if( -1 == device ) device = used_device[slot];
            if( device != used_device[slot] )
                report_error("the run %d executed on devices %d and %d", run, device, (int)used_device[slot]);
            if( gpu != (0 != (parsec_mca_device_get(used_device[slot])->type & PARSEC_DEV_CUDA)) )
                report_error("slot %d executed on device %d, expected a %s", slot, (int)used_device[slot],
                             gpu ? "GPU" : "CPU");
        }
    }

    parsec_dtd_task_class_release(dtd_tp, write_tc);
    if( NULL != second_write_tc ) parsec_dtd_task_class_release(dtd_tp, second_write_tc);
    if( NULL != check_tc ) parsec_dtd_task_class_release(dtd_tp, check_tc);
    parsec_dtd_data_collection_fini(A);
    free_data(dcA);
    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_taskpool_free(dtd_tp);
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int rank = 0, world = 1, nb_gpus = 0, nb_cases = 0, verbose = 0;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for( int i = 1; i < argc; i++ ) {
        if( 0 != strcmp(argv[i], "--verbose") ) continue;
        verbose = 1;
        for( int j = i; j < argc; j++ ) argv[j] = argv[j+1];
        argc--;
        break;
    }

    parsec = parsec_init(-1, &argc, &argv);

    for( unsigned int i = 0; i < parsec_nb_devices; i++ ) {
        if( parsec_mca_device_get(i)->type & PARSEC_DEV_CUDA ) nb_gpus++;
    }
    /* Every rank has to run the same cases */
    MPI_Allreduce(MPI_IN_PLACE, &nb_gpus, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if( world < 2 ) {
        parsec_warning("This test needs at least two ranks");
        parsec_fini(&parsec);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if( nb_gpus > 0 ) {
        cudaMallocManaged((void **)&gpu_errors, sizeof(int), cudaMemAttachGlobal);
        *gpu_errors = 0;
    }
    parsec_output(0, "\nChecking PARSEC_ATOMIC_WRITE followed by every access, with %d GPU(s) per rank\n", nb_gpus);

    const int owner = 0;
    const int other = world - 1;
    const int ranks[2] = { owner, other };

    for( int init = 0; init < 2; init++ ) {
        for( int run = 0; run < 2; run++ ) {
            for( int run_gpu = 0; run_gpu <= (nb_gpus > 0); run_gpu++ ) {
                test_case_t c = { .init_rank = ranks[init], .run_rank = ranks[run], .run_gpu = run_gpu };

                /* The flush is all that follows */
                c.follower = FOLLOW_FLUSH;
                c.follower_rank = c.run_rank;
                c.follower_gpu = c.run_gpu;
                if( verbose ) fprintf(stderr, "case %d\n", nb_cases);
                run_case(parsec, rank, world, &c);
                nb_cases++;

                /* A task that reads or updates the data, on the rank of the run or on the other one */
                for( c.follower = FOLLOW_INPUT; c.follower <= FOLLOW_INOUT; c.follower++ ) {
                    for( int where = 0; where < 2; where++ ) {
                        for( int gpu = 0; gpu <= (nb_gpus > 0); gpu++ ) {
                            c.follower_rank = where ? (c.run_rank == owner ? other : owner) : c.run_rank;
                            c.follower_gpu = gpu;
                            if( verbose ) fprintf(stderr, "case %d\n", nb_cases);
                            run_case(parsec, rank, world, &c);
                            nb_cases++;
                        }
                    }
                }

                /* Another run, on the other rank */
                c.follower = FOLLOW_ATOMIC;
                for( int gpu = 0; gpu <= (nb_gpus > 0); gpu++ ) {
                    c.follower_rank = (c.run_rank == owner) ? other : owner;
                    c.follower_gpu = gpu;
                    if( verbose ) fprintf(stderr, "case %d\n", nb_cases);
                    run_case(parsec, rank, world, &c);
                    nb_cases++;
                }
            }
        }
    }
    parsec_output(0, "  %d cases\n", nb_cases);

    if( NULL != gpu_errors ) cudaFree(gpu_errors);
    parsec_fini(&parsec);
    MPI_Finalize();

    if( nb_errors > 0 ) {
        fprintf(stderr, "dtd_test_atomic_write_matrix: %d error(s)\n", (int)nb_errors);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
