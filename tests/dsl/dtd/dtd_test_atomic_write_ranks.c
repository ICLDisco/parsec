/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/*
 * A run of PARSEC_ATOMIC_WRITE tasks is placed on a single rank, which does not have to
 * be the rank that owns the data. Runs placed on different ranks are ordered, and whatever
 * follows a run, a flush included, gets the data from the rank where the run executed.
 *
 * The tile is owned by rank 0, and "far" is the last rank (a rank other than the owner).
 * Every task is placed explicitly with a PARSEC_VALUE | PARSEC_AFFINITY parameter:
 *
 *   INOUT   initialize           on far            counter = BASE
 *   INPUT   readers              on far            -- must see BASE and no writer running
 *   ATOMIC  run A (8 tasks)      on far            -- the prologue waits for the readers,
 *                                                     and the data it sees is the initialized one
 *   INPUT   check                on rank 0         -- must see every write of run A
 *   ATOMIC  4 runs of 4 tasks    on ranks 0, 1, ...  -- every run starts when the previous one
 *                                                     is over, and sees all its writes
 *   INOUT   check                on the rank after the last run -- must see every write of the 4 runs
 *   ATOMIC  run F (4 tasks)      on far
 *   flush of the tile                              -- brings the result of run F to rank 0
 *
 * The tasks of a run must really overlap (when there is more than one core) on the
 * ranks that run more than one task of it.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>

#include "tests/tests_data.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#include <mpi.h>

#define NB_SLOTS 28
#define NB_READERS 4
#define TILE_SIZE_IN_INT32 32
#define BASE 100

static int TILE_FULL;

/* The content of the tile */
typedef struct {
    int32_t counter;
    int32_t spare[3];
    int32_t slot[NB_SLOTS];
} shared_t;

/* What the tasks executing on this rank observe about each other */
static volatile int32_t readers_running = 0;
static volatile int32_t writers_running = 0;
static volatile int32_t overlap_seen = 0;
static volatile int32_t nb_atomic_tasks_here = 0;
static volatile int32_t nb_errors = 0;
static int check_overlap = 0;

static void report_error(const char *format, ...)
{
    va_list args;
    char message[256];

    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    parsec_warning("%s", message);
    (void)parsec_atomic_fetch_inc_int32(&nb_errors);
}

static int
my_rank_of(parsec_task_t *this_task)
{
    return this_task->taskpool->context->my_rank;
}

/* INOUT: start from a known state */
int
task_initialize(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int where;

    parsec_dtd_unpack_args(this_task, &shared, &where);
    if( where != my_rank_of(this_task) )
        report_error("initialize was placed on rank %d, it ran on rank %d", where, my_rank_of(this_task));
    memset(shared, 0, sizeof(*shared));
    shared->counter = BASE;
    return PARSEC_HOOK_RETURN_DONE;
}

/* INPUT: no writer of a run before or after may run alongside, and the data is what was written before */
int
task_read(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int expected_counter, id, where;

    parsec_dtd_unpack_args(this_task, &shared, &expected_counter, &id, &where);
    if( where != my_rank_of(this_task) )
        report_error("reader %d was placed on rank %d, it ran on rank %d", id, where, my_rank_of(this_task));

    (void)parsec_atomic_fetch_inc_int32(&readers_running);
    if( 0 != writers_running )
        report_error("reader %d started while %d ATOMIC_WRITE tasks were running", id, writers_running);
    usleep(1000);  /* give an early writer the time to show up */
    if( expected_counter != shared->counter )
        report_error("reader %d expected the counter to be %d, it is %d", id, expected_counter, shared->counter);
    if( 0 != writers_running )
        report_error("%d ATOMIC_WRITE tasks were running while reader %d was", writers_running, id);
    (void)parsec_atomic_fetch_dec_int32(&readers_running);
    return PARSEC_HOOK_RETURN_DONE;
}

/* ATOMIC_WRITE: concurrent with the other tasks of its run, and with nothing else. It has to see the
 * result of all the runs before it, wherever they executed. */
int
task_atomic_write(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int slot, first_slot, run_size, where;

    parsec_dtd_unpack_args(this_task, &shared, &slot, &first_slot, &run_size, &where);
    if( where != my_rank_of(this_task) )
        report_error("ATOMIC_WRITE task %d was placed on rank %d, it ran on rank %d",
                     slot, where, my_rank_of(this_task));

    if( 0 != readers_running )
        report_error("ATOMIC_WRITE task %d started while %d readers were running", slot, readers_running);

    /* The previous content has to be here: the tasks of the runs before are over */
    if( shared->counter < BASE + first_slot )
        report_error("ATOMIC_WRITE task %d expected a counter of at least %d at its start, it is %d",
                     slot, BASE + first_slot, shared->counter);
    for( int before = 0; before < first_slot; before++ ) {
        if( before + 1 != shared->slot[before] )
            report_error("ATOMIC_WRITE task %d: slot %d of a previous run should be %d, it is %d",
                         slot, before, before + 1, shared->slot[before]);
    }

    (void)parsec_atomic_fetch_inc_int32(&nb_atomic_tasks_here);
    (void)parsec_atomic_fetch_inc_int32(&writers_running);

    if( check_overlap && run_size >= 2 && !overlap_seen ) {
        /* Wait for another task of the run, up to a generous timeout: if the tasks were
         * serialized, nobody ever comes. */
        for( int spin = 0; spin < 20000 && !overlap_seen; spin++ ) {
            if( writers_running >= 2 ) {
                overlap_seen = 1;
                break;
            }
            usleep(100);
        }
    }

    (void)parsec_atomic_fetch_inc_int32(&shared->counter);
    shared->slot[slot] = slot + 1;

    if( 0 != readers_running )
        report_error("%d readers were running while ATOMIC_WRITE task %d was", readers_running, slot);
    (void)parsec_atomic_fetch_dec_int32(&writers_running);
    return PARSEC_HOOK_RETURN_DONE;
}

/* INPUT or INOUT: everything written by the runs before has to be there, and no run is going on here */
int
task_check(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int expected_counter, nb_slots, where;

    parsec_dtd_unpack_args(this_task, &shared, &expected_counter, &nb_slots, &where);
    if( where != my_rank_of(this_task) )
        report_error("a check was placed on rank %d, it ran on rank %d", where, my_rank_of(this_task));

    if( 0 != writers_running )
        report_error("a check started while %d ATOMIC_WRITE tasks were running", writers_running);
    usleep(1000);  /* give an early writer the time to show up */
    if( expected_counter != shared->counter )
        report_error("rank %d: the counter should be %d once the runs are over, it is %d",
                     my_rank_of(this_task), expected_counter, shared->counter);
    for( int slot = 0; slot < nb_slots; slot++ ) {
        if( slot + 1 != shared->slot[slot] )
            report_error("rank %d: slot %d should be %d once the runs are over, it is %d",
                         my_rank_of(this_task), slot, slot + 1, shared->slot[slot]);
    }
    if( 0 != writers_running )
        report_error("%d ATOMIC_WRITE tasks were running while a check was", writers_running);
    return PARSEC_HOOK_RETURN_DONE;
}

static void
insert_run(parsec_taskpool_t *dtd_tp, parsec_data_collection_t *A, int first_slot, int run_size, int where)
{
    for( int i = 0; i < run_size; i++ ) {
        int slot = first_slot + i;
        parsec_dtd_insert_task(dtd_tp, task_atomic_write, 0, PARSEC_DEV_CPU, "ATOMIC_WRITE",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_ATOMIC_WRITE | TILE_FULL,
                               sizeof(int), &slot, PARSEC_VALUE,
                               sizeof(int), &first_slot, PARSEC_VALUE,
                               sizeof(int), &run_size, PARSEC_VALUE,
                               sizeof(int), &where, PARSEC_VALUE | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);
    }
}

static void
insert_check(parsec_taskpool_t *dtd_tp, parsec_data_collection_t *A, int access,
             int expected_counter, int nb_slots, int where)
{
    parsec_dtd_insert_task(dtd_tp, task_check, 0, PARSEC_DEV_CPU, "check",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), access | TILE_FULL,
                           sizeof(int), &expected_counter, PARSEC_VALUE,
                           sizeof(int), &nb_slots, PARSEC_VALUE,
                           sizeof(int), &where, PARSEC_VALUE | PARSEC_AFFINITY,
                           PARSEC_DTD_ARG_END);
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int rank, world, provided, rc;
    int cores = 4;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if( world < 2 ) {
        fprintf(stderr, "dtd_test_atomic_write_ranks needs at least 2 ranks\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if( NULL != argv[1] ) cores = atoi(argv[1]);

    parsec = parsec_init(cores, &argc, &argv);
    check_overlap = (parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_CORES) >= 2);

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    parsec_arena_datatype_t *adt = parsec_matrix_adt_new_rect(parsec_datatype_int32_t,
                                                              TILE_SIZE_IN_INT32, 1, TILE_SIZE_IN_INT32);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    /* One tile per rank, the one that matters (key 0) is owned by rank 0 */
    parsec_tiled_matrix_t *dcA = create_and_distribute_data(rank, world, 8, world);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");
    if( dcA->bsiz * parsec_datadist_getsizeoftype(dcA->mtype) != sizeof(shared_t) ) {
        parsec_fatal("The tile is %d bytes, the test expects %d\n",
                     (int)(dcA->bsiz * parsec_datadist_getsizeoftype(dcA->mtype)), (int)sizeof(shared_t));
    }
    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    const int far = world - 1;
    const int run_a = 8, nb_runs = 4, run_size = 4, run_f = 4;
    const int slots_after_a = run_a, slots_after_runs = run_a + nb_runs * run_size;
    const int total_slots = slots_after_runs + run_f;
    assert(total_slots <= NB_SLOTS);

    parsec_output(0, "\nChecking PARSEC_ATOMIC_WRITE runs placed on ranks other than the owner of the data (%d ranks)\n\n",
                  world);

    rc = parsec_context_add_taskpool(parsec, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    /* The data is first written on a rank that does not own it */
    parsec_dtd_insert_task(dtd_tp, task_initialize, 0, PARSEC_DEV_CPU, "initialize",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL,
                           sizeof(int), &far, PARSEC_VALUE | PARSEC_AFFINITY,
                           PARSEC_DTD_ARG_END);

    /* Readers, then a run on the same rank: it needs a prologue to wait for them */
    for( int id = 0; id < NB_READERS; id++ ) {
        int expected_counter = BASE;
        parsec_dtd_insert_task(dtd_tp, task_read, 0, PARSEC_DEV_CPU, "reader",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INPUT | TILE_FULL,
                               sizeof(int), &expected_counter, PARSEC_VALUE,
                               sizeof(int), &id, PARSEC_VALUE,
                               sizeof(int), &far, PARSEC_VALUE | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);
    }
    insert_run(dtd_tp, A, 0, run_a, far);

    /* The owner reads what the run did somewhere else */
    insert_check(dtd_tp, A, PARSEC_INPUT, BASE + slots_after_a, slots_after_a, 0);

    /* Runs on different ranks: each one starts when the previous one is over */
    for( int run = 0; run < nb_runs; run++ ) {
        insert_run(dtd_tp, A, slots_after_a + run * run_size, run_size, run % world);
    }
    /* The last writer is placed away from the last run, and from the rank of the run that follows */
    const int writer_rank = ((nb_runs - 1) % world + 1) % world;
    insert_check(dtd_tp, A, PARSEC_INOUT, BASE + slots_after_runs, slots_after_runs, writer_rank);

    /* A run on a rank that does not own the data, followed by a flush: the data has to come from there,
     * and not from the last writer, which is on another rank */
    insert_run(dtd_tp, A, slots_after_runs, run_f, far);
    parsec_dtd_data_flush_all(dtd_tp, A);

    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if( rank == (int)A->rank_of_key(A, 0) ) {
        parsec_data_copy_t *copy = parsec_data_get_copy(A->data_of_key(A, 0), 0);
        shared_t *shared = (shared_t *)parsec_data_copy_get_ptr(copy);
        if( BASE + total_slots != shared->counter )
            report_error("the flushed counter should be %d, it is %d", BASE + total_slots, shared->counter);
        for( int slot = 0; slot < total_slots; slot++ ) {
            if( slot + 1 != shared->slot[slot] )
                report_error("the flushed slot %d should be %d, it is %d", slot, slot + 1, shared->slot[slot]);
        }
    }

    /* a rank that executed several tasks of a run saw them overlap */
    if( check_overlap && nb_atomic_tasks_here >= 2 && !overlap_seen )
        report_error("rank %d: the ATOMIC_WRITE tasks of a run never ran concurrently", rank);

    parsec_dtd_data_collection_fini(A);
    free_data(dcA);
    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_taskpool_free(dtd_tp);
    parsec_fini(&parsec);
    MPI_Finalize();

    if( nb_errors > 0 ) {
        fprintf(stderr, "dtd_test_atomic_write_ranks: %d error(s)\n", (int)nb_errors);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
