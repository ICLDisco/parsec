/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/*
 * PARSEC_ATOMIC_WRITE behaves like PARSEC_OUTPUT towards the accesses that come
 * before and after it, and the tasks that use it consecutively run concurrently.
 *
 * One tile is accessed through the following sequence, and every task checks
 * from the inside what it is allowed to observe:
 *
 *   INOUT   initialize
 *   INPUT   readers          -- must see the initial value, and no writer running
 *   ATOMIC  group A          -- must not start before the readers are done
 *   INPUT   readers          -- must see every write of group A, and no writer running
 *   ATOMIC  group B          -- must not start before the readers are done
 *   INOUT   check            -- must see every write of groups A and B, and no writer running
 *   ATOMIC  group C
 *   INOUT   check            -- must see every write of group C
 *
 * With more than one rank, the tile is owned by rank 0, where all the ATOMIC_WRITE tasks
 * run, and tasks placed on rank 1 (through a second tile) read and update it right after
 * a group: they must see every write of the group.
 *
 * The tasks of a group must really overlap (when there is more than one core),
 * which a final counter alone cannot tell from a serialized execution.
 * Each task of a group writes its own slot of the tile, and increments a counter
 * atomically: the two ways the access type is meant to be used.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>

#include "tests/tests_data.h"
#include "tests/tests_timing.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/sys/atomic.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#define NB_SLOTS 28
#define NB_READERS 4
#define TILE_SIZE_IN_INT32 32

static int TILE_FULL;  /* the arena datatype of a tile: needed to move it between ranks */

/* The content of the tile */
typedef struct {
    int32_t counter;
    int32_t spare[3];
    int32_t slot[NB_SLOTS];
} shared_t;

/* What the tasks observe about each other, outside of the data they are ordered on */
static volatile int32_t readers_running = 0;
static volatile int32_t writers_running = 0;
static volatile int32_t max_writers_running = 0;
static volatile int32_t overlap_seen = 0;
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

/* Body of the tasks used to check that misuses of ATOMIC_WRITE are rejected: never executed. */
int
task_misuse(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;
    return PARSEC_HOOK_RETURN_DONE;
}

/* INOUT: start from a known state */
int
task_initialize(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;

    parsec_dtd_unpack_args(this_task, &shared);
    memset(shared, 0, sizeof(*shared));
    return PARSEC_HOOK_RETURN_DONE;
}

/* INPUT: the writers of the groups before and after must not run alongside */
int
task_read(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int expected_counter, id;

    parsec_dtd_unpack_args(this_task, &shared, &expected_counter, &id);

    (void)parsec_atomic_fetch_inc_int32(&readers_running);
    if( 0 != writers_running )
        report_error("reader %d (expecting %d) started while %d ATOMIC_WRITE tasks were running",
                     id, expected_counter, writers_running);
    usleep(1000);  /* give an early writer the time to show up */
    if( expected_counter != shared->counter )
        report_error("reader %d expected the counter to be %d, it is %d",
                     id, expected_counter, shared->counter);
    if( 0 != writers_running )
        report_error("%d ATOMIC_WRITE tasks were running while reader %d was",
                     writers_running, id);
    (void)parsec_atomic_fetch_dec_int32(&readers_running);
    return PARSEC_HOOK_RETURN_DONE;
}

/* ATOMIC_WRITE: concurrent with the other tasks of the group, and with nothing else */
int
task_atomic_write(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int slot;
    int32_t running, previous_max;

    parsec_dtd_unpack_args(this_task, &shared, &slot);

    if( 0 != readers_running )
        report_error("ATOMIC_WRITE task %d started while %d readers were running", slot, readers_running);

    running = parsec_atomic_fetch_inc_int32(&writers_running) + 1;
    do {
        previous_max = max_writers_running;
    } while( running > previous_max && !parsec_atomic_cas_int32(&max_writers_running, previous_max, running) );

    if( check_overlap && !overlap_seen ) {
        /* Wait for another task of the group, up to a generous timeout: if the tasks were
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

/* INOUT: everything written by the groups before has to be there, and the groups are over */
int
task_check(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    int expected_counter, nb_slots, new_counter;

    parsec_dtd_unpack_args(this_task, &shared, &expected_counter, &nb_slots, &new_counter);

    if( 0 != writers_running )
        report_error("checking task started while %d ATOMIC_WRITE tasks were running", writers_running);
    if( expected_counter != shared->counter )
        report_error("the counter should be %d once the group is over, it is %d",
                     expected_counter, shared->counter);
    for( int slot = 0; slot < nb_slots; slot++ ) {
        if( slot + 1 != shared->slot[slot] )
            report_error("slot %d should be %d once the group is over, it is %d",
                         slot, slot + 1, shared->slot[slot]);
    }
    shared->counter = new_counter;
    return PARSEC_HOOK_RETURN_DONE;
}

/* INPUT or INOUT from another rank: everything written by the group before has to be there.
 * The second flow is only there to place the task. */
int
task_remote_check(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    shared_t *shared;
    void *placement;
    int expected_counter, nb_slots;

    parsec_dtd_unpack_args(this_task, &shared, &placement, &expected_counter, &nb_slots);

    if( expected_counter != shared->counter )
        report_error("rank %d: the counter should be %d once the group is over, it is %d",
                     this_task->taskpool->context->my_rank, expected_counter, shared->counter);
    for( int slot = 0; slot < nb_slots; slot++ ) {
        if( slot + 1 != shared->slot[slot] )
            report_error("rank %d: slot %d should be %d once the group is over, it is %d",
                         this_task->taskpool->context->my_rank, slot, slot + 1, shared->slot[slot]);
    }
    return PARSEC_HOOK_RETURN_DONE;
}

static void
insert_remote_check(parsec_taskpool_t *dtd_tp, parsec_data_collection_t *A, int access,
                    int expected_counter, int nb_slots)
{
    parsec_dtd_insert_task(dtd_tp, task_remote_check, 0, PARSEC_DEV_CPU, "remote_check",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), access | TILE_FULL,
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 1), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                           sizeof(int), &expected_counter, PARSEC_VALUE,
                           sizeof(int), &nb_slots, PARSEC_VALUE,
                           PARSEC_DTD_ARG_END);
}

static void
insert_readers(parsec_taskpool_t *dtd_tp, parsec_data_collection_t *A, int expected_counter)
{
    for( int id = 0; id < NB_READERS; id++ ) {
        parsec_dtd_insert_task(dtd_tp, task_read, 0, PARSEC_DEV_CPU, "reader",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INPUT | TILE_FULL | PARSEC_AFFINITY,
                               sizeof(int), &expected_counter, PARSEC_VALUE,
                               sizeof(int), &id, PARSEC_VALUE,
                               PARSEC_DTD_ARG_END);
    }
}

static void
insert_group(parsec_taskpool_t *dtd_tp, parsec_data_collection_t *A, int first_slot, int nb_tasks)
{
    for( int i = 0; i < nb_tasks; i++ ) {
        int slot = first_slot + i;
        parsec_dtd_insert_task(dtd_tp, task_atomic_write, 0, PARSEC_DEV_CPU, "ATOMIC_WRITE",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_ATOMIC_WRITE | TILE_FULL | PARSEC_AFFINITY,
                               sizeof(int), &slot, PARSEC_VALUE,
                               PARSEC_DTD_ARG_END);
    }
}

static void
insert_check(parsec_taskpool_t *dtd_tp, parsec_data_collection_t *A,
             int expected_counter, int nb_slots, int new_counter)
{
    parsec_dtd_insert_task(dtd_tp, task_check, 0, PARSEC_DEV_CPU, "check",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                           sizeof(int), &expected_counter, PARSEC_VALUE,
                           sizeof(int), &nb_slots, PARSEC_VALUE,
                           sizeof(int), &new_counter, PARSEC_VALUE,
                           PARSEC_DTD_ARG_END);
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int rc;
    parsec_tiled_matrix_t *dcA;

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

    /* "--misuse <kind>" inserts a task that the runtime must reject, see Testings.cmake */
    const char *misuse = NULL;
    if( argc > 2 && 0 == strcmp(argv[1], "--misuse") ) {
        misuse = argv[2];
        cores = 1;
    } else if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    /* Creating parsec context and initializing dtd environment */
    parsec = parsec_init( cores, &argc, &argv );
    check_overlap = (parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_CORES) >= 2);

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    parsec_arena_datatype_t *adt = parsec_matrix_adt_new_rect(parsec_datatype_int32_t,
                                                              TILE_SIZE_IN_INT32, 1, TILE_SIZE_IN_INT32);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    /* One tile, large enough for a shared_t */
    dcA = create_and_distribute_data(rank, world, 8, (NULL != misuse || world > 1) ? 2 : 1);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");
    if( dcA->bsiz * parsec_datadist_getsizeoftype(dcA->mtype) != sizeof(shared_t) ) {
        parsec_fatal("The tile is %d bytes, the test expects %d\n",
                     (int)(dcA->bsiz * parsec_datadist_getsizeoftype(dcA->mtype)), (int)sizeof(shared_t));
    }

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    if( NULL != misuse ) {
        rc = parsec_context_add_taskpool( parsec, dtd_tp );
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        if( 0 == strcmp(misuse, "combined_access") ) {
            /* Access types are exclusive values, they cannot be or-ed together */
            parsec_dtd_insert_task(dtd_tp, task_misuse, 0, PARSEC_DEV_CPU, "misuse",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INPUT | PARSEC_ATOMIC_WRITE | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        } else if( 0 == strcmp(misuse, "two_atomic_flows") ) {
            parsec_dtd_insert_task(dtd_tp, task_misuse, 0, PARSEC_DEV_CPU, "misuse",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_ATOMIC_WRITE | PARSEC_AFFINITY,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 1), PARSEC_ATOMIC_WRITE,
                                   PARSEC_DTD_ARG_END);
        } else if( 0 == strcmp(misuse, "atomic_and_other_flow") ) {
            parsec_dtd_insert_task(dtd_tp, task_misuse, 0, PARSEC_DEV_CPU, "misuse",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_ATOMIC_WRITE | PARSEC_AFFINITY,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INPUT,
                                   PARSEC_DTD_ARG_END);
        } else if( 0 == strcmp(misuse, "atomic_on_other_rank") ) {
            /* The tile 0 is owned by rank 0, the task is placed on the owner of the tile 1 */
            parsec_dtd_insert_task(dtd_tp, task_misuse, 0, PARSEC_DEV_CPU, "misuse",
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_ATOMIC_WRITE,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 1), PARSEC_INOUT | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        } else {
            fprintf(stderr, "unknown misuse '%s'\n", misuse);
            return EXIT_FAILURE;
        }
        fprintf(stderr, "the runtime accepted the misuse '%s'\n", misuse);
        return EXIT_SUCCESS;  /* no error message: the test fails */
    }

    const int group_a = 20, group_b = 4, group_c = 6, group_d = 5;

    parsec_output( 0, "\nChecking PARSEC_ATOMIC_WRITE: groups of %d, %d and %d unordered tasks between "
                   "readers and exclusive writers%s\n\n", group_a, group_b, group_c,
                   check_overlap ? "" : " (single core: overlap is not checked)" );

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    parsec_dtd_insert_task(dtd_tp, task_initialize, 0, PARSEC_DEV_CPU, "initialize",
                           PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, 0), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                           PARSEC_DTD_ARG_END);

    insert_readers(dtd_tp, A, 0);
    insert_group(dtd_tp, A, 0, group_a);
    if( world > 1 ) insert_remote_check(dtd_tp, A, PARSEC_INPUT, group_a, group_a);
    insert_readers(dtd_tp, A, group_a);
    insert_group(dtd_tp, A, group_a, group_b);
    if( world > 1 ) insert_remote_check(dtd_tp, A, PARSEC_INOUT, group_a + group_b, group_a + group_b);
    insert_check(dtd_tp, A, group_a + group_b, group_a + group_b, 1000);
    insert_group(dtd_tp, A, 0, group_c);
    if( world > 1 ) insert_remote_check(dtd_tp, A, PARSEC_INPUT, 1000 + group_c, group_c);
    insert_check(dtd_tp, A, 1000 + group_c, group_c, 0);
    /* the data is flushed right after a run: the flush has to wait for it */
    insert_group(dtd_tp, A, 0, group_d);

    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    const int owner = (rank == A->rank_of_key(A, 0));  /* where the ATOMIC_WRITE tasks ran */
    if( owner ) {
        parsec_data_copy_t *copy = parsec_data_get_copy(A->data_of_key(A, 0), 0);
        shared_t *shared = (shared_t *)parsec_data_copy_get_ptr(copy);
        /* the last check reset the counter, only the last group counts */
        if( group_d != shared->counter ) {
            report_error("the flushed counter should be %d, it is %d", group_d, shared->counter);
        }
        for( int slot = 0; slot < group_d; slot++ ) {
            if( slot + 1 != shared->slot[slot] )
                report_error("the flushed slot %d should be %d, it is %d", slot, slot + 1, shared->slot[slot]);
        }
    }

    if( owner && check_overlap && !overlap_seen ) {
        report_error("the ATOMIC_WRITE tasks never ran concurrently (at most %d at a time)", max_writers_running);
    }

    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_taskpool_free( dtd_tp );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    if( nb_errors > 0 ) {
        fprintf(stderr, "dtd_test_atomic_write: %d error(s)\n", (int)nb_errors);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
