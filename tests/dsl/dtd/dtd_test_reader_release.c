/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/* Readers must keep a later writer from overtaking them.
 *
 * DTD gates a writer on the number of readers of the copy it is about to
 * overwrite. That count has to balance exactly: every reader is counted once
 * and takes itself out once. This test inserts a long pseudo-random sequence of
 * tasks that read two tiles and write a third, while the workers execute them,
 * and checks three things:
 *
 *   - the final value of every tile against a sequential reference,
 *   - while the bodies run, that no writer overlaps a reader of its tile and
 *     that every task sees the version of every tile that sequential
 *     execution would give it,
 *   - after the taskpool completes, that no copy has readers left over.
 *
 * It only fails when tasks complete while others are still being inserted,
 * that is with several execution streams, because a completed task is
 * recycled for a later insertion.
 */

#include "parsec/runtime.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "tests/tests_data.h"
#include "tests/tests_runtime.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#define NB_TILES                12
#define NB_OPERATIONS           3000
#define NB_ROUNDS               8
#define PERCENT_MIXING          70   /* operations that read two tiles, the rest only write */
#define INITIAL_VALUE(tile)     ((uint32_t)(tile) * 1000u + 7u)

typedef struct {
    int is_mixing;               /* reads two other tiles, otherwise only updates its own */
    int written_tile;
    int first_read_tile;
    int second_read_tile;
    int salt;
    int index;                   /* position in the sequence */
    int version_written;         /* writes to written_tile before this operation */
    int version_first_read;      /* writes to first_read_tile before this operation */
    int version_second_read;
} operation_t;

static operation_t operations[NB_OPERATIONS];
static uint32_t    expected_values[NB_TILES];

/* What the bodies observe while they run. */
static _Atomic int tile_version[NB_TILES];
static _Atomic int tile_active_readers[NB_TILES];
static _Atomic int tile_active_writers[NB_TILES];
static _Atomic int ordering_violations;

static uint32_t
mix_values(uint32_t old_value, uint32_t first, uint32_t second, int salt)
{
    return old_value * 31u + first * 17u + second * 13u + (uint32_t)salt;
}

static uint32_t
update_value(uint32_t old_value, int salt)
{
    return old_value * 3u + (uint32_t)salt;
}

static void
body_enters_read(int tile, int expected_version)
{
    atomic_fetch_add(&tile_active_readers[tile], 1);
    if( 0 != atomic_load(&tile_active_writers[tile]) ||
        expected_version != atomic_load(&tile_version[tile]) ) {
        atomic_fetch_add(&ordering_violations, 1);
    }
}

static void
body_leaves_read(int tile)
{
    atomic_fetch_sub(&tile_active_readers[tile], 1);
}

static void
body_enters_write(int tile, int expected_version)
{
    if( 0 != atomic_fetch_add(&tile_active_writers[tile], 1) ||
        0 != atomic_load(&tile_active_readers[tile]) ||
        expected_version != atomic_load(&tile_version[tile]) ) {
        atomic_fetch_add(&ordering_violations, 1);
    }
}

static void
body_leaves_write(int tile)
{
    atomic_fetch_add(&tile_version[tile], 1);
    atomic_fetch_sub(&tile_active_writers[tile], 1);
}

static int
body_mixing(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    uint32_t *written, *first, *second;
    int salt, index;
    parsec_dtd_unpack_args(this_task, &written, &first, &second, &salt, &index);
    const operation_t *operation = &operations[index];

    body_enters_read(operation->first_read_tile, operation->version_first_read);
    body_enters_read(operation->second_read_tile, operation->version_second_read);
    body_enters_write(operation->written_tile, operation->version_written);
    *written = mix_values(*written, *first, *second, salt);
    body_leaves_write(operation->written_tile);
    body_leaves_read(operation->first_read_tile);
    body_leaves_read(operation->second_read_tile);
    return PARSEC_HOOK_RETURN_DONE;
}

static int
body_updating(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    uint32_t *written;
    int salt, index;
    parsec_dtd_unpack_args(this_task, &written, &salt, &index);
    const operation_t *operation = &operations[index];

    body_enters_write(operation->written_tile, operation->version_written);
    *written = update_value(*written, salt);
    body_leaves_write(operation->written_tile);
    return PARSEC_HOOK_RETURN_DONE;
}

static unsigned
next_random(unsigned *state)
{
    *state = *state * 1103515245u + 12345u;
    return *state >> 8;
}

/* Draw the sequence and compute what a sequential execution would produce. */
static void
prepare_operations(void)
{
    unsigned random_state = 20260512u;
    int writes_so_far[NB_TILES] = {0};

    for( int tile = 0; tile < NB_TILES; tile++ ) {
        expected_values[tile] = INITIAL_VALUE(tile);
    }
    for( int index = 0; index < NB_OPERATIONS; index++ ) {
        operation_t *operation = &operations[index];
        operation->index = index;
        operation->is_mixing = (next_random(&random_state) % 100) < PERCENT_MIXING;
        operation->written_tile = next_random(&random_state) % NB_TILES;
        do {
            operation->first_read_tile = next_random(&random_state) % NB_TILES;
        } while( operation->first_read_tile == operation->written_tile );
        do {
            operation->second_read_tile = next_random(&random_state) % NB_TILES;
        } while( operation->second_read_tile == operation->written_tile ||
                 operation->second_read_tile == operation->first_read_tile );
        operation->salt = next_random(&random_state) % 1000;

        operation->version_written = writes_so_far[operation->written_tile];
        operation->version_first_read = writes_so_far[operation->first_read_tile];
        operation->version_second_read = writes_so_far[operation->second_read_tile];
        writes_so_far[operation->written_tile]++;

        if( operation->is_mixing ) {
            expected_values[operation->written_tile] =
                mix_values(expected_values[operation->written_tile],
                           expected_values[operation->first_read_tile],
                           expected_values[operation->second_read_tile], operation->salt);
        } else {
            expected_values[operation->written_tile] =
                update_value(expected_values[operation->written_tile], operation->salt);
        }
    }
}

/* One round: insert everything, wait, and report what went wrong (0 if nothing). */
static int
run_round(parsec_context_t *context, int round)
{
    int arena, failures = 0, rc;
    parsec_taskpool_t *taskpool = parsec_dtd_taskpool_new();

    parsec_arena_datatype_t *arena_datatype =
        parsec_matrix_adt_new_rect(parsec_datatype_int32_t, 1, 1, 1);
    parsec_dtd_attach_arena_datatype(context, arena_datatype, &arena);

    parsec_tiled_matrix_t *matrix = create_and_distribute_data(0, 1, 1, NB_TILES);
    parsec_data_collection_t *collection = (parsec_data_collection_t *)matrix;
    parsec_data_collection_set_key(collection, "A");
    parsec_dtd_data_collection_init(collection);

    rc = parsec_context_add_taskpool(context, taskpool);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    uint32_t *storage[NB_TILES];
    parsec_data_copy_t *host_copy[NB_TILES];
    for( int tile = 0; tile < NB_TILES; tile++ ) {
        parsec_data_t *data = collection->data_of_key(collection, collection->data_key(collection, tile, 0));
        host_copy[tile] = data->device_copies[0];
        storage[tile] = (uint32_t *)host_copy[tile]->device_private;
        *storage[tile] = INITIAL_VALUE(tile);
        atomic_store(&tile_version[tile], 0);
        atomic_store(&tile_active_readers[tile], 0);
        atomic_store(&tile_active_writers[tile], 0);
    }
    atomic_store(&ordering_violations, 0);

#define TILE_OF(tile_index) \
    parsec_dtd_tile_of(collection, collection->data_key(collection, (tile_index), 0))
    for( int index = 0; index < NB_OPERATIONS; index++ ) {
        operation_t *operation = &operations[index];
        if( operation->is_mixing ) {
            parsec_dtd_insert_task(taskpool, body_mixing, 0, PARSEC_DEV_CPU, "Mixing",
                                   PASSED_BY_REF, TILE_OF(operation->written_tile),
                                   PARSEC_INOUT | arena | PARSEC_AFFINITY,
                                   PASSED_BY_REF, TILE_OF(operation->first_read_tile),
                                   PARSEC_INPUT | arena,
                                   PASSED_BY_REF, TILE_OF(operation->second_read_tile),
                                   PARSEC_INPUT | arena,
                                   sizeof(int), &operation->salt, PARSEC_VALUE,
                                   sizeof(int), &operation->index, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
        } else {
            parsec_dtd_insert_task(taskpool, body_updating, 0, PARSEC_DEV_CPU, "Updating",
                                   PASSED_BY_REF, TILE_OF(operation->written_tile),
                                   PARSEC_INOUT | arena | PARSEC_AFFINITY,
                                   sizeof(int), &operation->salt, PARSEC_VALUE,
                                   sizeof(int), &operation->index, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END);
        }
    }
#undef TILE_OF
    parsec_dtd_data_flush_all(taskpool, collection);

    rc = parsec_taskpool_wait(taskpool);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    int wrong_tiles = 0, tiles_with_readers = 0;
    for( int tile = 0; tile < NB_TILES; tile++ ) {
        if( *storage[tile] != expected_values[tile] ) {
            wrong_tiles++;
        }
        if( 0 != host_copy[tile]->readers ) {
            tiles_with_readers++;
        }
    }
    if( 0 != wrong_tiles ) {
        parsec_output(0, "[round %d] FAILED: %d of %d tiles differ from the sequential result\n",
                      round, wrong_tiles, NB_TILES);
        failures++;
    }
    if( 0 != atomic_load(&ordering_violations) ) {
        parsec_output(0, "[round %d] FAILED: %d tasks ran out of order with respect to a reader or writer\n",
                      round, atomic_load(&ordering_violations));
        failures++;
    }
    if( 0 != tiles_with_readers ) {
        parsec_output(0, "[round %d] FAILED: %d tiles are left with a non-zero reader count\n",
                      round, tiles_with_readers);
        failures++;
    }

    parsec_taskpool_free(taskpool);
    parsec_dtd_data_collection_fini(collection);
    free_data(matrix);
    parsec_dtd_free_arena_datatype(context, arena);
    return failures;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int rank, world, cores = -1, rc, failures = 0;

    if( NULL != argv[1] ) {
        cores = atoi(argv[1]);
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    if( world > 1 ) {
        parsec_output(0, "DTD reader release is single process only, skipping\n");
        rc = parsec_tests_context_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
        return 0;
    }

    prepare_operations();

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for( int round = 0; round < NB_ROUNDS; round++ ) {
        failures += run_round(parsec, round);
    }

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    /* Before the context is finalized: this closes the output stream. */
    if( 0 != failures ) {
        parsec_output(0, "DTD reader release: %d checks FAILED\n", failures);
    } else {
        parsec_output(0, "DTD reader release: all checks passed\n");
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return (0 != failures) ? 1 : 0;
}
