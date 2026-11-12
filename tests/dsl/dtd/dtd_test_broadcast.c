/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/* Broadcast of a tile from a root to every other rank.
 *
 *   dtd_test_broadcast [cores] [PaRSEC options] [-- [--root N] [--scenario plain|late|writer|multi|multiset|multisingle]
 *                                                   [--topology star|chain|binomial]]
 *
 * The propagation tree is laid out relative to the root, so which rank holds
 * the data is worth varying (--root). Every rank checks the value it receives.
 * The topology is the default one unless asked for, in which case it is the one of
 * the taskpool.
 * The scenarios cover what a peer that forwards what it received has to cope
 * with:
 *   plain:  one flow, every consumer is a reader.
 *   late:   the consumers are inserted after the activation arrived, so a peer
 *           has to hold the activation back until its consumer exists.
 *   writer: the first peer below the root reads the tile and then writes it, while
 *           the peers below it still have to receive what the root sent.
 *   multi:  the broadcast task has two flows, both to the same destinations.
 *   multiset: the same, but only every other destination reads the first flow, so each
 *           flow has a set of destinations of its own, and the position of a rank in one
 *           is not its position in the other. DTD activates each flow of a task on its
 *           own, so these are two broadcasts, each with its own tree.
 *   multisingle: the same, but the first flow has a single destination, the first of the
 *           second flow's: nobody to forward to for that one, so no destination set.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <unistd.h>

#include "tests/tests_data.h"
#include "tests/tests_timing.h"
#include "tests/tests_runtime.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

/* IDs for the Arena Datatypes */
static int TILE_FULL;

typedef enum {
    SCENARIO_PLAIN,
    SCENARIO_LATE,
    SCENARIO_WRITER,
    SCENARIO_MULTI,
    SCENARIO_MULTISET,
    SCENARIO_MULTISINGLE
} scenario_t;

/* the value the second flow of the multi scenario starts with, on top of the root */
#define SECOND_FLOW_SEED 100

int
task_rank_0( parsec_execution_stream_t *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;

    parsec_dtd_unpack_args(this_task, &data);
    *data *= 2;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_0_multi( parsec_execution_stream_t *es,
                   parsec_task_t *this_task )
{
    (void)es;
    int *data, *second_data;

    parsec_dtd_unpack_args(this_task, &data, &second_data);
    *data *= 2;
    *second_data *= 2;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_1( parsec_execution_stream_t *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data;
    int *second_data;
    int expected;

    parsec_dtd_unpack_args(this_task, &expected, &data, &second_data);

    printf( "My rank: %d, data: %d\n", this_task->taskpool->context->my_rank, *data );
    if( expected != *data ) {
        parsec_fatal("rank %d received %d instead of %d",
                     this_task->taskpool->context->my_rank, *data, expected);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_1_multi( parsec_execution_stream_t *es,
                   parsec_task_t *this_task )
{
    (void)es;
    int *data, *second_data, *own_data;
    int expected, expected_second;

    parsec_dtd_unpack_args(this_task, &expected, &expected_second, &data, &second_data, &own_data);

    printf( "My rank: %d, data: %d %d\n", this_task->taskpool->context->my_rank, *data, *second_data );
    if( expected != *data || expected_second != *second_data ) {
        parsec_fatal("rank %d received %d %d instead of %d %d",
                     this_task->taskpool->context->my_rank, *data, *second_data,
                     expected, expected_second);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

/* Overwrites the tile it was given: whoever still has to read what the root
 * sent must not see this. */
int
task_writer( parsec_execution_stream_t *es,
             parsec_task_t *this_task )
{
    (void)es;
    int *data, *own_data;

    parsec_dtd_unpack_args(this_task, &data, &own_data);
    *data = -1;

    return PARSEC_HOOK_RETURN_DONE;
}

/* Value of the argument at the given position, if it is a number: the options
 * that PaRSEC itself parses (--mca ...) must not be taken for one. */
static int numeric_argument(int argc, char **argv, int position, int fallback)
{
    if( position < argc && NULL != argv[position] && isdigit((unsigned char)argv[position][0]) )
        return atoi(argv[position]);
    return fallback;
}

/* Value following the named option, or NULL if it is not given. */
static const char *named_argument(int argc, char **argv, const char *name)
{
    for( int a = 1; a + 1 < argc; a++ ) {
        if( 0 == strcmp(argv[a], name) )
            return argv[a + 1];
    }
    return NULL;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rank, world, cores;
    int nb, nt, rc, root = 0, expected, expected_second = 0;
    scenario_t scenario = SCENARIO_PLAIN;
    parsec_bcast_topology_t topology = PARSEC_BCAST_TOPOLOGY_DEFAULT;
    const char *value;
    parsec_tiled_matrix_t *dcA;
    parsec_arena_datatype_t *adt;

    nb = 1; /* tile_size */

    cores = numeric_argument(argc, argv, 1, -1);
    if( NULL != (value = named_argument(argc, argv, "--root")) ) {
        root = atoi(value);
    }
    if( NULL != (value = named_argument(argc, argv, "--scenario")) ) {
        if( 0 == strcmp(value, "plain") )       scenario = SCENARIO_PLAIN;
        else if( 0 == strcmp(value, "late") )   scenario = SCENARIO_LATE;
        else if( 0 == strcmp(value, "writer") ) scenario = SCENARIO_WRITER;
        else if( 0 == strcmp(value, "multi") )  scenario = SCENARIO_MULTI;
        else if( 0 == strcmp(value, "multiset") ) scenario = SCENARIO_MULTISET;
        else if( 0 == strcmp(value, "multisingle") ) scenario = SCENARIO_MULTISINGLE;
        else {
            fprintf(stderr, "Unknown scenario %s (plain, late, writer, multi, multiset or multisingle)\n", value);
            return EXIT_FAILURE;
        }
    }

    if( NULL != (value = named_argument(argc, argv, "--topology")) ) {
        if( 0 == strcmp(value, "star") )          topology = PARSEC_BCAST_TOPOLOGY_STAR;
        else if( 0 == strcmp(value, "chain") )    topology = PARSEC_BCAST_TOPOLOGY_CHAIN;
        else if( 0 == strcmp(value, "binomial") ) topology = PARSEC_BCAST_TOPOLOGY_BINOMIAL;
        else {
            fprintf(stderr, "Unknown topology %s (star, chain or binomial)\n", value);
            return EXIT_FAILURE;
        }
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    if( root < 0 || root >= world ) {
        if( 0 == rank ) {
            fprintf(stderr, "The root (%d) must be a rank between 0 and %d\n", root, world - 1);
        }
        (void)parsec_tests_context_fini(&parsec);
        return EXIT_FAILURE;
    }

    /* The tile (m, 0) lives on the rank m % world: the second flow of the multi
     * scenario uses a second tile of the root. */
    int two_flows = (SCENARIO_MULTI == scenario) || (SCENARIO_MULTISET == scenario) || (SCENARIO_MULTISINGLE == scenario);
    nt = two_flows ? 2 * world : world;
    int second_tile = root + world;
    /* the first peer below the root, which writes in the writer scenario */
    int writer_rank = (root + 1) % world;
    /* a tile that is written after having been read elsewhere has to be known
     * as such by every rank, the others only insert what they run */
    int all_insert = (SCENARIO_WRITER == scenario);

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    if( PARSEC_BCAST_TOPOLOGY_DEFAULT != topology ) {
        dtd_tp->bcast_topology = topology;
    }

    adt = parsec_matrix_adt_new_rect(
            parsec_datatype_int32_t, nb, 1, nb);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

    /* Correctness checking */
    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;
    int i;

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    // *********************
    /* the broadcast task doubles what the root seeded the tile with */
    expected = (root + 5) * 2;
    if( two_flows ) {
        expected_second = (root + SECOND_FLOW_SEED) * 2;
    }
    if( rank == root ) {
        parsec_output( 0, "Testing Broadcast, every node should end up having %d\n", expected );
    }

    if( root == rank ) {
        key = A->data_key(A, root, 0);
        data = A->data_of_key(A, key);
        gdata = data->device_copies[0];
        real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
        *real_data = 5+root;
        if( two_flows ) {
            key = A->data_key(A, second_tile, 0);
            data = A->data_of_key(A, key);
            gdata = data->device_copies[0];
            real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
            *real_data = root + SECOND_FLOW_SEED;
        }
    }

    if( rank == root) {
        printf("Root: %d\n\n", root );
    }

    if( two_flows ) {
        parsec_dtd_insert_task(dtd_tp, task_rank_0_multi, 0, PARSEC_DEV_CPU, "task_rank_0_multi",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, root),        PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, second_tile), PARSEC_INOUT | TILE_FULL,
                               PARSEC_DTD_ARG_END);
    } else {
        parsec_dtd_insert_task(dtd_tp, task_rank_0, 0, PARSEC_DEV_CPU, "task_rank_0",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, root), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);
    }

    /* The consumers of the broadcast: the root inserts one for every rank, each
     * other rank only its own. */
    for( i = 0; i < world; i++ ) {
        if( i == root ) continue;
        if( rank != root && rank != i && !all_insert ) continue;

        if( rank != root && SCENARIO_LATE == scenario ) {
            /* the activation is on its way, or already here, by the time the
             * task it is for gets inserted */
            usleep(200000);
        }
        /* whether the destination reads the first flow, the second one being read by all */
        int distance = (i - root + world) % world;
        int reads_first = (SCENARIO_MULTISET == scenario) ? (1 == (distance % 2))
                        : (SCENARIO_MULTISINGLE == scenario) ? (1 == distance) : 1;

        if( two_flows && !reads_first ) {
            parsec_dtd_insert_task(dtd_tp, task_rank_1, 0, PARSEC_DEV_CPU, "task_rank_1",
                                   sizeof(int), &expected_second, PARSEC_VALUE,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, second_tile), PARSEC_INPUT | TILE_FULL,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, i),           PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        } else if( two_flows ) {
            parsec_dtd_insert_task(dtd_tp, task_rank_1_multi, 0, PARSEC_DEV_CPU, "task_rank_1_multi",
                                   sizeof(int), &expected, PARSEC_VALUE,
                                   sizeof(int), &expected_second, PARSEC_VALUE,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, root),        PARSEC_INPUT | TILE_FULL,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, second_tile), PARSEC_INPUT | TILE_FULL,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, i),           PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        } else {
            parsec_dtd_insert_task(dtd_tp, task_rank_1, 0, PARSEC_DEV_CPU, "task_rank_1",
                                   sizeof(int), &expected, PARSEC_VALUE,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, root), PARSEC_INPUT | TILE_FULL,
                                   PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, i),    PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   PARSEC_DTD_ARG_END);
        }
    }

    if( SCENARIO_WRITER == scenario && (all_insert || rank == root || rank == writer_rank) && writer_rank != root ) {
        /* after its own read, and while the peers below still have to be served
         * from the copy it received */
        parsec_dtd_insert_task(dtd_tp, task_writer, 0, PARSEC_DEV_CPU, "task_writer",
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, root),        PARSEC_INOUT | TILE_FULL,
                               PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, writer_rank), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               PARSEC_DTD_ARG_END);
    }

    //******************
    parsec_dtd_data_flush_all( dtd_tp, A );

    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    parsec_taskpool_free( dtd_tp );

    parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini( A );
    free_data(dcA);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
