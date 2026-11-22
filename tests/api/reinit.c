/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * A process may bring the runtime up more than once: a library that owns
 * its PaRSEC context for the length of a solve, a test driver that runs
 * its cases one context at a time. parsec_fini() must then leave nothing
 * behind that the next parsec_init() trips over.
 *
 * Each round creates a context, runs a few DTD tasks in it, then DTD tasks
 * that move tiles between processes and a PTG taskpool (redistribute) whose
 * results it checks, and destroys the context. MPI can only be initialized
 * once, so the test initializes it here and keeps it up across the rounds,
 * instead of leaving it to parsec_init().
 */

#include "parsec.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/mca/device/device.h"
#include "parsec/utils/debug.h"
#include "parsec/sys/atomic.h"
#include "tests/tests_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NB_ROUNDS 3
#define NB_TASKS  100
/* The matrices of the data phase: M x N, MB x NB tiles. */
#define M  40
#define N  36
#define MB 8
#define NB 8

static int32_t counter;

static int
count_one(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;
    (void)parsec_atomic_fetch_inc_int32(&counter);
    return PARSEC_HOOK_RETURN_DONE;
}

/* The value element (i, j) of the matrices holds at the end of a round. */
static double expected(int i, int j)
{
    return 1000.0 * i + j + 1.0;
}

/* DTD body: adds a value to every element of a tile. */
static int
add_value(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    double *tile, value;
    (void)es;
    parsec_dtd_unpack_args(this_task, &tile, &value);
    for( int i = 0; i < MB * NB; i++ ) tile[i] += value;
    return PARSEC_HOOK_RETURN_DONE;
}

static void
matrix_init(parsec_matrix_block_cyclic_t *dc, const char *key,
            int mb, int nb, int p, int rank, int world)
{
    size_t bytes;
    parsec_matrix_block_cyclic_init(dc, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                    rank, mb, nb, M, N, 0, 0, M, N,
                                    p, world / p, 1, 1, 0, 0);
    bytes = (size_t)dc->super.nb_local_tiles * dc->super.bsiz * sizeof(double);
    dc->mat = parsec_data_allocate(bytes);
    memset(dc->mat, 0, bytes);
    parsec_data_collection_set_key(&dc->super.super, (char*)key);
    (void)key;  /* a no-op without PARSEC_PROF_TRACE */
}

static double *
local_tile(parsec_matrix_block_cyclic_t *dc, int m, int n)
{
    parsec_data_collection_t *d = &dc->super.super;
    return parsec_data_copy_get_ptr(parsec_data_get_copy(d->data_of(d, m, n), 0));
}

/* Number of local elements of dc that do not hold expected(). */
static int
check(parsec_matrix_block_cyclic_t *dc, int rank)
{
    parsec_data_collection_t *d = &dc->super.super;
    int mb = dc->super.mb, nb = dc->super.nb, wrong = 0;
    for( int m = 0; m < dc->super.mt; m++ ) {
        for( int n = 0; n < dc->super.nt; n++ ) {
            if( (uint32_t)rank != d->rank_of(d, m, n) ) continue;
            double *tile = local_tile(dc, m, n);
            for( int j = 0; j < nb && n * nb + j < N; j++ )
                for( int i = 0; i < mb && m * mb + i < M; i++ )
                    wrong += (tile[j * mb + i] != expected(m * mb + i, n * nb + j));
        }
    }
    return wrong;
}

/* src is filled by DTD tasks placed on the owners of its tiles, then copied
 * by redistribute into dst, which has other tiles and another process grid,
 * so that both phases move data between processes. */
static int
data_round(parsec_context_t *parsec, int round)
{
    parsec_matrix_block_cyclic_t src, dst;
    parsec_data_collection_t *d = &src.super.super;
    parsec_arena_datatype_t *adt;
    parsec_datatype_t dtt;
    parsec_taskpool_t *tp;
    parsec_task_class_t *tc;
    int rank = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_RANK);
    int world = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_NODES);
    int adt_id, rc, wrong;

    matrix_init(&src, "src", MB, NB, 1, rank, world);
    matrix_init(&dst, "dst", MB + 4, NB - 2, world, rank, world);
    /* Each tile starts with the offsets of its elements; a task adds the
     * value of the tile's first element. */
    for( int m = 0; m < src.super.mt; m++ )
        for( int n = 0; n < src.super.nt; n++ ) {
            if( (uint32_t)rank != d->rank_of(d, m, n) ) continue;
            double *tile = local_tile(&src, m, n);
            for( int j = 0; j < NB; j++ )
                for( int i = 0; i < MB; i++ )
                    tile[j * MB + i] = expected(i, j) - expected(0, 0);
        }

    parsec_dtd_data_collection_init(d);
    parsec_translate_matrix_type(PARSEC_MATRIX_DOUBLE, &dtt);
    adt = parsec_matrix_adt_new_rect(dtt, MB, NB, MB);
    rc = parsec_dtd_attach_arena_datatype(parsec, adt, &adt_id);
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_attach_arena_datatype");
    tp = parsec_dtd_taskpool_new();
    rc = parsec_context_add_taskpool(parsec, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    tc = parsec_dtd_create_task_class(tp, "add_value",
                                      PASSED_BY_REF, PARSEC_INOUT | PARSEC_AFFINITY | adt_id,
                                      sizeof(double), PARSEC_VALUE,
                                      PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, tc, PARSEC_DEV_CPU, add_value);
    for( int m = 0; m < src.super.mt; m++ )
        for( int n = 0; n < src.super.nt; n++ ) {
            double base = expected(m * MB, n * NB);
            parsec_dtd_insert_task_with_task_class(tp, tc, 0, PARSEC_DEV_ALL,
                PARSEC_INOUT | PARSEC_AFFINITY, parsec_dtd_tile_of(d, d->data_key(d, m, n)),
                PARSEC_VALUE, &base, PARSEC_DTD_ARG_END);
        }
    parsec_dtd_data_flush_all(tp, d);
    rc = parsec_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
    parsec_dtd_task_class_release(tp, tc);
    parsec_taskpool_free(tp);
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    parsec_dtd_data_collection_fini(d);
    parsec_dtd_free_arena_datatype(parsec, adt_id);
    wrong = check(&src, rank);
    if( wrong ) fprintf(stderr, "Round %d, rank %d: %d wrong elements after DTD\n", round, rank, wrong);

    tp = parsec_redistribute_New(&src.super, &dst.super, M, N, 0, 0, 0, 0);
    if( NULL == tp ) {
        fprintf(stderr, "Round %d: parsec_redistribute_New failed\n", round);
        wrong++;
    } else {
        rc = parsec_context_add_taskpool(parsec, tp);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        parsec_taskpool_free(tp);
        int w = check(&dst, rank);
        if( w ) fprintf(stderr, "Round %d, rank %d: %d wrong elements after redistribute\n", round, rank, w);
        wrong += w;
    }

    parsec_data_free(src.mat);
    parsec_tiled_matrix_destroy(&src.super);
    parsec_data_free(dst.mat);
    parsec_tiled_matrix_destroy(&dst.super);
    return wrong;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    parsec_taskpool_t *tp;
    int rc, ret = 0;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
#endif

    for( int round = 0; round < NB_ROUNDS; round++ ) {
        parsec = parsec_init(2, &argc, &argv);
        if( NULL == parsec ) {
            fprintf(stderr, "Round %d: parsec_init failed\n", round);
            ret = 1;
            break;
        }

        counter = 0;
        tp = parsec_dtd_taskpool_new();
        rc = parsec_context_add_taskpool(parsec, tp);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        /* No data: every process inserts and runs all the tasks */
        for( int i = 0; i < NB_TASKS; i++ ) {
            parsec_dtd_insert_task(tp, count_one, 0, PARSEC_DEV_CPU, "count_one",
                                   PARSEC_DTD_ARG_END);
        }

        rc = parsec_taskpool_wait(tp);
        PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        parsec_taskpool_free(tp);

        if( NB_TASKS != counter ) {
            fprintf(stderr, "Round %d: %d tasks ran, expected %d\n",
                    round, (int)counter, NB_TASKS);
            ret = 1;
        }
        if( 0 != data_round(parsec, round) ) {
            ret = 1;
        }

        rc = parsec_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_fini");
        printf("Round %d: context created, used and destroyed\n", round);
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
    return ret;
}
