/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * A taskpool handed out by a library _New() call, or by
 * parsec_dtd_taskpool_new(), can be dropped before it is ever added to a
 * context: an application error path, a binding whose garbage collector
 * runs first. Freeing it must neither abort nor leak, it must leave no
 * trace in the taskpool registry, and the runtime must still run
 * taskpools afterwards.
 */
#include "parsec.h"
#include "parsec/parsec_internal.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "tests/tests_runtime.h"

#include <stdio.h>

#define M  50
#define N  40
#define MB 8
#define NB 8

static void init_matrix(parsec_matrix_block_cyclic_t *dc, const char *key,
                        int mb, int nb, int rank, int world)
{
    parsec_matrix_block_cyclic_init(dc, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                    rank, mb, nb, M, N, 0, 0, M, N,
                                    1, world, 1, 1, 0, 0);
    dc->mat = parsec_data_allocate((size_t)dc->super.nb_local_tiles *
                                   (size_t)dc->super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dc->super.mtype));
    parsec_data_collection_set_key(&dc->super.super, key);
    (void)key;  /* a no-op without PARSEC_PROF_TRACE */
}

static void fini_matrix(parsec_matrix_block_cyclic_t *dc)
{
    parsec_data_free(dc->mat);
    parsec_tiled_matrix_destroy(&dc->super);
}

/* Number of elements of the M x N matrix this rank owns with the
 * 1 x world process grid used above. */
static size_t local_elements(const parsec_matrix_block_cyclic_t *dc, int rank, int world)
{
    size_t cols = 0;
    for( int n = 0; n < dc->super.nt; n++ ) {
        if( (n % world) != rank ) continue;
        cols += (n == dc->super.nt - 1) ? (size_t)(N - n * dc->super.nb) : (size_t)dc->super.nb;
    }
    return cols * M;
}

/* Free a taskpool that was never added to a context and check that its id
 * no longer resolves to anything. */
static int free_unenqueued(parsec_taskpool_t *tp, const char *what)
{
    uint32_t id = tp->taskpool_id;
    parsec_taskpool_t *found;

    parsec_taskpool_free(tp);
    found = parsec_taskpool_lookup(id);
    if( NULL != found ) {
        fprintf(stderr, "%s: taskpool id %u still registered after free (%p)\n",
                what, id, (void*)found);
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    parsec_taskpool_t *tp;
    parsec_matrix_block_cyclic_t src, dst, dst_small;
    int rank = 0, world = 1, rc, err = 0;
    size_t nelem, i, copied = 0, mismatched = 0;
    double *s, *d;

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    init_matrix(&src, "src", MB, NB, rank, world);
    init_matrix(&dst, "dst", MB, NB, rank, world);
    /* Different tile sizes select the general redistribute JDF rather than
     * the reshuffle one. */
    init_matrix(&dst_small, "dst_small", MB / 2, NB / 2, rank, world);

    nelem = (size_t)src.super.nb_local_tiles * (size_t)src.super.bsiz;
    s = (double*)src.mat;
    d = (double*)dst.mat;
    for( i = 0; i < nelem; i++ ) {
        s[i] = (double)(rank * nelem + i);
        d[i] = -1.0;
    }

    printf("PTG taskpool (redistribute_reshuffle) freed without being added to a context\n");
    tp = parsec_redistribute_New(&src.super, &dst.super, M, N, 0, 0, 0, 0);
    if( NULL == tp ) {
        fprintf(stderr, "parsec_redistribute_New (reshuffle) returned NULL\n");
        err++;
    } else {
        err += free_unenqueued(tp, "redistribute_reshuffle");
    }

    printf("PTG taskpool (redistribute) freed without being added to a context\n");
    tp = parsec_redistribute_New(&src.super, &dst_small.super, M, N, 0, 0, 0, 0);
    if( NULL == tp ) {
        fprintf(stderr, "parsec_redistribute_New (general) returned NULL\n");
        err++;
    } else {
        err += free_unenqueued(tp, "redistribute");
    }

    printf("DTD taskpool freed without being added to a context\n");
    tp = parsec_dtd_taskpool_new();
    err += free_unenqueued(tp, "dtd");

    printf("The runtime still runs a taskpool after that\n");
    tp = parsec_redistribute_New(&src.super, &dst.super, M, N, 0, 0, 0, 0);
    if( NULL == tp ) {
        fprintf(stderr, "parsec_redistribute_New (reshuffle) returned NULL\n");
        err++;
    } else {
        rc = parsec_context_add_taskpool(parsec, tp);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");
        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        parsec_taskpool_free(tp);

        /* Same distribution on both sides, so the local buffers match element
         * for element. Padding of the partial border tiles may or may not be
         * copied; only what was copied has to agree. */
        for( i = 0; i < nelem; i++ ) {
            if( -1.0 == d[i] ) continue;
            copied++;
            if( d[i] != s[i] ) mismatched++;
        }
        if( (0 != mismatched) || (copied < local_elements(&src, rank, world)) ) {
            fprintf(stderr, "rank %d: redistribute copied %zu elements (%zu expected at least), %zu mismatched\n",
                    rank, copied, local_elements(&src, rank, world), mismatched);
            err++;
        }
    }

    fini_matrix(&src);
    fini_matrix(&dst);
    fini_matrix(&dst_small);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    if( 0 == rank ) {
        printf("%s\n", (0 == err) ? "TEST PASSED" : "TEST FAILED");
    }
    return (0 == err) ? 0 : 1;
}
