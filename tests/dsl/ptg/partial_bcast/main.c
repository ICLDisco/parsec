/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "tests/tests_runtime.h"
#include "partial_bcast_wrapper.h"

/**
 * Broadcast payloads smaller than the receive type from rank 0 to every other
 * rank and check what arrived. Run with several processes; the broadcast
 * topology (runtime_comm_coll_bcast: 0 star, 1 chain, 2 binomial) decides
 * which ranks forward. "-m MASK" selects the PB_CASE_* cases (default: all).
 */
int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    parsec_matrix_block_cyclic_t dcA;
    parsec_taskpool_t *tp;
    int rank, world, rc, mask = PB_CASE_ALL;
    int32_t total = 0;
    const int nb = 64;        /* tile of nb x nb doubles */
    const int nodd = 901;     /* short payload: below PARSEC_DIST_SHORT_LIMIT KB, not a multiple of 8 */
    const int nbig = 20001;   /* rendezvous payload: above the short limit, not a multiple of 8 */
    const int nrb = 64;       /* elements of the resized-byte receive */

    rc = parsec_tests_context_init(1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    for( int i = 1; i + 1 < argc; i++ ) {  /* -m MASK; parsec_init() leaves its own --mca arguments in place */
        if( 0 == strcmp(argv[i], "-m") ) mask = (int)strtol(argv[++i], NULL, 0);
    }
#if defined(PARSEC_DIST_SHORT_LIMIT) && (PARSEC_DIST_SHORT_LIMIT == 0)
    /* A payload longer than the receive is only legal inside the activation
     * message; by rendezvous it would be a truncated receive. */
    mask &= ~PB_CASE_OVER;
#endif

    if( world < 2 ) {
        if( 0 == rank ) printf("partial_bcast needs at least two processes, nothing to check\n");
        rc = parsec_tests_context_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
        return 0;
    }

    /* One row of tiles, one per process: rank r owns tile (0, r). */
    parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE, rank,
                                    nb, nb, nb, world * nb, 0, 0, nb, world * nb,
                                    1, world, 1, 1, 0, 0);
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                   (size_t)dcA.super.bsiz * sizeof(double));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "A");
    if( 0 == rank ) {
        unsigned char *p = (unsigned char*)dcA.mat;
        for( size_t b = 0; b < (size_t)nb * nb * sizeof(double); b++ ) {
            p[b] = partial_bcast_pattern(b);
        }
    }

    tp = partial_bcast_new(&dcA, nb, world - 1, nodd, nbig, nrb, mask);

    rc = parsec_context_add_taskpool(parsec, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    parsec_taskpool_free(tp);

    rc = parsec_tests_allreduce(parsec, &partial_bcast_failures, &total, 1,
                                parsec_datatype_int32_t, PARSEC_TESTS_REDUCE_SUM);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_allreduce");
    if( 0 == rank ) {
        printf("partial_bcast: mask 0x%x on %d processes: %d failure%s\n",
               mask, world, total, (1 == total) ? "" : "s");
    }

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy(&dcA.super);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return total ? 1 : 0;
}
