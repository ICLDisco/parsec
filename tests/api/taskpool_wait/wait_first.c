/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * The first wait of the process targets a PTG taskpool instead of the context.
 *
 * In a single process run the main thread drives the communication engine
 * itself, from inside the wait or test calls. A PTG taskpool registers with the
 * communication engine even in that case, so the main thread has commands to
 * process from the start. This test makes sure the engine is ready for them
 * when no parsec_context_wait came first, with the taskpool completed through
 * parsec_taskpool_wait (default) or by polling parsec_taskpool_test (-t). A
 * second epoch then reuses the context the same way, to check that turning
 * the engine on again is harmless.
 */

#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "tests/tests_runtime.h"
#include "ptg_tp_wait_first.h"

#include <string.h>

static volatile int32_t completed = 0;

static int on_complete(parsec_taskpool_t *tp, void *data)
{
    (void)tp; (void)data;
    parsec_atomic_fetch_inc_int32(&completed);
    return 0;
}

static int run_epoch(parsec_context_t *parsec, parsec_matrix_block_cyclic_t *A,
                     int poll, int epoch)
{
    parsec_taskpool_t *tp;
    int32_t expected = completed + 1;
    int rc;

    tp = (parsec_taskpool_t*)parsec_ptg_tp_new(A, 1000, 5000);
    parsec_taskpool_set_complete_callback(tp, on_complete, NULL);
    rc = parsec_context_add_taskpool(parsec, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    if( poll ) {
        while( completed < expected ) {
            rc = parsec_taskpool_test(tp);
            if( rc < 0 ) {
                fprintf(stderr, "epoch %d: parsec_taskpool_test failed (%d)\n", epoch, rc);
                return 1;
            }
        }
    }
    rc = parsec_taskpool_wait(tp);
    if( rc < 0 ) {
        fprintf(stderr, "epoch %d: parsec_taskpool_wait failed (%d)\n", epoch, rc);
        return 1;
    }
    if( completed != expected ) {
        fprintf(stderr, "epoch %d: parsec_taskpool_wait returned before the taskpool completed\n", epoch);
        return 1;
    }
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    parsec_taskpool_free(tp);
    return 0;
}

int main(int argc, char *argv[])
{
    int err = 0, world_size = 1, my_rank = 0, poll = 0, nb = 4;
    parsec_matrix_block_cyclic_t A;
    parsec_context_t *parsec;

    for( int i = 1; i < argc; i++ ) {
        if( 0 == strcmp(argv[i], "-t") ) poll = 1;
    }

    err = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_MULTIPLE,
                                    NULL, NULL,
                                    &parsec, &my_rank, &world_size);
    PARSEC_CHECK_ERROR(err, "parsec_tests_context_init");

    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_INTEGER,
                                    PARSEC_MATRIX_TILE,
                                    my_rank,
                                    1, 1,   /* Tile size */
                                    nb*world_size, nb*world_size,   /* Global matrix size */
                                    0, 0,   /* Starting point in the global matrix */
                                    nb*world_size, nb*world_size,   /* Submatrix size */
                                    world_size, 1,   /* Process grid */
                                    1, 1,   /* k-cyclicity */
                                    0, 0);
    parsec_data_collection_set_key(&A.super.super, "A");

    if( 0 == my_rank ) {
        printf("PTG taskpool completed through parsec_taskpool_%s before any parsec_context_wait\n",
               poll ? "test" : "wait");
    }
    err += run_epoch(parsec, &A, poll, 0);
    err += run_epoch(parsec, &A, poll, 1);

    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&A);

    int rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
    if( 0 == my_rank ) {
        printf("%s\n", err ? "FAILED" : "All tests done");
    }
    return err;
}
