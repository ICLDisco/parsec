/*
 * Copyright (c) 2011-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec.h"
#include "parsec/runtime.h"
#include "parsec/data_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

static int
parsec_operator_print_id( struct parsec_execution_stream_s *es,
                          const void* src,
                          void* dest,
                          void* op_data, ... )
{
    va_list ap;
    int k, n, rank = 0;

    rank = parsec_context_query(es->virtual_process->parsec_context,
                                PARSEC_CONTEXT_QUERY_RANK);

    va_start(ap, op_data);
    k = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    printf( "tile (%d, %d) -> %p:%p thread %d of VP %d, process %d\n",
            k, n, src, dest, es->th_id, es->virtual_process->vp_id, rank );
    return 0;
}

int main( int argc, char* argv[] )
{
    parsec_context_t* parsec;
    parsec_taskpool_t* op;
    parsec_matrix_block_cyclic_t dcA;
    int cores = -1, world = 1, rank = 0;
    int mb = 100, nb = 100;
    int lm = 1000, ln = 1000;
    int rows = 1, rc;

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    parsec_matrix_block_cyclic_init( &dcA, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
                               rank, mb, nb, lm, ln, 0, 0, lm, ln,
                               rows, world/rows, 1, 1, 0, 0);
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                     (size_t)dcA.super.bsiz *
                                     (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));

    parsec_data_collection_set_key(&dcA.super.super, "A");
    op = parsec_map_operator_New((parsec_tiled_matrix_t*)&dcA,
                                  NULL,
                                  parsec_operator_print_id,
                                  "A");
    rc = parsec_context_add_taskpool(parsec, op);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(op);

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&dcA);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
