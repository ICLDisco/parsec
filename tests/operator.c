/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_internal.h"
#include "parsec/execution_stream.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/utils/debug.h"

static int
parsec_operator_print_id( struct parsec_execution_stream_s *es,
                          const void* src,
                          void* dest,
                          void* op_data, ... )
{
    va_list ap;
    int k, n, rank = 0;

#if defined(PARSEC_HAVE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

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
    two_dim_block_cyclic_t dcA;
    int cores = -1, world = 1, rank = 0;
    int mb = 100, nb = 100;
    int lm = 1000, ln = 1000;
    int rows = 1, rc;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    parsec = parsec_init(cores, &argc, &argv);

    two_dim_block_cyclic_init( &dcA, matrix_RealFloat, matrix_Tile,
                               world, rank, mb, nb, lm, ln, 0, 0, lm, ln, 1, 1, rows );
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                     (size_t)dcA.super.bsiz *
                                     (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));

    parsec_data_collection_set_key(&dcA.super.super, "A");
    op = parsec_map_operator_New((parsec_tiled_matrix_dc_t*)&dcA,
                                  NULL,
                                  parsec_operator_print_id,
                                  "A");
    rc = parsec_enqueue(parsec, op);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_map_operator_Destruct( op );

    parsec_fini(&parsec);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    return 0;
}
