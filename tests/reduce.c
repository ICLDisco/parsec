/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/execution_stream.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/datatype.h"
#include <math.h>
#include "parsec/data_dist/matrix/reduce.h"

#if 0
static int parsec_operator_print_id( struct parsec_execution_stream *es, void* data, void* op_data, ... )
{
    va_list ap;
    int k, n;

    va_start(ap, op_data);
    k = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    printf( "tile %s(%d, %d) -> %p:%p thread %d VP %d\n",
            (char*)op_data, k, n, data, op_data, es->th_id, es->virtual_process->vp_id );
    return 0;
}
#endif

int main( int argc, char* argv[] )
{
    parsec_context_t* parsec;
    int rc;
    parsec_taskpool_t* tp;
    two_dim_block_cyclic_t dcA;
    int cores = -1, world = 1, rank = 0;
    int nb = 100, ln = 900;
    int rows = 1;
    parsec_datatype_t newtype;

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
                               world, rank, nb, 1, ln, 1, 0, 0, ln, 1, 1, 1, rows );
    dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                     (size_t)dcA.super.bsiz *
                                     (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));

    parsec_data_collection_set_key(&dcA.super.super, "A");

    tp = (parsec_taskpool_t*)parsec_reduce_new((parsec_tiled_matrix_dc_t*)&dcA,
                                               (parsec_tiled_matrix_dc_t*)&dcA,
                                               NULL);
    /* Prepare the arena for the reduction */
    parsec_type_create_contiguous(nb, parsec_datatype_float_t, &newtype);
    parsec_arena_datatype_construct(&((parsec_reduce_taskpool_t*)tp)->arenas_datatypes[PARSEC_reduce_DEFAULT_ARENA],
                                    nb*sizeof(float),
                                    PARSEC_ARENA_ALIGNMENT_SSE,
                                    newtype);

    rc = parsec_context_add_taskpool(parsec, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(tp);

    parsec_type_free(&newtype);

    parsec_fini(&parsec);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif  /* defined(PARSEC_HAVE_MPI) */

    return 0;
}
