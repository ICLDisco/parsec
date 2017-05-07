/*
 * Copyright (c) 2011-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_internal.h"
#include "parsec/execution_unit.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int
parsec_operator_print_id( struct parsec_execution_unit_s *eu,
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
            k, n, src, dest, eu->th_id, eu->virtual_process->vp_id, rank );
    return 0;
}

int main( int argc, char* argv[] )
{
    parsec_context_t* parsec;
    int rc;
    parsec_handle_t* object;
    two_dim_block_cyclic_t ddescA;
    int cores = 4, world = 1, rank = 0;
    int mb = 100, nb = 100;
    int lm = 1000, ln = 1000;
    int rows = 1;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    parsec = parsec_init(cores, &argc, &argv);

    two_dim_block_cyclic_init( &ddescA, matrix_RealFloat, matrix_Tile,
                               world, rank, mb, nb, lm, ln, 0, 0, lm, ln, 1, 1, rows );
    ddescA.mat = parsec_data_allocate((size_t)ddescA.super.nb_local_tiles *
                                     (size_t)ddescA.super.bsiz *
                                     (size_t)parsec_datadist_getsizeoftype(ddescA.super.mtype));

    parsec_ddesc_set_key(&ddescA.super.super, "A");
    object = parsec_map_operator_New((tiled_matrix_desc_t*)&ddescA,
                                    NULL,
                                    parsec_operator_print_id,
                                    "A");
    rc = parsec_enqueue(parsec, (parsec_handle_t*)object);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_map_operator_Destruct( object );

    parsec_fini(&parsec);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    return 0;
}
