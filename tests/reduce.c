/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "dague/execution_unit.h"
#include "dague/arena.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dague/datatype.h"
#include <math.h>
#include "data_dist/matrix/reduce.h"

#if 0
static int dague_operator_print_id( struct dague_execution_unit *eu, void* data, void* op_data, ... )
{
    va_list ap;
    int k, n;

    va_start(ap, op_data);
    k = va_arg(ap, int);
    n = va_arg(ap, int);
    va_end(ap);
    printf( "tile %s(%d, %d) -> %p:%p thread %d VP %d\n",
            (char*)op_data, k, n, data, op_data, eu->th_id, eu->virtual_process->vp_id );
    return 0;
}
#endif

int main( int argc, char* argv[] )
{
    dague_context_t* dague;
    dague_handle_t* object;
    two_dim_block_cyclic_t ddescA;
    int cores = 2, world = 1, rank = 0;
    int nb = 100, ln = 900;
    int rows = 1;
    dague_datatype_t newtype;

#if defined(HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    dague = dague_init(cores, &argc, &argv);

    two_dim_block_cyclic_init( &ddescA, matrix_RealFloat, matrix_Tile,
                               world, rank, nb, 1, ln, 1, 0, 0, ln, 1, 1, 1, rows );
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles *
                                     (size_t)ddescA.super.bsiz *
                                     (size_t)dague_datadist_getsizeoftype(ddescA.super.mtype));

    dague_ddesc_set_key(&ddescA.super.super, "A");

    object = (dague_handle_t*)dague_reduce_new((tiled_matrix_desc_t*)&ddescA,
                                               (tiled_matrix_desc_t*)&ddescA,
                                               NULL);
    /* Prepare the arena for the reduction */
    dague_type_create_contiguous(nb, dague_datatype_float_t, &newtype);
#if defined(HAVE_MPI)
    MPI_Type_commit(&newtype);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(((dague_reduce_handle_t*)object)->arenas[DAGUE_reduce_DEFAULT_ARENA],
                          nb*sizeof(float),
                          DAGUE_ARENA_ALIGNMENT_SSE,
                          newtype);

    dague_enqueue(dague, (dague_handle_t*)object);

    dague_context_wait(dague);

    dague_map_operator_Destruct( object );

    dague_fini(&dague);

#if defined(HAVE_MPI)
    MPI_Finalize();
#endif  /* defined(HAVE_MPI) */

    return 0;
}
