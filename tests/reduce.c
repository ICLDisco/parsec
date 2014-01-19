/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "execution_unit.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include <math.h>

struct dague_reduce_object_t;
typedef struct dague_reduce_object_t dague_reduce_object_t;

extern dague_reduce_object_t *dague_reduce_new(dague_ddesc_t* R /* data R */, dague_ddesc_t* A /* data A */, int MT, int depth, void* ELEM_NEUTRE /* data ELEM_NEUTRE */);
extern void dague_reduce_destroy( dague_reduce_object_t *o );

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
    dague_object_t* object;
    two_dim_block_cyclic_t ddescA;
    int cores = 4, world = 1, rank = 0;
    int mb = 100, nb = 100;
    int lm = 900, ln = 900;
    int rows = 1;

#if defined(HAVE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    dague = dague_init(cores, &argc, &argv);

    two_dim_block_cyclic_init( &ddescA, matrix_RealFloat, matrix_Tile,
                               world, cores, rank, mb, nb, lm, ln, 0, 0, lm, ln, 1, 1, rows );
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles *
                                     (size_t)ddescA.super.bsiz *
                                     (size_t)dague_datadist_getsizeoftype(ddescA.super.mtype));

    dague_ddesc_set_key(&ddescA.super.super, "A");

    object = (dague_object_t*)dague_reduce_new((dague_ddesc_t*)&ddescA,
                                               (dague_ddesc_t*)&ddescA,
                                               ddescA.super.mt,
                                               (int)ceil(log(ddescA.super.mt) / log(2.0)),
                                               NULL);
    dague_enqueue(dague, (dague_object_t*)object);

    dague_progress(dague);

    dague_map_operator_Destruct( object );

    dague_fini(&dague);

    return 0;
}
