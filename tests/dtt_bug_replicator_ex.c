/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "dague/data_distribution.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dtt_bug_replicator.h"
#include "dague/arena.h"
#include <math.h>

#define N     10
#define NB    3

extern void dump_double_array(char* msg, double* mat, int i, int j, int nb, int mb, int lda);

#define PASTE_CODE_ALLOCATE_MATRIX(DDESC, COND, TYPE, INIT_PARAMS)      \
    TYPE##_t DDESC;                                                     \
    if(COND) {                                                          \
        TYPE##_init INIT_PARAMS;                                        \
        DDESC.mat = dague_data_allocate((size_t)DDESC.super.nb_local_tiles * \
                                        (size_t)DDESC.super.bsiz *      \
                                        (size_t)dague_datadist_getsizeoftype(DDESC.super.mtype)); \
        dague_ddesc_set_key((dague_ddesc_t*)&DDESC, #DDESC);            \
    }


int main( int argc, char** argv )
{
    dague_context_t* dague;
    dague_handle_t* handle;
#if defined(DAGUE_HAVE_MPI)
    MPI_Datatype tile_dtt, vdtt1, vdtt2, vdtt;
#endif
    dague_dtt_bug_replicator_handle_t *dtt_handle;;
    int nodes, rank, i, j;
    (void)argc; (void)argv;

#if defined(DAGUE_HAVE_MPI)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &nodes);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nodes = 1;
    rank = 0;
#endif

    dague = dague_init(1, &argc, &argv);
    assert( NULL != dague );

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_RealDouble, matrix_Tile,
                               nodes, rank, NB, NB, N, N, 0, 0,
                               N, N, 1, 1, 1));

    handle = (dague_handle_t*) (dtt_handle = dague_dtt_bug_replicator_new(&ddescA.super.super));
    assert( NULL != handle );

    /* initialize the first tile */
    if( 0 == rank ) {
        for( i = 0; i < NB; i++ )
            for( j = 0; j < NB; j++ )
                ((double*)ddescA.mat)[i * NB + j] = (double)(i * NB + j);
        dump_double_array("Original ", (double*)ddescA.mat, 0, 0, NB, NB, NB);
    }
#if defined(DAGUE_HAVE_MPI)
    dague_type_create_contiguous(NB*NB, dague_datatype_double_t, &tile_dtt);
    MPI_Type_set_name(tile_dtt, "TILE_DTT");
    MPI_Type_commit(&tile_dtt);
    dague_arena_construct(dtt_handle->arenas[DAGUE_dtt_bug_replicator_DTT1_ARENA],
                          NB*NB*sizeof(double),
                          DAGUE_ARENA_ALIGNMENT_SSE, tile_dtt);

    dague_type_create_vector(NB, 1, NB, dague_datatype_double_t, &vdtt1);
    dague_type_create_resized(vdtt1, 0, sizeof(dague_datatype_double_t), &vdtt2);
    dague_type_create_contiguous(NB, vdtt2, &vdtt);
    MPI_Type_set_name(vdtt, "TILE_DTT");
    MPI_Type_commit(&vdtt);
    dague_arena_construct(dtt_handle->arenas[DAGUE_dtt_bug_replicator_DTT2_ARENA],
                          NB*NB*sizeof(double),
                          DAGUE_ARENA_ALIGNMENT_SSE, vdtt);
#endif

    dague_enqueue( dague, handle );

    dague_context_wait(dague);

    dague_fini( &dague);
#if defined(DAGUE_HAVE_MPI)
    MPI_Finalize();
#endif
    return 0;
}
