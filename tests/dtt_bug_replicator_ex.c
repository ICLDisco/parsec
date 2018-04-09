/*
 * Copyright (c) 2013-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dtt_bug_replicator.h"
#include "parsec/arena.h"
#include <math.h>

#define N     10
#define NB    3

extern void dump_double_array(char* msg, double* mat, int i, int j, int nb, int mb, int lda);

#define PASTE_CODE_ALLOCATE_MATRIX(DC, COND, TYPE, INIT_PARAMS)      \
    TYPE##_t DC;                                                     \
    if(COND) {                                                          \
        TYPE##_init INIT_PARAMS;                                        \
        DC.mat = parsec_data_allocate((size_t)DC.super.nb_local_tiles * \
                                        (size_t)DC.super.bsiz *      \
                                        (size_t)parsec_datadist_getsizeoftype(DC.super.mtype)); \
        parsec_data_collection_set_key((parsec_data_collection_t*)&DC, #DC);            \
    }


int main( int argc, char** argv )
{
    parsec_context_t* parsec;
    parsec_taskpool_t* tp;
    parsec_datatype_t tile_dtt, vdtt1, vdtt2, vdtt;
    parsec_dtt_bug_replicator_taskpool_t *dtt_tp;
    int nodes, rank, i, j, rc;
    (void)argc; (void)argv;

#if defined(PARSEC_HAVE_MPI)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &nodes);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nodes = 1;
    rank = 0;
#endif

    parsec = parsec_init(1, &argc, &argv);
    assert( NULL != parsec );

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_RealDouble, matrix_Tile,
                               nodes, rank, NB, NB, N, N, 0, 0,
                               N, N, 1, 1, 1));

    tp = (parsec_taskpool_t*) (dtt_tp = parsec_dtt_bug_replicator_new(&dcA.super.super));
    assert( NULL != tp );

    /* initialize the first tile */
    if( 0 == rank ) {
        for( i = 0; i < NB; i++ )
            for( j = 0; j < NB; j++ )
                ((double*)dcA.mat)[i * NB + j] = (double)(i * NB + j);
        dump_double_array("Original ", (double*)dcA.mat, 0, 0, NB, NB, NB);
    }
    parsec_type_create_contiguous(NB*NB, parsec_datatype_double_t, &tile_dtt);
    parsec_arena_construct(dtt_tp->arenas[PARSEC_dtt_bug_replicator_DTT1_ARENA],
                           NB*NB*sizeof(double),
                           PARSEC_ARENA_ALIGNMENT_SSE, tile_dtt);

    parsec_type_create_vector(NB, 1, NB, parsec_datatype_double_t, &vdtt1);
    parsec_type_create_resized(vdtt1, 0, sizeof(parsec_datatype_double_t), &vdtt2);
    parsec_type_create_contiguous(NB, vdtt2, &vdtt);
    parsec_arena_construct(dtt_tp->arenas[PARSEC_dtt_bug_replicator_DTT2_ARENA],
                           NB*NB*sizeof(double),
                           PARSEC_ARENA_ALIGNMENT_SSE, vdtt);

    rc = parsec_enqueue( parsec, tp);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_type_free(&tile_dtt);
    parsec_type_free(&vdtt1);
    parsec_type_free(&vdtt2);
    parsec_type_free(&vdtt);

    parsec_fini( &parsec);
#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
    return 0;
}
