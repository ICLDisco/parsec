#include "dague.h"
#include "arena.h" /* get rid of this */
#include "data_distribution.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "touch.h"

#define BLOCK 10
#define N     100
#define TYPE  matrix_RealFloat

int main( int argc, char** argv )
{
    dague_context_t* dague;
    dague_touch_handle_t* object;
    two_dim_block_cyclic_t descA;

    (void)argc; (void)argv;
    dague = dague_init(1, NULL, NULL);
    assert( NULL != dague );

    two_dim_block_cyclic_init( &descA, TYPE, matrix_Tile,
                               1 /*nodes*/, 0 /*rank*/,
                               BLOCK, BLOCK, N, N,
                               0, 0, N, N, 1, 1, 1);
    descA.mat = dague_data_allocate( descA.super.nb_local_tiles *
                                     descA.super.bsiz *
                                     dague_datadist_getsizeoftype(TYPE) );

    object = dague_touch_new( &descA, 0 );
    assert( NULL != object );

    dague_arena_construct( object->arenas[DAGUE_touch_DEFAULT_ARENA],
                           descA.super.mb * descA.super.nb * dague_datadist_getsizeoftype(TYPE),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           DAGUE_DATATYPE_NULL);  /* change for distributed cases */

    dague_enqueue( dague, (dague_handle_t*)object );

    dague_progress(dague);

    free(descA.mat);

    dague_fini( &dague);

    return 0;
}
