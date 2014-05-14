#include "dague_internal.h"
#include <data_distribution.h>
#include <arena.h>

#if defined(HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "BT_reduction.h"
#include "BT_reduction_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] nb   tile size 
 * @param [IN] nt   number of tiles
 *
 * @return the dague object to schedule.
 */
dague_handle_t *BT_reduction_new(struct tiled_matrix_desc_t *A, int nb, int nt)
{
    dague_BT_reduction_handle_t *o = NULL;

    o = dague_BT_reduction_new(A, nb, nt);

#if defined(HAVE_MPI)
    {
        MPI_Aint extent;
    	MPI_Type_contiguous(nb, MPI_INT, &block);
        MPI_Type_commit(&block);
#if defined(HAVE_MPI_20)
        MPI_Aint lb = 0; 
        MPI_Type_get_extent(block, &lb, &extent);
#else
        MPI_Type_extent(block, &extent);
#endif  /* defined(HAVE_MPI_20) */
        dague_arena_construct(o->arenas[DAGUE_BT_reduction_DEFAULT_ARENA],
                              extent, DAGUE_ARENA_ALIGNMENT_SSE,
                              block);
    }
#endif

    return (dague_handle_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void BT_reduction_destroy(dague_handle_t *o)
{
#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}
