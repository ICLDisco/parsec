#include "dague_internal.h"
#include <data_distribution.h>
#include <arena.h>

#if defined(HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "a2a.h"
#include "a2a_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_object_t *a2a_new(tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, int size, int repeat)
{
    int worldsize;
    dague_a2a_object_t *o = NULL;
#if defined(HAVE_MPI)
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
#else
    worldsize = 1;
#endif
    
    if( repeat <= 0 ) {
        fprintf(stderr, "To work, A2A must do at least one exchange of at least one byte\n");
        return (dague_object_t*)o;
    }

    o = dague_a2a_new(A, B, repeat, worldsize);

#if defined(HAVE_MPI)
    {
        MPI_Aint extent;
    	MPI_Type_contiguous(size, MPI_INT, &block);
        MPI_Type_commit(&block);
#if defined(HAVE_MPI_20)
        MPI_Aint lb = 0; 
        MPI_Type_get_extent(block, &lb, &extent);
#else
        MPI_Type_extent(block, &extent);
#endif  /* defined(HAVE_MPI_20) */
        dague_arena_construct(o->arenas[DAGUE_a2a_DEFAULT_ARENA],
                              extent, DAGUE_ARENA_ALIGNMENT_SSE,
                              block);
    }
#endif

    return (dague_object_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void a2a_destroy(dague_object_t *o)
{
#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}
