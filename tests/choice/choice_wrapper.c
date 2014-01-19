#include "dague.h"
#include <data_distribution.h>
#include <arena.h>

#if defined(HAVE_MPI)
#include <mpi.h>
static MPI_Datatype block;
#endif
#include <stdio.h>

#include "choice.h"
#include "choice_wrapper.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_handle_t *choice_new(dague_ddesc_t *A, int size, int *decision, int nb, int world)
{
    dague_choice_handle_t *o = NULL;

    if( nb <= 0 || size <= 0 ) {
        fprintf(stderr, "To work, CHOICE nb and size must be > 0\n");
        return (dague_handle_t*)o;
    }

    o = dague_choice_new(A, nb, world, decision);

#if defined(HAVE_MPI)
    {
        MPI_Type_vector(1, size, size, MPI_BYTE, &block);
        MPI_Type_commit(&block);
        dague_arena_construct(o->arenas[DAGUE_choice_DEFAULT_ARENA],
                              size * sizeof(char), size * sizeof(char),
                              block);
    }
#endif

    return (dague_handle_t*)o;
}

/**
 * @param [INOUT] o the dague object to destroy
 */
void choice_destroy(dague_handle_t *o)
{
    dague_choice_handle_t *c = (dague_choice_handle_t*)o;
    (void)c;

#if defined(HAVE_MPI)
    MPI_Type_free( &block );
#endif

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}
