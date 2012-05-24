/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zgetrf_sp.h"



dague_object_t*
dplasma_zgetrf_sp_New(double criteria, tiled_matrix_desc_t *A, int *info)
{
    dague_object_t *dague_zgetrf_sp = NULL;

    *info = 0;
    dague_zgetrf_sp = (dague_object_t*)dague_zgetrf_sp_new(*A, (dague_ddesc_t*)A, criteria, info);

    dplasma_add2arena_tile(((dague_zgetrf_sp_object_t*)dague_zgetrf_sp)->arenas[DAGUE_zgetrf_sp_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zgetrf_sp;
}

void
dplasma_zgetrf_sp_Destruct( dague_object_t *o )
{
    dague_zgetrf_sp_object_t *dague_zgetrf_sp = (dague_zgetrf_sp_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zgetrf_sp->arenas[DAGUE_zgetrf_sp_DEFAULT_ARENA   ]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

int dplasma_zgetrf_sp( dague_context_t *dague, const double criteria, tiled_matrix_desc_t* ddescA)
{
    dague_object_t *dague_zgetrf_sp = NULL;
    int info = 0, ginfo = 0 ;

    dague_zgetrf_sp = dplasma_zgetrf_sp_New(criteria, ddescA, &info);

    if ( dague_zgetrf_sp != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgetrf_sp);
        dplasma_progress(dague);
        dplasma_zgetrf_sp_Destruct( dague_zgetrf_sp );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}
