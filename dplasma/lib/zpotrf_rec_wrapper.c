/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zpotrf_Url_rec.h"
#include "zpotrf_Lrl_rec.h"

dague_handle_t*
dplasma_zpotrf_rec_New(PLASMA_enum uplo, tiled_matrix_desc_t *A, int *info)
{
    dague_handle_t *dague_zpotrf_rec = NULL;

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    *info = 0;
    if ( uplo == PlasmaUpper ) {
        dague_zpotrf_rec = (dague_handle_t*)dague_zpotrf_Url_rec_new( uplo, (dague_ddesc_t*)A, info);
    } else {
        dague_zpotrf_rec = (dague_handle_t*)dague_zpotrf_Lrl_rec_new( uplo, (dague_ddesc_t*)A, info);
    }
    ((struct dague_zpotrf_Lrl_rec_handle*)dague_zpotrf_rec)->PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );

    dplasma_add2arena_tile(((dague_zpotrf_Url_rec_handle_t*)dague_zpotrf_rec)->arenas[DAGUE_zpotrf_Url_rec_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zpotrf_rec;
}

void
dplasma_zpotrf_rec_Destruct( dague_handle_t *o )
{
    dague_zpotrf_Url_rec_handle_t *opotrf = (dague_zpotrf_Url_rec_handle_t *)o;

    dplasma_datatype_undefine_type( &(opotrf->arenas[DAGUE_zpotrf_Url_rec_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

int dplasma_zpotrf_rec( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA)
{
    dague_handle_t *dague_zpotrf_rec = NULL;
    int info = 0, ginfo = 0 ;

    dague_zpotrf_rec = dplasma_zpotrf_rec_New(uplo, ddescA, &info);

    if ( dague_zpotrf_rec != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zpotrf_rec);
        dplasma_progress(dague);
        dplasma_zpotrf_rec_Destruct( dague_zpotrf_rec );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}

/*
 * Functions for advanced user allowing to choose right or left-looking variant
 */
dague_handle_t*
dplasma_zpotrfl_rec_New(const PLASMA_enum looking, PLASMA_enum uplo,
                    tiled_matrix_desc_t *A, int *info)
{
    dague_handle_t *dague_zpotrf_rec = NULL;

    *info = 0;

    if ( looking == PlasmaRight ) {
        if ( uplo == PlasmaUpper ) {
            dague_zpotrf_rec = (dague_handle_t*)dague_zpotrf_Url_rec_new( uplo, (dague_ddesc_t*)A, info);
        } else {
            dague_zpotrf_rec = (dague_handle_t*)dague_zpotrf_Lrl_rec_new( uplo, (dague_ddesc_t*)A, info);
        }
    } /* else { */
    /*     if ( uplo == PlasmaUpper ) { */
    /*         dague_zpotrf = (dague_handle_t*)dague_zpotrf_Ull_new( */
    /*             (dague_ddesc_t*)A,  */
    /*             pri_change, uplo, info,  */
    /*             A->m, A->n, A->mb, A->nb, A->mt, A->nt); */
    /*     } else { */
    /*         dague_zpotrf = (dague_handle_t*)dague_zpotrf_Lll_new( */
    /*             (dague_ddesc_t*)A,  */
    /*             pri_change, uplo, info,  */
    /*             A->m, A->n, A->mb, A->nb, A->mt, A->nt); */
    /*     } */
    /* } */
    ((struct dague_zpotrf_Lrl_rec_handle*)dague_zpotrf_rec)->PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );

    dplasma_add2arena_tile(((dague_zpotrf_Url_rec_handle_t*)dague_zpotrf_rec)->arenas[DAGUE_zpotrf_Url_rec_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zpotrf_rec;
}
