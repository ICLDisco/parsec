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

#include "zpotrf_Url.h"
#include "zpotrf_Lrl.h"
#include "zpotrf_ll.h"

dague_object_t* 
dplasma_zpotrf_New(const PLASMA_enum uplo, tiled_matrix_desc_t *A, int *info)
{
    dague_object_t *dague_zpotrf = NULL;
    int pri_change = dplasma_aux_get_priority( "POTRF", A );

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    *info = 0;
    if ( uplo == PlasmaUpper ) {
        dague_zpotrf = (dague_object_t*)dague_zpotrf_Url_new(
            (dague_ddesc_t*)A, 
            pri_change, uplo, info, 
            A->m, A->n, A->mb, A->nb, A->mt, A->nt);
    } else {
        dague_zpotrf = (dague_object_t*)dague_zpotrf_Lrl_new(
            (dague_ddesc_t*)A, 
            pri_change, uplo, info, 
            A->m, A->n, A->mb, A->nb, A->mt, A->nt);
    }
    
    dplasma_add2arena_tile(((dague_zpotrf_Url_object_t*)dague_zpotrf)->arenas[DAGUE_zpotrf_Url_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);
    
    return dague_zpotrf;
}
 
void
dplasma_zpotrf_Destruct( dague_object_t *o )
{
    dague_zpotrf_Url_object_t *opotrf = (dague_zpotrf_Url_object_t *)o;
    int uplo    = ((dague_zpotrf_Url_object_t *)o)->uplo;
    int looking = PlasmaRight; /*((dague_zpotrf_Url_object_t *)o)->uplo;*/
    
    dplasma_datatype_undefine_type( &(opotrf->arenas[DAGUE_zpotrf_Url_DEFAULT_ARENA]->opaque_dtt) );

    if (looking == PlasmaRight ) {
        if ( uplo == PlasmaUpper ) {
            dague_zpotrf_Url_destroy((dague_zpotrf_Url_object_t *)o);
        } else {
            dague_zpotrf_Lrl_destroy((dague_zpotrf_Lrl_object_t *)o);
        }
    } /* else { */
    /*     if ( uplo == PlasmaUpper ) { */
    /*         dague_zpotrf_Ull_destroy((dague_zpotrf_Ull_object_t *)o); */
    /*     } else { */
    /*         dague_zpotrf_Lll_destroy((dague_zpotrf_Lll_object_t *)o); */
    /*     } */
    /* } */
}

int dplasma_zpotrf( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA) 
{
    dague_object_t *dague_zpotrf = NULL;
    int info = 0, ginfo = 0 ;

    dague_zpotrf = dplasma_zpotrf_New(uplo, ddescA, &info);

    if ( dague_zpotrf != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zpotrf);
        dplasma_progress(dague);
        dplasma_zpotrf_Destruct( dague_zpotrf );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}

/*
+ * Functions for advanced user allowing to choose right or left-looking variant 
+ */
dague_object_t* 
dplasma_zpotrfl_New(const PLASMA_enum looking, const PLASMA_enum uplo, 
                    tiled_matrix_desc_t *A, int *info)
{
    dague_object_t *dague_zpotrf = NULL;
    int pri_change = dplasma_aux_get_priority( "POTRF", A );
 
    *info = 0;
   
    if ( looking == PlasmaRight ) {
        if ( uplo == PlasmaUpper ) {
            dague_zpotrf = (dague_object_t*)dague_zpotrf_Url_new(
                (dague_ddesc_t*)A, 
                pri_change, uplo, info, 
                A->m, A->n, A->mb, A->nb, A->mt, A->nt);
        } else {
            dague_zpotrf = (dague_object_t*)dague_zpotrf_Lrl_new(
                (dague_ddesc_t*)A, 
                pri_change, uplo, info, 
                A->m, A->n, A->mb, A->nb, A->mt, A->nt);
        }
    } /* else { */
    /*     if ( uplo == PlasmaUpper ) { */
    /*         dague_zpotrf = (dague_object_t*)dague_zpotrf_Ull_new( */
    /*             (dague_ddesc_t*)A,  */
    /*             pri_change, uplo, info,  */
    /*             A->m, A->n, A->mb, A->nb, A->mt, A->nt); */
    /*     } else { */
    /*         dague_zpotrf = (dague_object_t*)dague_zpotrf_Lll_new( */
    /*             (dague_ddesc_t*)A,  */
    /*             pri_change, uplo, info,  */
    /*             A->m, A->n, A->mb, A->nb, A->mt, A->nt); */
    /*     } */
    /* } */
    
    dplasma_add2arena_tile(((dague_zpotrf_Url_object_t*)dague_zpotrf)->arenas[DAGUE_zpotrf_Url_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);
    
    return dague_zpotrf;
}
