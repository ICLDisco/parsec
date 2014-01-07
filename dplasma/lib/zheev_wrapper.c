/*
 * Copyright (c) 2013      The University of Tennessee and The University
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

#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/diag_band_to_rect.h"
#include "dplasma/lib/zhbrdt.h"

/*    SUBROUTINE PZHEEV( JOBZ, UPLO, N, A, IA, JA, DESCA, W, Z, IZ, JZ, 
     $                   DESCZ, WORK, LWORK, RWORK, LRWORK, INFO ) */
dague_handle_t*
dplasma_zheev_New(PLASMA_enum jobz, PLASMA_enum uplo,
                  tiled_matrix_desc_t* A,
                  tiled_matrix_desc_t* W,
                  tiled_matrix_desc_t* Z,
                  int* info )
{
    /* TODO: remove this when implemented */
    if( jobz == PlasmaVec ) {
        dplasma_error("DPLASMA_zheev_New", "Non-blocking interface is not implemented (yet)");
        *info = -1;
        return NULL;
    }

    /* Check input arguments */
    if( jobz != PlasmaNoVec && jobz != PlasmaVec ) {
        dplasma_error("DPLASMA_zheev", "illegal value of jobz");
        *info = -1;
        return NULL;
    }
    /* TODO: remove this when implemented */
    if( jobz == PlasmaVec ) {
        dplasma_error("DPLASMA_zheev", "PlasmaVec jobz is not implemented (yet)");
        *info = -1;
        return NULL;
    }

    if( uplo != PlasmaLower && uplo != PlasmaUpper ) {
        dplasma_error("DPLASMA_zheev", "illegal value of uplo");
        *info = -2;
        return NULL;
    }

    if( PlasmaLower == uplo ) {
        dague_handle_t* zherbt_obj, * zhbrdt_obj;
        dague_diag_band_to_rect_handle_t* band2rect_obj;
        dague_handle_t* zheev_compound;
        sym_two_dim_block_cyclic_t* As = (sym_two_dim_block_cyclic_t*)A;
        int ib=A->nb/3;

        two_dim_block_cyclic_t* T = calloc(1, sizeof(two_dim_block_cyclic_t));
        two_dim_block_cyclic_init(T, matrix_ComplexDouble, matrix_Tile,
             A->super.nodes, A->super.myrank, ib, A->nb, A->mt*ib, A->n, 0, 0,
             A->mt*ib, A->n, As->grid.strows, As->grid.strows, As->grid.rows);
        T->mat = dague_data_allocate((size_t)T->super.nb_local_tiles *
                                     (size_t)T->super.bsiz *
                                     (size_t)dague_datadist_getsizeoftype(T->super.mtype));
        dague_ddesc_set_key((dague_ddesc_t*)T, "zheev_ddescT");

        zherbt_obj = (dague_handle_t*)dplasma_zherbt_New( uplo, ib, A, (tiled_matrix_desc_t*)T );
        band2rect_obj = dague_diag_band_to_rect_new((sym_two_dim_block_cyclic_t*)A, (two_dim_block_cyclic_t*)W,
                A->mt, A->nt, A->mb, A->nb, sizeof(dague_complex64_t));
        zhbrdt_obj = (dague_handle_t*)dplasma_zhbrdt_New(W);
        zheev_compound = dague_compose( zherbt_obj, (dague_handle_t*)band2rect_obj );
        zheev_compound = dague_compose( zheev_compound, zhbrdt_obj );

        dague_arena_t* arena = band2rect_obj->arenas[DAGUE_diag_band_to_rect_DEFAULT_ARENA];
        dplasma_add2arena_tile(arena,
                               A->mb*A->nb*sizeof(dague_complex64_t),
                               DAGUE_ARENA_ALIGNMENT_SSE,
                               MPI_DOUBLE_COMPLEX, A->mb);

        return zheev_compound;
    }
    else {
        /* TODO: remove this when implemented */
        dplasma_error("DPLASMA_zheev", "PlasmaUpper uplo is not implemented (yet)");
        *info = -1;
        return NULL;
    }
}

void
dplasma_zheev_Destruct( dague_handle_t *o )
{
#if 0
    two_dim_block_cyclic_t* T = ???
    dague_data_free(T->mat);
    dague_ddesc_destroy((dague_ddesc_t*)T); free(T);
    
    dplasma_datatype_undefine_type( &(((dague_diag_band_to_rect_handle_t *)o)->arenas[DAGUE_diag_band_to_rect_DEFAULT_ARENA]->opaque_dtt) );
#endif
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

int
dplasma_zheev( dague_context_t *dague, PLASMA_enum jobz, PLASMA_enum uplo,
                    tiled_matrix_desc_t* A, 
                    tiled_matrix_desc_t* W,
                    tiled_matrix_desc_t* Z,
                    int* info )
{
    dague_handle_t *dague_zheev = NULL;

    dague_zheev = dplasma_zheev_New( jobz, uplo, A, W, Z, info );

    if ( dague_zheev != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zheev);
        dplasma_progress(dague);
        dplasma_zheev_Destruct( dague_zheev );
        return 0;
    }
    else {
        return -101;
    }
}


