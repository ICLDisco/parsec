/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/diag_band_to_rect.h"
#include "dplasma/lib/zhbrdt.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zheev_New - TO FILL IN CORRECTLY BY THE PERSON WORKING ON IT !!!
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *
 * @param[in,out] A
 *
 * @param[out] info
 *
 *******************************************************************************
 *
 * @return
 *
 *******************************************************************************
 *
 * @sa dplasma_zheev
 * @sa dplasma_zheev_Destruct
 * @sa dplasma_cheev_New
 * @sa dplasma_dheev_New
 * @sa dplasma_sheev_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zheev_New(PLASMA_enum jobz, PLASMA_enum uplo,
                  tiled_matrix_desc_t* A,
                  tiled_matrix_desc_t* W,  /* Should be removed: internal workspace as T */
                  tiled_matrix_desc_t* Z,
                  int* info )
{
    (void)Z;
    *info = 0;

    /* Check input arguments */
    if( (jobz != PlasmaNoVec) && (jobz != PlasmaVec) ) {
        dplasma_error("dplasma_zheev_New", "illegal value of jobz");
        *info = -1;
        return NULL;
    }
    if( (uplo != PlasmaLower) && (uplo != PlasmaUpper) ) {
        dplasma_error("dplasma_zheev_New", "illegal value of uplo");
        *info = -2;
        return NULL;
    }

    /* TODO: remove those extra check when those options will be implemented */
    if( jobz == PlasmaVec ) {
        dplasma_error("dplasma_zheev_New", "PlasmaVec not implemented (yet)");
        *info = -1;
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
                               dague_datatype_double_complex_t, A->mb);

        return zheev_compound;
    }
    else {
        dplasma_error("dplasma_zheev_New", "PlasmaUpper not implemented (yet)");
        *info = -2;
        return NULL;
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zheev_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zheev_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zheev_New
 * @sa dplasma_zheev
 *
 ******************************************************************************/
void
dplasma_zheev_Destruct( dague_handle_t *handle )
{
#if 0
    two_dim_block_cyclic_t* T = ???
    dague_data_free(T->mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)T); free(T);

    dague_matrix_del2arena( ((dague_diag_band_to_rect_handle_t *)handle)->arenas[DAGUE_diag_band_to_rect_DEFAULT_ARENA] );
#endif
    dague_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zheev - TO FILL IN CORRECTLY BY THE PERSON WORKING ON IT !!!
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * COPY OF NEW INTERFACE
 *
 *******************************************************************************
 *
 * @return
 *
 *******************************************************************************
 *
 * @sa dplasma_zheev_New
 * @sa dplasma_zheev_Destruct
 * @sa dplasma_cheev
 * @sa dplasma_dheev
 * @sa dplasma_sheev
 *
 ******************************************************************************/
int
dplasma_zheev( dague_context_t *dague,
               PLASMA_enum jobz, PLASMA_enum uplo,
               tiled_matrix_desc_t* A,
               tiled_matrix_desc_t* W, /* Should be removed */
               tiled_matrix_desc_t* Z )
{
    dague_handle_t *dague_zheev = NULL;
    int info = 0;

    /* Check input arguments */
    if( (jobz != PlasmaNoVec) && (jobz != PlasmaVec) ) {
        dplasma_error("dplasma_zheev", "illegal value of jobz");
        return -1;
    }
    if( (uplo != PlasmaLower) && (uplo != PlasmaUpper) ) {
        dplasma_error("dplasma_zheev", "illegal value of uplo");
        return -2;
    }

    dague_zheev = dplasma_zheev_New( jobz, uplo, A, W, Z, &info );

    if ( dague_zheev != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zheev);
        dplasma_progress(dague);
        dplasma_zheev_Destruct( dague_zheev );
        return info;
    }
    else {
        return -101;
    }
}
