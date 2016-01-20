/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include <cblas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

static int
dplasma_zlascal_operator( dague_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         void *_A,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    dague_complex64_t *A     = (dague_complex64_t*)_A;
    dague_complex64_t  alpha = *((dague_complex64_t*)args);
    int i;
    int tempmm, tempnn, ldam;
    (void)eu;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );

    /* Overwrite uplo when outside the diagonal */
    if (m != n) {
        uplo = PlasmaUpperLower;
    }

    switch ( uplo ) {
    case PlasmaUpper:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( dplasma_imin( i+1, tempmm ), CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;

    case PlasmaLower:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( dplasma_imax( tempmm, tempmm-i ), CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;
    default:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( tempmm, CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;
    }

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlascal_New - Generates the handle that scales a matrix by a given scalar.
 *
 * See dplasma_map_New() for further information.
 *
 *  WARNINGS:
 *      - The computations are not done by this call.
 *      - This routine is equivalent to the pzlascal routine from ScaLapack and
 *  not to the zlascl/pzlascl routines from Lapack/ScaLapack.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *          = PlasmaUpperHessenberg: A is an upper Hessenberg matrix (Not
 *          supported)
 *
 * @param[in] alpha
 *          The scalar to use to scale the matrix.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the matrix A is scaled by alpha.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlascal_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal
 * @sa dplasma_zlascal_Destruct
 * @sa dplasma_clascal_New
 * @sa dplasma_dlascal_New
 * @sa dplasma_slascal_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zlascal_New( PLASMA_enum uplo,
                     dague_complex64_t alpha,
                     tiled_matrix_desc_t *A )
{
    dague_complex64_t *a = (dague_complex64_t*)malloc(sizeof(dague_complex64_t));
    *a = alpha;

    return dplasma_map_New( uplo, A, dplasma_zlascal_operator, (void*)a );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlascal_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zlascal_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal_New
 * @sa dplasma_zlascal
 *
 ******************************************************************************/
void
dplasma_zlascal_Destruct( dague_handle_t *handle )
{
    dplasma_map_Destruct(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlascal - Scales a matrix by a given scalar.
 *
 * See dplasma_map() for further information.
 *
 *  WARNINGS:
 *      - This routine is equivalent to the pzlascal routine from ScaLapack and
 *  not to the zlascl/pzlascl routines from Lapack/ScaLapack.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *          = PlasmaUpperHessenberg: A is an upper Hessenberg matrix (Not
 *          supported)
 *
 * @param[in] alpha
 *          The scalar to use to scale the matrix.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the matrix A is scaled by alpha.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal_New
 * @sa dplasma_zlascal_Destruct
 * @sa dplasma_clascal
 * @sa dplasma_dlascal
 * @sa dplasma_slascal
 *
 ******************************************************************************/
int
dplasma_zlascal( dague_context_t     *dague,
                 PLASMA_enum          uplo,
                 dague_complex64_t    alpha,
                 tiled_matrix_desc_t *A )
{
    dague_handle_t *dague_zlascal = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zlascal", "illegal value of type");
        return -2;
    }

    dague_zlascal = dplasma_zlascal_New(uplo, alpha, A);

    if ( dague_zlascal != NULL ) {
        dague_enqueue(dague, (dague_handle_t*)dague_zlascal);
        dplasma_progress(dague);
        dplasma_zlascal_Destruct( dague_zlascal );
    }
    return 0;
}
