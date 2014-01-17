/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map2.h"

static int
dplasma_zgeadd_operator( dague_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         const tiled_matrix_desc_t *descB,
                         const void *_A, void *_B,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    const dague_complex64_t *A     = (dague_complex64_t*)_A;
    dague_complex64_t       *B     = (dague_complex64_t*)_B;
    dague_complex64_t        alpha = *((dague_complex64_t*)args);
    int j;
    int tempmm, tempnn, ldam, ldbm;
    (void)eu;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );
    ldbm = BLKLDD( *descB, m );

    switch ( uplo ) {
    case PlasmaLower:
        for (j = 0; j < tempnn; j++, tempmm--, A+=ldam+1, B+=ldbm+1) {
            cblas_zaxpy(tempmm, CBLAS_SADDR(alpha), A, 1, B, 1);
        }
        break;
    case PlasmaUpper:
        for (j = 0; j < tempnn; j++, A+=ldam, B+=ldbm) {
            cblas_zaxpy(j+1, CBLAS_SADDR(alpha), A, 1, B, 1);
        }
        break;
    case PlasmaUpperLower:
    default:
        for (j = 0; j < tempnn; j++, A+=ldam, B+=ldbm) {
            cblas_zaxpy(tempmm, CBLAS_SADDR(alpha), A, 1, B, 1);
        }
    }

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgeadd_New - Generates an object that computes the operation B =
 * alpha * A + B
 *
 * See dplasma_map2_New() for further information.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower
 *          triangular:
 *          = PlasmaUpperLower: All matrix A is referenced;
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The descriptor of the distributed matrix A of size M-by-N.
 *
 * @param[in,out] B
 *          The descriptor of the distributed matrix B of size M-by-N.
 *          On exit, the matrix B data are overwritten by the result of alpha*A+B
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgeadd_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeadd
 * @sa dplasma_zgeadd_Destruct
 * @sa dplasma_cgeadd_New
 * @sa dplasma_dgeadd_New
 * @sa dplasma_sgeadd_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgeadd_New( PLASMA_enum uplo,
                    dague_complex64_t alpha,
                    const tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *B)
{
    dague_complex64_t *a = (dague_complex64_t*)malloc(sizeof(dague_complex64_t));
    *a = alpha;

    return dplasma_map2_New(uplo, A, B,
                            dplasma_zgeadd_operator, (void *)a);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlacpy_Destruct - Free the data structure associated to an object
 *  created with dplasma_zlacpy_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlacpy_New
 * @sa dplasma_zlacpy
 *
 ******************************************************************************/
void
dplasma_zgeadd_Destruct( dague_handle_t *o )
{
    dplasma_map2_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgeadd - Generates an object that computes the operation B =
 * alpha * A + B
 *
 * See dplasma_map2() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower
 *          triangular:
 *          = PlasmaUpperLower: All matrix A is referenced;
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The descriptor of the distributed matrix A of size M-by-N.
 *
 * @param[in,out] B
 *          The descriptor of the distributed matrix B of size M-by-N.
 *          On exit, the matrix B data are overwritten by the result of alpha*A+B
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeadd_New
 * @sa dplasma_zgeadd_Destruct
 * @sa dplasma_cgeadd
 * @sa dplasma_dgeadd
 * @sa dplasma_sgeadd
 *
 ******************************************************************************/
int
dplasma_zgeadd( dague_context_t *dague,
                PLASMA_enum uplo,
                dague_complex64_t alpha,
                const tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *B)
{
    dague_handle_t *dague_zgeadd = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_zgeadd", "illegal value of uplo");
        return -2;
    }

    dague_zgeadd = dplasma_zgeadd_New(uplo, alpha, A, B);

    if ( dague_zgeadd != NULL )
    {
        dague_enqueue(dague, (dague_handle_t*)dague_zgeadd);
        dplasma_progress(dague);
        dplasma_zgeadd_Destruct( dague_zgeadd );
    }
    return 0;
}
