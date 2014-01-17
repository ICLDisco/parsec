/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */
#include "dague_internal.h"
#include <lapacke.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map2.h"

static int
dplasma_zlacpy_operator( dague_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         const tiled_matrix_desc_t *descB,
                         const void *_A, void *_B,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    int tempmm, tempnn, ldam, ldbm;
    const dague_complex64_t *A = (const dague_complex64_t*)_A;
    dague_complex64_t       *B = (dague_complex64_t*)_B;
    (void)eu;
    (void)args;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );
    ldbm = BLKLDD( *descB, m );

    LAPACKE_zlacpy_work(
        LAPACK_COL_MAJOR, lapack_const( uplo ), tempmm, tempnn, A, ldam, B, ldbm);

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zlacpy_New - Generates an object that performs a copy of the matrix A
 * into the matrix B.
 *
 * See dplasma_map2_New() for further information.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is copied:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is refrenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed original matrix A. Any tiled matrix
 *          descriptor can be used. However, if the data is stored in column
 *          major, the tile distribution must match the one of the matrix B.
 *
 * @param[in,out] B
 *          Descriptor of the distributed destination matrix B. Any tiled matrix
 *          descriptor can be used, with no specific storage.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlacpy_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlacpy
 * @sa dplasma_zlacpy_Destruct
 * @sa dplasma_clacpy_New
 * @sa dplasma_dlacpy_New
 * @sa dplasma_slacpy_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zlacpy_New( PLASMA_enum uplo,
                    const tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *B)
{
    dague_handle_t* object;

    object = dplasma_map2_New(uplo, A, B,
                              dplasma_zlacpy_operator, NULL );

    return object;
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
dplasma_zlacpy_Destruct( dague_handle_t *o )
{
    dplasma_map2_Destruct( o );
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zlacpy - Generates an object that performs a copy of the matrix A
 * into the matrix B.
 *
 * See dplasma_map2() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is copied:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is refrenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed original matrix A. Any tiled matrix
 *          descriptor can be used. However, if the data is stored in column
 *          major, the tile distribution must match the one of the matrix B.
 *
 * @param[in,out] B
 *          Descriptor of the distributed destination matrix B. Any tiled matrix
 *          descriptor can be used, with no specific storage.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlacpy_New
 * @sa dplasma_zlacpy_Destruct
 * @sa dplasma_clacpy
 * @sa dplasma_dlacpy
 * @sa dplasma_slacpy
 *
 ******************************************************************************/
int
dplasma_zlacpy( dague_context_t *dague,
                PLASMA_enum uplo,
                const tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *B)
{
    dague_handle_t *dague_zlacpy = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_zlacpy", "illegal value of uplo");
        return -2;
    }

    if ( (A->m > B->m) || (A->n > B->n) ) {
        dplasma_error("dplasma_zlacpy", "illegal matrix A (B can't contain A)");
        return -3;
    }

    dague_zlacpy = dplasma_zlacpy_New(uplo, A, B);

    if ( dague_zlacpy != NULL )
    {
        dague_enqueue(dague, (dague_handle_t*)dague_zlacpy);
        dplasma_progress(dague);
        dplasma_zlacpy_Destruct( dague_zlacpy );
    }
    return 0;
}
