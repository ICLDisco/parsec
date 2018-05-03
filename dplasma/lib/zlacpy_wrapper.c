/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include <lapacke.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map2.h"

static int
dplasma_zlacpy_operator( parsec_execution_stream_t *es,
                         const parsec_tiled_matrix_dc_t *descA,
                         const parsec_tiled_matrix_dc_t *descB,
                         const void *_A, void *_B,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    int tempmm, tempnn, ldam, ldbm;
    const parsec_complex64_t *A = (const parsec_complex64_t*)_A;
    parsec_complex64_t       *B = (parsec_complex64_t*)_B;
    (void)es;
    (void)args;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( descA, m );
    ldbm = BLKLDD( descB, m );

    LAPACKE_zlacpy_work(
        LAPACK_COL_MAJOR, lapack_const( uplo ), tempmm, tempnn, A, ldam, B, ldbm);

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlacpy_New - Generates an taskpool that performs a copy of the matrix A
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
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
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
parsec_taskpool_t*
dplasma_zlacpy_New( PLASMA_enum uplo,
                    const parsec_tiled_matrix_dc_t *A,
                    parsec_tiled_matrix_dc_t *B)
{
    parsec_taskpool_t* tp;

    tp = dplasma_map2_New(uplo, PlasmaNoTrans, A, B,
                          dplasma_zlacpy_operator, NULL );

    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlacpy_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zlacpy_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlacpy_New
 * @sa dplasma_zlacpy
 *
 ******************************************************************************/
void
dplasma_zlacpy_Destruct( parsec_taskpool_t *tp )
{
    dplasma_map2_Destruct(tp);
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlacpy - Generates an taskpool that performs a copy of the matrix A
 * into the matrix B.
 *
 * See dplasma_map2() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
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
dplasma_zlacpy( parsec_context_t *parsec,
                PLASMA_enum uplo,
                const parsec_tiled_matrix_dc_t *A,
                parsec_tiled_matrix_dc_t *B)
{
    parsec_taskpool_t *parsec_zlacpy = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_zlacpy", "illegal value of uplo");
        return -2;
    }

    if ( (A->m > B->m) || (A->n > B->n) ) {
        dplasma_error("dplasma_zlacpy", "illegal matrix A (B is smaller than A)");
        return -3;
    }

    parsec_zlacpy = dplasma_zlacpy_New(uplo, A, B);

    if ( parsec_zlacpy != NULL )
    {
        parsec_context_add_taskpool(parsec, (parsec_taskpool_t*)parsec_zlacpy);
        dplasma_wait_until_completion(parsec);
        dplasma_zlacpy_Destruct( parsec_zlacpy );
    }
    return 0;
}
