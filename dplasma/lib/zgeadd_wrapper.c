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

struct zgeadd_args_s {
    dague_complex64_t          alpha;
    const tiled_matrix_desc_t *descA;
    tiled_matrix_desc_t       *descB;
};
typedef struct zgeadd_args_s zgeadd_args_t;

static int
dplasma_zgeadd_operator( dague_execution_unit_t *eu,
                         const void *_A, void *_B,
                         void *op_data, ... )
{
    va_list ap;
    zgeadd_args_t *args = (zgeadd_args_t*)op_data;
    PLASMA_enum uplo;
    int j, m, n;
    int tempmm, tempnn, ldam, ldbm;
    const tiled_matrix_desc_t *descA;
    tiled_matrix_desc_t *descB;
    dague_complex64_t *A = (dague_complex64_t*)_A;
    dague_complex64_t *B = (dague_complex64_t*)_B;
    (void)eu;
    va_start(ap, op_data);
    uplo = va_arg(ap, PLASMA_enum);
    m    = va_arg(ap, int);
    n    = va_arg(ap, int);
    va_end(ap);

    descA = args->descA;
    descB = args->descB;
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );
    ldbm = BLKLDD( *descB, m );

    switch ( uplo ) {
    case PlasmaLower:
        for (j = 0; j < tempnn; j++, tempmm--, A+=ldam+1, B+=ldbm+1) {
            cblas_zaxpy(tempmm, CBLAS_SADDR(args->alpha), A, 1, B, 1);
        }
        break;
    case PlasmaUpper:
        for (j = 0; j < tempnn; j++, A+=ldam, B+=ldbm) {
            cblas_zaxpy(j+1, CBLAS_SADDR(args->alpha), A, 1, B, 1);
        }
        break;
    case PlasmaUpperLower:
    default:
        for (j = 0; j < tempnn; j++, A+=ldam, B+=ldbm) {
            cblas_zaxpy(tempmm, CBLAS_SADDR(args->alpha), A, 1, B, 1);
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
dague_object_t*
dplasma_zgeadd_New( PLASMA_enum uplo,
                    dague_complex64_t alpha,
                    const tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *B)
{
    dague_object_t* object;
    zgeadd_args_t *params = (zgeadd_args_t*)malloc(sizeof(zgeadd_args_t));

    params->alpha = alpha;
    params->descA = A;
    params->descB = B;

    object = dplasma_map2_New(uplo, A, B,
                              dplasma_zgeadd_operator, (void *)params);

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
dplasma_zgeadd_Destruct( dague_object_t *o )
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
    dague_object_t *dague_zgeadd = NULL;

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
        dague_enqueue(dague, (dague_object_t*)dague_zgeadd);
        dplasma_progress(dague);
        dplasma_zgeadd_Destruct( dague_zgeadd );
    }
    return 0;
}
