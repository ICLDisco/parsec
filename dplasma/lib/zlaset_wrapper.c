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
#include <lapacke.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

struct zlaset_args_s {
    dague_complex64_t alpha;
    dague_complex64_t beta;
    tiled_matrix_desc_t *descA;
};
typedef struct zlaset_args_s zlaset_args_t;

static int
dplasma_zlaset_operator( struct dague_execution_unit *eu,
                         void *_A,
                         void *op_data, ... )
{
    va_list ap;
    PLASMA_enum uplo;
    int m, n;
    int tempmm, tempnn, ldam;
    tiled_matrix_desc_t *descA;
    dague_complex64_t alpha, beta;
    zlaset_args_t *args = (zlaset_args_t*)op_data;
    dague_complex64_t *A = (dague_complex64_t*)_A;
    (void)eu;

    va_start(ap, op_data);
    uplo = va_arg(ap, PLASMA_enum);
    m    = va_arg(ap, int);
    n    = va_arg(ap, int);
    va_end(ap);

    descA = args->descA;
    alpha = args->alpha;
    beta  = args->beta;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );

    if (m == n) {
        LAPACKE_zlaset_work(
            LAPACK_COL_MAJOR, lapack_const( uplo ), tempmm, tempnn,
            alpha, beta, A, ldam);
    } else {
        LAPACKE_zlaset_work(
            LAPACK_COL_MAJOR, 'A', tempmm, tempnn,
            alpha, alpha, A, ldam);
    }
    return 0;
}

/***************************************************************************/
/**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zlaset_New - Sets the elements of the matrix A on the diagonal
 *  to beta and on the off-diagonals to alpha
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which elements of the matrix are to be set
 *          = PlasmaUpper: Upper part of A is set;
 *          = PlasmaLower: Lower part of A is set;
 *          = PlasmaUpperLower: ALL elements of A are set.
 *
 * @param[in] alpha
 *         The constant to which the off-diagonal elements are to be set.
 *
 * @param[in] beta
 *         The constant to which the diagonal elements are to be set.
 *
 * @param[in,out] A
 *         On entry, the M-by-N tile A.
 *         On exit, A has been set accordingly.
 *
 **/
dague_object_t*
dplasma_zlaset_New( PLASMA_enum uplo,
                    dague_complex64_t alpha,
                    dague_complex64_t beta,
                    tiled_matrix_desc_t *A )
{
    zlaset_args_t *params = (zlaset_args_t*)malloc(sizeof(zlaset_args_t));

    params->alpha = alpha;
    params->beta  = beta;
    params->descA = A;

    return dplasma_map_New( uplo, A, dplasma_zlaset_operator, params );
}

void
dplasma_zlaset_Destruct( dague_object_t *o )
{
    dplasma_map_Destruct( o );
}

int
dplasma_zlaset( dague_context_t *dague,
                PLASMA_enum uplo,
                dague_complex64_t alpha,
                dague_complex64_t beta,
                tiled_matrix_desc_t *A )
{
    dague_object_t *dague_zlaset = NULL;

    dague_zlaset = dplasma_zlaset_New(uplo, alpha, beta, A);

    dague_enqueue(dague, (dague_object_t*)dague_zlaset);
    dplasma_progress(dague);

    dplasma_zlaset_Destruct( dague_zlaset );
    return 0;
}
