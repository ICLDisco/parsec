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

struct zlacpy_args_s {
    const tiled_matrix_desc_t *descA;
    tiled_matrix_desc_t *descB;
};
typedef struct zlacpy_args_s zlacpy_args_t;

static int
dplasma_zlacpy_operator( struct dague_execution_unit *eu,
                       const void *_A, void *_B,
                       void *op_data, ... )
{
    va_list ap;
    zlacpy_args_t *args = (zlacpy_args_t*)op_data;
    PLASMA_enum uplo;
    int m, n;
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

    LAPACKE_zlacpy_work(
        LAPACK_COL_MAJOR, lapack_const( uplo ), tempmm, tempnn, A, ldam, B, ldbm);

    return 0;
}

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zlacpy_New - Generate a random matrix by tiles.
 *
 *******************************************************************************
 *
 * @param[out] A
 *          On exit, The random hermitian matrix A generated.
 *
 * @param[in] seed
 *          The seed used in the random generation.
 *
 ******************************************************************************/
dague_object_t*
dplasma_zlacpy_New( PLASMA_enum uplo,
                    const tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *B)
{
    dague_object_t* object;
    zlacpy_args_t *params = (zlacpy_args_t*)malloc(sizeof(zlacpy_args_t));

    params->descA = A;
    params->descB = B;

    object = dplasma_map2_New(uplo, A, B,
                              dplasma_zlacpy_operator,
                              (void *)params);

    return object;
}

void
dplasma_zlacpy_Destruct( dague_object_t *o )
{
    dplasma_map2_Destruct( o );
}

int
dplasma_zlacpy( dague_context_t *dague,
                PLASMA_enum uplo,
                const tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *B)
{
    dague_object_t *dague_zlacpy = NULL;

    dague_zlacpy = dplasma_zlacpy_New(uplo, A, B);

    dague_enqueue(dague, (dague_object_t*)dague_zlacpy);
    dplasma_progress(dague);

    dplasma_zlacpy_Destruct( dague_zlacpy );
    return 0;
}
