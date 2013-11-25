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
#include <cblas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

struct zplgsy_args_s {
    tiled_matrix_desc_t   *descA;
    dague_complex64_t      bump;
    unsigned long long int seed;
};
typedef struct zplgsy_args_s zplgsy_args_t;

static int
dplasma_zplgsy_operator( struct dague_execution_unit *eu,
                         void *_A,
                         void *op_data, ... )
{
    va_list ap;
    PLASMA_enum uplo;
    int m, n;
    int tempmm, tempnn, ldam;
    tiled_matrix_desc_t *descA;
    zplgsy_args_t     *args = (zplgsy_args_t*)op_data;
    dague_complex64_t *A    = (dague_complex64_t*)_A;
    (void)eu;

    va_start(ap, op_data);
    uplo = va_arg(ap, PLASMA_enum);
    m    = va_arg(ap, int);
    n    = va_arg(ap, int);
    va_end(ap);

    (void)uplo;
    descA  = args->descA;
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam   = BLKLDD( *descA, m );

    CORE_zplgsy(
        args->bump, tempmm, tempnn, A, ldam,
        descA->m, m*descA->mb, n*descA->nb, args->seed );

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zplgsy_New - Generates the object that generates a random symmetric
 * matrix by tiles.
 *
 * See dplasma_map_New() for further information.
 *
 *  WARNINGS: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] bump
 *          The value to add to the diagonal to be sure
 *          to have a positive definite matrix.
 *
 * @param[in] uplo
 *          Specifies which elements of the matrix are to be set
 *          = PlasmaUpper: Upper part of A is set;
 *          = PlasmaLower: Lower part of A is set;
 *          = PlasmaUpperLower: ALL elements of A are set.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to generate. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the symmetric matrix A generated.
 *
 * @param[in] seed
 *          The seed used in the random generation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zplgsy_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zplgsy
 * @sa dplasma_zplgsy_Destruct
 * @sa dplasma_cplgsy_New
 * @sa dplasma_dplgsy_New
 * @sa dplasma_splgsy_New
*
 ******************************************************************************/
dague_object_t*
dplasma_zplgsy_New( dague_complex64_t bump, PLASMA_enum uplo,
                    tiled_matrix_desc_t *A,
                    unsigned long long int seed)
{
    zplgsy_args_t *params = (zplgsy_args_t*)malloc(sizeof(zplgsy_args_t));

    params->descA = A;
    params->bump  = bump;
    params->seed  = seed;

    return dplasma_map_New( uplo, A, dplasma_zplgsy_operator, params );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zplgsy_Destruct - Free the data structure associated to an object
 *  created with dplasma_zplgsy_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zplgsy_New
 * @sa dplasma_zplgsy
 *
 ******************************************************************************/
void
dplasma_zplgsy_Destruct( dague_object_t *o )
{
    dplasma_map_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zplgsy - Generates a random symmetric matrix by tiles.
 *
 * See dplasma_map() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] bump
 *          The value to add to the diagonal to be sure
 *          to have a positive definite matrix.
 *
 * @param[in] uplo
 *          Specifies which elements of the matrix are to be set
 *          = PlasmaUpper: Upper part of A is set;
 *          = PlasmaLower: Lower part of A is set;
 *          = PlasmaUpperLower: ALL elements of A are set.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to generate. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the symmetric matrix A generated.
 *
 * @param[in] seed
 *          The seed used in the random generation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zplgsy_New
 * @sa dplasma_zplgsy_Destruct
 * @sa dplasma_cplgsy
 * @sa dplasma_dplgsy
 * @sa dplasma_splgsy
 *
 ******************************************************************************/
int
dplasma_zplgsy( dague_context_t *dague,
                dague_complex64_t bump, PLASMA_enum uplo,
                tiled_matrix_desc_t *A,
                unsigned long long int seed)
{
    dague_object_t *dague_zplgsy = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zplgsy", "illegal value of type");
        return -3;
    }

    dague_zplgsy = dplasma_zplgsy_New( bump, uplo, A, seed );

    if ( dague_zplgsy != NULL ) {
        dague_enqueue(dague, (dague_object_t*)dague_zplgsy);
        dplasma_progress(dague);
        dplasma_zplgsy_Destruct( dague_zplgsy );
    }
    return 0;
}
