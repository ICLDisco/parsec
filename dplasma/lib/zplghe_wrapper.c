/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c
 *
 */

#include "parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

struct zplghe_args_s {
    double                 bump;
    unsigned long long int seed;
};
typedef struct zplghe_args_s zplghe_args_t;

static int
dplasma_zplghe_operator( parsec_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         void *_A,
                         PLASMA_enum uplo, int m, int n,
                         void *op_data )
{
    int tempmm, tempnn, ldam;
    zplghe_args_t     *args = (zplghe_args_t*)op_data;
    parsec_complex64_t *A    = (parsec_complex64_t*)_A;
    (void)eu;
    (void)uplo;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam   = BLKLDD( descA, m );

    CORE_zplghe(
        args->bump, tempmm, tempnn, A, ldam,
        descA->m, m*descA->mb, n*descA->nb, args->seed );

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zplghe_New - Generates the handle that generates a random hermitian
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
 *          On exit, the hermitian matrix A generated.
 *
 * @param[in] seed
 *          The seed used in the random generation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zplghe_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zplghe
 * @sa dplasma_zplghe_Destruct
 * @sa dplasma_cplghe_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zplghe_New( double bump, PLASMA_enum uplo,
                    tiled_matrix_desc_t *A,
                    unsigned long long int seed)
{
    zplghe_args_t *params = (zplghe_args_t*)malloc(sizeof(zplghe_args_t));

    params->bump  = bump;
    params->seed  = seed;

    return dplasma_map_New( uplo, A, dplasma_zplghe_operator, params );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zplghe_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zplghe_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zplghe_New
 * @sa dplasma_zplghe
 *
 ******************************************************************************/
void
dplasma_zplghe_Destruct( parsec_handle_t *handle )
{
    dplasma_map_Destruct(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zplghe - Generates a random hermitian matrix by tiles.
 *
 * See dplasma_map() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
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
 *          On exit, the hermitian matrix A generated.
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
 * @sa dplasma_zplghe_New
 * @sa dplasma_zplghe_Destruct
 * @sa dplasma_cplghe
 *
 ******************************************************************************/
int
dplasma_zplghe( parsec_context_t *parsec,
                double bump, PLASMA_enum uplo,
                tiled_matrix_desc_t *A,
                unsigned long long int seed)
{
    parsec_handle_t *parsec_zplghe = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zplghe", "illegal value of type");
        return -3;
    }

    parsec_zplghe = dplasma_zplghe_New( bump, uplo, A, seed );

    if ( parsec_zplghe != NULL ) {
        parsec_enqueue(parsec, (parsec_handle_t*)parsec_zplghe);
        dplasma_progress(parsec);
        dplasma_zplghe_Destruct( parsec_zplghe );
    }
    return 0;
}
