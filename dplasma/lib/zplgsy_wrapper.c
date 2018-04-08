/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

struct zplgsy_args_s {
    parsec_complex64_t      bump;
    unsigned long long int seed;
};
typedef struct zplgsy_args_s zplgsy_args_t;

static int
dplasma_zplgsy_operator( parsec_execution_stream_t *es,
                         const parsec_tiled_matrix_dc_t *descA,
                         void *_A,
                         PLASMA_enum uplo, int m, int n,
                         void *op_data )
{
    int tempmm, tempnn, ldam;
    zplgsy_args_t     *args = (zplgsy_args_t*)op_data;
    parsec_complex64_t *A    = (parsec_complex64_t*)_A;
    (void)es;
    (void)uplo;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam   = BLKLDD( descA, m );

    CORE_zplgsy(
        args->bump, tempmm, tempnn, A, ldam,
        descA->m, m*descA->mb, n*descA->nb, args->seed );

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zplgsy_New - Generates the taskpool that generates a random symmetric
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
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
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
parsec_taskpool_t*
dplasma_zplgsy_New( parsec_complex64_t bump, PLASMA_enum uplo,
                    parsec_tiled_matrix_dc_t *A,
                    unsigned long long int seed)
{
    zplgsy_args_t *params = (zplgsy_args_t*)malloc(sizeof(zplgsy_args_t));

    params->bump  = bump;
    params->seed  = seed;

    return dplasma_map_New( uplo, A, dplasma_zplgsy_operator, params );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zplgsy_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zplgsy_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zplgsy_New
 * @sa dplasma_zplgsy
 *
 ******************************************************************************/
void
dplasma_zplgsy_Destruct( parsec_taskpool_t *tp )
{
    dplasma_map_Destruct(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zplgsy - Generates a random symmetric matrix by tiles.
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
dplasma_zplgsy( parsec_context_t *parsec,
                parsec_complex64_t bump, PLASMA_enum uplo,
                parsec_tiled_matrix_dc_t *A,
                unsigned long long int seed)
{
    parsec_taskpool_t *parsec_zplgsy = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zplgsy", "illegal value of type");
        return -3;
    }

    parsec_zplgsy = dplasma_zplgsy_New( bump, uplo, A, seed );

    if ( parsec_zplgsy != NULL ) {
        parsec_enqueue(parsec, (parsec_taskpool_t*)parsec_zplgsy);
        dplasma_wait_until_completion(parsec);
        dplasma_zplgsy_Destruct( parsec_zplgsy );
    }
    return 0;
}
