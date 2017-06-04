/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include "parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

struct zplrnt_args_s {
    int                    diagdom;
    unsigned long long int seed;
};
typedef struct zplrnt_args_s zplrnt_args_t;

static int
dplasma_zplrnt_operator( parsec_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         void *_A,
                         PLASMA_enum uplo, int m, int n,
                         void *op_data )
{
    int tempmm, tempnn, ldam;
    zplrnt_args_t     *args = (zplrnt_args_t*)op_data;
    parsec_complex64_t *A    = (parsec_complex64_t*)_A;
    (void)eu;
    (void)uplo;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam   = BLKLDD( descA, m );

    CORE_zplrnt(
        tempmm, tempnn, A, ldam,
        descA->m, m*descA->mb, n*descA->nb, args->seed );

    if (args->diagdom && (m == n))
    {
        parsec_complex64_t  alpha;
        int maxmn = dplasma_imax( descA->m, descA->n );
        int i;

#if defined(PRECISION_z) || defined(PRECISION_c)
        int nvir  = descA->m + descA->n - 1;
        alpha = (double)nvir + I * (double)maxmn;
#else
        alpha = maxmn;
#endif

        for(i=0; i<dplasma_imin(tempmm, tempnn); i++) {
            (*A) += alpha;
            A += (ldam+1);
        }
    }

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zplrnt_New - Generates the handle that generates a random general
 * matrix by tiles.
 *
 * See dplasma_map_New() for further information.
 *
 *  WARNINGS: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] diagdom
 *          Specify if the diagonal is increased by max(M,N) or not to get a
 *          diagonal dominance.
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
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zplrnt_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zplrnt
 * @sa dplasma_zplrnt_Destruct
 * @sa dplasma_cplrnt_New
 * @sa dplasma_dplrnt_New
 * @sa dplasma_splrnt_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zplrnt_New( int diagdom,
                    tiled_matrix_desc_t *A,
                    unsigned long long int seed)
{
    zplrnt_args_t *params = (zplrnt_args_t*)malloc(sizeof(zplrnt_args_t));

    params->diagdom = diagdom;
    params->seed    = seed;

    return dplasma_map_New( PlasmaUpperLower, A, dplasma_zplrnt_operator, params );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zplrnt_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zplrnt_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zplrnt_New
 * @sa dplasma_zplrnt
 *
 ******************************************************************************/
void
dplasma_zplrnt_Destruct( parsec_handle_t *handle )
{
    dplasma_map_Destruct(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zplrnt - Generates a random general matrix by tiles.
 *
 * See dplasma_map() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] diagdom
 *          Specify if the diagonal is increased by max(M,N) or not to get a
 *          diagonal dominance.
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
 * @sa dplasma_zplrnt_New
 * @sa dplasma_zplrnt_Destruct
 * @sa dplasma_cplrnt
 * @sa dplasma_dplrnt
 * @sa dplasma_splrnt
 *
 ******************************************************************************/
int
dplasma_zplrnt( parsec_context_t *parsec,
                int diagdom,
                tiled_matrix_desc_t *A,
                unsigned long long int seed)
{
    parsec_handle_t *parsec_zplrnt = NULL;

    parsec_zplrnt = dplasma_zplrnt_New(diagdom, A, seed);

    parsec_enqueue(parsec, (parsec_handle_t*)parsec_zplrnt);
    dplasma_progress(parsec);

    dplasma_zplrnt_Destruct( parsec_zplrnt );
    return 0;
}
