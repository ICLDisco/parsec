/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include <cblas.h>
#include "dplasma.h"
#include "dplasmatypes.h"

#include "map.h"

static int
dplasma_zlascal_operator( parsec_execution_stream_t *es,
                         const parsec_tiled_matrix_dc_t *descA,
                         void *_A,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    parsec_complex64_t *A     = (parsec_complex64_t*)_A;
    parsec_complex64_t  alpha = *((parsec_complex64_t*)args);
    int i;
    int tempmm, tempnn, ldam;
    (void)es;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( descA, m );

    /* Overwrite uplo when outside the diagonal */
    if (m != n) {
        uplo = PlasmaUpperLower;
    }

    switch ( uplo ) {
    case PlasmaUpper:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( dplasma_imin( i+1, tempmm ), CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;

    case PlasmaLower:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( dplasma_imax( tempmm, tempmm-i ), CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;
    default:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( tempmm, CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;
    }

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlascal_New - Generates the taskpool that scales a matrix by a given scalar.
 *
 * See dplasma_map_New() for further information.
 *
 *  WARNINGS:
 *      - The computations are not done by this call.
 *      - This routine is equivalent to the pzlascal routine from ScaLapack and
 *  not to the zlascl/pzlascl routines from Lapack/ScaLapack.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *          = PlasmaUpperHessenberg: A is an upper Hessenberg matrix (Not
 *          supported)
 *
 * @param[in] alpha
 *          The scalar to use to scale the matrix.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the matrix A is scaled by alpha.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zlascal_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal
 * @sa dplasma_zlascal_Destruct
 * @sa dplasma_clascal_New
 * @sa dplasma_dlascal_New
 * @sa dplasma_slascal_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zlascal_New( PLASMA_enum uplo,
                     parsec_complex64_t alpha,
                     parsec_tiled_matrix_dc_t *A )
{
    parsec_complex64_t *a = (parsec_complex64_t*)malloc(sizeof(parsec_complex64_t));
    *a = alpha;

    return dplasma_map_New( uplo, A, dplasma_zlascal_operator, (void*)a );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlascal_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zlascal_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal_New
 * @sa dplasma_zlascal
 *
 ******************************************************************************/
void
dplasma_zlascal_Destruct( parsec_taskpool_t *tp )
{
    dplasma_map_Destruct(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlascal - Scales a matrix by a given scalar.
 *
 * See dplasma_map() for further information.
 *
 *  WARNINGS:
 *      - This routine is equivalent to the pzlascal routine from ScaLapack and
 *  not to the zlascl/pzlascl routines from Lapack/ScaLapack.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *          = PlasmaUpperHessenberg: A is an upper Hessenberg matrix (Not
 *          supported)
 *
 * @param[in] alpha
 *          The scalar to use to scale the matrix.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the matrix A is scaled by alpha.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal_New
 * @sa dplasma_zlascal_Destruct
 * @sa dplasma_clascal
 * @sa dplasma_dlascal
 * @sa dplasma_slascal
 *
 ******************************************************************************/
int
dplasma_zlascal( parsec_context_t     *parsec,
                 PLASMA_enum          uplo,
                 parsec_complex64_t    alpha,
                 parsec_tiled_matrix_dc_t *A )
{
    parsec_taskpool_t *parsec_zlascal = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zlascal", "illegal value of type");
        return -2;
    }

    parsec_zlascal = dplasma_zlascal_New(uplo, alpha, A);

    if ( parsec_zlascal != NULL ) {
        parsec_context_add_taskpool(parsec, (parsec_taskpool_t*)parsec_zlascal);
        dplasma_wait_until_completion(parsec);
        dplasma_zlascal_Destruct( parsec_zlascal );
    }
    return 0;
}
