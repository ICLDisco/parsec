/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zpotrf_U.h"
#include "zpotrf_L.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zpotrf_setrecursive - Set the recursive size parameter to enable
 *  recursive DAGs.
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to modify.
 *          On exit, the modified object.
 *
 * @param[in] hmb
 *          The tile size to use for the smaller recursive call.
 *          hmb must be > 0, otherwise nothing is changed.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf
 *
 ******************************************************************************/
void
dplasma_zpotrf_setrecursive( dague_handle_t *o, int hmb )
{
    dague_zpotrf_L_handle_t *dague_zpotrf = (dague_zpotrf_L_handle_t *)o;

    if (hmb > 0) {
        dague_zpotrf->smallnb = hmb;
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf_New - Generates the object that Computes the Cholesky
 * factorization of a symmetric positive definite (or Hermitian positive
 * definite in the complex case) matrix A, with or without recursive calls.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 * WARNING: The computations are not done by this call.
 *
 * If you want to enable the recursive DAGs, don't forget to set the recursive
 * tile size and to synchonize the handle ids after the computations since those
 * are for now local. You can follow the code of dplasma_zpotrf_rec() as an
 * example to do this.
 *
 * Hierarchical DAG Scheduling for Hybrid Distributed Systems; Wu, Wei and
 * Bouteiller, Aurelien and Bosilca, George and Faverge, Mathieu and Dongarra,
 * Jack. 29th IEEE International Parallel & Distributed Processing Symposium,
 * May 2015. (https://hal.inria.fr/hal-0107835)
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 * @param[out] info
 *          Address where to store the output information of the factorization,
 *          this is not synchronized between the nodes, and might not be set
 *          when function exists.
 *          On DAG completion:
 *              - info = 0 on all nodes if successful.
 *              - info > 0 if the leading minor of order i of A is not positive
 *                definite, so the factorization could not be completed, and the
 *                solution has not been computed. Info will be equal to i on the
 *                node that owns the diagonal element (i,i), and 0 on all other
 *                nodes.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zpotrf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf_New
 * @sa dplasma_dpotrf_New
 * @sa dplasma_spotrf_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zpotrf_New( PLASMA_enum uplo,
                    tiled_matrix_desc_t *A,
                    int *info )
{
    dague_zpotrf_L_handle_t *dague_zpotrf = NULL;
    dague_handle_t *o = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaUpper) && (uplo != PlasmaLower)) {
        dplasma_error("dplasma_zpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    *info = 0;
    if ( uplo == PlasmaUpper ) {
        o = (dague_handle_t*)dague_zpotrf_U_new( uplo, (dague_ddesc_t*)A, info);
    } else {
        o = (dague_handle_t*)dague_zpotrf_L_new( uplo, (dague_ddesc_t*)A, info);
    }

    dague_zpotrf = (dague_zpotrf_L_handle_t*)o;
    dague_zpotrf->PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == dague_zpotrf->PRI_CHANGE)
      dague_zpotrf->PRI_CHANGE = A->nt;
    dplasma_add2arena_tile( dague_zpotrf->arenas[DAGUE_zpotrf_L_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    return o;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zpotrf_Destruct - Free the data structure associated to an object
 *  created with dplasma_zpotrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf
 *
 ******************************************************************************/
void
dplasma_zpotrf_Destruct( dague_handle_t *o )
{
    dague_zpotrf_L_handle_t *dague_zpotrf = (dague_zpotrf_L_handle_t *)o;

    dague_matrix_del2arena( dague_zpotrf->arenas[DAGUE_zpotrf_L_DEFAULT_ARENA] );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf - Computes the Cholesky factorization of a symmetric positive
 * definite (or Hermitian positive definite in the complex case) matrix A.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf
 * @sa dplasma_dpotrf
 * @sa dplasma_spotrf
 *
 ******************************************************************************/
int
dplasma_zpotrf( dague_context_t *dague,
                PLASMA_enum uplo,
                tiled_matrix_desc_t *A )
{
    dague_handle_t *dague_zpotrf = NULL;
    int info = 0, ginfo = 0 ;

    dague_zpotrf = dplasma_zpotrf_New( uplo, A, &info );

    if ( dague_zpotrf != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zpotrf);
        dplasma_progress(dague);
        dplasma_zpotrf_Destruct( dague_zpotrf );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf_rec - Computes the Cholesky factorization of a symmetric
 * positive definite (or Hermitian positive definite in the complex case) matrix
 * An, using the recursive DAGs feature if hmb is smaller than A.mb or A.nb.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 * @param[in] hmb
 *          The tile size to use for the smaller recursive call.
 *          If hmb <= 0 or hmb > A.mb, the classic algorithm without recursive
 *          calls is applied.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf
 * @sa dplasma_dpotrf
 * @sa dplasma_spotrf
 *
 ******************************************************************************/
int
dplasma_zpotrf_rec( dague_context_t *dague,
                    PLASMA_enum uplo,
                    tiled_matrix_desc_t *A, int hmb )
{
    dague_handle_t *dague_zpotrf = NULL;
    int info = 0, ginfo = 0 ;

    dague_zpotrf = dplasma_zpotrf_New( uplo, A, &info );
    if ( dague_zpotrf != NULL )
    {
        dplasma_zpotrf_setrecursive( (dague_handle_t*)dague_zpotrf, hmb );
        dague_enqueue( dague, (dague_handle_t*)dague_zpotrf);
        dplasma_progress(dague);
        dplasma_zpotrf_Destruct( dague_zpotrf );
        dague_handle_sync_ids(); /* recursive DAGs are not synchronous on ids */
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}
