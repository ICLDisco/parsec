/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zlange_one_cyclic.h"
#include "zlange_frb_cyclic.h"

static inline int dague_imax(int a, int b) { return (a > b) ? a : b; };
static inline int dague_imin(int a, int b) { return (a < b) ? a : b; };

static inline void *fake_data_of(struct dague_ddesc *mat, ...)
{
    return (void*)mat;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlange_New - Generates the object that computes the value
 *
 *     zlange = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
 *              (
 *              ( norm1(A),         NORM = PlasmaOneNorm
 *              (
 *              ( normI(A),         NORM = PlasmaInfNorm
 *              (
 *              ( normF(A),         NORM = PlasmaFrobeniusNorm
 *
 *  where norm1 denotes the one norm of a matrix (maximum column sum),
 *  normI denotes the infinity norm of a matrix (maximum row sum) and
 *  normF denotes the Frobenius norm of a matrix (square root of sum
 *  of squares). Note that max(abs(A(i,j))) is not a consistent matrix
 *  norm.
 *
 *  WARNING: The computations are not done by this call
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          = PlasmaMaxNorm: Max norm
 *          = PlasmaOneNorm: One norm
 *          = PlasmaInfNorm: Infinity norm
 *          = PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] A
 *          The descriptor of the matrix A.
 *          Must be a two_dim_rectangle_cyclic matrix.
 *
 * @param[out] result
 *          The norm described above. Might not be set when the function returns.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlange_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlange
 * @sa dplasma_zlange_Destruct
 * @sa dplasma_clange_New
 * @sa dplasma_dlange_New
 * @sa dplasma_slange_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zlange_New( PLASMA_enum ntype,
                    int P, int Q,
                    const tiled_matrix_desc_t *A,
                    double *result )
{
    int m, n, mb, nb, elt;
    two_dim_block_cyclic_t *Tdist;
    dague_object_t *dague_zlange = NULL;

    if ( (ntype != PlasmaMaxNorm) && (ntype != PlasmaOneNorm)
        && (ntype != PlasmaInfNorm) && (ntype != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlange", "illegal value of ntype");
        return NULL;
    }
    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        dplasma_error("dplasma_zlange", "illegal type of descriptor for A");
        return NULL;
    }

    /* Warning: Pb with smb/snb when mt/nt lower than P/Q */
    switch( ntype ) {
    case PlasmaFrobeniusNorm:
        mb = 2;
        nb = 1;
        m  = A->mt;
        n  = Q;
        elt = 2;
        break;
    case PlasmaInfNorm:
        mb = A->mb;
        nb = 1;
        m  = A->mt;
        n  = Q;
        elt = 1;
        break;
    case PlasmaOneNorm:
        mb = 1;
        nb = A->nb;
        m  = P;
        n  = dague_imax(A->nt, Q);
        elt = 1;
        break;
    case PlasmaMaxNorm:
    default:
        mb = 1;
        nb = 1;
        m  = A->mt;
        n  = Q;
        elt = 1;
    }

    /* Create a copy of the A descriptor that is general to avoid problem when
     * accessing not referenced part of the matrix */
    /* Create the task distribution */
    Tdist = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

    two_dim_block_cyclic_init(
        Tdist, matrix_RealDouble, matrix_Tile,
        A->super.nodes, A->super.cores, A->super.myrank,
        1, 1, /* Dimensions of the tiles              */
        m, n, /* Dimensions of the matrix             */
        0, 0, /* Starting points (not important here) */
        m, n, /* Dimensions of the submatrix          */
        1, 1, P);

    Tdist->super.super.data_of = fake_data_of;

    /* Create the DAG */
    switch( ntype ) {
    case PlasmaOneNorm:
        dague_zlange = (dague_object_t*)dague_zlange_one_cyclic_new(
            P, Q, ntype, (dague_ddesc_t*)A, (dague_ddesc_t*)Tdist, result);
        break;

    case PlasmaMaxNorm:
    case PlasmaInfNorm:
    case PlasmaFrobeniusNorm:
    default:
        dague_zlange = (dague_object_t*)dague_zlange_frb_cyclic_new(
            P, Q, ntype, (dague_ddesc_t*)A, (dague_ddesc_t*)Tdist, result);
    }

    /* Set the datatypes */
    dplasma_add2arena_tile(((dague_zlange_frb_cyclic_object_t*)dague_zlange)->arenas[DAGUE_zlange_frb_cyclic_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);
    dplasma_add2arena_rectangle(((dague_zlange_frb_cyclic_object_t*)dague_zlange)->arenas[DAGUE_zlange_frb_cyclic_COL_ARENA],
                                mb * nb * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE, mb, nb, -1);
    dplasma_add2arena_rectangle(((dague_zlange_frb_cyclic_object_t*)dague_zlange)->arenas[DAGUE_zlange_frb_cyclic_ELT_ARENA],
                                elt * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE, elt, 1, -1);

    return (dague_object_t*)dague_zlange;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlange_Destruct - Free the data structure associated to an object
 *  created with dplasma_zlange_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlange_New
 * @sa dplasma_zlange
 *
 ******************************************************************************/
void
dplasma_zlange_Destruct( dague_object_t *o )
{
    dague_zlange_frb_cyclic_object_t *dague_zlange = (dague_zlange_frb_cyclic_object_t *)o;

    dague_ddesc_destroy( dague_zlange->Tdist );
    free( dague_zlange->Tdist );

    dplasma_datatype_undefine_type( &(dague_zlange->arenas[DAGUE_zlange_frb_cyclic_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlange->arenas[DAGUE_zlange_frb_cyclic_COL_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlange->arenas[DAGUE_zlange_frb_cyclic_ELT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlange - Computes the value
 *
 *     zlange = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
 *              (
 *              ( norm1(A),         NORM = PlasmaOneNorm
 *              (
 *              ( normI(A),         NORM = PlasmaInfNorm
 *              (
 *              ( normF(A),         NORM = PlasmaFrobeniusNorm
 *
 *  where norm1 denotes the one norm of a matrix (maximum column sum),
 *  normI denotes the infinity norm of a matrix (maximum row sum) and
 *  normF denotes the Frobenius norm of a matrix (square root of sum
 *  of squares). Note that max(abs(A(i,j))) is not a consistent matrix
 *  norm.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] norm
 *          = PlasmaMaxNorm: Max norm
 *          = PlasmaOneNorm: One norm
 *          = PlasmaInfNorm: Infinity norm
 *          = PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] A
 *          The descriptor of the matrix A.
 *          Must be a two_dim_rectangle_cyclic matrix.
 *
*******************************************************************************
 *
 * @return
 *          \retval the computed norm described above.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlange_New
 * @sa dplasma_zlange_Destruct
 * @sa dplasma_clange
 * @sa dplasma_dlange
 * @sa dplasma_slange
 *
 ******************************************************************************/
double
dplasma_zlange( dague_context_t *dague,
                PLASMA_enum ntype,
                const tiled_matrix_desc_t *A)
{
    double result = 0.;
    dague_object_t *dague_zlange = NULL;

    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    int Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    if ( (ntype != PlasmaMaxNorm) && (ntype != PlasmaOneNorm)
        && (ntype != PlasmaInfNorm) && (ntype != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlange", "illegal value of ntype");
        return -2.;
    }
    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        dplasma_error("dplasma_zlange", "illegal type of descriptor for A");
        return -3.;
    }

    dague_zlange = dplasma_zlange_New(ntype, P, Q, A, &result);

    if ( dague_zlange != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zlange);
        dplasma_progress(dague);
        dplasma_zlange_Destruct( dague_zlange );
    }

    return result;
}

