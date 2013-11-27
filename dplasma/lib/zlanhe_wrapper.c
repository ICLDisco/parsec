/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> z c
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"

#include "zlansy.h"

static inline void *fake_data_of(struct dague_ddesc *mat, ...)
{
    return (void*)mat;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlanhe_New - Generates the object that computes the value
 *
 *     zlanhe = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
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
 *          The descriptor of the hermitian matrix A.
 *          Must be a two_dim_rectangle_cyclic or sym_two_dim_rectangle_cyclic
 *          matrix
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
 *          destroy with dplasma_zlanhe_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanhe
 * @sa dplasma_zlanhe_Destruct
 * @sa dplasma_clanhe_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zlanhe_New( PLASMA_enum ntype,
                    PLASMA_enum uplo,
                    const tiled_matrix_desc_t *A,
                    double *result )
{
    int P, Q, mb, nb, elt;
    two_dim_block_cyclic_t *Tdist;
    dague_object_t *dague_zlanhe = NULL;

    if ( (ntype != PlasmaMaxNorm) && (ntype != PlasmaOneNorm)
        && (ntype != PlasmaInfNorm) && (ntype != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of ntype");
        return NULL;
    }
    if ( (uplo != PlasmaUpper) && (uplo != PlasmaLower) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of uplo");
        return NULL;
    }
    if ( !(A->dtype & ( two_dim_block_cyclic_type | sym_two_dim_block_cyclic_type)) ) {
        dplasma_error("dplasma_zlanhe", "illegal type of descriptor for A");
        return NULL;
    }

    P = ((sym_two_dim_block_cyclic_t*)A)->grid.rows;
    Q = ((sym_two_dim_block_cyclic_t*)A)->grid.cols;

    /* Warning: Pb with smb/snb when mt/nt lower than P/Q */
    switch( ntype ) {
    case PlasmaFrobeniusNorm:
        mb = 2;
        nb = 1;
        elt = 2;
        break;
    case PlasmaInfNorm:
    case PlasmaOneNorm:
        mb = A->mb;
        nb = 1;
        elt = 1;
        break;
    case PlasmaMaxNorm:
    default:
        mb = 1;
        nb = 1;
        elt = 1;
    }

    /* Create a copy of the A descriptor that is general to avoid problem when
     * accessing not referenced part of the matrix */
    /* Create the task distribution */
    Tdist = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

    two_dim_block_cyclic_init(
        Tdist, matrix_RealDouble, matrix_Tile,
        A->super.nodes, A->super.cores, A->super.myrank,
        1, 1,       /* Dimensions of the tiles              */
        A->mt, P*Q, /* Dimensions of the matrix             */
        0, 0,       /* Starting points (not important here) */
        A->mt, P*Q, /* Dimensions of the submatrix          */
        1, 1, P);

    Tdist->super.super.data_of = fake_data_of;

    /* Create the DAG */
    dague_zlanhe = (dague_object_t*)dague_zlansy_new(
        P, Q, ntype, uplo, PlasmaConjTrans,
        (dague_ddesc_t*)A,
        (dague_ddesc_t*)Tdist,
        result);

    /* Set the datatypes */
    dplasma_add2arena_tile(((dague_zlansy_object_t*)dague_zlanhe)->arenas[DAGUE_zlansy_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);
    dplasma_add2arena_rectangle(((dague_zlansy_object_t*)dague_zlanhe)->arenas[DAGUE_zlansy_COL_ARENA],
                                mb * nb * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE, mb, nb, -1);
    dplasma_add2arena_rectangle(((dague_zlansy_object_t*)dague_zlanhe)->arenas[DAGUE_zlansy_ELT_ARENA],
                                elt * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE, elt, 1, -1);

    return (dague_object_t*)dague_zlanhe;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlanhe_Destruct - Free the data structure associated to an object
 *  created with dplasma_zlanhe_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanhe_New
 * @sa dplasma_zlanhe
 *
 ******************************************************************************/
void
dplasma_zlanhe_Destruct( dague_object_t *o )
{
    dague_zlansy_object_t *dague_zlanhe = (dague_zlansy_object_t *)o;

    dague_ddesc_destroy( dague_zlanhe->Tdist );
    free( dague_zlanhe->Tdist );

    dplasma_datatype_undefine_type( &(dague_zlanhe->arenas[DAGUE_zlansy_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlanhe->arenas[DAGUE_zlansy_COL_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlanhe->arenas[DAGUE_zlansy_ELT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlanhe - Computes the value
 *
 *     zlanhe = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
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
 *          The descriptor of the hermitian matrix A.
 *          Must be a two_dim_rectangle_cyclic or sym_two_dim_rectangle_cyclic
 *          matrix
 *
*******************************************************************************
 *
 * @return
 *          \retval the computed norm described above.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanhe_New
 * @sa dplasma_zlanhe_Destruct
 * @sa dplasma_clanhe
 *
 ******************************************************************************/
double
dplasma_zlanhe( dague_context_t *dague,
                PLASMA_enum ntype,
                PLASMA_enum uplo,
                const tiled_matrix_desc_t *A)
{
    double result = 0.;
    dague_object_t *dague_zlanhe = NULL;

    if ( (ntype != PlasmaMaxNorm) && (ntype != PlasmaOneNorm)
        && (ntype != PlasmaInfNorm) && (ntype != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of ntype");
        return -2.;
    }
    if ( (uplo != PlasmaUpper) && (uplo != PlasmaLower) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of uplo");
        return -3.;
    }
    if ( !(A->dtype & ( two_dim_block_cyclic_type | sym_two_dim_block_cyclic_type)) ) {
        dplasma_error("dplasma_zlanhe", "illegal type of descriptor for A");
        return -4.;
    }
    if ( A->m != A->n ) {
        dplasma_error("dplasma_zlanhe", "illegal matrix A (not square)");
        return -5.;
    }

    dague_zlanhe = dplasma_zlanhe_New(ntype, uplo, A, &result);

    if ( dague_zlanhe != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zlanhe);
        dplasma_progress(dague);
        dplasma_zlanhe_Destruct( dague_zlanhe );
    }

    return result;
}

