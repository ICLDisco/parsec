/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zlansy.h"

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t
 *
 *  dplasm_zlansy_New - computes the value
 *
 *     zlansy = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
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
 *          Must be a two_dim_rectangle_cyclic or sym_two_dim_rectangle_cyclic
 *          matrix
 *
 * @param[out] result
 *          The norm described above. Might not be set when the function returns.
 *
 *******************************************************************************
 *
 * @return
 *          \retval Pointer to the dague object describing the operation.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlansy
 * @sa dplasma_clansy_New
 * @sa dplasma_dlansy_New
 * @sa dplasma_slansy_New
 *
 ******************************************************************************/
dague_handle_t* dplasma_zlansy_New( PLASMA_enum ntype,
                                    PLASMA_enum uplo,
                                    const tiled_matrix_desc_t *A,
                                    double *result )
{
    int P, Q, m, n, mb, nb, elt;
    two_dim_block_cyclic_t *Wcol;
    two_dim_block_cyclic_t *Welt;
    dague_handle_t *dague_zlansy = NULL;

    if ( (ntype != PlasmaMaxNorm) && (ntype != PlasmaOneNorm)
        && (ntype != PlasmaInfNorm) && (ntype != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlansy", "illegal value of ntype");
        return NULL;
    }
    if ( (uplo != PlasmaUpper) && (uplo != PlasmaLower) ) {
        dplasma_error("dplasma_zlansy", "illegal value of uplo");
        return NULL;
    }
    if ( !(A->dtype & ( two_dim_block_cyclic_type | sym_two_dim_block_cyclic_type)) ) {
        dplasma_error("dplasma_zlansy", "illegal type of descriptor for A");
        return NULL;
    }

    P = ((sym_two_dim_block_cyclic_t*)A)->grid.rows;
    Q = ((sym_two_dim_block_cyclic_t*)A)->grid.cols;

    /* Create the workspace */
    Wcol = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    Welt = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

    /* Warning: Pb with smb/snb when mt/nt lower than P/Q */
    switch( ntype ) {
    case PlasmaFrobeniusNorm:
        mb = 2;
        nb = 1;
        m  = A->mt * mb;
        n  = P*Q;
        elt = 2;
        break;
    case PlasmaInfNorm:
    case PlasmaOneNorm:
        mb = A->mb;
        nb = 1;
        m  = A->mt * mb;
        n  = P*Q;
        elt = 1;
        break;
    case PlasmaMaxNorm:
    default:
        mb = 1;
        nb = 1;
        m  = A->mt;
        n  = P*Q;
        elt = 1;
    }

    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*Wcol), two_dim_block_cyclic,
        (Wcol, matrix_RealDouble, matrix_Tile,
         A->super.nodes, A->super.myrank,
         mb,  nb,   /* Dimesions of the tile                */
         m,   n,    /* Dimensions of the matrix             */
         0,   0,    /* Starting points (not important here) */
         m,   n,    /* Dimensions of the submatrix          */
         1, 1, 1));

    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*Welt), two_dim_block_cyclic,
        (Welt, matrix_RealDouble, matrix_Tile,
         A->super.nodes, A->super.myrank,
         elt,   1,  /* Dimesions of the tile                */
         elt*P, Q,  /* Dimensions of the matrix             */
         0,     0,  /* Starting points (not important here) */
         elt*P, Q,  /* Dimensions of the submatrix          */
         1, 1, P));

    /* Create the DAG */
    dague_zlansy = (dague_handle_t*)dague_zlansy_new(
        P, Q, ntype, uplo, PlasmaTrans,
        (dague_ddesc_t*)A,
        (dague_ddesc_t*)Wcol,
        (dague_ddesc_t*)Welt,
        result);

    /* Set the datatypes */
    dplasma_add2arena_tile(((dague_zlansy_handle_t*)dague_zlansy)->arenas[DAGUE_zlansy_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);
    dplasma_add2arena_rectangle(((dague_zlansy_handle_t*)dague_zlansy)->arenas[DAGUE_zlansy_COL_ARENA],
                                mb * nb * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE, mb, nb, -1);
    dplasma_add2arena_rectangle(((dague_zlansy_handle_t*)dague_zlansy)->arenas[DAGUE_zlansy_ELT_ARENA],
                                elt * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE, elt, 1, -1);

    return (dague_handle_t*)dague_zlansy;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zlansy_Destruct - Clean the data structures associated to a
 *  zlansy dague object.
 *
 *******************************************************************************
 *
 * @param[in] o
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlansy_New
 * @sa dplasma_zlansy
 * @sa dplasma_clansy_Destruct
 * @sa dplasma_dlansy_Destruct
 * @sa dplasma_slansy_Destruct
 *
 ******************************************************************************/
void
dplasma_zlansy_Destruct( dague_handle_t *o )
{
    dague_zlansy_handle_t *dague_zlansy = (dague_zlansy_handle_t *)o;
    two_dim_block_cyclic_t *Wcol = (two_dim_block_cyclic_t*)(dague_zlansy->Wcol);
    two_dim_block_cyclic_t *Welt = (two_dim_block_cyclic_t*)(dague_zlansy->Welt);

    dplasma_datatype_undefine_type( &(dague_zlansy->arenas[DAGUE_zlansy_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlansy->arenas[DAGUE_zlansy_COL_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlansy->arenas[DAGUE_zlansy_ELT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);

    dague_data_free( Wcol->mat );
    dague_ddesc_destroy( (dague_ddesc_t*)Wcol );
    free( Wcol );

    dague_data_free( Welt->mat );
    dague_ddesc_destroy( (dague_ddesc_t*)Welt );
    free( Welt );
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zlansy - Synchronous version of dplasma_zlansy_New
 *
 *******************************************************************************
 *
 * @param[in] dague
 *          Dague context to which submit the DAG object.
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
 *          Must be a two_dim_rectangle_cyclic or sym_two_dim_rectangle_cyclic
 *          matrix
 *
*******************************************************************************
 *
 * @return
 *          \retval The computed norm
 *
 *******************************************************************************
 *
 * @sa dplasma_zlansy_Destruct
 * @sa dplasma_zlansy_New
 * @sa dplasma_clansy
 * @sa dplasma_dlansy
 * @sa dplasma_slansy
 *
 ******************************************************************************/
double dplasma_zlansy( dague_context_t *dague,
                       PLASMA_enum ntype,
                       PLASMA_enum uplo,
                       const tiled_matrix_desc_t *A)
{
    double result;
    dague_handle_t *dague_zlansy = NULL;

    dague_zlansy = dplasma_zlansy_New(ntype, uplo, A, &result);

    if ( dague_zlansy != NULL ) {
        dague_enqueue( dague, (dague_handle_t*)dague_zlansy);
        dplasma_progress(dague);
        dplasma_zlansy_Destruct( dague_zlansy );
    }

    return result;
}

