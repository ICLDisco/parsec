/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zgetrf_fusion.h"

#define LDV  5
#define IB 40

dague_object_t* dplasma_zgetrf_fusion_New( tiled_matrix_desc_t *A,
                                           tiled_matrix_desc_t *IPIV,
                                           int P,
                                           int Q,
                                           int *info )
{
    dague_object_t *dague_zgetrf_fusion = NULL;
    two_dim_block_cyclic_t *UMAT, *LMAX, *V, *BUFFER, *ACOPY;
    int mb = A->mb, nb = A->nb;

    /* Create the workspaces */
    UMAT = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*UMAT), two_dim_block_cyclic,
        (UMAT, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         IB,  nb ,          /* Dimesions of the tile                */
         IB*P, nb*Q,      /* Dimensions of the matrix             */
         0,    0,          /* Starting points (not important here) */
         IB*P, nb*Q,      /* Dimensions of the submatrix          */
         1, 1, P));


    LMAX = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*LMAX), two_dim_block_cyclic,
        (LMAX, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         1,  nb +1,        /* Dimesions of the tile                */
         A->mt, (nb+1)*Q,  /* Dimensions of the matrix             */
         0,    0,          /* Starting points (not important here) */
         A->mt, (nb+1)*Q,  /* Dimensions of the submatrix          */
         1, 1, P));

    V = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*V), two_dim_block_cyclic,
        (V, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         LDV,   nb,      /* Dimesions of the tile                */
         LDV*P, nb*Q,    /* Dimensions of the matrix             */
         0,    0,        /* Starting points (not important here) */
         LDV*P, nb*Q,    /* Dimensions of the submatrix          */
         1, 1, P));

    BUFFER = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*BUFFER), two_dim_block_cyclic,
        (BUFFER, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         mb,   nb,      /* Dimesions of the tile                */
         mb*P, nb*A->nt,/* Dimensions of the matrix             */
         0,    0,       /* Starting points (not important here) */
         mb*P, nb*A->nt,/* Dimensions of the submatrix          */
         1, 1, P));

    ACOPY  = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*ACOPY), two_dim_block_cyclic,
        (ACOPY, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         mb,   nb,      /* Dimesions of the tile                */
         mb*P, nb*A->nt,/* Dimensions of the matrix             */
         0,    0,       /* Starting points (not important here) */
         mb*P, nb*A->nt,/* Dimensions of the submatrix          */
         1, 1, P));


    *info = 0;
    dague_zgetrf_fusion = (dague_object_t*)dague_zgetrf_fusion_new((dague_ddesc_t*)A,
                                                             (dague_ddesc_t*)IPIV,
                                                             (dague_ddesc_t*)UMAT,
                                                             (dague_ddesc_t*)LMAX,
                                                             (dague_ddesc_t*)V,
                                                             (dague_ddesc_t*)BUFFER,
                                                             (dague_ddesc_t*)ACOPY,
                                                             IB,
                                                             P,
                                                             Q,
                                                             info);

    /* A */
    dplasma_add2arena_tile( ((dague_zgetrf_fusion_object_t*)dague_zgetrf_fusion)->arenas[DAGUE_zgetrf_fusion_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* SWAP */
    dplasma_add2arena_rectangle( ((dague_zgetrf_fusion_object_t*)dague_zgetrf_fusion)->arenas[DAGUE_zgetrf_fusion_SWAP_ARENA],
                                 LDV*nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, LDV, nb, -1 );

    /* MAXL */
    dplasma_add2arena_rectangle( ((dague_zgetrf_fusion_object_t*)dague_zgetrf_fusion)->arenas[DAGUE_zgetrf_fusion_MAXL_ARENA],
                                 (nb+1)*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, 1, nb+1, -1 );

    /* UMES */
    dplasma_add2arena_rectangle( ((dague_zgetrf_fusion_object_t*)dague_zgetrf_fusion)->arenas[DAGUE_zgetrf_fusion_UMES_ARENA],
                                 IB*nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, IB, nb, -1 );

    /* PIVOT */
    dplasma_add2arena_rectangle( ((dague_zgetrf_fusion_object_t*)dague_zgetrf_fusion)->arenas[DAGUE_zgetrf_fusion_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    return (dague_object_t*)dague_zgetrf_fusion;
}

void
dplasma_zgetrf_fusion_Destruct( dague_object_t *o )
{
    dague_zgetrf_fusion_object_t *dague_zgetrf_fusion = (dague_zgetrf_fusion_object_t *)o;
    two_dim_block_cyclic_t *desc;

    desc = (two_dim_block_cyclic_t*)(dague_zgetrf_fusion->UMAT);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_zgetrf_fusion->UMAT );
    free( dague_zgetrf_fusion->UMAT );

    desc = (two_dim_block_cyclic_t*)(dague_zgetrf_fusion->LMAX);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_zgetrf_fusion->LMAX );
    free( dague_zgetrf_fusion->LMAX );

    desc = (two_dim_block_cyclic_t*)(dague_zgetrf_fusion->V);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_zgetrf_fusion->V );
    free( dague_zgetrf_fusion->V );

    desc = (two_dim_block_cyclic_t*)(dague_zgetrf_fusion->BUFFER);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_zgetrf_fusion->BUFFER );
    free( dague_zgetrf_fusion->BUFFER );

    desc= (two_dim_block_cyclic_t*)(dague_zgetrf_fusion->ACOPY);
    dague_data_free( desc->mat );
    dague_ddesc_destroy( dague_zgetrf_fusion->ACOPY );
    free( dague_zgetrf_fusion->ACOPY );

    dplasma_datatype_undefine_type( &(dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_SWAP_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_MAXL_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_UMES_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_PIVOT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

int dplasma_zgetrf_fusion( dague_context_t *dague,
                   tiled_matrix_desc_t *A,
                   tiled_matrix_desc_t *IPIV)
{
    int info = 0, ginfo = 0 ;
    dague_object_t *dague_zgetrf_fusion = NULL;

    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    int Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    dague_zgetrf_fusion = dplasma_zgetrf_fusion_New(A, IPIV, P, Q, &info);

    if ( dague_zgetrf_fusion != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgetrf_fusion);
        dplasma_progress(dague);
        dplasma_zgetrf_fusion_Destruct( dague_zgetrf_fusion );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}

