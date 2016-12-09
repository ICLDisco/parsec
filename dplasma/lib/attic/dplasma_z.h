/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _DPLASMA_Z_EXTENDED_H_
#define _DPLASMA_Z_EXTENDED_H_

int  dplasma_zgetrf_hincpiv(      parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int *INFO);
int  dplasma_zgetrf_hpp(          parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int *INFO);
int  dplasma_zgetrf_hpp2(         parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int *INFO);
int  dplasma_zgetrf_hpp_multithrd(parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int *INFO);
void dplasma_ztrsmpl_hincpiv(     parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *B, int *INFO);
int  dplasma_ztrsmpl_hpp(         parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *LT, int *INFO);
int  dplasma_ztrsmpl_hpp2(        parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int *INFO);

parsec_handle_t* dplasma_zgetrf_hincpiv_New(      dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int *INFO);
parsec_handle_t* dplasma_zgetrf_hpp_New(          dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int *INFO);
parsec_handle_t* dplasma_zgetrf_hpp2_New(         dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int *INFO);
parsec_handle_t* dplasma_zgetrf_hpp_multithrd_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int *INFO);
parsec_handle_t* dplasma_ztrsmpl_hincpiv_New(     dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *LT, int *INFO);
parsec_handle_t* dplasma_ztrsmpl_hpp_New(         dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *LT, int *INFO);
parsec_handle_t* dplasma_ztrsmpl_hpp2_New(        dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int *INFO);

void dplasma_zgetrf_hincpiv_Destruct( parsec_handle_t *o );
void dplasma_zgetrf_hpp_Destruct( parsec_handle_t *o );
void dplasma_zgetrf_hpp2_Destruct( parsec_handle_t *o );
void dplasma_zgetrf_hpp_multithrd_Destruct( parsec_handle_t *o );
void dplasma_ztrsmpl_hincpiv_Destruct( parsec_handle_t *o );
void dplasma_ztrsmpl_hpp_Destruct( parsec_handle_t *o );
void dplasma_ztrsmpl_hpp2_Destruct( parsec_handle_t *o );

#endif /* _DPLASMA_Z_EXTENDED_H_ */
