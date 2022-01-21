/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef __TWO_DIM_RECTANGLE_CYCLIC_H__
#error "Deprecated headers should not be included directly!"
#endif // __TWO_DIM_RECTANGLE_CYCLIC_H__

typedef parsec_matrix_block_cyclic_t two_dim_block_cyclic_t __parsec_attribute_deprecated__("Use parsec_matrix_block_cyclic_t");

static inline
void two_dim_block_cyclic_init(parsec_matrix_block_cyclic_t * twoDBCdesc,
                               parsec_matrix_type_t mtype,
                               parsec_matrix_storage_t storage,
                               int myrank,
                               int mb,    int nb,   /* Tile size */
                               int lm,    int ln,   /* Global matrix size (what is stored)*/
                               int i,     int j,    /* Staring point in the global matrix */
                               int m,     int n,    /* Submatrix size (the one concerned by the computation */
                               int p,     int q,    /* process process grid*/
                               int kp,    int kq,   /* k-cyclicity */
                               int ip,    int jq)   /* starting point on the process grid*/
    __parsec_attribute_deprecated__("Use parsec_matrix_block_cyclic_init");

static inline
void two_dim_block_cyclic_init(parsec_matrix_block_cyclic_t * twoDBCdesc,
                               parsec_matrix_type_t mtype,
                               parsec_matrix_storage_t storage,
                               int myrank,
                               int mb,    int nb,   /* Tile size */
                               int lm,    int ln,   /* Global matrix size (what is stored)*/
                               int i,     int j,    /* Staring point in the global matrix */
                               int m,     int n,    /* Submatrix size (the one concerned by the computation */
                               int p,     int q,    /* process process grid*/
                               int kp,    int kq,   /* k-cyclicity */
                               int ip,    int jq)
{
    parsec_matrix_block_cyclic_init(twoDBCdesc, mtype, storage, myrank,
                                    mb, nb, lm, ln, i, j, m, n, p, q,
                                    kp, kq, ip, jq);
}

static inline
void two_dim_block_cyclic_lapack_init(parsec_matrix_block_cyclic_t * twoDBCdesc,
                                      parsec_matrix_type_t mtype,
                                      parsec_matrix_storage_t storage,
                                      int myrank,
                                      int mb,   int nb,   /* Tile size */
                                      int lm,   int ln,   /* Global matrix size (what is stored)*/
                                      int i,    int j,    /* Staring point in the global matrix */
                                      int m,    int n,    /* Submatrix size (the one concerned by the computation */
                                      int p,     int q,   /* process process grid*/
                                      int kp,    int kq,  /* k-cyclicity */
                                      int ip,    int jq,  /* starting point on the process grid*/
                                      int mloc, int nloc) /* number of local rows and cols of the matrix */
    __parsec_attribute_deprecated__("Use parsec_matrix_block_cyclic_lapack_init");

static inline
void two_dim_block_cyclic_lapack_init(parsec_matrix_block_cyclic_t * twoDBCdesc,
                                      parsec_matrix_type_t mtype,
                                      parsec_matrix_storage_t storage,
                                      int myrank,
                                      int mb,   int nb,   /* Tile size */
                                      int lm,   int ln,   /* Global matrix size (what is stored)*/
                                      int i,    int j,    /* Staring point in the global matrix */
                                      int m,    int n,    /* Submatrix size (the one concerned by the computation */
                                      int p,     int q,   /* process process grid*/
                                      int kp,    int kq,  /* k-cyclicity */
                                      int ip,    int jq,  /* starting point on the process grid*/
                                      int mloc, int nloc)
{
    parsec_matrix_block_cyclic_lapack_init(twoDBCdesc, mtype, storage, myrank,
                                           mb, nb, lm, ln, i, j, m, n, p, q,
                                           kp, kq, ip, jq, mloc, nloc);
}

static inline
void two_dim_block_cyclic_kview( parsec_matrix_block_cyclic_t* target,
                                 parsec_matrix_block_cyclic_t* origin,
                                 int kp, int kq )
    __parsec_attribute_deprecated__("Use parsec_matrix_block_cyclic_kview");

static inline
void two_dim_block_cyclic_kview( parsec_matrix_block_cyclic_t* target,
                                 parsec_matrix_block_cyclic_t* origin,
                                 int kp, int kq )
{
    parsec_matrix_block_cyclic_kview(target, origin, kp, kq);
}
