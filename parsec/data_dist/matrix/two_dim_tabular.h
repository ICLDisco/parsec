/*
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWODTD_H__
#define __TWODTD_H__

#include "parsec/data_internal.h"
#include "parsec/data_dist/matrix/matrix.h"

BEGIN_C_DECLS

/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

typedef struct parsec_two_dim_td_table_elem_s {
    uint32_t             rank;
    int32_t              vpid;
    int32_t              pos;
    void                *data;
} parsec_two_dim_td_table_elem_t;

typedef parsec_two_dim_td_table_elem_t two_dim_td_table_elem_t __parsec_attribute_deprecated__("Use parsec_two_dim_td_table_elem_t");

typedef struct parsec_two_dim_td_table_s {
    int nbelem;
    parsec_two_dim_td_table_elem_t elems[1]; /**< Elements of table are arranged column major. */
} parsec_two_dim_td_table_t;

typedef parsec_two_dim_td_table_t two_dim_td_table_t __parsec_attribute_deprecated__("Use parsec_two_dim_td_table_t");

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct parsec_matrix_tabular_s {
    parsec_tiled_matrix_t super;
    int user_table;
    parsec_two_dim_td_table_t *tiles_table;
} parsec_matrix_tabular_t;

typedef parsec_matrix_tabular_t two_dim_tabular_t __parsec_attribute_deprecated__("Use parsec_matrix_tabular_t");

/**
 * Initialize the description of a tabular abribtrary distribution
 * @param dc matrix description structure, already allocated, that will be initialize
 * @param nodes number of nodes
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of row in a tile
 * @param nb number of column in a tile
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param j starting column index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param n numbr of column of the entire submatrix
 * @param table associative table to assign tiles to all ranks. Can be NULL.
 *        In that case, you need to call set_table or set_random_table before
 *        using that descriptor.
 */

void parsec_matrix_tabular_init(parsec_matrix_tabular_t * dc,
                          parsec_matrix_type_t mtype,
                          unsigned int nodes, unsigned int myrank,
                          unsigned int mb, unsigned int nb,
                          unsigned int lm, unsigned int ln,
                          unsigned int i, unsigned int j,
                          unsigned int m, unsigned int n,
                          parsec_two_dim_td_table_t *table );

void two_dim_tabular_init(parsec_matrix_tabular_t * dc,
                          parsec_matrix_type_t mtype,
                          unsigned int nodes, unsigned int myrank,
                          unsigned int mb, unsigned int nb,
                          unsigned int lm, unsigned int ln,
                          unsigned int i, unsigned int j,
                          unsigned int m, unsigned int n,
                          parsec_two_dim_td_table_t *table )
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_init");

void parsec_matrix_tabular_destroy(parsec_matrix_tabular_t *tdc);
void parsec_matrix_tabular_set_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table);
void parsec_matrix_tabular_set_user_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table);
void parsec_matrix_tabular_set_random_table(parsec_matrix_tabular_t *dc, unsigned int seed);
void parsec_matrix_tabular_clone_table_structure(parsec_matrix_tabular_t *Src, parsec_matrix_tabular_t *Dst);

void two_dim_tabular_destroy(parsec_matrix_tabular_t *tdc)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_destroy");
void two_dim_tabular_set_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_set_table");
void two_dim_tabular_set_user_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_set_user_table");
void two_dim_tabular_set_random_table(parsec_matrix_tabular_t *dc, unsigned int seed)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_set_random_table");
void two_dim_td_table_clone_table_structure(parsec_matrix_tabular_t *Src, parsec_matrix_tabular_t *Dst)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_clone_table_structure");

END_C_DECLS

#endif /* __TWODTD_H__ */
