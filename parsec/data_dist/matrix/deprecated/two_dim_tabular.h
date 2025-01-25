/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWODTD_H__
#error "Deprecated headers should not be included directly!"
#endif // __TWODTD_H__

typedef parsec_two_dim_td_table_elem_t two_dim_td_table_elem_t __parsec_attribute_deprecated__("Use parsec_two_dim_td_table_elem_t");

typedef parsec_two_dim_td_table_t two_dim_td_table_t __parsec_attribute_deprecated__("Use parsec_two_dim_td_table_t");

typedef parsec_matrix_tabular_t two_dim_tabular_t __parsec_attribute_deprecated__("Use parsec_matrix_tabular_t");

static inline
void two_dim_tabular_init(parsec_matrix_tabular_t * dc,
                          parsec_matrix_type_t mtype,
                          unsigned int nodes, unsigned int myrank,
                          unsigned int mb, unsigned int nb,
                          unsigned int lm, unsigned int ln,
                          unsigned int i, unsigned int j,
                          unsigned int m, unsigned int n,
                          parsec_two_dim_td_table_t *table )
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_init");

static inline
void two_dim_tabular_init(parsec_matrix_tabular_t * dc,
                          parsec_matrix_type_t mtype,
                          unsigned int nodes, unsigned int myrank,
                          unsigned int mb, unsigned int nb,
                          unsigned int lm, unsigned int ln,
                          unsigned int i, unsigned int j,
                          unsigned int m, unsigned int n,
                          parsec_two_dim_td_table_t *table )
{
    parsec_matrix_tabular_init(dc, mtype, nodes, myrank,
                               mb, nb, lm, ln, i, j, m, n, table);
}

static inline
void two_dim_tabular_destroy(parsec_matrix_tabular_t *tdc)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_destroy");

static inline
void two_dim_tabular_destroy(parsec_matrix_tabular_t *tdc)
{
    parsec_matrix_tabular_destroy(tdc);
}

static inline
void two_dim_tabular_set_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_set_table");

static inline
void two_dim_tabular_set_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
{
    parsec_matrix_tabular_set_table(dc, table);
}

static inline
void two_dim_tabular_set_user_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_set_user_table");

static inline
void two_dim_tabular_set_user_table(parsec_matrix_tabular_t *dc, parsec_two_dim_td_table_t *table)
{
    parsec_matrix_tabular_set_user_table(dc, table);
}

static inline
void two_dim_tabular_set_random_table(parsec_matrix_tabular_t *dc, unsigned int seed)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_set_random_table");

static inline
void two_dim_tabular_set_random_table(parsec_matrix_tabular_t *dc, unsigned int seed)
{
    parsec_matrix_tabular_set_random_table(dc, seed);
}

static inline
void two_dim_td_table_clone_table_structure(parsec_matrix_tabular_t *Src, parsec_matrix_tabular_t *Dst)
    __parsec_attribute_deprecated__("Use parsec_matrix_tabular_clone_table_structure");

static inline
void two_dim_td_table_clone_table_structure(parsec_matrix_tabular_t *Src, parsec_matrix_tabular_t *Dst)
{
    parsec_matrix_tabular_clone_table_structure(Src, Dst);
}
