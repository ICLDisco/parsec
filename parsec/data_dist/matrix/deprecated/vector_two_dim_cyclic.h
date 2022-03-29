/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __VECTOR_TWO_DIM_CYCLIC_H__
#error "Deprecated headers should not be included directly!"
#endif  // __VECTOR_TWO_DIM_CYCLIC_H__


typedef enum vector_distrib {
    matrix_VectorRow  __parsec_enum_attribute_deprecated__("Use PARSEC_VECTOR_DISTRIB_ROW"),
    matrix_VectorCol  __parsec_enum_attribute_deprecated__("Use PARSEC_VECTOR_DISTRIB_COL"),
    matrix_VectorDiag __parsec_enum_attribute_deprecated__("Use PARSEC_VECTOR_DISTRIB_DIAG")
} vector_distrib_t __parsec_attribute_deprecated__("Use parsec_vector_two_dim_cyclic_t_distrib_t");

typedef parsec_vector_two_dim_cyclic_t vector_two_dim_cyclic_t __parsec_attribute_deprecated__("Use parsec_vector_two_dim_cyclic_t");

static inline
void vector_two_dim_cyclic_init(parsec_vector_two_dim_cyclic_t * vdesc,
                                parsec_matrix_type_t    mtype,
                                enum parsec_vector_two_dim_cyclic_distrib_t distrib,
                                int myrank,
                                int mb, int lm, int i, int m,
                                int P, int Q )
    __parsec_attribute_deprecated__("Use parsec_vector_two_dim_cyclic_init");

static inline
void vector_two_dim_cyclic_init(parsec_vector_two_dim_cyclic_t * vdesc,
                                parsec_matrix_type_t    mtype,
                                enum parsec_vector_two_dim_cyclic_distrib_t distrib,
                                int myrank,
                                int mb, int lm, int i, int m,
                                int P, int Q )
{
    parsec_vector_two_dim_cyclic_init(vdesc, mtype, distrib, myrank,
                                        mb, lm, i, m, P, Q);
}
