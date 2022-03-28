/*
 * Copyright (c) 2010-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _MATRIX_H_
#error "Deprecated headers must not be included directly!"
#endif // _MATRIX_H_


enum
__parsec_attribute_deprecated__("Use parsec_matrix_type_t instead")
matrix_type {
    matrix_Byte          __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_BYTE instead") =  PARSEC_MATRIX_BYTE, /**< unsigned char  */
    matrix_Integer       __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_INTEGER instead") = PARSEC_MATRIX_INTEGER, /**< signed int     */
    matrix_RealFloat     __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_FLOAT instead") = PARSEC_MATRIX_FLOAT, /**< float          */
    matrix_RealDouble    __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_DOUBLE instead") = PARSEC_MATRIX_DOUBLE, /**< double         */
    matrix_ComplexFloat  __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_COMPLEX_FLOAT instead") = PARSEC_MATRIX_COMPLEX_FLOAT, /**< complex float  */
    matrix_ComplexDouble __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_COMPLEX_DOUBLE instead") = PARSEC_MATRIX_COMPLEX_DOUBLE  /**< complex double */
};

enum
__parsec_attribute_deprecated__("Use parsec_matrix_storage_t instead")
matrix_storage {
    matrix_Lapack __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_LAPACK instead") = PARSEC_MATRIX_LAPACK, /**< LAPACK Layout or Column Major  */
    matrix_Tile   __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_TILE instead") = PARSEC_MATRIX_TILE, /**< Tile Layout or Column-Column Rectangular Block (CCRB) */
};

enum
__parsec_attribute_deprecated__("Use parsec_matrix_uplo_t instead")
matrix_uplo {
    matrix_Upper      __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_UPPER instead") = PARSEC_MATRIX_UPPER,
    matrix_Lower      __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_LOWER instead") = PARSEC_MATRIX_LOWER,
    matrix_UpperLower __parsec_enum_attribute_deprecated__("Use PARSEC_MATRIX_FULL instead")  = PARSEC_MATRIX_FULL
};

enum {
    parsec_tiled_matrix_dc_type    __parsec_enum_attribute_deprecated__("Use parsec_matrix_type instead") = parsec_matrix_type,
    two_dim_block_cyclic_type      __parsec_enum_attribute_deprecated__("Use parsec_matrix_block_cyclic_type instead") = parsec_matrix_block_cyclic_type,
    sym_two_dim_block_cyclic_type  __parsec_enum_attribute_deprecated__("Use parsec_matrix_sym_block_cyclic_type instead") = parsec_matrix_sym_block_cyclic_type,
    two_dim_tabular_type           __parsec_enum_attribute_deprecated__("Use parsec_matrix_tabular_type instead") = parsec_matrix_tabular_type
};

typedef parsec_tiled_matrix_t parsec_tiled_matrix_dc_t __parsec_attribute_deprecated__("Use parsec_tiled_matrix_t instead");

static inline
void parsec_tiled_matrix_dc_init( parsec_tiled_matrix_t *tdesc, parsec_matrix_type_t dtyp, parsec_matrix_storage_t storage,
                             int matrix_distribution_type, int nodes, int myrank,
                             int mb, int nb, int lm, int ln, int i,  int j, int m,  int n)
    __parsec_attribute_deprecated__("Use parsec_tiled_matrix_init");

static inline
void parsec_tiled_matrix_dc_init( parsec_tiled_matrix_t *tdesc, parsec_matrix_type_t dtyp, parsec_matrix_storage_t storage,
                             int matrix_distribution_type, int nodes, int myrank,
                             int mb, int nb, int lm, int ln, int i,  int j, int m,  int n)
{
    parsec_tiled_matrix_init(tdesc, dtyp, storage, matrix_distribution_type,
                             nodes, myrank, mb, nb, lm, ln, i, j, m, n);
}

static inline
void parsec_tiled_matrix_dc_destroy( parsec_tiled_matrix_t *tdesc )
    __parsec_attribute_deprecated__("Use parsec_tiled_matrix_destroy");

static inline
void parsec_tiled_matrix_dc_destroy( parsec_tiled_matrix_t *tdesc )
{
    parsec_tiled_matrix_destroy(tdesc);
}

static inline
parsec_tiled_matrix_t *tiled_matrix_submatrix( parsec_tiled_matrix_t *tdesc, int i, int j, int m, int n)
    __parsec_attribute_deprecated__("Use parsec_tiled_matrix_submatrix");

static inline
parsec_tiled_matrix_t *tiled_matrix_submatrix( parsec_tiled_matrix_t *tdesc, int i, int j, int m, int n)
{
    return parsec_tiled_matrix_submatrix(tdesc, i, j, m, n);
}

static inline
int  tiled_matrix_data_write(parsec_tiled_matrix_t *tdesc, char *filename)
    __parsec_attribute_deprecated__("Use parsec_tiled_matrix_data_write");

static inline
int  tiled_matrix_data_write(parsec_tiled_matrix_t *tdesc, char *filename)
{
    return parsec_tiled_matrix_data_write(tdesc, filename);
}

static inline
int  tiled_matrix_data_read(parsec_tiled_matrix_t *tdesc, char *filename)
    __parsec_attribute_deprecated__("Use parsec_tiled_matrix_data_read");

static inline
int  tiled_matrix_data_read(parsec_tiled_matrix_t *tdesc, char *filename)
{
    return parsec_tiled_matrix_data_read(tdesc, filename);
}

typedef parsec_tiled_matrix_unary_op_t tiled_matrix_unary_op_t __parsec_attribute_deprecated__("Use parsec_tiled_matrix_unary_op_t instead");

typedef parsec_tiled_matrix_binary_op_t tiled_matrix_binary_op_t __parsec_attribute_deprecated__("Use parsec_tiled_matrix_binary_op_t instead");

static inline
parsec_data_t*
parsec_matrix_create_data(parsec_tiled_matrix_t* matrix,
                         void* ptr,
                         int pos,
                         parsec_data_key_t key)
    __parsec_attribute_deprecated__("Use parsec_tiled_matrix_create_data");

static inline
parsec_data_t*
parsec_matrix_create_data(parsec_tiled_matrix_t* matrix,
                         void* ptr,
                         int pos,
                         parsec_data_key_t key)
{
    return parsec_tiled_matrix_create_data(matrix, ptr, pos, key);
}

static inline
void
parsec_matrix_destroy_data( parsec_tiled_matrix_t* matrix )
    __parsec_attribute_deprecated__("Use parsec_tiled_matrix_destroy_data");

static inline
void
parsec_matrix_destroy_data( parsec_tiled_matrix_t* matrix )
{
    parsec_tiled_matrix_destroy_data(matrix);
}

static inline
int parsec_matrix_add2arena( parsec_arena_datatype_t *adt, parsec_datatype_t oldtype,
                             parsec_matrix_uplo_t uplo, int diag,
                             unsigned int m, unsigned int n, unsigned int ld,
                             size_t alignment, int resized )
    __parsec_attribute_deprecated__("Use parsec_add2arena");

static inline
int parsec_matrix_add2arena( parsec_arena_datatype_t *adt, parsec_datatype_t oldtype,
                             parsec_matrix_uplo_t uplo, int diag,
                             unsigned int m, unsigned int n, unsigned int ld,
                             size_t alignment, int resized )
{
    return parsec_add2arena(adt, oldtype, uplo, diag, m, n, ld, alignment, resized);
}

static inline
int parsec_matrix_del2arena( parsec_arena_datatype_t *adt )
    __parsec_attribute_deprecated__("Use parsec_del2arena");

static inline
int parsec_matrix_del2arena( parsec_arena_datatype_t *adt )
{
    return parsec_del2arena(adt);
}

/* deprecated */
#define parsec_matrix_add2arena_tile( _adt_ , _oldtype_, _m_ ) \
    parsec_matrix_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_FULL, 0, (_m_), (_m_), (_m_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )

#define parsec_matrix_add2arena_upper( _adt_ , _oldtype_, diag, _n_ ) \
    parsec_matrix_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_UPPER, (_diag_), (_n_), (_n_), (_n_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )

#define parsec_matrix_add2arena_lower( _adt_ , _oldtype_, diag, _n_ ) \
    parsec_matrix_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_LOWER, (_diag_), (_n_), (_n_), (_n_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )

#define parsec_matrix_add2arena_rect( _adt_ , _oldtype_, _m_, _n_, _ld_ ) \
    parsec_matrix_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_FULL, 0, (_m_), (_n_), (_ld_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )
