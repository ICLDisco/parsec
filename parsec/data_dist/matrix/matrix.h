/*
 * Copyright (c) 2010-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "parsec/parsec_config.h"
#include <stdarg.h>
#include <stdio.h>
#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/data.h"
#include "parsec/datatype.h"
#include "parsec/parsec_internal.h"

BEGIN_C_DECLS

struct parsec_execution_stream_s;
struct parsec_taskpool_s;

typedef enum parsec_matrix_type_e {
    PARSEC_MATRIX_BYTE          = 0, /**< unsigned char  */
    PARSEC_MATRIX_INTEGER       = 1, /**< signed int     */
    PARSEC_MATRIX_FLOAT     = 2, /**< float          */
    PARSEC_MATRIX_DOUBLE    = 3, /**< double         */
    PARSEC_MATRIX_COMPLEX_FLOAT  = 4, /**< complex float  */
    PARSEC_MATRIX_COMPLEX_DOUBLE = 5  /**< complex double */
} parsec_matrix_type_t;

typedef enum parsec_matrix_storage_e {
    PARSEC_MATRIX_LAPACK        = 0, /**< LAPACK Layout or Column Major  */
    PARSEC_MATRIX_TILE          = 1, /**< Tile Layout or Column-Column Rectangular Block (CCRB) */
} parsec_matrix_storage_t;

/**
 * Put our own definition of Upper/Lower/General values mathing the
 * Cblas/Plasma/... ones to avoid the dependency
 */
typedef enum parsec_matrix_uplo_e {
    PARSEC_MATRIX_UPPER      = 121,
    PARSEC_MATRIX_LOWER      = 122,
    PARSEC_MATRIX_FULL       = 123
} parsec_matrix_uplo_t;

/**
 * Obtain the size in bytes of a matrix type.
 */
static inline int parsec_datadist_getsizeoftype(parsec_matrix_type_t type)
{
    int size = -1;
    switch( type ) {
    case PARSEC_MATRIX_BYTE          : parsec_type_size(parsec_datatype_int8_t, &size); break;
    case PARSEC_MATRIX_INTEGER       : parsec_type_size(parsec_datatype_int32_t, &size); break;
    case PARSEC_MATRIX_FLOAT     : parsec_type_size(parsec_datatype_float_t, &size); break;
    case PARSEC_MATRIX_DOUBLE    : parsec_type_size(parsec_datatype_double_t, &size); break;
    case PARSEC_MATRIX_COMPLEX_FLOAT  : parsec_type_size(parsec_datatype_complex_t, &size); break;
    case PARSEC_MATRIX_COMPLEX_DOUBLE : parsec_type_size(parsec_datatype_double_complex_t, &size); break;
    /* If you want to add more types, note that size=extent is true only for predefined datatypes.
     * also, for non-predefined datatypes, you'd want to check for errors from
     * parsec_type_size() */
    default:
        return -1;
    }
    return size;
}

/**
 * Convert from a matrix type to a more traditional PaRSEC type usable for
 * creating arenas.
 */
static inline int parsec_translate_matrix_type( parsec_matrix_type_t mt, parsec_datatype_t* dt )
{
    switch(mt) {
    case PARSEC_MATRIX_BYTE:          *dt = parsec_datatype_int8_t; break;
    case PARSEC_MATRIX_INTEGER:       *dt = parsec_datatype_int32_t; break;
    case PARSEC_MATRIX_FLOAT:     *dt = parsec_datatype_float_t; break;
    case PARSEC_MATRIX_DOUBLE:    *dt = parsec_datatype_double_t; break;
    case PARSEC_MATRIX_COMPLEX_FLOAT:  *dt = parsec_datatype_complex_t; break;
    case PARSEC_MATRIX_COMPLEX_DOUBLE: *dt = parsec_datatype_double_complex_t; break;
    default:
        fprintf(stderr, "%s:%d Unknown matrix_type (%d)\n", __func__, __LINE__, mt);
        return PARSEC_ERR_BAD_PARAM;
    }
    return PARSEC_SUCCESS;
}

enum {
  parsec_matrix_type = 0x01,
  parsec_matrix_block_cyclic_type = 0x2,
  parsec_matrix_sym_block_cyclic_type = 0x4,
  parsec_matrix_tabular_type = 0x8
};

typedef struct parsec_tiled_matrix_s {
    parsec_data_collection_t super;
    parsec_data_t**       data_map;   /**< map of the data */
    parsec_matrix_type_t     mtype;      /**< precision of the matrix */
    parsec_matrix_storage_t  storage;    /**< storage of the matrix   */
    int dtype;          /**< Distribution type of descriptor      */
    int tileld;         /**< leading dimension of each tile (Should be a function depending on the row) */
    int mb;             /**< number of rows in a tile */
    int nb;             /**< number of columns in a tile */
    int bsiz;           /**< size in elements including padding of a tile - derived parameter */
    int lm;             /**< number of rows of the entire matrix */
    int ln;             /**< number of columns of the entire matrix */
    int lmt;            /**< number of tile rows of the entire matrix - derived parameter */
    int lnt;            /**< number of tile columns of the entire matrix - derived parameter */
    int llm;            /**< number of rows of the matrix stored locally - derived parameter */
    int lln;            /**< number of columns of the matrix stored locally - derived parameter */
    int i;              /**< row index to the beginning of the submatrix */
    int j;              /**< column index to the beginning of the submatrix */
    int m;              /**< number of rows of the submatrix */
    int n;              /**< number of columns of the submatrix */
    int mt;             /**< number of tile rows of the submatrix - derived parameter */
    int nt;             /**< number of tile columns of the submatrix - derived parameter */
    int nb_local_tiles; /**< number of tile handled locally */
    int slm;            /**< number of local rows of the submatrix */
    int sln;            /**< number of local columns of the submatrix */
} parsec_tiled_matrix_t;

void parsec_tiled_matrix_init( parsec_tiled_matrix_t *tdesc, parsec_matrix_type_t dtyp, parsec_matrix_storage_t storage,
                             int matrix_distribution_type, int nodes, int myrank,
                             int mb, int nb, int lm, int ln, int i,  int j, int m,  int n);

void parsec_tiled_matrix_destroy( parsec_tiled_matrix_t *tdesc );

parsec_tiled_matrix_t *parsec_tiled_matrix_submatrix( parsec_tiled_matrix_t *tdesc, int i, int j, int m, int n);

int  parsec_tiled_matrix_data_write(parsec_tiled_matrix_t *tdesc, char *filename);

int  parsec_tiled_matrix_data_read(parsec_tiled_matrix_t *tdesc, char *filename);

typedef int (*parsec_operator_t)( struct parsec_execution_stream_s *es,
                                  const void* src,
                                  void* dst,
                                  void* op_data,
                                  ... );

typedef int (*parsec_tiled_matrix_unary_op_t )( struct parsec_execution_stream_s *es,
                                         const parsec_tiled_matrix_t *desc1,
                                         void *data1,
                                         int uplo, int m, int n,
                                         void *args );

typedef int (*parsec_tiled_matrix_binary_op_t)( struct parsec_execution_stream_s *es,
                                         const parsec_tiled_matrix_t *desc1,
                                         const parsec_tiled_matrix_t *desc2,
                                         const void *data1, void *data2,
                                         int uplo, int m, int n,
                                         void *args );

extern struct parsec_taskpool_s*
parsec_map_operator_New(const parsec_tiled_matrix_t* src,
                       parsec_tiled_matrix_t* dest,
                       parsec_operator_t op,
                       void* op_data);

extern struct parsec_taskpool_s*
parsec_reduce_col_New( const parsec_tiled_matrix_t* src,
                      parsec_tiled_matrix_t* dest,
                      parsec_operator_t op,
                      void* op_data );

extern void parsec_reduce_col_Destruct( struct parsec_taskpool_s *o );

extern struct parsec_taskpool_s*
parsec_reduce_row_New( const parsec_tiled_matrix_t* src,
                      parsec_tiled_matrix_t* dest,
                      parsec_operator_t op,
                      void* op_data );
extern void parsec_reduce_row_Destruct( struct parsec_taskpool_s *o );

extern struct parsec_taskpool_s*
parsec_apply_New(     parsec_matrix_uplo_t uplo,
                      parsec_tiled_matrix_t* A,
                      parsec_tiled_matrix_unary_op_t operation,
                      void* op_args );

extern void parsec_apply_Destruct( struct parsec_taskpool_s *o );

extern int
parsec_apply( parsec_context_t *parsec,
             parsec_matrix_uplo_t uplo,
             parsec_tiled_matrix_t *A,
             parsec_tiled_matrix_unary_op_t operation,
             void *op_args );
/**
 * @brief Non-blocking function of redistribute for PTG
 *
 * @param [in] source: source distribution, already distributed and allocated
 * @param [out] target: target distribution, redistributed and allocated
 * @param [in] size_row: row size to be redistributed
 * @param [in] size_col: column size to be redistributed
 * @param [in] disi_source: row displacement in source
 * @param [in] disj_source: column displacement in source
 * @param [in] disi_target: row displacement in target
 * @param [in] disj_target: column displacement in target
 * @return the parsec object to schedule.
 */
parsec_taskpool_t*
parsec_redistribute_New(parsec_tiled_matrix_t *source,
                        parsec_tiled_matrix_t *target,
                        int size_row, int size_col,
                        int disi_source, int disj_source,
                        int disi_target, int disj_target);

/**
 * @brief Cases other than that of parsec_redistribute_ss_Destruct
 * @param [inout] the parsec object to destroy
 */
void parsec_redistribute_Destruct(parsec_taskpool_t *taskpool);

/**
 * @brief Redistribute source to target of PTG
 *
 * @details
 * Source and target could be ANY distribuiton with ANY displacement
 * in both source and target.
 *
 * @param [in] source: source distribution, already distributed and allocated
 * @param [out] target: target distribution, redistributed and allocated
 * @param [in] size_row: row size to be redistributed
 * @param [in] size_col: column size to be redistributed
 * @param [in] disi_source: row displacement in source
 * @param [in] disj_source: column displacement in source
 * @param [in] disi_target: row displacement in target
 * @param [in] disj_target: column displacement in target
 */
int parsec_redistribute(parsec_context_t *parsec,
                        parsec_tiled_matrix_t *source,
                        parsec_tiled_matrix_t *target,
                        int size_row, int size_col,
                        int disi_source, int disj_source,
                        int disi_target, int disj_target);

/**
 * @brief Non-blocking function of redistribute for DTD
 *
 * @details
 * Source and target could be ANY distribuiton with ANY displacement
 * in both source and target.
 *
 * @param [in] source: source distribution, already distributed and allocated
 * @param [out] target: target distribution, redistributed and allocated
 * @param [in] size_row: row size to be redistributed
 * @param [in] size_col: column size to be redistributed
 * @param [in] disi_source: row displacement in source
 * @param [in] disj_source: column displacement in source
 * @param [in] disi_target: row displacement in target
 * @param [in] disj_target: column displacement in target
 */
int parsec_redistribute_dtd_New(parsec_context_t *parsec,
                            parsec_tiled_matrix_t *source,
                            parsec_tiled_matrix_t *target,
                            int size_row, int size_col,
                            int disi_source, int disj_source,
                            int disi_target, int disj_target);

/**
 * @brief Redistribute source to target of DTD
 *
 * @details
 * Source and target could be ANY distribuiton with ANY displacement
 * in both source and target.
 *
 * @param [in] source: source distribution, already distributed and allocated
 * @param [out] target: target distribution, redistributed and allocated
 * @param [in] size_row: row size to be redistributed
 * @param [in] size_col: column size to be redistributed
 * @param [in] disi_source: row displacement in source
 * @param [in] disj_source: column displacement in source
 * @param [in] disi_target: row displacement in target
 * @param [in] disj_target: column displacement in target
 */
int parsec_redistribute_dtd(parsec_context_t *parsec,
                            parsec_tiled_matrix_t *source,
                            parsec_tiled_matrix_t *target,
                            int size_row, int size_col,
                            int disi_source, int disj_source,
                            int disi_target, int disj_target);

/*
 * Macro to get the block leading dimension
 */
#define BLKLDD( _desc_, _m_ ) ( (_desc_)->storage == PARSEC_MATRIX_TILE ? (_desc_)->mb : (_desc_)->llm )
#define TILED_MATRIX_KEY( _desc_, _m_, _n_ ) ( ((parsec_data_collection_t*)(_desc_))->data_key( ((parsec_data_collection_t*)(_desc_)), (_m_), (_n_) ) )

/**
 * Helper functions to allocate and retrieve pointers to the parsec_data_t and
 * the corresponding copies.
 */
parsec_data_t*
parsec_tiled_matrix_create_data(parsec_tiled_matrix_t* matrix,
                         void* ptr,
                         int pos,
                         parsec_data_key_t key);

void
parsec_tiled_matrix_destroy_data( parsec_tiled_matrix_t* matrix );

/**
 * Helper function to create datatypes for matrices with different shapes. This generic
 * function allow for the creation of vector of vector data, or a tile in a submatrice.
 * The m and n are the size of the tile, while the LDA is the size of the submatrix.
 * The resized parameter indicates the need to resize the resulting data. A negative
 * resized indicates that no resize if necessary, while any positive value will resize
 * the resulting datatype to resized times the size of the oldtype. This allows for the
 * creation of column major data layouts in a row major storage.
 * The extent of the output datatype is set on the extent argument.
 */
int parsec_matrix_define_datatype(parsec_datatype_t *newtype, parsec_datatype_t oldtype,
                                  parsec_matrix_uplo_t uplo, int diag,
                                  unsigned int m, unsigned int n, unsigned int ld,
                                  int resized,
                                  ptrdiff_t * extent);

/**
 * Helper functions to create both the datatype and the arena of matrices with different
 * shapes. Datatypes are created as defined by parsec_matrix_define_datatype and the arena
 * is created using the datatype extent. The alignment indicates the restrictions related
 * to the alignment of the allocated data by the arena.
 */
int parsec_add2arena( parsec_arena_datatype_t *adt, parsec_datatype_t oldtype,
                             parsec_matrix_uplo_t uplo, int diag,
                             unsigned int m, unsigned int n, unsigned int ld,
                             size_t alignment, int resized );

int parsec_del2arena( parsec_arena_datatype_t *adt );

#define parsec_add2arena_tile( _adt_ , _oldtype_, _m_ ) \
    parsec_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_FULL, 0, (_m_), (_m_), (_m_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )

#define parsec_add2arena_upper( _adt_ , _oldtype_, diag, _n_ ) \
    parsec_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_UPPER, (_diag_), (_n_), (_n_), (_n_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )

#define parsec_add2arena_lower( _adt_ , _oldtype_, diag, _n_ ) \
    parsec_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_LOWER, (_diag_), (_n_), (_n_), (_n_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )

#define parsec_add2arena_rect( _adt_ , _oldtype_, _m_, _n_, _ld_ ) \
    parsec_add2arena( (_adt_), (_oldtype_), PARSEC_MATRIX_FULL, 0, (_m_), (_n_), (_ld_), PARSEC_ARENA_ALIGNMENT_SSE, -1 )

/* include deprecated symbols */
#include "parsec/data_dist/matrix/deprecated/matrix.h"

END_C_DECLS
#endif /* _MATRIX_H_  */
