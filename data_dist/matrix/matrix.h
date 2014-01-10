/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "dague_config.h"
#include <stdarg.h>
#include <assert.h>
#include "precision.h"
#include "data_distribution.h"
#include "data.h"
#include "vpmap.h"

BEGIN_C_DECLS

struct dague_object_t;

enum matrix_type {
    matrix_Byte          = 0, /**< unsigned char  */
    matrix_Integer       = 1, /**< signed int     */
    matrix_RealFloat     = 2, /**< float          */
    matrix_RealDouble    = 3, /**< double         */
    matrix_ComplexFloat  = 4, /**< complex float  */
    matrix_ComplexDouble = 5  /**< complex double */
};

enum matrix_storage {
    matrix_Lapack        = 0, /**< LAPACK Layout or Column Major  */
    matrix_Tile          = 1, /**< Tile Layout or Column-Column Rectangular Block (CCRB) */
};

static inline int dague_datadist_getsizeoftype(enum matrix_type type)
{
    switch( type ) {
    case matrix_Byte          : return sizeof(char);
    case matrix_Integer       : return sizeof(int);
    case matrix_RealFloat     : return sizeof(float);
    case matrix_RealDouble    : return sizeof(double);
    case matrix_ComplexFloat  : return sizeof(dague_complex32_t);
    case matrix_ComplexDouble : return sizeof(dague_complex64_t);
    default:
        return -1;
    }
}

#define tiled_matrix_desc_type        0x01
#define two_dim_block_cyclic_type     0x02
#define sym_two_dim_block_cyclic_type 0x04
#define two_dim_tabular_type          0x08

typedef struct tiled_matrix_desc_t {
    dague_ddesc_t super;
    dague_data_t**       data_map;   /**< map of the data */
    enum matrix_type     mtype;      /**< precision of the matrix */
    enum matrix_storage  storage;    /**< storage of the matrix   */
    int dtype;          /**< Distribution type of descriptor      */
    int tileld;         /**< leading dimension of each tile (Should be a function depending on the row) */
    int mb;             /**< number of rows in a tile */
    int nb;             /**< number of columns in a tile */
    int bsiz;           /**< size in elements including padding of a tile - derived parameter */
    int lm;             /**< number of rows of the entire matrix */
    int ln;             /**< number of columns of the entire matrix */
    int lmt;            /**< number of tile rows of the entire matrix - derived parameter */
    int lnt;            /**< number of tile columns of the entire matrix - derived parameter */
    int llm;            /**< number of rows of the matrix stored localy - derived parameter */
    int lln;            /**< number of columns of the matrix stored localy - derived parameter */
    int i;              /**< row index to the beginning of the submatrix */
    int j;              /**< column indes to the beginning of the submatrix */
    int m;              /**< number of rows of the submatrix */
    int n;              /**< number of columns of the submatrix */
    int mt;             /**< number of tile rows of the submatrix - derived parameter */
    int nt;             /**< number of tile columns of the submatrix - derived parameter */
    int nb_local_tiles; /**< number of tile handled locally */
} tiled_matrix_desc_t;

void tiled_matrix_desc_init( tiled_matrix_desc_t *tdesc, enum matrix_type dtyp, enum matrix_storage storage,
                             int matrix_distribution_type, int nodes, int myrank,
                             int mb, int nb, int lm, int ln, int i,  int j, int m,  int n);

void tiled_matrix_desc_destroy( tiled_matrix_desc_t *tdesc );

tiled_matrix_desc_t *tiled_matrix_submatrix( tiled_matrix_desc_t *tdesc, int i, int j, int m, int n);

int  tiled_matrix_data_write(tiled_matrix_desc_t *tdesc, char *filename);
int  tiled_matrix_data_read(tiled_matrix_desc_t *tdesc, char *filename);

static inline int32_t tiled_matrix_get_vpid(tiled_matrix_desc_t *tdesc, int pos)
{
    assert( vpmap_get_nb_vp() > 0 );
    assert( pos <= tdesc->nb_local_tiles );
    (void)tdesc;
    return pos % vpmap_get_nb_vp();
}

struct dague_execution_unit_s;
typedef int (*dague_operator_t)( struct dague_execution_unit_s *eu, const void* src, void* dst, void* op_data, ... );

extern struct dague_handle_t*
dague_map_operator_New(const tiled_matrix_desc_t* src,
                       tiled_matrix_desc_t* dest,
                       dague_operator_t op,
                       void* op_data);

extern void
dague_map_operator_Destruct( struct dague_handle_t* o );

extern struct dague_handle_t*
dague_reduce_col_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      dague_operator_t op,
                      void* op_data );

extern void dague_reduce_col_Destruct( struct dague_handle_t *o );

extern struct dague_handle_t*
dague_reduce_row_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      dague_operator_t op,
                      void* op_data );
extern void dague_reduce_row_Destruct( struct dague_handle_t *o );

/*
 * Macro to get the block leading dimension
 */
#define BLKLDD( _desc_, _m_ ) ( (_desc_).storage == matrix_Tile ? (_desc_).mb : (_desc_).llm )
#define TILED_MATRIX_KEY( _desc_, _m_, _n_ ) ( ((dague_ddesc_t*)(_desc_))->data_key( ((dague_ddesc_t*)(_desc_)), (_m_), (_n_) ) )

/**
 * Helper functions to allocate and retrieve pointers to the dague_data_t and
 * the corresponding copies.
 */
static inline dague_data_t*
dague_matrix_create_data(tiled_matrix_desc_t* matrix,
                         void* ptr,
                         int pos,
                         dague_data_key_t key)
{
    dague_data_t* data = matrix->data_map[pos];

    if( NULL == data ) {
        dague_data_copy_t* data_copy = OBJ_NEW(dague_data_copy_t);
        data = OBJ_NEW(dague_data_t);

        data_copy->coherency_state = DATA_COHERENCY_OWNED;
        data_copy->original = data;
        data_copy->device_private = ptr;

        data->owner_device = 0;
        data->key = key;
        data->nb_elts = matrix->bsiz * dague_datadist_getsizeoftype(matrix->mtype);
        data->device_copies[0] = data_copy;

        if( !dague_atomic_cas(&matrix->data_map[pos], NULL, data) ) {
            free(data_copy);
            free(data);
            data = matrix->data_map[pos];
        }
    } else {
        /* Do we have a copy of this data */
        if( NULL == data->device_copies[0] ) {
            dague_data_copy_t* data_copy = dague_data_copy_new(data, 0);
            data_copy->device_private = ptr;
        }
    }
    return data;
}

END_C_DECLS
#endif /* _MATRIX_H_  */
