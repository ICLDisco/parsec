/*
 * Copyright (c) 2010      The University of Tennessee and The University
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
#include "vpmap.h"

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

typedef struct tiled_matrix_desc_t {
    dague_ddesc_t super;
    struct _dague_data** data_map;   /**< map of the data */
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
    int i;              /**< row index to the beginning of the submatrix */
    int j;              /**< column indes to the beginning of the submatrix */
    int m;              /**< number of rows of the submatrix */
    int n;              /**< number of columns of the submatrix */
    int mt;             /**< number of tile rows of the submatrix - derived parameter */
    int nt;             /**< number of tile columns of the submatrix - derived parameter */
    int nb_local_tiles; /**< number of tile handled locally */
} tiled_matrix_desc_t;

void tiled_matrix_desc_init( tiled_matrix_desc_t *tdesc, enum matrix_type dtyp, enum matrix_storage storage,
                             int matrix_distribution_type, int nodes, int cores, int myrank,
                             int mb, int nb, int lm, int ln, int i,  int j, int m,  int n);

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

struct dague_execution_unit;
typedef int (*dague_operator_t)( struct dague_execution_unit *eu, const void* src, void* dst, void* op_data, ... );

extern struct dague_object_t*
dague_map_operator_New(const tiled_matrix_desc_t* src,
                       tiled_matrix_desc_t* dest,
                       dague_operator_t op,
                       void* op_data);

extern void
dague_map_operator_Destruct( struct dague_object_t* o );

extern struct dague_object_t*
dague_reduce_col_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      dague_operator_t operator,
                      void* op_data );

extern void dague_reduce_col_Destruct( struct dague_object_t *o );

extern struct dague_object_t*
dague_reduce_row_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      dague_operator_t operator,
                      void* op_data );
extern void dague_reduce_row_Destruct( struct dague_object_t *o );

/*
 * Macro to get the block leading dimension
 */
#define BLKLDD( _desc_, _m_ ) ( (_desc_).storage == matrix_Tile ? (_desc_).mb : (_desc_).lm )

#endif /* _MATRIX_H_  */
