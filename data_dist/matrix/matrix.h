/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _MATRIX_H_ 
#define _MATRIX_H_ 

#include <stdarg.h>
#include "dague_config.h"
#include "precision.h"
#include "data_distribution.h"

// TODO: This type is weird/broken, it needs to be fixed at some point (was not conceived originally to compare 2 matrices...)
enum matrix_type {
    matrix_Byte          = sizeof(char),              /**< unsigned char */
    matrix_Integer       = sizeof(int),               /**< signed int */
    matrix_RealFloat     = sizeof(float),             /**< float */
    matrix_RealDouble    = sizeof(double),            /**< double */
    matrix_ComplexFloat  = sizeof(Dague_Complex32_t), /**< complex float */
    matrix_ComplexDouble = sizeof(Dague_Complex64_t)  /**< complex double */
};

typedef struct tiled_matrix_desc_t {
    dague_ddesc_t super;
    enum matrix_type mtype;      /**< precision of the matrix */
    unsigned int mb;             /**< number of rows in a tile */
    unsigned int nb;             /**< number of columns in a tile */
    unsigned int bsiz;           /**< size in elements including padding of a tile - derived parameter */
    unsigned int lm;             /**< number of rows of the entire matrix */
    unsigned int ln;             /**< number of columns of the entire matrix */
    unsigned int lmt;            /**< number of tile rows of the entire matrix - derived parameter */
    unsigned int lnt;            /**< number of tile columns of the entire matrix - derived parameter */
    unsigned int i;              /**< row index to the beginning of the submatrix */
    unsigned int j;              /**< column indes to the beginning of the submatrix */
    unsigned int m;              /**< number of rows of the submatrix */
    unsigned int n;              /**< number of columns of the submatrix */
    unsigned int mt;             /**< number of tile rows of the submatrix - derived parameter */
    unsigned int nt;             /**< number of tile columns of the submatrix - derived parameter */
    unsigned int nb_local_tiles; /**< number of tile handled locally */
} tiled_matrix_desc_t;

/**
 * Generate the tile (row, col) int the buffer position.
 */

void create_tile_zero(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);
void matrix_ztile(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);
void matrix_ztile_cholesky(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);

void matrix_ctile(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);
void matrix_ctile_cholesky(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);



void matrix_dtile(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);
void matrix_dtile_cholesky(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);

void matrix_stile(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);
void matrix_stile_cholesky(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col, unsigned long long int seed);

/**
 * Generate the full distributed matrix using all nodes/cores available.
 * The generated matrix is symetric positive and diagonal dominant.
 */
void generate_tiled_random_sym_pos_mat(tiled_matrix_desc_t * Mdesc, unsigned long long int seed);
/**
 * Generate the full distributed matrix using all nodes/cores available.
 * The generated matrix if symetric.
 */
void generate_tiled_random_sym_mat( tiled_matrix_desc_t * Mdesc, unsigned long long int seed );
/**
 * Generate the full distributed matrix using all nodes/cores available.
 */
void generate_tiled_random_mat(tiled_matrix_desc_t * Mdesc, unsigned long long int seed);

/**
 * Generate the full distributed matrix using all nodes/cores available. Zeroing the matrix.
 */
void generate_tiled_zero_mat(tiled_matrix_desc_t * Mdesc);


/**
 * Set diagonal value of a double matrix to val.
 */
void pddiagset(tiled_matrix_desc_t * Mdesc, double val);



int data_write(tiled_matrix_desc_t * Ddesc, char * filename);

int data_read(tiled_matrix_desc_t * Ddesc, char * filename);

#ifdef HAVE_MPI
void matrix_zcompare_dist_data(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b);
void matrix_ccompare_dist_data(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b);
void matrix_dcompare_dist_data(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b);
void matrix_scompare_dist_data(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b);
#else
#define matrix_zcompare_dist_data(...) do {} while(0)
#define matrix_ccompare_dist_data(...) do {} while(0)
#define matrix_dcompare_dist_data(...) do {} while(0)
#define matrix_scompare_dist_data(...) do {} while(0)
#endif

struct dague_execution_unit;
typedef int (*dague_operator_t)( struct dague_execution_unit *eu, void* data, void* op_data, ... );

extern struct dague_object_t*
dague_apply_operator_New(tiled_matrix_desc_t* A,
                         dague_operator_t op,
                         void* op_data);
extern void
dague_apply_operator_Destruct( struct dague_object_t* o );
extern struct dague_object_t*
dague_reduce_col_New( tiled_matrix_desc_t* A,
                      tiled_matrix_desc_t* res,
                      dague_operator_t operator,
                      void* op_data );
extern void dague_reduce_col_Destruct( struct dague_object_t *o );
extern struct dague_object_t*
dague_reduce_row_New( tiled_matrix_desc_t* A,
                      tiled_matrix_desc_t* res,
                      dague_operator_t operator,
                      void* op_data );
extern void dague_reduce_row_Destruct( struct dague_object_t *o );

#endif /* _MATRIX_H_  */
