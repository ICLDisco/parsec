/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _MATRIX_H_ 
#define _MATRIX_H_ 

#include <stdarg.h>
#include "../data_distribution.h"

// TODO: This type is weird/broken, it needs to be fixed at some point
enum matrix_type {
    matrix_Byte           = sizeof(char),   /**< unsigned char */
    matrix_Integer        = sizeof(int),    /**< signed int */
    matrix_RealFloat      = sizeof(float),  /**< float */
    matrix_RealDouble     = sizeof(double), /**< double */
    matrix_ComplexFloat   = sizeof(float),  /**< complex float */
    matrix_ComplexDouble  = sizeof(double)  /**< complex double */
};

typedef struct tiled_matrix_desc_t {
    dague_ddesc_t super;
    enum matrix_type mtype;  /**< precision of the matrix */
    unsigned int mb;         /**< number of rows in a tile */
    unsigned int nb;         /**< number of columns in a tile */
    unsigned int ib;         /**< number of columns in an inner block */
    unsigned int bsiz;       /**< size in elements including padding of a tile - derived parameter */
    unsigned int lm;         /**< number of rows of the entire matrix */
    unsigned int ln;         /**< number of columns of the entire matrix */
    unsigned int lmt;        /**< number of tile rows of the entire matrix - derived parameter */
    unsigned int lnt;        /**< number of tile columns of the entire matrix - derived parameter */
    unsigned int i;          /**< row index to the beginning of the submatrix */
    unsigned int j;          /**< column indes to the beginning of the submatrix */
    unsigned int m;          /**< number of rows of the submatrix */
    unsigned int n;          /**< number of columns of the submatrix */
    unsigned int mt;         /**< number of tile rows of the submatrix - derived parameter */
    unsigned int nt;         /**< number of tile columns of the submatrix - derived parameter */
} tiled_matrix_desc_t;

/**
 * Generate the tile (row, col) int the buffer position.
 */

void create_tile_zero(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col);
void create_tile_lu_float(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col);
void create_tile_lu_double(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col);
void create_tile_cholesky_float(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col);
void create_tile_cholesky_double(tiled_matrix_desc_t * Ddesc, void * position,  unsigned int row, unsigned int col);

/**
 * Generate the full distributed matrix using all nodes/cores available.
 * The matrix generated is symetric positive and diagonal dominant.
 */
void generate_tiled_random_sym_pos_mat(tiled_matrix_desc_t * Mdesc);

/**
 * Generate the full distributed matrix using all nodes/cores available.
 */
void generate_tiled_random_mat(tiled_matrix_desc_t * Mdesc);

/**
 * Generate the full distributed matrix using all nodes/cores available. Zeroing the matrix.
 */
void generate_tiled_zero_mat(tiled_matrix_desc_t * Mdesc);

/**
 * allocate a buffer to hold the matrix
 */
void* dague_allocate_matrix(size_t matrix_size );

int data_write(tiled_matrix_desc_t * Ddesc, char * filename);

int data_read(tiled_matrix_desc_t * Ddesc, char * filename);

#ifdef USE_MPI
void compare_dist_data(tiled_matrix_desc_t * a, tiled_matrix_desc_t * b);
#endif
#endif /* _MATRIX_H_  */
