/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __SUBTILE_H__
#define __SUBTILE_H__

#include "parsec/data_dist/matrix/matrix.h"

BEGIN_C_DECLS

/**
 * Exploit the tiled_matrix_desc to recursively split a single tile part of a
 * parsec_tiled_matrix_dc_t
 */
typedef struct subtile_desc_s {
    parsec_tiled_matrix_dc_t super;
    void *mat;      /**< pointer to the beginning of the matrix */
    int vpid;
} subtile_desc_t;

/**
 * Initialize a descriptor to apply a recursive call on a single tile of a more
 * general tile descriptor.
 *
 * @param[in] tdesc
 *        tiled_matrix_descriptor which owns the tile that will be split into
 *        smaller tiles.
 *
 * @param[in] mt
 *        Row coordinate of the tile to split int the larger matrix.
 *
 * @param[in] nt
 *        Column coordinate of the tile to split int the larger matrix.
 *
 * @param[in] mb
 *        Number of rows in each subtiles
 *
 * @param[in] nb
 *        Number of columns in each subtiles
 *
 * @param[in] i
 *        Row index of the first element of the submatrix. 0 beeing the first
 *        row of the original tile.
 *
 * @param[in] j
 *        Column index of the first element of the submatrix. 0 beeing the first
 *        row of the original tile.
 *
 * @param[in] m
 *        Number of rows in the submatrix.
 *
 * @param[in] n
 *        Number of columns in the submatrix.
 *
 * @return
 *       Descriptor of the tile (mt, nt) of tdesc split in tiles of size mb by
 *       nb.
 *
 */
subtile_desc_t *subtile_desc_create( const parsec_tiled_matrix_dc_t *tdesc,
                                     int mt, int nt,
                                     int mb, int nb,
                                     int i,  int j,
                                     int m,  int n );

END_C_DECLS

#endif /* __TWO_DIM_RECTANGLE_CYCLIC_H__*/
