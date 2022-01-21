/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWO_DIM_RECTANGLE_CYCLIC_BAND_H__
#error "Deprecated headers should not be included directly!"
#endif // __TWO_DIM_RECTANGLE_CYCLIC_BAND_H__


typedef parsec_matrix_block_cyclic_band_t two_dim_block_cyclic_band_t __parsec_attribute_deprecated__("Use parsec_matrix_block_cyclic_t");

static inline
void two_dim_block_cyclic_band_init(
    parsec_matrix_block_cyclic_band_t *desc,
    int nodes, int myrank, int band_size )
    __parsec_attribute_deprecated__("Use parsec_matrix_block_cyclic_band_init");


static inline
void two_dim_block_cyclic_band_init(
    parsec_matrix_block_cyclic_band_t *desc,
    int nodes, int myrank, int band_size )
{
    parsec_matrix_block_cyclic_band_init(desc, nodes, myrank, band_size);
}
