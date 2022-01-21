/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef __SYM_TWO_DIM_RECTANGLE_CYCLIC_BAND_H__
#error "Deprecated header must not be included directly!"
#endif // __SYM_TWO_DIM_RECTANGLE_CYCLIC_BAND_H__

typedef parsec_matrix_sym_block_cyclic_band_t sym_two_dim_block_cyclic_band_t __parsec_attribute_deprecated__("Use parsec_matrix_sym_block_cyclic_band_t");

static inline
void sym_two_dim_block_cyclic_band_init( parsec_matrix_sym_block_cyclic_band_t *desc,
                                     int nodes, int myrank, int band_size )
    __parsec_attribute_deprecated__("Use parsec_matrix_sym_block_cyclic_band_init");

void sym_two_dim_block_cyclic_band_init( parsec_matrix_sym_block_cyclic_band_t *desc,
                                     int nodes, int myrank, int band_size )
{
    parsec_matrix_sym_block_cyclic_band_init(desc, nodes, myrank, band_size);
}

