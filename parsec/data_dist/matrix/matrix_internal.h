/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __MATRIX_INTERNAL_H__
#define __MATRIX_INTERNAL_H__

#include "parsec/runtime.h"
#include "parsec/data.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"

BEGIN_C_DECLS

/* Also use in *band* structure */
void parsec_matrix_block_cyclic_key2coords(parsec_data_collection_t *desc,
                                           parsec_data_key_t key, int *m, int *n);


size_t parsec_matrix_sym_block_cyclic_coord2pos(
    parsec_matrix_sym_block_cyclic_t *dc,
    int m,
    int n);


END_C_DECLS

#endif /* __MATRIX_INTERNAL_H__*/
