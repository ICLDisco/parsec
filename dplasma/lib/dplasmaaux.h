/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 */

#ifndef _DPLASMAAUX_H_INCLUDED
#define _DPLASMAAUX_H_INCLUDED

int dplasma_aux_get_priority_limit( char* function, const parsec_tiled_matrix_dc_t* dc );

int dplasma_aux_getGEMMLookahead( parsec_tiled_matrix_dc_t *A );

#endif

