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

int dplasma_aux_get_priority_limit( char* function, const tiled_matrix_desc_t* ddesc );

int dplasma_aux_getGEMMLookahead( tiled_matrix_desc_t *A );

#endif

