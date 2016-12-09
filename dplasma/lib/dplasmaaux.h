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


#define PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(DDESC, TYPE, INIT_PARAMS)   \
    {                                                                   \
        TYPE##_init INIT_PARAMS;                                        \
        DDESC.mat = parsec_data_allocate((size_t)DDESC.super.nb_local_tiles * \
                                        (size_t)DDESC.super.bsiz *      \
                                        (size_t)parsec_datadist_getsizeoftype(DDESC.super.mtype)); \
        parsec_ddesc_set_key((parsec_ddesc_t*)&DDESC, #DDESC);            \
    }


#endif

