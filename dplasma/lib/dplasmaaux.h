#ifndef _DPLASMAAUX_H_INCLUDED
#define _DPLASMAAUX_H_INCLUDED

#include "data_dist/matrix/matrix.h"

int dplasma_aux_get_priority( char* function, const tiled_matrix_desc_t* ddesc );   


#define PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(DDESC, TYPE, INIT_PARAMS)   \
    {                                                                   \
        TYPE##_init INIT_PARAMS;                                        \
        DDESC.mat = dague_data_allocate((size_t)DDESC.super.nb_local_tiles * \
                                        (size_t)DDESC.super.bsiz *      \
                                        (size_t)dague_datadist_getsizeoftype(DDESC.super.mtype)); \
        dague_ddesc_set_key((dague_ddesc_t*)&DDESC, #DDESC);            \
    }


#endif

