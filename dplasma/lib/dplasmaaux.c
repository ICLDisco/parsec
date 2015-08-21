/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 */
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */

#include "dplasmaaux.h"

int dplasma_aux_get_priority_limit( char* function, const tiled_matrix_desc_t* ddesc )
{
    char *v;
    char keyword[strlen(function)+2];

    if( NULL == function || NULL == ddesc )
        return 0;

    switch( ddesc->mtype ) {
    case matrix_RealFloat:
        sprintf(keyword, "S%s", function);
        break;
    case matrix_RealDouble:
        sprintf(keyword, "D%s", function);
        break;
    case matrix_ComplexFloat:
        sprintf(keyword, "C%s", function);
        break;
    case matrix_ComplexDouble:
        sprintf(keyword, "Z%s", function);
        break;
    default:
        return 0;
    }

    if( (v = getenv(keyword)) != NULL ) {
        return atoi(v);
    }
    return 0;
}

