/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 */

#include "dplasma.h"
#include "parsec/vpmap.h"
#include <math.h>

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#include "dplasmaaux.h"

#if defined(PARSEC_HAVE_MPI)
/*
 * dplasma falls back to MPI_COMM_WORLD by default.
 * This is sub-optimal, as this does not provide an opportunity
 * to insulate dplasma communications from potential others, but
 * this allows to maintain the behavior that dplasma does not
 * need initialization / finalization.
 *
 * The dplasma API provides two functions to provide and free
 * a dplasma-specific communicator if needed (these should be called
 * before any other dplasma call and after any dplasma call, respectively)
 */

static MPI_Comm dplasma_comm = MPI_COMM_WORLD;
void *dplasma_pcomm = &dplasma_comm;

int dplasma_aux_dup_comm(void *_psrc)
{
    MPI_Comm *src = (MPI_Comm*)_psrc;
    return MPI_Comm_dup(*src, &dplasma_comm);
}

int dplasma_aux_free_comm(void)
{
    return MPI_Comm_free(&dplasma_comm);
}
#else
void *dplasma_pcomm = NULL;
int dplasma_aux_dup_comm(void *comm)
{
    return -1;
}

int dplasma_aux_free_comm(void)
{
    return -1;
}
#endif


int
dplasma_aux_get_priority_limit( char* function, const parsec_tiled_matrix_dc_t* dc )
{
    char *v;
    char *keyword;

    if( NULL == function || NULL == dc )
        return 0;

    keyword = alloca( strlen(function)+2 );
    
    switch( dc->mtype ) {
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

int
dplasma_aux_getGEMMLookahead( parsec_tiled_matrix_dc_t *A )
{
    /**
     * Assume that the number of threads per node is constant, and compute the
     * look ahead based on the global information to get the same one on all
     * nodes.
     */
    int nbunits = vpmap_get_nb_total_threads() * A->super.nodes;
    double alpha =  3. * (double)nbunits / ( A->mt * A->nt );

    if ( A->super.nodes == 1 ) {
        /* No look ahaead */
        return dplasma_imax( A->mt, A->nt );
    }
    else {
        /* Look ahaed of at least 2, and that provides 3 tiles per computational units */
        return dplasma_imax( ceil( alpha ), 2 );
    }
}

