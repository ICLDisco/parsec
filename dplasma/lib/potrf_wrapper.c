/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#ifdef DAGCOMPLEX
#ifdef DAGDOUBLE
#include "zpotrf_rl.h"
#include "zpotrf_ll.h"
#else /* DAGSINGLE */
#include "cpotrf_rl.h"
#include "cpotrf_ll.h"
#endif
#else /* DAGREAL */
#ifdef DAGDOUBLE
#include "dpotrf_rl.h"
#include "dpotrf_ll.h"
#else
#include "spotrf_rl.h"
#include "spotrf_ll.h"
#endif
#endif

dague_object_t* 
DAGUEprefix(potrf_rl_New)(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO)
{
    dagueprefix(potrf_rl_object_t)* object;
    dague_remote_dep_datatype_t default_ddt;
    int pri_change = dplasma_aux_get_priority( "POTRF", ddescA );


    object = dagueprefix(potrf_rl_new)( (dague_ddesc_t*)ddescA, 
                                        ddescA->nb, ddescA->nt, pri_change, uplo, INFO );
    dplasma_aux_create_tile_type(MPITYPE, ddescA->nb, &default_ddt);
    dague_arena_construct(object->arenas[DAGUEprefix(potrf_rl_DEFAULT_ARENA)], ddescA->nb*ddescA->nb*sizeof(TYPENAME), 
                          DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
    return (dague_object_t*)object;
}

dague_object_t* 
DAGUEprefix(potrf_ll_New)(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO)
{
    dagueprefix(potrf_ll_object_t)* object;
    dague_remote_dep_datatype_t default_ddt;
    int pri_change = dplasma_aux_get_priority( "POTRF", ddescA );

    object = dagueprefix(potrf_ll_new)( (dague_ddesc_t*)ddescA, 
                                        ddescA->nb, ddescA->nt, pri_change, uplo, INFO );
    dplasma_aux_create_tile_type(MPITYPE, ddescA->nb, &default_ddt);
    dague_arena_construct(object->arenas[DAGUEprefix(potrf_ll_DEFAULT_ARENA)], ddescA->nb*ddescA->nb*sizeof(TYPENAME), 
                          DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
    return (dague_object_t*)object;
}

dague_object_t* 
DAGUEprefix(potrf_New)(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO)
{
    dagueprefix(potrf_rl_object_t)* object;
    dague_remote_dep_datatype_t default_ddt;
    int pri_change = dplasma_aux_get_priority( "POTRF", ddescA );


    object = dagueprefix(potrf_rl_new)( (dague_ddesc_t*)ddescA, 
                                        ddescA->nb, ddescA->nt, pri_change, uplo, INFO );
    dplasma_aux_create_tile_type(MPITYPE, ddescA->nb, &default_ddt);
    dague_arena_construct(object->arenas[DAGUEprefix(potrf_rl_DEFAULT_ARENA)], ddescA->nb*ddescA->nb*sizeof(TYPENAME), 
                          DAGUE_ARENA_ALIGNMENT_SSE, &default_ddt);
    return (dague_object_t*)object;
}
