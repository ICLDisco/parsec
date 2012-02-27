/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
  
#include "zgebut.h"
#include "zhebut.h"
#include "rbt_mapping.h"

typedef struct seg_dague_ddesc{
 dague_ddesc_t super; /* include/data_distribution.h */
 tiled_matrix_desc_t *A_org; /* data_dist/matrix/matrix.h */
 seg_info_t seg_info;
}dague_seg_ddesc_t;

static void *dague_rbt_rank_of(dague_ddesc_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile, offset;
    va_list ap;
    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segment_to_tile(seg_dague_ddsec_t *seg_ddesc, m_seg, n_seg, &m_tile, &n_tile, &offset);

    /* TODO: if not distributed, return 0 */

    return desc->A_org->rank_of(desc->A_org, m_tile, n_tile);
}


/*
 * Segments can be handled in two ways:
 * Case 1: The MPI datatype starts from the beginning of the tile (of the original ddesc) and
 *         uses an offset to get to the beginning of the data of the segment (and a stride).
 * Case 2: The MPI datatype starts from the beginning of the data of the segment (and uses a
 *         stride so it has mb as lda).
 *
 * In case 1, dague_rbt_data_of() should return a pointer to the beginning of the original tile,
 * i.e., it should return the same thing as data_of() of the original ddesc for the tile that
 * the given segment falls in.
 * In case 2, dague_rbt_data_of() should return a pointer to the beginning of the segment,
 * i.e. add the offset to the return value of data_of() of the original ddesc.
 * The choice between case 1 and case 2 is made in dplasma_datatype_define_subarray(), so
 * these two functions must always correspond.
 */
static void *dague_rbt_data_of(dague_ddesc_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile;
    va_list ap;
    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segment_to_tile(seg_dague_ddsec_t *seg_ddesc, m_seg, n_seg, &m_tile, &n_tile, &offset);

    return desc->A_org->data_of(desc->A_org, m_tile, n_tile);
}


#if defined(HAVE_MPI)
/*
 * Don't change this function without updating dague_rbt_data_of().
 * Look at the comments at dague_rbt_data_of() for details.
 */
int dplasma_datatype_define_subarray( dague_remote_dep_datatype_t oldtype,
                                      unsigned int tile_mb,
                                      unsigned int tile_nb,
                                      unsigned int seg_mb,
                                      unsigned int seg_nb,
                                      unsigned int m_off,
                                      unsigned int n_off,
                                      dague_remote_dep_datatype_t* newtype )
{
    int sizes[2], subsizes[2], starts[2]; 
 
    sizes[0]    = tile_mb;
    sizes[1]    = tile_nb; 
    subsizes[0] = seg_mb;
    subsizes[1] = seg_nb; 
    starts[0]   = m_off;
    starts[1]   = n_off;

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_FORTRAN, oldtype, newtype); 
    MPI_Type_commit(newtype);
#if defined(HAVE_MPI_20)
    do{
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "SEG %s %4u*%4u+%4u [%4ux%4u]", oldtype_name, seg_mb, seg_nb, seg_off, tile_mb, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }while(0); /* just for the scope */
#endif  /* defined(HAVE_MPI_20) */
    return 0;
}

#else /* HAVE MPI */
# error "No, no. Have MPI. Really."
#endif


/* HE for Hermitian */

/*
 * dplasma_zhebut_New()
 */
dague_object_t* 
dplasma_zhebut_New( tiled_matrix_desc_t *A, int i_block, int j_block, int level, int *info)
{
    dague_object_t *dague_zhebut = NULL;
    dague_seg_ddesc_t *seg_descA;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    if( nt%2 ){
        dplasma_error("dplasma_zhebut_New", "illegal number of tiles in matrix");
        return NULL;
    }

    /* copy the ddesc part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(dague_ddesc_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_rbt_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(A->lm, A->nb, level, i_block, j_block);

    dague_zhebut = (dague_object_t *)dague_zhebut_new(*seg_descA, (dague_ddesc_t*)seg_descA, nt, nt);
    
    /* keep this, although it's useless, because we're not sure what happens if we leave it empty */
    dplasma_add2arena_tile(((dague_zhebut_object_t*)dague_zhebut)->arenas[DAGUE_zhebut_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    for(i=0; i<36; i++){
        dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        int m_off, n_off, m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_indo, A->mb, A->nb, i, &m_off, &n_off, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zhebut_object_t*)dague_zhebut)->arenas[DAGUE_zhebut_MIN_ARENA_INDEX + i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, A->mb, A->nb,
                                              seg_mb, seg_nb, m_off, n_off,
                                              &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        }

    }

    return dague_zhebut;
}

void
dplasma_zhebut_Destruct( dague_object_t *o )
{
    dague_zhebut_object_t *obut = (dague_zhebut_object_t *)o;
    
    dplasma_datatype_undefine_type( &(obut->arenas[DAGUE_zhebut_DEFAULT_ARENA]->opaque_dtt) );

    dague_zhebut_destroy(obut);
}

/* GE for General */

/*
 * dplasma_zgebut_New() 
 */
dague_object_t* 
dplasma_zgebut_New( tiled_matrix_desc_t *A, int i_block, int j_block, int level, int *info)
{
    dague_object_t *dague_zgebut = NULL;
    dague_seg_ddesc_t *seg_descA;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    if( nt%2 ){
        dplasma_error("dplasma_zhebut_New", "illegal number of tiles in matrix");
        return NULL;
    }

    /* copy the ddesc part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(dague_ddesc_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_rbt_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(A->lm, A->nb, level, i_block, j_block);

    dague_zgebut = (dague_object_t *)dague_zgebut_new(*seg_descA, (dague_ddesc_t*)seg_descA, nt, nt);
    
    /* keep this, although it's useless, because we're not sure what happens if we leave it empty */
    dplasma_add2arena_tile(((dague_zgebut_object_t*)dague_zgebut)->arenas[DAGUE_zgebut_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    for(i=0; i<36; i++){
        dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        int m_off, n_off, m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_indo, A->mb, A->nb, i, &m_off, &n_off, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zgebut_object_t*)dague_zgebut)->arenas[DAGUE_zgebut_MIN_ARENA_INDEX + i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, A->mb, A->nb,
                                              seg_mb, seg_nb, m_off, n_off,
                                              &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        }

    }

    return dague_zgebut;
}

void
dplasma_zgebut_Destruct( dague_object_t *o )
{
    dague_zgebut_object_t *obut = (dague_zgebut_object_t *)o;
    
    dplasma_datatype_undefine_type( &(obut->arenas[DAGUE_zgebut_DEFAULT_ARENA]->opaque_dtt) );

    dague_zgebut_destroy(obut);
}

/*
 * Blocking Interface
 */

static dague_object_t **iterate_ops(tiled_matrix_desc_t *A, int curlevel,
	      				 	int maxlevel, int i_block, int j_block,
					       	dague_object_t **subop,
					       	dague_context_t *dague, 
						int destroy, int *info)
{
    if(curlevel == maxlevel){
        if( i_block == j_block ){
	    if( destroy ){
	        dplasma_zhebut_Destruct(*subop);
	    }else{
	        *subop = dplasma_zhebut_New(A, i_block, j_block, curlevel, info);
	    }
	}else{
	    if( destroy ){
	        dplasma_zgebut_Destruct(*subop);
	    }else{
	        *subop = dplasma_zgebut_New(A, i_block, j_block, curlevel, info);
	    }
	}
	if( !destroy ){
            dague_enqueue(dague, *subop);
	}
        return subop+1;
    }else{
        if( i_block == j_block ){
            subop = iterate_ops(A, curlevel+1, maxlevel, 2*i_block,   2*j_block,   subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, 2*i_block+1, 2*j_block,   subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, 2*i_block+1, 2*j_block+1, subop, dague, destroy, info);
	}else{
            subop = iterate_ops(A, curlevel+1, maxlevel, 2*i_block,   2*j_block,   subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, 2*i_block+1, 2*j_block,   subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, 2*i_block,   2*j_block+1, subop, dague, destroy, info);
            subop = iterate_ops(A, curlevel+1, maxlevel, 2*i_block+1, 2*j_block+1, subop, dague, destroy, info);
	}
        return subop;
    }

}


int dplasma_zhebut(dague_context_t *dague, tiled_matrix_desc_t *A, int level)
{
    dague_object_t **subop;
    int info = 0;
    int nbhe = 1<<level;
    int nbge = (1<<(level-1))*((1<<level)-1);
    int final_nt = A->nt/nbhe;
    if( final_nt == 0 ){
        fprintf(stderr,"Too many butterflies. Death by starvation.\n");
        return -1;
    }     
    if( A->nt%nbhe != 0 ){
        fprintf(stderr,"Please use a matrix size that is divisible by 2^level.\n");
        return -1;
    }     

    subop = (dague_object_t **)malloc((nbhe+nbge) * sizeof(dague_object_t*));

    
    (void)iterate_ops(A, 0, level, 0, 0, subop, dague, 0, &info);    
    dplasma_progress(dague);
    (void)iterate_ops(A, 0, level, 0, 0, subop, dague, 1, &info);    
    free(subop);
    return info;
}

