/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include <math.h>
#include <stdlib.h>
#include <cblas.h>

#include "dplasma/lib/butterfly_map.h"
#include "dplasma/lib/zhebut.h"
#include "dplasma/lib/zgebut.h"
#include "dplasma/lib/zgebmm.h"
#include <lapacke.h>

#if (DAGUE_zhebut_ARENA_INDEX_MIN != 0) || (DAGUE_zgebut_ARENA_INDEX_MIN != 0)
#error Current zhebut can work only if not using named types.
#endif

#define CREATE_N_ENQUEUE 0x0
#define DESTRUCT         0x1


/*
 ***************************************************
 * RBT rank_of and data_of 
 ***************************************************
 */
static uint32_t dague_rbt_rank_of(dague_ddesc_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile;
    uintptr_t offset;
    va_list ap;
    dague_seg_ddesc_t *segA;
    dague_ddesc_t *A;
    tiled_matrix_desc_t *orgA_super;

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (dague_seg_ddesc_t *)desc;
    A = (dague_ddesc_t *)(segA->A_org.sym_two_DBC); 

    orgA_super = &(segA->A_org.sym_two_DBC->super);

    segment_to_tile(segA, orgA_super, m_seg, n_seg, &m_tile, &n_tile, &offset);

#if defined(DISTRIBUTED)
    return A->rank_of(A, m_tile, n_tile);
#else
    return 0;
#endif
}


/*
 * The following code assumes that:
 * + data_dist/matrix/sym_two_dim_rectange_cyclic.h:sym_twoDBC_coordinates_to_position() is up to date
 */
static dague_data_t *dague_rbt_data_of(dague_ddesc_t *desc, ...)
{
    int m_seg, n_seg, m_tile, n_tile, position;
    uintptr_t offset, seg_start;
    va_list ap;
    dague_seg_ddesc_t *segA;
    dague_ddesc_t *A;
    sym_two_dim_block_cyclic_t *segA_super;
    tiled_matrix_desc_t *orgA_super;
    

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (dague_seg_ddesc_t *)desc;
    A = (dague_ddesc_t *)(segA->A_org.sym_two_DBC);

    segA_super = &(segA->super.sym_two_DBC);
    orgA_super = &(segA->A_org.sym_two_DBC->super);

    segment_to_tile(segA, orgA_super, m_seg, n_seg, &m_tile, &n_tile, &offset);

    dague_data_t *data = A->data_of(A, m_tile, n_tile);

    seg_start = offset*sizeof(dague_complex64_t) + (uintptr_t)(DAGUE_DATA_COPY_GET_PTR(data->device_copies[0]));

    /*
     * Map the segment and tile into a position in the data_map[] array.
     * DO NOT CHANGE this mapping without also changing the allocation of the data_map[] array in the _New() function.
     */
    int tile_position = sym_twoDBC_coordinates_to_position(segA_super, m_tile, n_tile);
    int seg_off = m_seg%3 + (n_seg%3)*3; /* 3x3 is the maximum decomposition of a tile into segments */
    position = 9*tile_position + seg_off;

    int key = (n_tile*(orgA_super->mb)+m_tile)*9 + seg_off;
    dague_data_t *rslt = dague_matrix_create_data(&(segA_super->super), (void *)seg_start, position, key);
    return rslt;
}

/* 
 ***************************************************
 * BMM rank_of() and data_of()
 ***************************************************
 */
static uint32_t dague_bmm_rank_of(dague_ddesc_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile;
    uintptr_t offset;
    va_list ap;
    dague_seg_ddesc_t *segA;
    dague_ddesc_t *A;
    tiled_matrix_desc_t *orgA_super;

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (dague_seg_ddesc_t *)desc;
    A = (dague_ddesc_t *)(segA->A_org.two_DBC); 

    orgA_super = &(segA->A_org.two_DBC->super);

    segment_to_tile(segA, orgA_super, m_seg, n_seg, &m_tile, &n_tile, &offset);

#if defined(DISTRIBUTED)
    return A->rank_of(A, m_tile, n_tile);
#else
    return 0;
#endif
}

/*
 * The following code assumes that:
 * + data_dist/matrix/two_dim_rectange_cyclic.h:twoDBC_coordinates_to_position() is up to date
 */
static dague_data_t *dague_bmm_data_of(dague_ddesc_t *desc, ...)
{
    int m_seg, n_seg, m_tile, n_tile, position;
    uintptr_t offset, seg_start;
    va_list ap;
    dague_seg_ddesc_t *segA;
    dague_ddesc_t *A;
    two_dim_block_cyclic_t *segA_super;
    tiled_matrix_desc_t *orgA_super;

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (dague_seg_ddesc_t *)desc;
    A = (dague_ddesc_t *)(segA->A_org.two_DBC);

    segA_super = &(segA->super.two_DBC);
    orgA_super = &(segA->A_org.two_DBC->super);

    segment_to_tile(segA, orgA_super, m_seg, n_seg, &m_tile, &n_tile, &offset);

    dague_data_t *data = A->data_of(A, m_tile, n_tile);

    seg_start = offset*sizeof(dague_complex64_t) + (uintptr_t)(DAGUE_DATA_COPY_GET_PTR(data->device_copies[0]));

    /*
     * Map the segment and tile into a position in the data_map[] array.
     * DO NOT CHANGE this mapping without also changing the allocation of the data_map[] array in the _New() function.
     */
    int tile_position = twoDBC_coordinates_to_position(segA_super, m_tile, n_tile);
    int seg_off = m_seg%3 + (n_seg%3)*3; /* 3x3 is the maximum decomposition of a tile into segments */
    position = 9*tile_position + seg_off;

    int key = (n_tile*(orgA_super->mb)+m_tile)*9 + seg_off;
    dague_data_t *rslt = dague_matrix_create_data(&(segA_super->super), (void *)seg_start, position, key);
    return rslt;
}


#if defined(HAVE_MPI)
/*
 * Don't change this function (or rather how the data is packed) without updating
 * the offset calculation in segment_to_tile() that is called in dague_rbt_data_of().
 */
static int dplasma_datatype_define_subarray( dague_remote_dep_datatype_t oldtype,
                                             unsigned int tile_mb,
                                             unsigned int tile_nb,
                                             unsigned int seg_mb,
                                             unsigned int seg_nb,
                                             dague_remote_dep_datatype_t* newtype )
{
    MPI_Type_vector (seg_nb, seg_mb, tile_mb, oldtype, newtype);

    MPI_Type_commit(newtype);
#if defined(HAVE_MPI_20)
    do{
        char newtype_name[MPI_MAX_OBJECT_NAME], oldtype_name[MPI_MAX_OBJECT_NAME];
        int len;

        MPI_Type_get_name(oldtype, oldtype_name, &len);
        snprintf(newtype_name, MPI_MAX_OBJECT_NAME, "SEG %s %3u*%3u [%3ux%3u]", oldtype_name, seg_mb, seg_nb, tile_mb, tile_nb);
        MPI_Type_set_name(*newtype, newtype_name);
    }while(0); /* just for the scope */
#endif  /* defined(HAVE_MPI_20) */

    return 0;
}

#endif


/*
 * dplasma_zhebut_New()
 * The input matrix A needs to be sym_two_dim_block_cyclic_t because of the way we map elements of
 * the A->data_map[] to (m,n) coordinates in the matrix.
 */
dague_handle_t*
dplasma_zhebut_New( sym_two_dim_block_cyclic_t *A, PLASMA_Complex64_t *U_but_vec, int i_block, int j_block, int level, int *info)
{
    dague_handle_t *dague_zhebut = NULL;
    dague_seg_ddesc_t *seg_descA;
    dague_memory_pool_t* pool_0;
    PLASMA_Complex64_t *U_before, *U_after;
    tiled_matrix_desc_t *tiled_A = &(A->super);
    int i, mt, nt, N, lcl_nt, lcl_seg_cnt;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    memcpy(seg_descA, A, sizeof(sym_two_dim_block_cyclic_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_rbt_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org.sym_two_DBC = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(tiled_A, level, i_block, j_block);

    /*
     * Here we could iterate over all dague_data_t entries of A->data_map[] (that correspond to the local tiles)
     * and create corresponding dague_data_t entries in seg_descA->data_map[] for the local segments
     * (and there will probably be multiple segments per tile). However, we will do this in a per case
     * basis inside the data_of() function.
     */
    lcl_nt = tiled_A->nb_local_tiles;
    lcl_seg_cnt = 9*lcl_nt;

    ((tiled_matrix_desc_t *)seg_descA)->nb_local_tiles = lcl_seg_cnt;
    ((tiled_matrix_desc_t *)seg_descA)->data_map = (dague_data_t**)calloc(lcl_seg_cnt, sizeof(dague_data_t*));

    N  = tiled_A->lm;
    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_0, tiled_A->mb * tiled_A->nb * sizeof(dague_complex64_t) );

    U_before = &U_but_vec[level*N];
    U_after  = &U_but_vec[level*N];

    dague_zhebut = (dague_handle_t *)dague_zhebut_new(*seg_descA, (dague_ddesc_t*)seg_descA, U_before, U_after, nt, mt, pool_0);


    for(i=0; i<36; i++){
#if defined(HAVE_MPI)
        dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zhebut_handle_t*)dague_zhebut)->arenas[i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, tiled_A->mb, tiled_A->nb,
                                              m_sz, n_sz, &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        } else
#endif
        {
            /* Oops, yet another arena allocated by the generated code for nothing
             *   We free it for it. */
            free( ((dague_zhebut_handle_t*)dague_zhebut)->arenas[DAGUE_zhebut_ARENA_INDEX_MIN + i]);
            ((dague_zhebut_handle_t*)dague_zhebut)->arenas[DAGUE_zhebut_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    return dague_zhebut;
}

void
dplasma_zhebut_Destruct( dague_handle_t *o )
{
    int i;
    dague_zhebut_handle_t *obut = (dague_zhebut_handle_t *)o;

    for(i=0; i<36; i++){
        if( NULL != obut->arenas[i] ){
            free( obut->arenas[i] );
            obut->arenas[i] = NULL;
        }
    }

    DAGUE_INTERNAL_HANDLE_DESTRUCT(obut);
}

/* GE for General */

/*
 * dplasma_zgebut_New()
 */
dague_handle_t*
dplasma_zgebut_New( sym_two_dim_block_cyclic_t *A, PLASMA_Complex64_t *U_but_vec, int i_block, int j_block, int level, int *info)
{
    dague_handle_t *dague_zgebut = NULL;
    dague_seg_ddesc_t *seg_descA;
    dague_memory_pool_t *pool_0;
    int i, mt, nt, N, lcl_nt, lcl_seg_cnt;
    tiled_matrix_desc_t *tiled_A = &(A->super);
    PLASMA_Complex64_t *U_before, *U_after;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    memcpy(seg_descA, A, sizeof(sym_two_dim_block_cyclic_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_rbt_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org.sym_two_DBC = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(tiled_A, level, i_block, j_block);

    /* nb_local_tiles is in tiled_matrix_desc_t which is the super of sym_two_dim_block_cyclic_t */
    lcl_nt = tiled_A->nb_local_tiles;
    lcl_seg_cnt = 9*lcl_nt;

    ((tiled_matrix_desc_t *)seg_descA)->nb_local_tiles = lcl_seg_cnt;
    ((tiled_matrix_desc_t *)seg_descA)->data_map = (dague_data_t**)calloc(lcl_seg_cnt, sizeof(dague_data_t*));


    N  = tiled_A->lm;
    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    U_before = &U_but_vec[level*N];
    U_after  = &U_but_vec[level*N];

    pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_0, tiled_A->mb * tiled_A->nb * sizeof(dague_complex64_t) );

    dague_zgebut = (dague_handle_t *)dague_zgebut_new(*seg_descA, (dague_ddesc_t*)seg_descA, U_before, U_after, nt, mt, pool_0);

    for(i=0; i<36; i++){
#if defined(HAVE_MPI)
       dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zgebut_handle_t*)dague_zgebut)->arenas[i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, tiled_A->mb, tiled_A->nb,
                                              m_sz, n_sz, &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        } else
#endif
        {
            free(((dague_zgebut_handle_t*)dague_zgebut)->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i]);
            ((dague_zgebut_handle_t*)dague_zgebut)->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    return dague_zgebut;
}

void
dplasma_zgebut_Destruct( dague_handle_t *o )
{
    int i;
    dague_zgebut_handle_t *obut = (dague_zgebut_handle_t *)o;

    for(i=0; i<36; i++){
        if( NULL != obut->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] ){
            free( obut->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] );
            obut->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    DAGUE_INTERNAL_HANDLE_DESTRUCT(obut);
}

/*
 * dplasma_zgebmm_New()
 */
dague_handle_t*
dplasma_zgebmm_New( two_dim_block_cyclic_t *A, PLASMA_Complex64_t *U_but_vec, int i_block, int j_block, int level, int trans, int *info)
{
    dague_handle_t *dague_zgebmm = NULL;
    dague_seg_ddesc_t *seg_descA;
    dague_memory_pool_t *pool_0;
    tiled_matrix_desc_t *tiled_A = &(A->super);
    int i, mt, nt, N, lcl_nt, lcl_seg_cnt;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    memcpy(seg_descA, A, sizeof(two_dim_block_cyclic_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_bmm_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_bmm_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org.two_DBC = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(tiled_A, level, i_block, j_block);

    lcl_nt = tiled_A->nb_local_tiles;
    lcl_seg_cnt = 9*lcl_nt;

    ((tiled_matrix_desc_t *)seg_descA)->nb_local_tiles = lcl_seg_cnt;
    ((tiled_matrix_desc_t *)seg_descA)->data_map = (dague_data_t**)calloc(lcl_seg_cnt, sizeof(dague_data_t*));

    N  = tiled_A->lm;
    U_but_vec = &U_but_vec[level*N];

    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_0, tiled_A->mb * tiled_A->nb * sizeof(dague_complex64_t) );

    dague_zgebmm = (dague_handle_t *)dague_zgebmm_new(*seg_descA, (dague_ddesc_t*)seg_descA, U_but_vec, nt, mt, trans, pool_0);

    for(i=0; i<36; i++){
#if defined(HAVE_MPI)
       dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zgebmm_handle_t*)dague_zgebmm)->arenas[i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, tiled_A->mb, tiled_A->nb,
                                              m_sz, n_sz, &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        } else
#endif
        {
            free(((dague_zgebmm_handle_t*)dague_zgebmm)->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i]);
            ((dague_zgebmm_handle_t*)dague_zgebmm)->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    return dague_zgebmm;
}

void
dplasma_zgebmm_Destruct( dague_handle_t *o )
{
    int i;
    dague_zgebmm_handle_t *obmm = (dague_zgebmm_handle_t *)o;

    for(i=0; i<36; i++){
        if( NULL != obmm->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] ){
            free( obmm->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] );
            obmm->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    DAGUE_INTERNAL_HANDLE_DESTRUCT(obmm);
}



/*
 * Blocking Interface
 */

static dague_handle_t **iterate_ops(sym_two_dim_block_cyclic_t *A, int tmp_level,
                                    int target_level, int i_block, int j_block,
                                    dague_handle_t **subop,
                                    dague_context_t *dague,
                                    PLASMA_Complex64_t *U_but_vec,
                                    int destroy, int *info)
{
    if(tmp_level == target_level){
        if( (i_block == j_block) ){
            if( destroy ){
                dplasma_zhebut_Destruct(*subop);
            }else{
                *subop = dplasma_zhebut_New(A, U_but_vec, i_block, j_block, target_level, info);
            }
        }else{
            if( destroy ){
                dplasma_zgebut_Destruct(*subop);
            }else{
                *subop = dplasma_zgebut_New(A, U_but_vec, i_block, j_block, target_level, info);
            }
        }
        if( !destroy ){
            dague_enqueue(dague, *subop);
        }
        return subop+1;
    }else{
        if( i_block == j_block ){
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block,   2*j_block,   subop, dague, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block,   subop, dague, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block+1, subop, dague, U_but_vec, destroy, info);
        }else{
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block,   2*j_block,   subop, dague, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block,   subop, dague, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block,   2*j_block+1, subop, dague, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block+1, subop, dague, U_but_vec, destroy, info);
        }
        return subop;
    }

}

static void RBT_zrandom(int N, PLASMA_Complex64_t *V)
{
    int i;

    for (i=0; i<N; i++){
        V[i] = (PLASMA_Complex64_t)exp(((random()/(double)RAND_MAX)-0.5)/10.0);
    }

}


int dplasma_zhebut(dague_context_t *dague, sym_two_dim_block_cyclic_t *A, PLASMA_Complex64_t **U_but_ptr, int levels)
{
    dague_handle_t **subop;
    PLASMA_Complex64_t *U_but_vec, beta;
    int cur_level, N;
    int info = 0;
    int nbhe = 1<<levels;
    int nbge = (1<<(levels-1))*((1<<levels)-1);
    tiled_matrix_desc_t *tiled_A = &(A->super);
    int final_nt = tiled_A->nt/(2*nbhe);
#if defined(DEBUG_BUTTERFLY)
    int i;
#endif
    if( final_nt == 0 ){
        fprintf(stderr,"Too many butterflies. Death by starvation.\n");
        return -1;
    }
    if( tiled_A->ln%nbhe != 0 ){
        fprintf(stderr,"Please use a matrix size that is divisible by %d\n - Current Matrix size=%d\n - Number of Hermitian Blocks for this level of RBT=%d\n", 2*nbhe, tiled_A->ln, nbhe);
        return -1;
    }

    N = tiled_A->lm;

    subop = (dague_handle_t **)malloc((nbhe+nbge) * sizeof(dague_handle_t*));
    U_but_vec = (PLASMA_Complex64_t *)malloc( (levels+1)*N*sizeof(PLASMA_Complex64_t) );
    *U_but_ptr = U_but_vec;
    srandom(0);
    RBT_zrandom((levels+1)*N, U_but_vec);

    beta = (PLASMA_Complex64_t)pow(1.0/sqrt(2.0), levels);
    cblas_zscal(levels*N, CBLAS_SADDR(beta), U_but_vec, 1);
#if defined(DEBUG_BUTTERFLY)
    for(i=0; i<levels*N; i++){
        printf("U[%d]: %lf\n",i,creal(U_but_vec[i]));
    }
#endif

    for(cur_level = levels; cur_level >=0; cur_level--){
        nbhe = 1<<cur_level;
        nbge = (1<<(cur_level-1))*((1<<cur_level)-1);
        final_nt = tiled_A->nt/(2*nbhe);
        if( final_nt == 0 ){
            fprintf(stderr,"Too many butterflies. Death by starvation.\n");
            return -1;
        }
        if( tiled_A->ln%nbhe != 0 ){
            fprintf(stderr,"Please use a matrix size that is divisible by %d\n - Current Matrix size=%d\n - Number of Hermitian Blocks for this level of RBT=%d\n", 2*nbhe, tiled_A->ln, nbhe);
            return -1;
        }

#if defined(DEBUG_BUTTERFLY)
        printf("\n  =====  Applying Butterfly at level %d\n\n", cur_level);
        fflush(stdout);
#endif

        subop = (dague_handle_t **)malloc((nbhe+nbge) * sizeof(dague_handle_t*));
        (void)iterate_ops(A, 0, cur_level, 0, 0, subop, dague, U_but_vec, CREATE_N_ENQUEUE, &info);
        dplasma_progress(dague);
        (void)iterate_ops(A, 0, cur_level, 0, 0, subop, dague, NULL, DESTRUCT, &info);
        free(subop);

#if defined(DEBUG_BUTTERFLY)
        printf("\n\n -+-+-+> Matrix after level %d\n\n", cur_level);
        dplasma_zprint(dague, PlasmaLower, A);
        printf("\n\n");
#endif

        if( info != 0 ){
            fprintf(stderr,"Terminating the application of butterflies at level: %d\n", cur_level);
            return info;
        }
    }

    return info;
}


