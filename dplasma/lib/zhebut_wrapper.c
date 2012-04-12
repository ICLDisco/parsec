/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <math.h>
#include <stdlib.h>
#include "dague.h"
#include <plasma.h>
#include <cblas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

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

/* Global matrix holding the butterflies.  It is a concatanation of L+1 vectors */
PLASMA_Complex64_t *U_but_vec;

static uint32_t dague_rbt_rank_of(dague_ddesc_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile;
    uintptr_t offset;
    va_list ap;
    dague_seg_ddesc_t *segA;
    dague_ddesc_t *A;

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (dague_seg_ddesc_t *)desc;
    A = (dague_ddesc_t *)(segA->A_org);

    segment_to_tile(segA, m_seg, n_seg, &m_tile, &n_tile, &offset);

    /* TODO: if not distributed, return 0 */

    return A->rank_of(A, m_tile, n_tile);
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
 *
 * Currently we are using Case 2.
 */
static void *dague_rbt_data_of(dague_ddesc_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile;
    uintptr_t offset, data_start;
    va_list ap;
    dague_seg_ddesc_t *segA;
    dague_ddesc_t *A;

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (dague_seg_ddesc_t *)desc;
    A = (dague_ddesc_t *)(segA->A_org);

    segment_to_tile(segA, m_seg, n_seg, &m_tile, &n_tile, &offset);

    data_start = offset*sizeof(Dague_Complex64_t) + (uintptr_t)A->data_of(A, m_tile, n_tile);

    /*
    fprintf(stderr, "Dataof (%d, %d) -> (%d, %d): %p + %llu * %u = %p\n",
            m_seg, n_seg, m_tile, n_tile,
            A->data_of(A, m_tile, n_tile), offset, sizeof(Dague_Complex64_t), data_start);
    */

    return (void *)data_start;
}

#if defined(HAVE_MPI)
/*
 * Don't change this function without updating dague_rbt_data_of().
 * Look at the comments at dague_rbt_data_of() for details.
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


/* HE for Hermitian */

/*
 * dplasma_zhebut_New()
 */
dague_object_t*
dplasma_zhebut_New( tiled_matrix_desc_t *A, int i_block, int j_block, int level, int *info)
{
    dague_object_t *dague_zhebut = NULL;
    dague_seg_ddesc_t *seg_descA;
    dague_memory_pool_t* pool_0;
    int i, mt, nt;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    /* copy the ddesc part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(dague_ddesc_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_rbt_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(A, level, i_block, j_block);

    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    /*
    fprintf(stderr,"Inserting zhebut(%d,%d) with mt=%d,nt=%d\n",i_block, j_block, mt, nt);
    */

    pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_0, A->mb * A->nb * sizeof(Dague_Complex64_t) );

    dague_zhebut = (dague_object_t *)dague_zhebut_new(*seg_descA, (dague_ddesc_t*)seg_descA, nt, mt, pool_0);


    for(i=0; i<36; i++){
#if defined(HAVE_MPI)
        dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zhebut_object_t*)dague_zhebut)->arenas[i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, A->mb, A->nb,
                                              m_sz, n_sz, &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        } else
#endif
        {
            /* Oops, yet another arena allocated by the generated code for nothing
             *   We free it for it. */
            free( ((dague_zhebut_object_t*)dague_zhebut)->arenas[DAGUE_zhebut_ARENA_INDEX_MIN + i]);
            ((dague_zhebut_object_t*)dague_zhebut)->arenas[DAGUE_zhebut_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    return dague_zhebut;
}

void
dplasma_zhebut_Destruct( dague_object_t *o )
{
    int i;
    dague_zhebut_object_t *obut = (dague_zhebut_object_t *)o;

    for(i=0; i<36; i++){
        if( NULL != obut->arenas[i] ){
            free( obut->arenas[i] );
            obut->arenas[i] = NULL;
        }
    }

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
    dague_memory_pool_t *pool_0;
    int i, mt, nt;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    /* copy the ddesc part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(dague_ddesc_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_rbt_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(A, level, i_block, j_block);

    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_0, A->mb * A->nb * sizeof(Dague_Complex64_t) );

    dague_zgebut = (dague_object_t *)dague_zgebut_new(*seg_descA, (dague_ddesc_t*)seg_descA, nt, mt, pool_0);

    for(i=0; i<36; i++){
#if defined(HAVE_MPI)
       dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zgebut_object_t*)dague_zgebut)->arenas[i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, A->mb, A->nb,
                                              m_sz, n_sz, &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        } else
#endif
        {
            free(((dague_zgebut_object_t*)dague_zgebut)->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i]);
            ((dague_zgebut_object_t*)dague_zgebut)->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    return dague_zgebut;
}

void
dplasma_zgebut_Destruct( dague_object_t *o )
{
    int i;
    dague_zgebut_object_t *obut = (dague_zgebut_object_t *)o;

    for(i=0; i<36; i++){
        if( NULL != obut->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] ){
            free( obut->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] );
            obut->arenas[DAGUE_zgebut_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    dague_zgebut_destroy(obut);
}

/*
 * dplasma_zgebmm_New()
 */
dague_object_t*
dplasma_zgebmm_New( tiled_matrix_desc_t *A, int i_block, int j_block, int level, int trans, int *info)
{
    dague_object_t *dague_zgebmm = NULL;
    dague_seg_ddesc_t *seg_descA;
    dague_memory_pool_t *pool_0;
    int i, mt, nt;

    (void)info;

    seg_descA = (dague_seg_ddesc_t *)calloc(1, sizeof(dague_seg_ddesc_t));

    /* copy the ddesc part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(dague_ddesc_t));
    /* overwrite the rank_of() and data_of() */
    ((dague_ddesc_t *)seg_descA)->rank_of = dague_rbt_rank_of;
    ((dague_ddesc_t *)seg_descA)->data_of = dague_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = dague_rbt_calculate_constants(A, level, i_block, j_block);

    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( pool_0, A->mb * A->nb * sizeof(Dague_Complex64_t) );

    dague_zgebmm = (dague_object_t *)dague_zgebmm_new(*seg_descA, (dague_ddesc_t*)seg_descA, nt, mt, trans, pool_0);

    for(i=0; i<36; i++){
#if defined(HAVE_MPI)
       dague_arena_t *arena;
        dague_remote_dep_datatype_t newtype;
        MPI_Aint extent = 0;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((dague_zgebmm_object_t*)dague_zgebmm)->arenas[i];
            dplasma_datatype_define_subarray( MPI_DOUBLE_COMPLEX, A->mb, A->nb,
                                              m_sz, n_sz, &newtype );
            dplasma_get_extent(newtype, &extent);
            dague_arena_construct(arena, extent, DAGUE_ARENA_ALIGNMENT_SSE, newtype);
        } else
#endif
        {
            free(((dague_zgebmm_object_t*)dague_zgebmm)->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i]);
            ((dague_zgebmm_object_t*)dague_zgebmm)->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    return dague_zgebmm;
}

void
dplasma_zgebmm_Destruct( dague_object_t *o )
{
    int i;
    dague_zgebmm_object_t *obmm = (dague_zgebmm_object_t *)o;

    for(i=0; i<36; i++){
        if( NULL != obmm->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] ){
            free( obmm->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] );
            obmm->arenas[DAGUE_zgebmm_ARENA_INDEX_MIN + i] = NULL;
        }
    }

    dague_zgebmm_destroy(obmm);
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
        if( (i_block == j_block) ){
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

static void RBT_zrandom(int N, PLASMA_Complex64_t *V)
{
    int i;

    for (i=0; i<N; i++){
        V[i] = (PLASMA_Complex64_t)exp(((random()/(double)RAND_MAX)-0.5)/10.0);
        printf("U_but_vec[%d]: %lf\n",i,V[i]);
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
    if( A->ln%nbhe != 0 ){
        fprintf(stderr,"Please use a matrix size that is divisible by 2^level: ln=%d, nbhe=%d\n", A->ln, nbhe);
        return -1;
    }

    subop = (dague_object_t **)malloc((nbhe+nbge) * sizeof(dague_object_t*));
    U_but_vec = (PLASMA_Complex64_t *)malloc( (level+1)*(A->lm)*sizeof(PLASMA_Complex64_t) );
    srandom(0);
    RBT_zrandom((level+1)*(A->lm), U_but_vec);

    (void)iterate_ops(A, 0, level, 0, 0, subop, dague, CREATE_N_ENQUEUE, &info);
    dplasma_progress(dague);
    (void)iterate_ops(A, 0, level, 0, 0, subop, dague, DESTRUCT, &info);
    free(subop);
    return info;
}


