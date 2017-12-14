/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <core_blas.h>
#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "parsec/private_mempool.h"

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>

#include "dplasma/lib/butterfly_map.h"
#include "dplasma/lib/zhebut.h"
#include "dplasma/lib/zgebut.h"
#include "dplasma/lib/zgebmm.h"
#include <lapacke.h>

#if (PARSEC_zhebut_ARENA_INDEX_MIN != 0) || (PARSEC_zgebut_ARENA_INDEX_MIN != 0) || (PARSEC_zgebmm_ARENA_INDEX_MIN != 0)
#error Current zhebut can work only if not using named types.
#endif

#define CREATE_N_ENQUEUE 0x0
#define DESTRUCT         0x1


static uint32_t parsec_rbt_rank_of(parsec_data_collection_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile;
    uintptr_t offset;
    va_list ap;
    parsec_seg_dc_t *segA;
    parsec_data_collection_t *A;

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (parsec_seg_dc_t *)desc;
    A = (parsec_data_collection_t *)(segA->A_org);

    segment_to_tile(segA, m_seg, n_seg, &m_tile, &n_tile, &offset);

    /* TODO: if not distributed, return 0 */

    return A->rank_of(A, m_tile, n_tile);
}


/*
 * Segments can be handled in two ways:
 * Case 1: The MPI datatype starts from the beginning of the tile (of the original dc) and
 *         uses an offset to get to the beginning of the data of the segment (and a stride).
 * Case 2: The MPI datatype starts from the beginning of the data of the segment (and uses a
 *         stride so it has mb as lda).
 *
 * In case 1, parsec_rbt_data_of() should return a pointer to the beginning of the original tile,
 * i.e., it should return the same thing as data_of() of the original dc for the tile that
 * the given segment falls in.
 * In case 2, parsec_rbt_data_of() should return a pointer to the beginning of the segment,
 * i.e. add the offset to the return value of data_of() of the original dc.
 * The choice between case 1 and case 2 is made in dplasma_datatype_define_subarray(), so
 * these two functions must always correspond.
 *
 * Currently we are using Case 2.
 */
static parsec_data_t *parsec_rbt_data_of(parsec_data_collection_t *desc, ...){
    int m_seg, n_seg, m_tile, n_tile;
    uintptr_t offset, data_start;
    va_list ap;
    parsec_seg_dc_t *segA;
    parsec_data_collection_t *A;

    va_start(ap, desc);
    m_seg = va_arg(ap, int);
    n_seg = va_arg(ap, int);
    va_end(ap);

    segA = (parsec_seg_dc_t *)desc;
    A = &segA->A_org->super;

    segment_to_tile(segA, m_seg, n_seg, &m_tile, &n_tile, &offset);

    data_start = offset*sizeof(parsec_complex64_t) + (uintptr_t)A->data_of(A, m_tile, n_tile);

    /*
    fprintf(stderr, "Dataof (%d, %d) -> (%d, %d): %p + %llu * %u = %p\n",
            m_seg, n_seg, m_tile, n_tile,
            A->data_of(A, m_tile, n_tile), offset, sizeof(parsec_complex64_t), data_start);
    */

    return (void *)data_start;
}

/* HE for Hermitian */

/*
 * dplasma_zhebut_New()
 */
parsec_taskpool_t*
dplasma_zhebut_New( parsec_tiled_matrix_dc_t *A, PLASMA_Complex64_t *U_but_vec, int i_block, int j_block, int level, int *info)
{
    parsec_taskpool_t *parsec_zhebut = NULL;
    parsec_seg_dc_t *seg_descA;
    parsec_memory_pool_t* pool_0;
    PLASMA_Complex64_t *U_before, *U_after;
    int i, mt, nt, N;

    (void)info;

    seg_descA = (parsec_seg_dc_t *)calloc(1, sizeof(parsec_seg_dc_t));

    /* copy the parsec_tiled_matrix_dc_t part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(parsec_tiled_matrix_dc_t));
    /* overwrite the rank_of() and data_of() */
    seg_descA->super.super.rank_of = parsec_rbt_rank_of;
    seg_descA->super.super.data_of = parsec_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = parsec_rbt_calculate_constants(A, level, i_block, j_block);

    N  = A->lm;
    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    pool_0 = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( pool_0, A->mb * A->nb * sizeof(parsec_complex64_t) );

    U_before = &U_but_vec[level*N];
    U_after  = &U_but_vec[level*N];

    parsec_zhebut = (parsec_taskpool_t *)parsec_zhebut_new(seg_descA,U_before, U_after, nt, mt, pool_0);

    for(i=0; i<36; i++){
        parsec_arena_t *arena;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(&seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((parsec_zhebut_taskpool_t*)parsec_zhebut)->arenas[i];
            parsec_matrix_add2arena_rect( arena, parsec_datatype_double_complex_t, m_sz, A->nb, A->mb );
        }
    }

    return parsec_zhebut;
}

void
dplasma_zhebut_Destruct( parsec_taskpool_t *tp )
{
    int i;
    parsec_zhebut_taskpool_t *obut = (parsec_zhebut_taskpool_t *)tp;

    for(i=0; i<36; i++){
        parsec_matrix_del2arena( obut->arenas[i] );
    }

    parsec_taskpool_free(tp);
}

/* GE for General */

/*
 * dplasma_zgebut_New()
 */
parsec_taskpool_t*
dplasma_zgebut_New( parsec_tiled_matrix_dc_t *A, PLASMA_Complex64_t *U_but_vec, int i_block, int j_block, int level, int *info)
{
    parsec_taskpool_t *parsec_zgebut = NULL;
    parsec_seg_dc_t *seg_descA;
    parsec_memory_pool_t *pool_0;
    int i, mt, nt, N;
    PLASMA_Complex64_t *U_before, *U_after;

    (void)info;

    seg_descA = (parsec_seg_dc_t *)calloc(1, sizeof(parsec_seg_dc_t));

    /* copy the parsec_tiled_matrix_dc_t part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(parsec_tiled_matrix_dc_t));
    /* overwrite the rank_of() and data_of() */
    seg_descA->super.super.rank_of = parsec_rbt_rank_of;
    seg_descA->super.super.data_of = parsec_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = parsec_rbt_calculate_constants(A, level, i_block, j_block);

    N  = A->lm;
    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    U_before = &U_but_vec[level*N];
    U_after  = &U_but_vec[level*N];

    pool_0 = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( pool_0, A->mb * A->nb * sizeof(parsec_complex64_t) );

    parsec_zgebut = (parsec_taskpool_t *)parsec_zgebut_new(seg_descA, U_before, U_after, nt, mt, pool_0);

    for(i=0; i<36; i++){
        parsec_arena_t *arena;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(&seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((parsec_zgebut_taskpool_t*)parsec_zgebut)->arenas[i];
            parsec_matrix_add2arena_rect( arena, parsec_datatype_double_complex_t, m_sz, A->nb, A->mb );
        }
    }

    return parsec_zgebut;
}

void
dplasma_zgebut_Destruct( parsec_taskpool_t *tp )
{
    int i;
    parsec_zgebut_taskpool_t *obut = (parsec_zgebut_taskpool_t *)tp;

    for(i=0; i<36; i++){
        parsec_matrix_del2arena( obut->arenas[i] );
    }

    parsec_taskpool_free(tp);
}

/*
 * dplasma_zgebmm_New()
 */
parsec_taskpool_t*
dplasma_zgebmm_New( parsec_tiled_matrix_dc_t *A, PLASMA_Complex64_t *U_but_vec, int i_block, int j_block, int level, int trans, int *info)
{
    parsec_taskpool_t *parsec_zgebmm = NULL;
    parsec_seg_dc_t *seg_descA;
    parsec_memory_pool_t *pool_0;
    int i, mt, nt, N;

    (void)info;

    seg_descA = (parsec_seg_dc_t *)calloc(1, sizeof(parsec_seg_dc_t));

    /* copy the parsec_tiled_matrix_dc_t part of A into seg_descA */
    memcpy(seg_descA, A, sizeof(parsec_tiled_matrix_dc_t));
    /* overwrite the rank_of() and data_of() */
    seg_descA->super.super.rank_of = parsec_rbt_rank_of;
    seg_descA->super.super.data_of = parsec_rbt_data_of;
    /* store a pointer to A itself */
    seg_descA->A_org = A;
    /* store the level */
    seg_descA->level = level;
    /* store the segment info */
    seg_descA->seg_info = parsec_rbt_calculate_constants(A, level, i_block, j_block);

    /*
    printf("Apllying zgebmm() in block %d,%d\n",i_block, j_block);
    */

    N  = A->lm;
    U_but_vec = &U_but_vec[level*N];

    mt = seg_descA->seg_info.tot_seg_cnt_m;
    nt = seg_descA->seg_info.tot_seg_cnt_n;

    pool_0 = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( pool_0, A->mb * A->nb * sizeof(parsec_complex64_t) );

    parsec_zgebmm = (parsec_taskpool_t *)parsec_zgebmm_new(seg_descA, U_but_vec, nt, mt, trans, pool_0);

    for(i=0; i<36; i++){
        parsec_arena_t *arena;
        int type_exists;
        unsigned int m_sz, n_sz;

        type_exists = type_index_to_sizes(&seg_descA->seg_info, i, &m_sz, &n_sz);

        if( type_exists ){
            arena = ((parsec_zgebmm_taskpool_t*)parsec_zgebmm)->arenas[i];
            parsec_matrix_add2arena_rect( arena, parsec_datatype_double_complex_t, m_sz, A->nb, A->mb );
        }
    }

    return parsec_zgebmm;
}

void
dplasma_zgebmm_Destruct( parsec_taskpool_t *tp )
{
    int i;
    parsec_zgebmm_taskpool_t *obmm = (parsec_zgebmm_taskpool_t *)tp;

    for(i=0; i<36; i++){
        parsec_matrix_del2arena( obmm->arenas[i] );
    }

    parsec_taskpool_free(tp);
}



/*
 * Blocking Interface
 */

static parsec_taskpool_t **iterate_ops(parsec_tiled_matrix_dc_t *A, int tmp_level,
                                    int target_level, int i_block, int j_block,
                                    parsec_taskpool_t **subop,
                                    parsec_context_t *parsec,
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
            parsec_enqueue(parsec, *subop);
        }
        return subop+1;
    }else{
        if( i_block == j_block ){
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block,   2*j_block,   subop, parsec, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block,   subop, parsec, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block+1, subop, parsec, U_but_vec, destroy, info);
        }else{
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block,   2*j_block,   subop, parsec, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block,   subop, parsec, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block,   2*j_block+1, subop, parsec, U_but_vec, destroy, info);
            subop = iterate_ops(A, tmp_level+1, target_level, 2*i_block+1, 2*j_block+1, subop, parsec, U_but_vec, destroy, info);
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


int dplasma_zhebut(parsec_context_t *parsec, parsec_tiled_matrix_dc_t *A, PLASMA_Complex64_t **U_but_ptr, int levels)
{
    parsec_taskpool_t **subop;
    PLASMA_Complex64_t *U_but_vec, beta;
    int cur_level, N;
    int info = 0;
    int nbhe = 1<<levels;
    int nbge;
    int final_nt = A->nt/(2*nbhe);
#if defined(DEBUG_BUTTERFLY)
    int i;
#endif
    nbge = (levels>0) ? (1<<(levels-1))*((1<<levels)-1) : 0;
    
    if( final_nt == 0 ){
        fprintf(stderr,"Too many butterflies. Death by starvation.\n");
        return -1;
    }
    if( A->ln%nbhe != 0 ){
        fprintf(stderr,"Please use a matrix size that is divisible by %d\n - Current Matrix size=%d\n - Number of Hermitian Blocks for this level of RBT=%d\n", 2*nbhe, A->ln, nbhe);
        return -1;
    }

    N = A->lm;

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
        nbge = cur_level > 0 ? (1<<(cur_level-1))*((1<<cur_level)-1) : 0;
        final_nt = A->nt/(2*nbhe);
        if( final_nt == 0 ){
            fprintf(stderr,"Too many butterflies. Death by starvation.\n");
            return -1;
        }
        if( A->ln%nbhe != 0 ){
            fprintf(stderr,"Please use a matrix size that is divisible by %d\n - Current Matrix size=%d\n - Number of Hermitian Blocks for this level of RBT=%d\n", 2*nbhe, A->ln, nbhe);
            return -1;
        }

#if defined(DEBUG_BUTTERFLY)
        printf("\n  =====  Applying Butterfly at level %d\n\n", cur_level);
        fflush(stdout);
#endif

        subop = (parsec_taskpool_t **)malloc((nbhe+nbge) * sizeof(parsec_taskpool_t*));
        (void)iterate_ops(A, 0, cur_level, 0, 0, subop, parsec, U_but_vec, CREATE_N_ENQUEUE, &info);
        dplasma_wait_until_completion(parsec);
        (void)iterate_ops(A, 0, cur_level, 0, 0, subop, parsec, NULL, DESTRUCT, &info);
        free(subop);

#if defined(DEBUG_BUTTERFLY)
        printf("\n\n -+-+-+> Matrix after level %d\n\n", cur_level);
        dplasma_zprint(parsec, PlasmaLower, A);
        printf("\n\n");
#endif

        if( info != 0 ){
            fprintf(stderr,"Terminating the application of butterflies at level: %d\n", cur_level);
            return info;
        }
    }

    return info;
}


