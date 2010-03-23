/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "remote_dep.h"
#include "scheduling.h"
#include "execution_unit.h"
#include <stdio.h>
#include <string.h>

/* Clear the already forwarded remote dependency matrix */
static inline void remote_dep_reset_forwarded( dplasma_execution_unit_t* eu_context )
{
    /*DEBUG(("fw reset\tcontext %p\n", (void*) eu_context));*/
    memset(eu_context->remote_dep_fw_mask, 0, eu_context->master_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void remote_dep_mark_forwarded( dplasma_execution_unit_t* eu_context, int rank )
{
    int boffset;
    uint32_t mask;
    
    /*DEBUG(("fw mark\tREMOTE rank %d\n", rank));*/
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    eu_context->remote_dep_fw_mask[boffset] |= mask;
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int remote_dep_is_forwarded( dplasma_execution_unit_t* eu_context, int rank )
{
    int boffset;
    uint32_t mask;
    
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    /*DEBUG(("fw test\tREMOTE rank %d (value=%x)\n", rank, (int) (eu_context->remote_dep_fw_mask[boffset] & mask)));*/
    return (int) ((eu_context->remote_dep_fw_mask[boffset] & mask) != 0);
}


/* make sure we don't leave before serving all data deps */
static inline void remote_dep_inc_flying_messages(dplasma_context_t* ctx)
{
    dplasma_atomic_inc_32b( &ctx->taskstodo );
}

/* allow for termination when all deps have been served */
static inline void remote_dep_dec_flying_messages(dplasma_context_t* ctx)
{
    dplasma_atomic_dec_32b( &ctx->taskstodo );
}


#ifdef USE_MPI
#include "remote_dep_mpi.c" 

#else 
#endif /* NO TRANSPORT */


#ifdef DISTRIBUTED
int dplasma_remote_dep_init(dplasma_context_t* context)
{
    int i;
    int np;
    
    np = (int32_t) remote_dep_init(context);
    if(np > 1)
    {
        context->remote_dep_fw_mask_sizeof = ((np + 31) / 32) * sizeof(uint32_t);
        for(i = 0; i < context->nb_cores; i++)
        {
            dplasma_execution_unit_t *eu = context->execution_units[i];
            eu->remote_dep_fw_mask = (uint32_t*) calloc(1, context->remote_dep_fw_mask_sizeof);
        }
    }
    else 
    {
        context->remote_dep_fw_mask_sizeof = 0; /* hoping memset(0b) is fast */
    }
    return np;
}

int dplasma_remote_dep_fini(dplasma_context_t* context)
{
    int i;        
    
    if(context->nb_nodes > 1)
    {
        for(i = 0; i < context->nb_cores; i++)
        {
            free(context->execution_units[i]->remote_dep_fw_mask);
        }
    }
    return remote_dep_fini(context);
}

int dplasma_remote_dep_on(dplasma_context_t* context)
{
    return remote_dep_on(context);
}

int dplasma_remote_dep_off(dplasma_context_t* context)
{
    return remote_dep_off(context);
}

int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context)
{
    return remote_dep_progress(eu_context);
}


int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* exec_context,
                                dplasma_remote_deps_t* remote_deps,
                                uint32_t remote_deps_count )
{
    dplasma_t* function = exec_context->function;
    int i, j, k, count, array_index, bit_index, current_mask;
    
    remote_dep_reset_forwarded(eu_context);
    
    remote_deps->output_count = remote_deps_count;
    remote_deps->msg.deps = (uintptr_t) remote_deps;
    remote_deps->msg.function = (uintptr_t) function;
    for(i = 0; i < function->nb_locals; i++)
    {
        remote_deps->msg.locals[i] = exec_context->locals[i];
    }
    
    for( i = 0; i < remote_deps_count; i++ ) {
        if( function->inout[i] == NULL ) break;  /* we're done ... hopefully */
        if( 0 == remote_deps->output[i].count ) continue;  /* no deps for this output */
        array_index = 0;
        for( j = count = 0; count < remote_deps->output[i].count; j++ ) {
            current_mask = remote_deps->output[i].rank_bits[array_index];
            if( 0 == current_mask ) continue;  /* no bits here */
            for( bit_index = 0; (bit_index < (8 * sizeof(uint32_t))) && (current_mask != 0); bit_index++ ) {
                if( current_mask & (1 << bit_index) ) {
                    int rank = (array_index * sizeof(uint32_t) * 8) + bit_index;
                    assert(rank >= 0);
                    assert(rank < eu_context->master_context->nb_nodes);

                    current_mask ^= (1 << bit_index);
                    count++;

                    gc_data_ref(remote_deps->output[i].data);
                    if(remote_dep_is_forwarded(eu_context, rank))
                    {
                       continue;
                    }
                    remote_dep_mark_forwarded(eu_context, rank);
                    remote_dep_inc_flying_messages(eu_context->master_context); /* TODO: check this counting for multiple deps */
                    remote_dep_send(rank, remote_deps);
                }
            }
            array_index++;
        }
    }
}


dplasma_atomic_lifo_t remote_deps_freelist;
uint32_t max_dep_count, max_nodes_number, elem_size;

int remote_deps_allocation_init(int np, int max_output_deps)
{ /* compute the maximum size of the dependencies array */
    max_dep_count = max_output_deps;
    max_nodes_number = np;
    elem_size = sizeof(dplasma_remote_deps_t) +
                max_dep_count * (sizeof(uint32_t) + sizeof(gc_data_t*) + 
                                 sizeof(uint32_t*) + 
                                 sizeof(uint32_t) * (max_nodes_number + 31)/32);
    dplasma_atomic_lifo_construct(&remote_deps_freelist);
}

#endif /* DISTRIBUTED */



#define HEAVY_DEBUG
#if defined(DPLASMA_DEBUG) && defined(HEAVY_DEBUG)
#define HDEBUG( args ) do { args ; } while(0)
#else
#define HDEBUG( args ) do {} while(0)
#endif 

/* THIS IS ALWAYS NEEDED: DPC is not distributed, hence doesn't define it, but
 * requires it to genrerate correct precompiled code */
int dplasma_remote_dep_get_rank_preds(const expr_t **predicates,
                                      expr_t **rowpred,
                                      expr_t **colpred, 
                                      symbol_t **rowsize,
                                      symbol_t **colsize)
{
    int pred_index;
    symbol_t *rowSymbol, *colSymbol;
    rowSymbol = dplasma_search_global_symbol( "rowRANK" );
    colSymbol = dplasma_search_global_symbol( "colRANK" );
    *rowpred = *colpred = NULL;
    
    if(NULL == rowSymbol) return -1;
    if(NULL == colSymbol) return -2;

    
    /* compute matching colRank and rowRank from predicates */
    for( pred_index = 0;
         (pred_index < MAX_PRED_COUNT) && (NULL != predicates[pred_index]);
         pred_index++ )
    {
        if( EXPR_SUCCESS == expr_depend_on_symbol(predicates[pred_index], rowSymbol) ) {
            assert(EXPR_IS_BINARY(predicates[pred_index]->op));
            assert(*rowpred == NULL);
            
            if( EXPR_SUCCESS == expr_depend_on_symbol(predicates[pred_index]->bop1, rowSymbol) )
            {
                *rowpred = predicates[pred_index]->bop2;
            }
            else
            {
                *rowpred = predicates[pred_index]->bop1;
            }
        } 
        else if( EXPR_SUCCESS == expr_depend_on_symbol(predicates[pred_index], colSymbol) ) 
        {
            assert(EXPR_IS_BINARY(predicates[pred_index]->op));
            assert(*colpred == NULL);
            if( EXPR_SUCCESS == expr_depend_on_symbol(predicates[pred_index]->bop1, colSymbol) )
            {
                *colpred = predicates[pred_index]->bop2;
            }
            else
            {
                *colpred = predicates[pred_index]->bop1;
            }
        } 
        else 
        {
            HDEBUG(         DEBUG(("SKIP\t"));expr_dump(stdout, predicates[pred_index]);DEBUG(("\n")));
        }
    }

    if(NULL == *rowpred) return -1;
    if(NULL == *colpred) return -2;

    *rowsize = dplasma_search_global_symbol( "GRIDrows" );
    *colsize = dplasma_search_global_symbol( "GRIDcols" );
    if(NULL == *rowsize) return -3;
    if(NULL == *colsize) return -4;
    
    return 0;
}

