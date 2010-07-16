/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "remote_dep.h"
#include "scheduling.h"
#include "execution_unit.h"
#include <stdio.h>
#include <string.h>

/* Clear the already forwarded remote dependency matrix */
static inline void remote_dep_reset_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps )
{
    /*DEBUG(("fw reset\tcontext %p\n", (void*) eu_context));*/
    memset(rdeps->remote_dep_fw_mask, 0, eu_context->master_context->remote_dep_fw_mask_sizeof);
}

/* Mark a rank as already forwarded the termination of the current task */
static inline void remote_dep_mark_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps, int rank )
{
    int boffset;
    uint32_t mask;
    
    /*DEBUG(("fw mark\tREMOTE rank %d\n", rank));*/
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    rdeps->remote_dep_fw_mask[boffset] |= mask;
}

/* Check if rank has already been forwarded the termination of the current task */
static inline int remote_dep_is_forwarded( dague_execution_unit_t* eu_context, dague_remote_deps_t* rdeps, int rank )
{
    int boffset;
    uint32_t mask;
    
    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    assert(boffset <= eu_context->master_context->remote_dep_fw_mask_sizeof);
    /*DEBUG(("fw test\tREMOTE rank %d (value=%x)\n", rank, (int) (eu_context->remote_dep_fw_mask[boffset] & mask)));*/
    return (int) ((rdeps->remote_dep_fw_mask[boffset] & mask) != 0);
}


/* make sure we don't leave before serving all data deps */
static inline void remote_dep_inc_flying_messages(dague_context_t* ctx)
{
    dague_atomic_inc_32b( &ctx->taskstodo );
}

/* allow for termination when all deps have been served */
static inline void remote_dep_dec_flying_messages(dague_context_t* ctx)
{
    dague_atomic_dec_32b( &ctx->taskstodo );
}

#define DAGUE_COLLECTIVE_TYPE_CHAINPIPELINE
#undef  DAGUE_COLLECTIVE_TYPE_BINOMIAL

#ifdef USE_MPI
#include "remote_dep_mpi.c" 

#else 
#endif /* NO TRANSPORT */


#ifdef DISTRIBUTED
int dague_remote_dep_init(dague_context_t* context)
{
    int i;
    int np;
    
    np = (int32_t) remote_dep_init(context);
    if(np > 1)
    {
        context->remote_dep_fw_mask_sizeof = ((np + 31) / 32) * sizeof(uint32_t);
/*        for(i = 0; i < context->nb_cores; i++)
        {
            dague_execution_unit_t *eu = context->execution_units[i];
            eu->remote_dep_fw_mask = (uint32_t*) calloc(1, context->remote_dep_fw_mask_sizeof);
        }*/
    }
    else 
    {
        context->remote_dep_fw_mask_sizeof = 0; /* hoping memset(0b) is fast */
    }
    return np;
}

int dague_remote_dep_fini(dague_context_t* context)
{
    int i;        
    
/*    if(context->nb_nodes > 1)
    {
        for(i = 0; i < context->nb_cores; i++)
        {
            free(context->execution_units[i]->remote_dep_fw_mask);
        }
    }*/
    return remote_dep_fini(context);
}

int dague_remote_dep_on(dague_context_t* context)
{
    return remote_dep_on(context);
}

int dague_remote_dep_off(dague_context_t* context)
{
    return remote_dep_off(context);
}

int dague_remote_dep_progress(dague_execution_unit_t* eu_context)
{
    return remote_dep_progress(eu_context);
}

#ifdef DAGUE_COLLECTIVE
static inline int remote_dep_bcast_chainpipeline_child(int me, int him)
{
    assert(him >= 0);
    if(me == -1) return 0;
    if(him == me+1) return 1;
    return 0;
}

static inline int remote_dep_bcast_binonial_child(int me, int him)
{
    int k, mask;
    
    /* flush out the easy cases first */
    assert(him >= 0);
    if(him == 0) return 0; /* root is child to nobody */
    if(me == -1) return 0; /* I don't even know who I am yet... */
    
    /* look for the leftmost 1 bit */
    for(k = sizeof(int) * 8 - 1; k >= 0; k--)
    {
        mask = 0x1<<k;
        if(him & mask)
        {
            him ^= mask;
            break;
        }
    }
    /* is the remainder suffix "me" ? */
    return him == me;
}
# ifdef DAGUE_COLLECTIVE_TYPE_CHAINPIPELINE
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_chainpipeline_child(me, him)
# elif defined(DPLASMA_COLLECTIVE_TYPE_BINOMIAL)
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_binonial_child(me, him)
# else
#  error "INVALID COLLECTIVE TYPE. YOU MUST DEFINE ONE COLLECTIVE TYPE WHEN ENABLING COLLECTIVES"
# endif
#else
static inline int remote_dep_bcast_star_child(int me, int him)
{
    if(me == 0) return 1;
    else return 0;
}
#  define remote_dep_bcast_child(me, him) remote_dep_bcast_star_child(me, him)
#endif

int dague_remote_dep_activate(dague_execution_unit_t* eu_context,
                                const dague_execution_context_t* exec_context,
                                dague_remote_deps_t* remote_deps,
                                uint32_t remote_deps_count )
{
    const dague_t* function = exec_context->function;
    int i, me, him, count, array_index, bit_index, current_mask;
    
#if defined(DAGUE_DEBUG)
    char tmp[128];
    
    /* make valgrind happy */
    memset(&remote_deps->msg, 0, sizeof(remote_dep_wire_activate_t));
#endif

    remote_dep_reset_forwarded(eu_context, remote_deps);
    
    remote_deps->output_count = remote_deps_count;
    remote_deps->msg.deps = (uintptr_t) remote_deps;
    remote_deps->msg.function = (uintptr_t) function;
    for(i = 0; i < function->nb_locals; i++)
    {
        remote_deps->msg.locals[i] = exec_context->locals[i];
    }
    remote_dep_get_datatypes(remote_deps);
    
    for( i = 0; remote_deps_count; i++) {
        if( 0 == remote_deps->output[i].count ) continue;
        
        if(remote_deps->root == eu_context->master_context->my_rank) me = 0;
        else me = -1; 
        him = 0;

        for( array_index = count = 0; count < remote_deps->output[i].count; array_index++ ) {
            current_mask = remote_deps->output[i].rank_bits[array_index];
            if( 0 == current_mask ) continue;  /* no bits here */
            for( bit_index = 0; (bit_index < (8 * sizeof(uint32_t))) && (current_mask != 0); bit_index++ ) {
                if( current_mask & (1 << bit_index) ) {
                    int rank = (array_index * sizeof(uint32_t) * 8) + bit_index;
                    assert(rank >= 0);
                    assert(rank < eu_context->master_context->nb_nodes);

                    current_mask ^= (1 << bit_index);
                    count++;
                    remote_deps_count--;

                    DEBUG((" TOPO\t%s\troot=%d\t%d (d%d) -? %d (dna)\n", dague_service_to_string(exec_context, tmp, 128), remote_deps->root, eu_context->master_context->my_rank, me, rank));
                    
                    /* root already knows but falsely appear in this bitfield */
                    if(rank == remote_deps->root) continue;

                    if((me == -1) && (rank > eu_context->master_context->my_rank))
                    {
                        /* the next bit points after me, so I know my dense rank now */
                        me = ++him;
                    }
                    him++;
                    
                    if(remote_dep_bcast_child(me, him))
                    {
                        DEBUG((" TOPO\t%s\troot=%d\t%d (d%d) -> %d (d%d)\n", dague_service_to_string(exec_context, tmp, 128), remote_deps->root, eu_context->master_context->my_rank, me, rank, him));
                        
                        gc_data_ref(remote_deps->output[i].data);
                        if(remote_dep_is_forwarded(eu_context, remote_deps, rank))
                        {
                            continue;
                        }
                        remote_dep_inc_flying_messages(eu_context->master_context);
                        remote_dep_mark_forwarded(eu_context, remote_deps, rank);
                        remote_dep_send(rank, remote_deps);
                    }
#ifdef DAGUE_DEBUG
                    else {
                        DEBUG((" TOPO\t%s\troot=%d\t%d (d%d) ][ %d (d%d)\n", dague_service_to_string(exec_context, tmp, 128), remote_deps->root, eu_context->master_context->my_rank, me, rank, him));
                    }
#endif
                }
            }
        }
    }
    return 0;
}


dague_atomic_lifo_t remote_deps_freelist;
uint32_t max_dep_count, max_nodes_number, elem_size;

int remote_deps_allocation_init(int np, int max_output_deps)
{ /* compute the maximum size of the dependencies array */
    max_dep_count = max_output_deps;
    max_nodes_number = np;
    elem_size = sizeof(dague_remote_deps_t) +
                max_dep_count * (sizeof(uint32_t) + sizeof(gc_data_t*) + 
                                 sizeof(uint32_t*) + sizeof(dague_remote_dep_datatype_t*) +
                                 sizeof(uint32_t) * (max_nodes_number + 31)/32) +
                sizeof(uint32_t) * (max_nodes_number + 31)/32;
    dague_atomic_lifo_construct(&remote_deps_freelist);
    return 0;
}

#endif /* DISTRIBUTED */



#define HEAVY_DEBUG
#if defined(DAGUE_DEBUG) && defined(HEAVY_DEBUG)
#define HDEBUG( args ) do { args ; } while(0)
#else
#define HDEBUG( args ) do {} while(0)
#endif 


/* THIS IS ALWAYS NEEDED: DPC is not distributed, hence doesn't define it, but
 * requires it to genrerate correct precompiled code */
int dague_remote_dep_get_rank_preds(const dague_object_t *dague_object,
                                    const expr_t **predicates,
                                    const expr_t **rowpred,
                                    const expr_t **colpred, 
                                    const symbol_t **rowsize,
                                    const symbol_t **colsize)
{
    int pred_index;
    symbol_t *rowSymbol, *colSymbol;
    rowSymbol = dague_search_global_symbol( "rowRANK" );
    colSymbol = dague_search_global_symbol( "colRANK" );
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
            HDEBUG(         DEBUG(("SKIP\t")); expr_dump(stdout, dague_object, predicates[pred_index]); DEBUG(("\n")));
        }
    }

    if(NULL == *rowpred) return -1;
    if(NULL == *colpred) return -2;

    *rowsize = dague_search_global_symbol( "GRIDrows" );
    *colsize = dague_search_global_symbol( "GRIDcols" );
    if(NULL == *rowsize) return -3;
    if(NULL == *colsize) return -4;
    
    return 0;
}

