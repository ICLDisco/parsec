/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "remote_dep.h"
#include "scheduling.h"
#include <stdio.h>

enum {
    REMOTE_DEP_ACTIVATE_TAG,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_PUT_DATA_TAG,
} dplasma_remote_dep_tag_t;

static int __remote_dep_init(dplasma_context_t* context);
static int __remote_dep_fini(dplasma_context_t* context);


#ifdef USE_MPI

#include "remote_dep_mpi.c" 

#else 
/* This is just failsafe, to compile when no transport is selected.
 *  In this mode, remote dependencies don't work, 
 *  use it only for single node shared memory multicore
 */

static int __remote_dep_init(dplasma_context_t* context)
{
    return 1;
}

static int __remote_dep_fini(dplasma_context_t* context)
{
    return 1;
}

int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                const dplasma_execution_context_t* exec_context,
                                const param_t* dest_param )
{
    int rank;
    
    rank = dplasma_remote_dep_compute_grid_rank(eu_context, origin, exec_context);
    return dplasma_remote_dep_activate_rank(eu_context, origin, origin_param, 
                                            exec_context, dest_param, rank);
}

int dplasma_remote_dep_activate_rank(dplasma_execution_unit_t* eu_context, 
                                     const dplasma_execution_context_t* origin, 
                                     const param_t* origin_param,
                                     const dplasma_execution_context_t* exec_context, 
                                     const param_t* dest_param, 
                                     int rank)
{
    /* return some error and be loud
     * we should never get called in multicore mode */
    int i;
    char tmp[128];
    char tmp2[128];
    dplasma_t* function = exec_context->function;
    
    fprintf(stderr, "/!\\ REMOTE DEPENDENCY DETECTED: %s activates %s and predicates states it should be executed on rank %d.\n    Remote dependencies are NOT ENABLED in this build!\n",
            dplasma_service_to_string(origin, tmp, 128),
            dplasma_service_to_string(exec_context, tmp2, 128),
            rank);
    for(i = 0; i < function->nb_locals; i++)
    {
        symbol_dump(function->locals[i], "\tPREDICATE VARS:\t");
    }
    symbol_dump_all("\tGLOBAL SYMBOLS:\t");
    return -1;
}

#endif


#ifdef DISTRIBUTED
/* Note for Pierre: this is not MPI specific and should not go to 
 * remote_dep_mpi.c. I fixed the warnings and legitimate concerns about dirty 
 * tricks with nb_nodes another way */
int dplasma_remote_dep_init(dplasma_context_t* context)
{
    int i;
    
    context->nb_nodes = (int32_t) __remote_dep_init(context);
    if(context->nb_nodes > 1)
    {
        context->remote_dep_fw_mask_sizeof = (context->nb_nodes + sizeof(char) - 1) / sizeof(char);
        for(i = 0; i < context->nb_cores; i++)
        {
            dplasma_execution_unit_t *eu = &context->execution_units[i];
            eu->remote_dep_fw_mask = (char *) malloc(context->remote_dep_fw_mask_sizeof);
            dplasma_remote_dep_reset_forwarded(eu);
        }
    }
    else 
    {
        context->remote_dep_fw_mask_sizeof = 0;
    }
    return context->nb_nodes;
}

int dplasma_remote_dep_fini(dplasma_context_t* context)
{
    int i;        
    
    if(context->nb_nodes > 1)
    {
        for(i = 0; i < context->nb_cores; i++)
        {
            free(context->execution_units[i].remote_dep_fw_mask);
        }
    }
    return __remote_dep_fini(context);
}
#endif

#define HEAVY_DEBUG
#if defined(_DEBUG) && defined(HEAVY_DEBUG)
#define HDEBUG( args ) do { args ; } while(0)
#else
#define HDEBUG( args ) do {} while(0)
#endif 

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

int dplasma_remote_dep_compute_grid_rank(dplasma_execution_unit_t* eu_context,
                                         const dplasma_execution_context_t* origin,
                                         const dplasma_execution_context_t* exec_context)
{
    int i, pred_index;
    int rank;
    int ranks[2] = { -1, -1 };
    const expr_t **predicates = (const expr_t**) exec_context->function->preds;
    expr_t *expr;
    symbol_t *symbols[2];
    symbols[0] = dplasma_search_global_symbol( "colRANK" );
    symbols[1] = dplasma_search_global_symbol( "rowRANK" );
    int gridcols;
    
    assert(NULL != symbols[0]);
    assert(NULL != symbols[1]);

    /* compute matching colRank and rowRank from predicates */
    for( pred_index = 0;
        (pred_index < MAX_PRED_COUNT) && (NULL != predicates[pred_index]);
        pred_index++ ) 
    {
        for( i = 0; i < 2; i++ ) 
        {            
            if( EXPR_SUCCESS != expr_depend_on_symbol(predicates[pred_index], symbols[i]) )
            {
HDEBUG(         DEBUG(("SKIP\t"));expr_dump(stdout, predicates[pred_index]);DEBUG(("\n")));
                continue;
            }
            assert(EXPR_IS_BINARY(predicates[pred_index]->op));
            
            if( EXPR_SUCCESS == expr_depend_on_symbol(predicates[pred_index]->bop1, symbols[i]) )
            {
                expr = predicates[pred_index]->bop2;
            }
            else
            {                    
                expr = predicates[pred_index]->bop1;
            }
            
            assert(ranks[i] == -1);
HDEBUG(     DEBUG(("expr[%d]:\t", i));expr_dump(stdout, expr);DEBUG(("\n")));
            if( EXPR_SUCCESS != expr_eval(expr,
                                          exec_context->locals, MAX_LOCAL_COUNT,
                                          &ranks[i]) ) 
            {
                DEBUG(("EVAL FAILED FOR EXPR\t"));
                expr_dump(stdout, expr);
                DEBUG(("\n"));
                return -1;
            }
        }
    }
    assert((ranks[0] != -1) && (ranks[1] != -1));
    
    expr = (expr_t *) dplasma_search_global_symbol("GRIDcols");
    assert(NULL != expr);
    expr = (expr_t *) ((symbol_t *) expr)->min;
    if (EXPR_SUCCESS != expr_eval(expr, NULL, 0, &gridcols) )
    {
        DEBUG(("EVAL FAILED FOR EXPR\t"));
        expr_dump(stdout, expr);
        DEBUG(("\n"));
    }
    
    rank = ranks[0] + ranks[1] * gridcols;
    
    DEBUG(("gridrank = %d ( %d + %d x %d )\n", rank, ranks[0], ranks[1], gridcols));
    
    return rank;
}
