/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "remote_dep.h"
#include <mpi.h>

enum {
    REMOTE_DEP_ACTIVATE_TAG,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_PUT_DATA_TAG,
} dplasma_remote_dep_tag_t;

static MPI_Comm dep_comm;
static MPI_Request dep_req;
#define dep_dtt MPI_BYTE
static dplasma_execution_context_t dep_buff

int dplasma_dependency_management_init(void)
{
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    MPI_Irecv_init(&dep_buff, 1, dep_dtt, MPI_ANY_SOURCE, DEP_ACTIVATE_TAG, dep_comm, &dep_req);
    MPI_Start(&dep_req, 1);
}

int dplasma_dependency_management_fini(void)
{
    MPI_Request_cancel(&dep_req);
    MPI_Request_free(&dep_req);
    MPI_Comm_free(&dep_comm);
}

    
#ifdef HEAVY_DEBUG
#define HDEBUG( args ) do { args } while(0)
#else
#define HDEBUG( args ) do {} while(0)
#endif 
    
int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                dplasma_execution_context_t* exec_context,
                                const param_t* dest_param )
{
#ifdef _DEBUG
    char tmp[128];
    char tmp2[128];
#endif
    int i, pred_index;
    int mpi_rank;
    int ranks[2] = { -1, -1 };
    const expr_t **predicates = (const expr_t**) exec_context->function->preds;
    expr_t *expr;
    symbol_t *symbols[2];
    symbols[0] = dplasma_search_global_symbol( "colRANK" );
    symbols[1] = dplasma_search_global_symbol( "rowRANK" );
    int gridcols;
    
    assert(NULL != symbols[0]);
    assert(NULL != symbols[1]);
    
HDEBUG( 
    dplasma_t* function = exec_context->function;
           
    symbol_dump_all("ALL SYMBOLS::::");
           
    DEBUG(("REMOTE DEPENDENCY DETECTED %s (var %s=%d violates locality predicate) - from %s\n", dplasma_service_to_string(exec_context, tmp, 128), function->locals[dep]->name, exec_context->locals[dep].value, dplasma_service_to_string(origin, tmp2, 128)));
    for(i = 0; i < function->nb_locals; i++)
    {
        symbol_dump(function->locals[i], "DEP VAR:\t");
    }
);
    
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
    
    mpi_rank = ranks[0] + ranks[1] * gridcols;
    
    DEBUG(("%s -> %s\ttrigger REMOTE process rank %d\n", dplasma_service_to_string(origin, tmp2, 128), dplasma_service_to_string(exec_context, tmp, 128), mpi_rank ));
    
    MPI_Send(origin, sizeof(origin), dep_dtt, mpi_rank, DEP_ACTIVATE_TAG, dep_comm);
    
    return 0;
}

int dplasma_dependency_management_test_activation(DPLASMA_desc *Ddesc, dplasma_execution_context_t *new)
{
    MPI_Status status;
    int flag;
    
    MPI_Test(&dep_req, &flag, &status);
    if(flag)
    {
        
        
    }
    else
    {
        new = NULL; 
        
    }
    
}

