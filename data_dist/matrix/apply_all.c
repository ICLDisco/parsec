/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "matrix.h"
#include "dague_prof_grapher.h"
#include <scheduling.h>

#if defined(DAGUE_PROF_TRACE)
int rtt_profiling_array[2] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \
   dague_profile_ddesc_info_t info;                         \
   info.desc = (dague_ddesc_t*)refdesc;                     \
   info.id = refid;                                         \
   dague_profiling_trace(context->eu_profile,               \
                         __dague_object->super.super.profiling_array[(key)],\
                         eid, (void*)&info);                \
  } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif

typedef struct dague_matrix_operator_object {
    dague_object_t       super;
    tiled_matrix_desc_t* A /* data A */;
    volatile uint32_t    next_k;
    dague_operator_t     op;
    void*                op_data;
} dague_matrix_operator_object_t;

typedef struct __dague_matrix_operator_object {
    dague_matrix_operator_object_t super;
} __dague_matrix_operator_object_t;

static const param_t param_of_apply;
static const dague_t dague_matrix_operator;

#define A(k,n)  (((dague_ddesc_t*)__dague_object->super.A)->data_of((dague_ddesc_t*)__dague_object->super.A, (k), (n)))

static inline uint32_t apply_op_hash(const dague_matrix_operator_object_t *o, int k, int n )
{
    return o->A->mt * k + n;
}

static inline int minexpr_of_row_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)__dague_object_parent;
  (void)assignments;
  return __dague_object->super.A->i;
}
static const expr_t minexpr_of_row = {
  .op = EXPR_OP_INLINE,
  .flags = 0x0,
  .inline_func = minexpr_of_row_fct
};
static inline int maxexpr_of_row_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)__dague_object_parent;

  (void)__dague_object;
  (void)assignments;
  return __dague_object->super.A->mt;
}
static const expr_t maxexpr_of_row = {
  .op = EXPR_OP_INLINE,
  .flags = 0x0,
  .inline_func = maxexpr_of_row_fct
};
static const symbol_t symb_row = {
    .min = &minexpr_of_row,
    .max = &maxexpr_of_row,
    .flags = DAGUE_SYMBOL_IS_STANDALONE
};

static inline int minexpr_of_column_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)__dague_object_parent;
    (void)assignments;
    return __dague_object->super.A->j;
}

static const expr_t minexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .flags = 0x0,
    .inline_func = minexpr_of_column_fct
};

static inline int maxexpr_of_column_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)__dague_object_parent;

    (void)__dague_object;
    (void)assignments;
    return __dague_object->super.A->nt;
}
static const expr_t maxexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .flags = 0x0,
    .inline_func = maxexpr_of_column_fct
};
static const symbol_t symb_column = {
    .min = &minexpr_of_column,
    .max = &maxexpr_of_column,
    .flags = DAGUE_SYMBOL_IS_STANDALONE
};

static inline int pred_of_apply_all_as_expr_fct(const dague_object_t *__dague_object_parent,
                                                const assignment_t *assignments)
{
    const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)__dague_object_parent;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_object;
    (void)assignments;
    /* Compute Predicate */
    return 1;
}
static const expr_t pred_of_apply_all_as_expr = {
    .op = EXPR_OP_INLINE,
    .flags = 0x0,
    .inline_func = pred_of_apply_all_as_expr_fct
};

static inline int
expr_of_p1_for_param_of_apply_dep_in_fct(const dague_object_t *__dague_object_parent,
                                         const assignment_t *assignments)
{
    const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)__dague_object_parent;
    int k = assignments[0].value;

    (void)__dague_object;
    (void)assignments;
    return k;
}
static const expr_t expr_of_p1_for_param_of_apply_dep_in = {
    .op = EXPR_OP_INLINE,
    .flags = 0x0,
    .inline_func = expr_of_p1_for_param_of_apply_dep_in_fct
};
static const dep_t param_of_apply_dep_in = {
  .cond = NULL,
  .dague = &dague_matrix_operator,
  .param = &param_of_apply,
  .datatype_index = 0,
  .call_params = {
        &expr_of_p1_for_param_of_apply_dep_in
    }
};

static inline int
expr_of_p1_for_param_of_apply_dep_out_fct(const dague_object_t *__dague_object_parent,
                                          const assignment_t *assignments)
{
    const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)__dague_object_parent;
    int k = assignments[0].value;

    (void)__dague_object;
    (void)assignments;
    return (k + 1);
}
static const expr_t expr_of_p1_for_param_of_apply_dep_out = {
    .op = EXPR_OP_INLINE,
    .flags = 0x0,
    .inline_func = expr_of_p1_for_param_of_apply_dep_out_fct
};
static const dep_t param_of_apply_dep_out = {
    .cond = NULL,
    .dague = &dague_matrix_operator,
    .param = &param_of_apply,
    .datatype_index = 0,
    .call_params = {
        &expr_of_p1_for_param_of_apply_dep_out
    }
};

static const param_t param_of_apply = {
    .name = "I",
    .sym_type = SYM_INOUT,
    .access_type = ACCESS_RW,
    .param_mask = 0x1,
    .dep_in  = { &param_of_apply_dep_in },
    .dep_out = { &param_of_apply_dep_out }
};

static dague_ontask_iterate_t
add_task_to_list(struct dague_execution_unit *eu_context,
                 dague_execution_context_t *newcontext,
                 dague_execution_context_t *oldcontext,
                 int param_index, int outdep_index,
                 int rank_src, int rank_dst,
                 dague_arena_t* arena,
                 void *param)
{
    dague_execution_context_t** pready_list = (dague_execution_context_t**)param;
    dague_execution_context_t* new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( eu_context->context_mempool );
    dague_thread_mempool_t* mpool = new_context->mempool_owner;

    memcpy( new_context, newcontext, sizeof(dague_execution_context_t) );
    new_context->mempool_owner = mpool;

    dague_list_add_single_elem_by_priority( pready_list, new_context );
    (void)arena; (void)oldcontext; (void)param_index; (void)outdep_index; (void)rank_src; (void)rank_dst;
    return DAGUE_ITERATE_STOP;
}

static void iterate_successors(dague_execution_unit_t *eu,
                               dague_execution_context_t *exec_context,
                               dague_ontask_function_t *ontask,
                               void *ontask_arg)
{
    __dague_matrix_operator_object_t *__dague_object = (__dague_matrix_operator_object_t*)exec_context->dague_object;
    int k = exec_context->locals[0].value;
    int n = exec_context->locals[1].value+1;
    dague_execution_context_t nc;

    /* If this is the last n, try to move to the next k */
    for( ; k < (int)__dague_object->super.A->nt; n = 0) {
        for( ; n < (int)__dague_object->super.A->mt; n++ ) {
            int is_local = (__dague_object->super.A->super.myrank ==
                            ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A,
                                                                               k, n));
            if( !is_local ) continue;
            /* Here we go, one ready local task */
            nc.locals[0].value = k;
            nc.locals[1].value = n;
            nc.function = &dague_matrix_operator /*this*/;
            nc.dague_object = exec_context->dague_object;
            nc.priority = 0;
            ontask(eu, &nc, exec_context, 0, 0,
                   __dague_object->super.A->super.myrank,
                   __dague_object->super.A->super.myrank, NULL, ontask_arg);
            return;
        }
        /* Go to the next row ... atomically */
        k = dague_atomic_inc_32b( &__dague_object->super.next_k );
    }
}

static int release_deps(dague_execution_unit_t *eu,
                        dague_execution_context_t *exec_context,
                        int action_mask,
                        dague_remote_deps_t *deps,
                        dague_arena_chunk_t **data)
{
    dague_execution_context_t* ready_list = NULL;

    iterate_successors(eu, exec_context, add_task_to_list, &ready_list);

    if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
        if( NULL != ready_list ) {
            __dague_schedule(eu, ready_list, !(DAGUE_ACTION_NO_PLACEHOLDER & action_mask));
            ready_list = NULL;
        }
    }

    if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS) {
        (void)AUNREF(exec_context->data[0].data);
    }

    assert( NULL == ready_list );
    (void)deps; (void)data;
    return 1;
}

static int hook_of(dague_execution_unit_t *context,
                   dague_execution_context_t *exec_context)
{
    const __dague_matrix_operator_object_t *__dague_object = (const __dague_matrix_operator_object_t*)exec_context->dague_object;
    const dague_matrix_operator_object_t *dague_object = ( const dague_matrix_operator_object_t * )exec_context->dague_object;
    int k = exec_context->locals[0].value;
    int n = exec_context->locals[1].value;
    dague_arena_chunk_t* arena = (dague_arena_chunk_t*) A(k,n);
    void* data = ADATA(arena);

    exec_context->data[0].data = arena;
    exec_context->data[0].data_repo = NULL;

#if !defined(DAGUE_PROF_DRY_BODY)
    TAKE_TIME(context, 2*exec_context->function->function_id,
              apply_op_hash( dague_object, k, n ), dague_object->A,
              ((dague_ddesc_t*)(dague_object->A))->data_key((dague_ddesc_t*)dague_object->A, k, n) );
    __dague_object->super.op( context, data, __dague_object->super.op_data, k, n );
#endif
    (void)context;
    return 0;
}

static int complete_hook(dague_execution_unit_t *context,
                         dague_execution_context_t *exec_context)
{
    const __dague_matrix_operator_object_t *__dague_object = ( const __dague_matrix_operator_object_t * )exec_context->dague_object;
    const dague_matrix_operator_object_t *dague_object = ( const dague_matrix_operator_object_t * )exec_context->dague_object;
    int k = exec_context->locals[0].value;
    int n = exec_context->locals[1].value;
    (void)k; (void)n;

    TAKE_TIME(context, 2*exec_context->function->function_id+1, apply_op_hash( dague_object, k, n ), NULL, 0);
      
    dague_prof_grapher_task(exec_context, context->eu_id, k+n);

    release_deps(context, exec_context,
                 (DAGUE_ACTION_RELEASE_REMOTE_DEPS |
                  DAGUE_ACTION_RELEASE_LOCAL_DEPS |
                  DAGUE_ACTION_RELEASE_LOCAL_REFS |
                  DAGUE_ACTION_DEPS_MASK),
                 NULL, &exec_context->data[0].data);

    return 0;
}

static const dague_t dague_matrix_operator = {
    .name = "apply",
    .deps = 0,
    .flags = 0x0,
    .function_id = 0,
    .dependencies_goal = 0x1,
    .nb_locals = 2,
    .nb_params = 2,
    .params = { &symb_row, &symb_column },
    .locals = { &symb_row, &symb_column },
    .pred = &pred_of_apply_all_as_expr,
    .priority = NULL,
    .in = { &param_of_apply },
    .out = { &param_of_apply },
    .iterate_successors = iterate_successors,
    .release_deps = release_deps,
    .hook = hook_of,
    .complete_execution = complete_hook,
};

static void dague_apply_operator_startup_fn(dague_context_t *context, 
                                            dague_object_t *dague_object,
                                            dague_execution_context_t** startup_list)
{
    __dague_matrix_operator_object_t *__dague_object = (__dague_matrix_operator_object_t*)dague_object;
    dague_execution_context_t fake_context;
    dague_execution_context_t *ready_list;
    int k = 0, n = 0, count = 0;
    dague_execution_unit_t* eu;

    /* If this is the last n, try to move to the next k */
    for( ; k < (int)__dague_object->super.A->nt; n = 0) {
        eu = context->execution_units[count];
        ready_list = NULL;

        for( ; n < (int)__dague_object->super.A->mt; n++ ) {
            int is_local = (__dague_object->super.A->super.myrank ==
                            ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A,
                                                                               k, n));
            if( !is_local ) continue;
            /* Here we go, one ready local task */
            fake_context.locals[0].value = k;
            fake_context.locals[1].value = n;
            fake_context.function = &dague_matrix_operator /*this*/;
            fake_context.dague_object = dague_object;
            fake_context.priority = 0;
            add_task_to_list(eu, &fake_context, NULL, 0, 0,
                             __dague_object->super.A->super.myrank,
                             __dague_object->super.A->super.myrank, NULL, (void*)&ready_list);
            __dague_schedule( eu, ready_list, 0 );
            count++;
            break;
        }
        /* Go to the next row ... atomically */
        k = dague_atomic_inc_32b( &__dague_object->super.next_k );
        if( count == context->nb_cores )
            break;
    }

    *startup_list = NULL;
}

struct dague_object_t*
dague_apply_operator_new(tiled_matrix_desc_t* A,
                         dague_operator_t op,
                         void* op_data)
{
    __dague_matrix_operator_object_t *res = (__dague_matrix_operator_object_t*)calloc(1, sizeof(__dague_matrix_operator_object_t));

    res->super.A = A;
    res->super.op = op;
    res->super.op_data = op_data;

    res->super.super.object_id = 1111;
    res->super.super.nb_local_tasks = A->nb_local_tiles;
    res->super.super.startup_hook = dague_apply_operator_startup_fn;
    return (struct dague_object_t*)res;
}

void dague_apply_operator_destroy( struct dague_object_t* o )
{
    (void)o;
}
