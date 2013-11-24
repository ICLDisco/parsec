/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "debug.h"
#include "remote_dep.h"
#include "matrix.h"
#include "dague_prof_grapher.h"
#include <scheduling.h>
#include <datarepo.h>

#if defined(DAGUE_PROF_TRACE)
int dague_map_operator_profiling_array[2] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \
   dague_profile_ddesc_info_t info;                         \
   info.desc = (dague_ddesc_t*)refdesc;                     \
   info.id = refid;                                         \
   DAGUE_PROFILING_TRACE(context->eu_profile,               \
                         __dague_object->super.super.profiling_array[(key)],\
                         eid, __dague_object->super.super.object_id, (void*)&info);  \
  } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif

typedef struct dague_map_operator_object {
    dague_object_t             super;
    const tiled_matrix_desc_t* src;
          tiled_matrix_desc_t* dest;
    volatile uint32_t          next_k;
    dague_operator_t           op;
    void*                      op_data;
} dague_map_operator_object_t;

typedef struct __dague_map_operator_object {
    dague_map_operator_object_t super;
} __dague_map_operator_object_t;

static const dague_flow_t flow_of_map_operator;
static const dague_function_t dague_map_operator;

#define src(k,n)  (((dague_ddesc_t*)__dague_object->super.src)->data_of((dague_ddesc_t*)__dague_object->super.src, (k), (n)))
#define dest(k,n)  (((dague_ddesc_t*)__dague_object->super.dest)->data_of((dague_ddesc_t*)__dague_object->super.dest, (k), (n)))

#if defined(DAGUE_PROF_TRACE)
static inline uint32_t map_operator_op_hash(const __dague_map_operator_object_t *o, int k, int n )
{
    return o->super.src->mt * k + n;
}
#endif  /* defined(DAGUE_PROF_TRACE) */

static inline int minexpr_of_row_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;
    (void)assignments;
    return __dague_object->super.src->i;
}
static const expr_t minexpr_of_row = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = minexpr_of_row_fct }
};
static inline int maxexpr_of_row_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;

    (void)__dague_object;
    (void)assignments;
    return __dague_object->super.src->mt;
}
static const expr_t maxexpr_of_row = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = maxexpr_of_row_fct }
};
static const symbol_t symb_row = {
    .min = &minexpr_of_row,
    .max = &maxexpr_of_row,
    .flags = DAGUE_SYMBOL_IS_STANDALONE
};

static inline int minexpr_of_column_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;
    (void)assignments;
    return __dague_object->super.src->j;
}

static const expr_t minexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = minexpr_of_column_fct }
};

static inline int maxexpr_of_column_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;

    (void)__dague_object;
    (void)assignments;
    return __dague_object->super.src->nt;
}
static const expr_t maxexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = maxexpr_of_column_fct }
};
static const symbol_t symb_column = {
    .min = &minexpr_of_column,
    .max = &maxexpr_of_column,
    .flags = DAGUE_SYMBOL_IS_STANDALONE
};

static inline int pred_of_map_operator_all_as_expr_fct(const dague_object_t *__dague_object_parent,
                                                const assignment_t *assignments)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)__dague_object_parent;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_object;
    (void)assignments;
    /* Compute Predicate */
    return 1;
}
static const expr_t pred_of_map_operator_all_as_expr = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = pred_of_map_operator_all_as_expr_fct }
};

static inline int
expr_of_p1_for_flow_of_map_operator_dep_in_fct(const dague_object_t *__dague_object_parent,
                                                const assignment_t *assignments)
{
    (void)__dague_object_parent;
    return assignments[0].value;
}
static const expr_t expr_of_p1_for_flow_of_map_operator_dep_in = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = expr_of_p1_for_flow_of_map_operator_dep_in_fct }
};
static const dep_t flow_of_map_operator_dep_in = {
    .cond = NULL,
    .function_id = 0,  /* dague_map_operator.function_id */
    .flow = &flow_of_map_operator,
    .datatype = { .type = { .cst = 0 }, .layout = { .fct = NULL}, .count = { .cst = 1 }, .displ = { .cst = 0 } },
    .call_params = {
        &expr_of_p1_for_flow_of_map_operator_dep_in
    }
};

static inline int
expr_of_p1_for_flow_of_map_operator_dep_out_fct(const dague_object_t *__dague_object_parent,
                                                 const assignment_t *assignments)
{
    (void)__dague_object_parent;
    return (assignments[0].value + 1);
}
static const expr_t expr_of_p1_for_flow_of_map_operator_dep_out = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = expr_of_p1_for_flow_of_map_operator_dep_out_fct }
};
static const dep_t flow_of_map_operator_dep_out = {
    .cond = NULL,
    .function_id = 0,  /* dague_map_operator.function_id */
    .dep_index = 1,
    .flow = &flow_of_map_operator,
    .datatype = { .type = { .cst = 0 }, .layout = { .fct = NULL}, .count = { .cst = 1 }, .displ = { .cst = 0 } },
    .call_params = {
        &expr_of_p1_for_flow_of_map_operator_dep_out
    }
};

static const dague_flow_t flow_of_map_operator = {
    .name = "I",
    .sym_type = SYM_INOUT,
    .flow_flags = ACCESS_RW,
    .flow_index = 0,
    .dep_in  = { &flow_of_map_operator_dep_in },
    .dep_out = { &flow_of_map_operator_dep_out }
};

static dague_ontask_iterate_t
add_task_to_list(struct dague_execution_unit *eu_context,
                 const dague_execution_context_t *newcontext,
                 const dague_execution_context_t *oldcontext,
                 const dep_t* dep,
                 dague_dep_data_description_t* data,
                 int rank_src, int rank_dst,
                 int vpid_dst,
                 void *_ready_lists)
{
    dague_execution_context_t** pready_list = (dague_execution_context_t**)_ready_lists;
    dague_execution_context_t* new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( eu_context->context_mempool );
    dague_thread_mempool_t* mpool = new_context->mempool_owner;

    memcpy( new_context, newcontext, sizeof(dague_execution_context_t) );
    new_context->mempool_owner = mpool;

    pready_list[vpid_dst] = (dague_execution_context_t*)dague_list_item_ring_push_sorted( (dague_list_item_t*)(pready_list[vpid_dst]),
                                                                                          (dague_list_item_t*)new_context,
                                                                                          dague_execution_context_priority_comparator );

    (void)oldcontext; (void)dep; (void)rank_src; (void)rank_dst; (void)vpid_dst; (void)data;
    return DAGUE_ITERATE_STOP;
}

static void iterate_successors(dague_execution_unit_t *eu,
                               dague_execution_context_t *this_task,
                               uint32_t action_mask,
                               dague_ontask_function_t *ontask,
                               void *ontask_arg)
{
    __dague_map_operator_object_t *__dague_object = (__dague_map_operator_object_t*)this_task->dague_object;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value+1;
    dague_execution_context_t nc;

    nc.priority = 0;
    nc.data[0].data_repo = NULL;
    nc.data[0].data_repo = NULL;
    /* If this is the last n, try to move to the next k */
    for( ; k < (int)__dague_object->super.src->nt; n = 0) {
        for( ; n < (int)__dague_object->super.src->mt; n++ ) {
            if( __dague_object->super.src->super.myrank !=
                ((dague_ddesc_t*)__dague_object->super.src)->rank_of((dague_ddesc_t*)__dague_object->super.src,
                                                                     k, n) )
                continue;
            int vpid =  ((dague_ddesc_t*)__dague_object->super.src)->vpid_of((dague_ddesc_t*)__dague_object->super.src,
                                                                             k, n);
            /* Here we go, one ready local task */
            nc.locals[0].value = k;
            nc.locals[1].value = n;
            nc.function = &dague_map_operator /*this*/;
            nc.dague_object = this_task->dague_object;
            nc.data[0].data = this_task->data[0].data;
            nc.data[1].data = this_task->data[1].data;

            ontask(eu, &nc, this_task, &flow_of_map_operator_dep_out, NULL,
                   __dague_object->super.src->super.myrank,
                   __dague_object->super.src->super.myrank,
                   vpid,
                   ontask_arg);
            return;
        }
        /* Go to the next row ... atomically */
        k = dague_atomic_inc_32b( &__dague_object->super.next_k );
    }
    (void)action_mask;
}

static int release_deps(dague_execution_unit_t *eu,
                        dague_execution_context_t *this_task,
                        uint32_t action_mask,
                        dague_remote_deps_t *deps)
{
    dague_execution_context_t** ready_list;
    int i;

    ready_list = (dague_execution_context_t **)calloc(sizeof(dague_execution_context_t *),
                                                     vpmap_get_nb_vp());

    iterate_successors(eu, this_task, action_mask, add_task_to_list, ready_list);

    if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
        for(i = 0; i < vpmap_get_nb_vp(); i++) {
            if( NULL != ready_list[i] ) {
                if( i == eu->virtual_process->vp_id )
                    __dague_schedule(eu, ready_list[i]);
                else
                    __dague_schedule(eu->virtual_process->dague_context->virtual_processes[i]->execution_units[0], 
                                     ready_list[i]);
            }
        }
    }

    if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS) {
        (void)AUNREF(this_task->data[0].data);
    }

    free(ready_list);

    (void)deps;
    return 1;
}

static int data_lookup(dague_execution_unit_t *context,
                       dague_execution_context_t *this_task)
{
    const __dague_map_operator_object_t *__dague_object = (__dague_map_operator_object_t*)this_task->dague_object;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;

    (void)context;

    if( NULL != __dague_object->super.src ) {
        this_task->data[0].data = (dague_arena_chunk_t*) src(k,n);
        this_task->data[0].data_repo = NULL;
    }
    if( NULL != __dague_object->super.dest ) {
        this_task->data[1].data = (dague_arena_chunk_t*) dest(k,n);
        this_task->data[1].data_repo = NULL;
    }
    return 0;
}

static int hook_of(dague_execution_unit_t *context,
                   dague_execution_context_t *this_task)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t*)this_task->dague_object;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    const void* src_data = NULL;
    void* dest_data = NULL;

    if( NULL != __dague_object->super.src ) {
        src_data = ADATA(this_task->data[0].data);
    }
    if( NULL != __dague_object->super.dest ) {
        dest_data = ADATA(this_task->data[1].data);
    }

#if !defined(DAGUE_PROF_DRY_BODY)
    TAKE_TIME(context, 2*this_task->function->function_id,
              map_operator_op_hash( __dague_object, k, n ), __dague_object->super.src,
              ((dague_ddesc_t*)(__dague_object->super.src))->data_key((dague_ddesc_t*)__dague_object->super.src, k, n) );
    __dague_object->super.op( context, src_data, dest_data, __dague_object->super.op_data, k, n );
#endif
    (void)context;
    return 0;
}

static int complete_hook(dague_execution_unit_t *context,
                         dague_execution_context_t *this_task)
{
    const __dague_map_operator_object_t *__dague_object = (const __dague_map_operator_object_t *)this_task->dague_object;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    (void)k; (void)n; (void)__dague_object;

    TAKE_TIME(context, 2*this_task->function->function_id+1, map_operator_op_hash( __dague_object, k, n ), NULL, 0);

    dague_prof_grapher_task(this_task, context->th_id, context->virtual_process->vp_id, k+n);

    release_deps(context, this_task,
                 (DAGUE_ACTION_RELEASE_REMOTE_DEPS |
                  DAGUE_ACTION_RELEASE_LOCAL_DEPS |
                  DAGUE_ACTION_RELEASE_LOCAL_REFS |
                  DAGUE_ACTION_DEPS_MASK),
                 NULL);

    return 0;
}

static __dague_chore_t __dague_map_chores = {
    .evaluate = NULL,
    .hook = hook_of,
};

static const dague_function_t dague_map_operator = {
    .name = "map_operator",
    .flags = 0x0,
    .function_id = 0,
    .nb_incarnations = 1,
    .nb_parameters = 2,
    .nb_locals = 2,
    .dependencies_goal = 0x1,
    .params = { &symb_row, &symb_column },
    .locals = { &symb_row, &symb_column },
    .pred = &pred_of_map_operator_all_as_expr,
    .priority = NULL,
    .in = { &flow_of_map_operator },
    .out = { &flow_of_map_operator },
    .init = NULL,
    .key = NULL,
    .prepare_input = data_lookup,
    .incarnations = &__dague_map_chores,
    .iterate_successors = iterate_successors,
    .release_deps = release_deps,
    .complete_execution = complete_hook,
    .fini = NULL,
};

static void dague_map_operator_startup_fn(dague_context_t *context,
                                          dague_object_t *dague_object,
                                          dague_execution_context_t** startup_list)
{
    __dague_map_operator_object_t *__dague_object = (__dague_map_operator_object_t*)dague_object;
    dague_execution_context_t fake_context;
    dague_execution_context_t *ready_list;
    int k = 0, n = 0, count = 0, vpid = 0;
    dague_execution_unit_t* eu;

    *startup_list = NULL;
    fake_context.function = &dague_map_operator;
    fake_context.dague_object = dague_object;
    fake_context.priority = 0;
    fake_context.data[0].data_repo = NULL;
    fake_context.data[0].data      = NULL;
    fake_context.data[1].data_repo = NULL;
    fake_context.data[1].data      = NULL;
    for( vpid = 0; vpid < vpmap_get_nb_vp(); vpid++ ) {
        /* If this is the last n, try to move to the next k */
        count = 0;
        for( ; k < (int)__dague_object->super.src->nt; n = 0) {
            for( ; n < (int)__dague_object->super.src->mt; n++ ) {
                if (__dague_object->super.src->super.myrank !=
                    ((dague_ddesc_t*)__dague_object->super.src)->rank_of((dague_ddesc_t*)__dague_object->super.src,
                                                                         k, n) )
                    continue;

                if( vpid != ((dague_ddesc_t*)__dague_object->super.src)->vpid_of((dague_ddesc_t*)__dague_object->super.src,
                                                                                 k, n) )
                    continue;
                /* Here we go, one ready local task */
                ready_list = NULL;
                eu = context->virtual_processes[vpid]->execution_units[count];
                fake_context.locals[0].value = k;
                fake_context.locals[1].value = n;
                add_task_to_list(eu, &fake_context, NULL, &flow_of_map_operator_dep_out, NULL,
                                 __dague_object->super.src->super.myrank, -1,
                                 0, (void*)&ready_list);
                __dague_schedule( eu, ready_list );
                count++;
                if( count == context->virtual_processes[vpid]->nb_cores )
                    goto done;
                break;
            }
            /* Go to the next row ... atomically */
            k = dague_atomic_inc_32b( &__dague_object->super.next_k );
        }
    done:  continue;
    }
    return;
}

/**
 * Apply the operator op on all tiles of the src matrix. The src matrix is const, the
 * result is supposed to be pushed on the dest matrix. However, any of the two matrices
 * can be NULL, and then the data is reported as NULL in the corresponding op
 * floweter.
 */
struct dague_object_t*
dague_map_operator_New(const tiled_matrix_desc_t* src,
                       tiled_matrix_desc_t* dest,
                       dague_operator_t op,
                       void* op_data)
{
    __dague_map_operator_object_t *res = (__dague_map_operator_object_t*)calloc(1, sizeof(__dague_map_operator_object_t));

    if( (NULL == src) && (NULL == dest) )
        return NULL;
    /* src and dest should have similar distributions */
    /* TODO */

    res->super.src     = src;
    res->super.dest    = dest;
    res->super.op      = op;
    res->super.op_data = op_data;

#  if defined(DAGUE_PROF_TRACE)
    res->super.super.profiling_array = dague_map_operator_profiling_array;
    if( -1 == dague_map_operator_profiling_array[0] ) {
        dague_profiling_add_dictionary_keyword("operator", "fill:CC2828",
                                               sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                               (int*)&res->super.super.profiling_array[0 + 2 * dague_map_operator.function_id],
                                               (int*)&res->super.super.profiling_array[1 + 2 * dague_map_operator.function_id]);
    }
#  endif /* defined(DAGUE_PROF_TRACE) */

    res->super.super.object_id = 1111;
    res->super.super.nb_local_tasks = src->nb_local_tiles;
    res->super.super.startup_hook = dague_map_operator_startup_fn;
    return (struct dague_object_t*)res;
}

void dague_map_operator_Destruct( struct dague_object_t* o )
{
#if defined(DAGUE_PROF_TRACE)
    char* filename = NULL;
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    asprintf(&filename, "%s.%d.profile", "operator", rank);
#else
    asprintf(&filename, "%s.profile", "operator");
#endif
    dague_profiling_dump_dbp(filename);
    free(filename);
#endif  /* defined(DAGUE_PROF_TRACE) */
    (void)o;
}
