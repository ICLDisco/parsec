/*
 * Copyright (c) 2011-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/remote_dep.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/parsec_prof_grapher.h"
#include "parsec/scheduling.h"
#include "parsec/datarepo.h"
#include "parsec/mca/device/device.h"
#include "parsec/vpmap.h"
#include "parsec/data_internal.h"
#include "parsec/interfaces/interface.h"
#include "parsec/execution_stream.h"

#if defined(PARSEC_PROF_TRACE)
int parsec_map_operator_profiling_array[2] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {              \
        parsec_profile_data_collection_info_t info;                    \
        info.desc = (parsec_data_collection_t*)refdesc;                \
        info.id = refid;                                               \
        PARSEC_PROFILING_TRACE(context->es_profile,                    \
                               __tp->super.profiling_array[(key)],      \
                               eid, __tp->super.taskpool_id, (void*)&info); \
    } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif

typedef struct parsec_map_operator_taskpool {
    parsec_taskpool_t               super;
    const parsec_tiled_matrix_dc_t* src;
          parsec_tiled_matrix_dc_t* dest;
    volatile int32_t                next_n;
    parsec_operator_t               op;
    void*                           op_data;
} parsec_map_operator_taskpool_t;

static const parsec_flow_t flow_of_map_operator;
static const parsec_task_class_t parsec_map_operator;

static parsec_key_t map_operator_make_key(const parsec_taskpool_t *tp, const parsec_assignment_t *as);

#define src(k,n)  (((parsec_data_collection_t*)__tp->src)->data_of((parsec_data_collection_t*)__tp->src, (k), (n)))
#define dest(k,n)  (((parsec_data_collection_t*)__tp->dest)->data_of((parsec_data_collection_t*)__tp->dest, (k), (n)))

static inline int minexpr_of_row_fct(const parsec_taskpool_t *tp, const parsec_assignment_t *assignments)
{
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)tp;
    (void)assignments;
    return __tp->src->i;
}
static const parsec_expr_t minexpr_of_row = {
    .op = PARSEC_EXPR_OP_INLINE,
    .u_expr.v_func = { .type = 0,
                       .func = { .inline_func_int32 = minexpr_of_row_fct }
    }
};
static inline int maxexpr_of_row_fct(const parsec_taskpool_t *tp, const parsec_assignment_t *assignments)
{
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)tp;

    (void)__tp;
    (void)assignments;
    return __tp->src->mt;
}
static const parsec_expr_t maxexpr_of_row = {
    .op = PARSEC_EXPR_OP_INLINE,
    .u_expr.v_func = { .type = 0,
                       .func = { .inline_func_int32 = maxexpr_of_row_fct }
    }
};
static const parsec_symbol_t symb_row = {
    .min = &minexpr_of_row,
    .max = &maxexpr_of_row,
    .flags = PARSEC_SYMBOL_IS_STANDALONE
};

static inline int minexpr_of_column_fct(const parsec_taskpool_t *tp, const parsec_assignment_t *assignments)
{
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)tp;
    (void)assignments;
    return __tp->src->j;
}

static const parsec_expr_t minexpr_of_column = {
    .op = PARSEC_EXPR_OP_INLINE,
    .u_expr.v_func = { .type = 0,
                       .func = { .inline_func_int32 = minexpr_of_column_fct }
    }
};

static inline int maxexpr_of_column_fct(const parsec_taskpool_t *tp, const parsec_assignment_t *assignments)
{
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)tp;

    (void)__tp;
    (void)assignments;
    return __tp->src->nt;
}
static const parsec_expr_t maxexpr_of_column = {
    .op = PARSEC_EXPR_OP_INLINE,
    .u_expr.v_func = { .type = 0,
                       .func = { .inline_func_int32 = maxexpr_of_column_fct }
    }
};
static const parsec_symbol_t symb_column = {
    .min = &minexpr_of_column,
    .max = &maxexpr_of_column,
    .flags = PARSEC_SYMBOL_IS_STANDALONE
};

static inline int affinity_of_map_operator(parsec_task_t *this_task,
                                           parsec_data_ref_t *ref)
{
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    ref->dc = (parsec_data_collection_t*)__tp->src;
    ref->key = ref->dc->data_key(ref->dc, k, n);
    return 1;
}

static inline int initial_data_of_map_operator(parsec_task_t *this_task,
                                               parsec_data_ref_t *refs)
{
    int __flow_nb = 0;
    parsec_data_collection_t *__d;
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;

    __d = (parsec_data_collection_t*)__tp->src;
    refs[__flow_nb].dc = __d;
    refs[__flow_nb].key = __d->data_key(__d, k, n);
    __flow_nb++;

    return __flow_nb;
}

static inline int final_data_of_map_operator(parsec_task_t *this_task,
                                             parsec_data_ref_t *data_refs)
{
    int __flow_nb = 0;
    parsec_data_collection_t *__d;
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;

    __d = (parsec_data_collection_t*)__tp->dest;
    data_refs[__flow_nb].dc = __d;
    data_refs[__flow_nb].key = __d->data_key(__d, k, n);
    __flow_nb++;

    return __flow_nb;
}

static const parsec_dep_t flow_of_map_operator_dep_in = {
    .cond = NULL,
    .task_class_id = 0,  /* parsec_map_operator.task_class_id */
    .flow = &flow_of_map_operator,
};

static const parsec_dep_t flow_of_map_operator_dep_out = {
    .cond = NULL,
    .task_class_id = 0,  /* parsec_map_operator.task_class_id */
    .dep_index = 1,
    .flow = &flow_of_map_operator,
};

static const parsec_flow_t flow_of_map_operator = {
    .name = "I",
    .sym_type = PARSEC_SYM_INOUT,
    .flow_flags = PARSEC_FLOW_ACCESS_RW,
    .flow_index = 0,
    .dep_in  = { &flow_of_map_operator_dep_in },
    .dep_out = { &flow_of_map_operator_dep_out }
};

static parsec_ontask_iterate_t
add_task_to_list(parsec_execution_stream_t *es,
                 const parsec_task_t *newcontext,
                 const parsec_task_t *oldcontext,
                 const parsec_dep_t* dep,
                 parsec_dep_data_description_t* data,
                 int rank_src, int rank_dst,
                 int vpid_dst,
                 void *_ready_lists)
{
    parsec_task_t** pready_list = (parsec_task_t**)_ready_lists;
    parsec_task_t* new_context = (parsec_task_t*)parsec_thread_mempool_allocate( es->context_mempool );

    PARSEC_COPY_EXECUTION_CONTEXT(new_context, newcontext);
    new_context->status = PARSEC_TASK_STATUS_NONE;
    pready_list[vpid_dst] = (parsec_task_t*)parsec_list_item_ring_push_sorted( (parsec_list_item_t*)(pready_list[vpid_dst]),
                                                                               (parsec_list_item_t*)new_context,
                                                                               parsec_execution_context_priority_comparator );

    (void)oldcontext; (void)dep; (void)rank_src; (void)rank_dst; (void)vpid_dst; (void)data;
    return PARSEC_ITERATE_STOP;
}

static void iterate_successors(parsec_execution_stream_t *es,
                               const parsec_task_t *this_task,
                               uint32_t action_mask,
                               parsec_ontask_function_t *ontask,
                               void *ontask_arg)
{
    parsec_map_operator_taskpool_t *__tp = (parsec_map_operator_taskpool_t*)this_task->taskpool;
    int m = this_task->locals[0].value+1;
    int n = this_task->locals[1].value;
    parsec_task_t nt;

    nt.priority = 0;
    nt.chore_id = 0;
    nt.data[0].data_repo = NULL;  /* src  */
    nt.data[1].data_repo = NULL;  /* dst */
    /* If this is the last n, try to move to the next k */
    for( ; n < (int)__tp->src->nt; m = 0) {
        for( ; m < (int)__tp->src->mt; m++ ) {
            if( __tp->src->super.myrank !=
                ((parsec_data_collection_t*)__tp->src)->rank_of((parsec_data_collection_t*)__tp->src,
                                                                m, n) )
                continue;
            int vpid =  ((parsec_data_collection_t*)__tp->src)->vpid_of((parsec_data_collection_t*)__tp->src,
                                                                        m, n);
            /* Here we go, one ready local task */
            nt.locals[0].value = m;
            nt.locals[1].value = n;
            nt.task_class = &parsec_map_operator /*this*/;
            nt.taskpool = this_task->taskpool;
            nt.data[0].data_in = this_task->data[0].data_out;  /* src */
            nt.data[1].data_in = this_task->data[1].data_out;  /* dst */

            ontask(es, &nt, this_task, &flow_of_map_operator_dep_out, NULL,
                   __tp->src->super.myrank,
                   __tp->src->super.myrank,
                   vpid,
                   ontask_arg);
            return;
        }
        /* Go to the next column ... atomically */
        n = parsec_atomic_fetch_inc_int32( &__tp->next_n ) + 1;
    }
    (void)action_mask;
}

static int release_deps(parsec_execution_stream_t *es,
                        parsec_task_t *this_task,
                        uint32_t action_mask,
                        parsec_remote_deps_t *deps)
{
    parsec_task_t** ready_list;
    int i;

    PARSEC_PINS(es, RELEASE_DEPS_BEGIN, (parsec_task_t *) this_task);

    ready_list = alloca(sizeof(parsec_task_t *) * es->virtual_process->parsec_context->nb_vp);
    for(i = 0; i < es->virtual_process->parsec_context->nb_vp; ready_list[i++] = NULL);

    iterate_successors(es, this_task, action_mask, add_task_to_list, ready_list);

    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
        for(i = 0; i < es->virtual_process->parsec_context->nb_vp; i++) {
            if( NULL == ready_list[i] )
                continue;
            if( i == es->virtual_process->vp_id )
                __parsec_schedule(es, ready_list[i], 0);
            else
                __parsec_schedule(es->virtual_process->parsec_context->virtual_processes[i]->execution_streams[0],
                                  ready_list[i], 0);
            ready_list[i] = NULL;
        }
    }

    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS) {
        const parsec_map_operator_taskpool_t *__tp = (parsec_map_operator_taskpool_t*)this_task->taskpool;

        /**
         * There is no repo to be release in this instance, so instead just release the
         * reference of the data copy (if such a copy exists).
         *
         * data_repo_entry_used_once( eu, this_task->data[0].data_repo, this_task->data[0].data_repo->key );
         */
        if( NULL != __tp->src ) {
            PARSEC_DATA_COPY_RELEASE(this_task->data[0].data_in);
        }
        if( NULL != __tp->dest ) {
            PARSEC_DATA_COPY_RELEASE(this_task->data[1].data_in);
        }
    }
    PARSEC_PINS(es, RELEASE_DEPS_END, (parsec_task_t *) this_task);
    (void)deps;
    return 0;
}

static int data_lookup(parsec_execution_stream_t *es,
                       parsec_task_t *this_task)
{
    const parsec_map_operator_taskpool_t *__tp = (parsec_map_operator_taskpool_t*)this_task->taskpool;
    int m = this_task->locals[0].value;
    int n = this_task->locals[1].value;

    (void)es;

    if( NULL != __tp->src ) {
        this_task->data[0].data_in   = parsec_data_get_copy(src(m,n), 0);
        this_task->data[0].data_repo = NULL;
        this_task->data[0].data_out  = NULL;
        PARSEC_OBJ_RETAIN(this_task->data[0].data_in);
    }
    if( NULL != __tp->dest ) {
        this_task->data[1].data_in   = parsec_data_get_copy(dest(m,n), 0);
        this_task->data[1].data_repo = NULL;
        this_task->data[1].data_out  = this_task->data[1].data_in;
        PARSEC_OBJ_RETAIN(this_task->data[1].data_in);
    }
    return 0;
}

static int hook_of(parsec_execution_stream_t *es,
                   parsec_task_t *this_task)
{
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t*)this_task->taskpool;
    int m = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    int rc = PARSEC_HOOK_RETURN_DONE;
    const void* src_data = NULL;
    void* dest_data = NULL;

    if( NULL != __tp->src ) {
        src_data = PARSEC_DATA_COPY_GET_PTR(this_task->data[0].data_in);
    }
    if( NULL != __tp->dest ) {
        dest_data = PARSEC_DATA_COPY_GET_PTR(this_task->data[1].data_in);
    }

#if !defined(PARSEC_PROF_DRY_BODY)
    TAKE_TIME(es, 2*this_task->task_class->task_class_id,
              parsec_hash_table_generic_64bits_key_hash( map_operator_make_key(this_task->taskpool, this_task->locals), NULL ), __tp->src,
              ((parsec_data_collection_t*)(__tp->src))->data_key((parsec_data_collection_t*)__tp->src, m, n) );
    rc = __tp->op( es, src_data, dest_data, __tp->op_data, m, n );
#endif
    (void)es; (void)rc;
    return 0;
}

static int complete_hook(parsec_execution_stream_t *es,
                         parsec_task_t *this_task)
{
    const parsec_map_operator_taskpool_t *__tp = (const parsec_map_operator_taskpool_t *)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    (void)k; (void)n; (void)__tp;

    TAKE_TIME(es, 2*this_task->task_class->task_class_id+1,
              parsec_hash_table_generic_64bits_key_hash( map_operator_make_key(this_task->taskpool, this_task->locals), NULL ),
              NULL, 0);

#if defined(PARSEC_PROF_GRAPHER)
    parsec_prof_grapher_task(this_task, es->th_id, es->virtual_process->vp_id, k+n);
#endif  /* defined(PARSEC_PROF_GRAPHER) */

    release_deps(es, this_task,
                 (PARSEC_ACTION_RELEASE_REMOTE_DEPS |
                  PARSEC_ACTION_RELEASE_LOCAL_DEPS |
                  PARSEC_ACTION_RELEASE_LOCAL_REFS |
                  PARSEC_ACTION_DEPS_MASK),
                 NULL);

    return 0;
}

static __parsec_chore_t __parsec_map_operator_chores[] = {
    { .type     = PARSEC_DEV_CPU,
      .evaluate = NULL,
      .hook     = hook_of },
    { .type     = PARSEC_DEV_NONE,
      .evaluate = NULL,
      .hook     = NULL },
};

static parsec_key_t map_operator_make_key(const parsec_taskpool_t *tp, const parsec_assignment_t *as)
{
    (void)tp;
    return (parsec_key_t)(uintptr_t)(((uint64_t)as[0].value << 32) | (uint64_t)as[1].value);
}

static char *map_operator_key_print(char *buffer, size_t buffer_size, parsec_key_t _key, void *user_data)
{
    uint64_t key = (uint64_t)(uintptr_t)_key;
    (void)user_data;
    snprintf(buffer, buffer_size, "map_operator(%d, %d)", (int)(key >> 32), (int)(key & 0xFFFFFFFF));
    return buffer;
}

static parsec_key_fn_t __parsec_map_operator_key_functions = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = map_operator_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

static const parsec_task_class_t parsec_map_operator = {
    .name = "map_operator",
    .flags = 0x0,
    .task_class_id = 0,
    .nb_parameters = 2,
    .nb_locals = 2,
    .dependencies_goal = 0x1,
    .params = { &symb_row, &symb_column },
    .locals = { &symb_row, &symb_column },
    .data_affinity = affinity_of_map_operator,
    .initial_data = initial_data_of_map_operator,
    .final_data = final_data_of_map_operator,
    .priority = NULL,
    .in = { &flow_of_map_operator },
    .out = { &flow_of_map_operator },
    .make_key = map_operator_make_key,
    .key_functions = &__parsec_map_operator_key_functions,
    .prepare_input = data_lookup,
    .incarnations = __parsec_map_operator_chores,
    .iterate_successors = iterate_successors,
    .release_deps = release_deps,
    .complete_execution = complete_hook,
    .release_task = parsec_release_task_to_mempool_update_nbtasks,
    .fini = NULL,
};

static void parsec_map_operator_startup_fn(parsec_context_t *context,
                                           parsec_taskpool_t *tp,
                                           parsec_task_t** startup_list)
{
    parsec_map_operator_taskpool_t *__tp = (parsec_map_operator_taskpool_t*)tp;
    parsec_task_t fake_context;
    parsec_task_t *ready_list;
    int m = 0, n = 0, count = 0, vpid = 0;
    parsec_execution_stream_t* es;

    *startup_list = NULL;
    fake_context.task_class = &parsec_map_operator;
    fake_context.taskpool = tp;
    fake_context.priority = 0;
    fake_context.chore_id = 0;
    fake_context.data[0].data_repo = NULL;  /* src */
    fake_context.data[0].data_in   = NULL;
    fake_context.data[1].data_repo = NULL;  /* dst */
    fake_context.data[1].data_in   = NULL;

    /**
     * Generate one local task per core. Each task will then take care of creating all
     * the remaining tasks for the same column of the matrix, and upon completion of
     * all tasks on the column htey will move to the next row. The row index is marshalled
     * using atomic operations to avoid conflicts between generators for different
     * completed tasks.
     */
    for( vpid = 0; vpid < context->nb_vp; vpid++ ) {
        /* If this is the last m, try to move to the next n */
        count = 0;
        for( ; n < (int)__tp->src->nt; ) {
            for( m = 0; m < (int)__tp->src->mt; m++ ) {
                if (__tp->src->super.myrank !=
                    ((parsec_data_collection_t*)__tp->src)->rank_of((parsec_data_collection_t*)__tp->src,
                                                                    m, n) )
                    continue;

                if( vpid != ((parsec_data_collection_t*)__tp->src)->vpid_of((parsec_data_collection_t*)__tp->src,
                                                                            m, n) )
                    continue;
                /* Here we go, one ready local task */
                ready_list = NULL;
                es = context->virtual_processes[vpid]->execution_streams[count];
                fake_context.locals[0].value = m;
                fake_context.locals[1].value = n;
                add_task_to_list(es, &fake_context, NULL, &flow_of_map_operator_dep_out, NULL,
                                 __tp->src->super.myrank, -1,
                                 0 /* here this must always be zero due to ready_list */, (void*)&ready_list);
                __parsec_schedule( es, ready_list, 0 );
                count++;
                if( count == context->virtual_processes[vpid]->nb_cores )
                    goto done;
                break;
            }
            /* Go to the next row ... atomically */
            n = parsec_atomic_fetch_inc_int32( &__tp->next_n ) + 1;
        }
    done:  continue;
    }
}

static void parsec_map_operator_destructor( parsec_map_operator_taskpool_t* tp )
{
    free(tp->super.task_classes_array);
    tp->super.task_classes_array = NULL;
    tp->super.nb_task_classes = 0;
    PARSEC_OBJ_DESTRUCT((parsec_taskpool_t*)tp);
    free(tp);
}

/**
 * Apply the operator op on all tiles of the src matrix. The src matrix is const, the
 * result is supposed to be pushed on the dest matrix. However, any of the two matrices
 * can be NULL, and then the data is reported as NULL in the corresponding op
 * flow.
 */
parsec_taskpool_t*
parsec_map_operator_New(const parsec_tiled_matrix_dc_t* src,
                        parsec_tiled_matrix_dc_t* dest,
                        parsec_operator_t op,
                        void* op_data)
{
    parsec_map_operator_taskpool_t *tp;

    /* src and dest should have similar distributions */

    /* TODO */
    tp =  (parsec_map_operator_taskpool_t*)calloc(1, sizeof(parsec_map_operator_taskpool_t));
    PARSEC_OBJ_CONSTRUCT((parsec_taskpool_t*)tp, parsec_taskpool_t);
    tp->src     = src;
    tp->dest    = dest;
    tp->op      = op;
    tp->op_data = op_data;
    tp->super.taskpool_name = strdup("map_operator");
    tp->super.taskpool_type = PARSEC_TASKPOOL_TYPE_PTG;

#  if defined(PARSEC_PROF_TRACE)
    tp->super.profiling_array = parsec_map_operator_profiling_array;
    if( -1 == parsec_map_operator_profiling_array[0] ) {
        parsec_profiling_add_dictionary_keyword("operator", "fill:CC2828",
                                                sizeof(parsec_profile_data_collection_info_t), PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                                (int*)&tp->super.profiling_array[0 + 2 * parsec_map_operator.task_class_id],
                                                (int*)&tp->super.profiling_array[1 + 2 * parsec_map_operator.task_class_id]);
    }
#  endif /* defined(PARSEC_PROF_TRACE) */

    tp->super.taskpool_id = 1111;
    tp->super.nb_tasks = src->nb_local_tiles;
    tp->super.nb_pending_actions = 1;  /* for all local tasks */
    tp->super.startup_hook = parsec_map_operator_startup_fn;
    tp->super.destructor = (parsec_destruct_fn_t) parsec_map_operator_destructor;
    tp->super.nb_task_classes = 1;
    tp->super.task_classes_array = (const parsec_task_class_t **)
        malloc(tp->super.nb_task_classes * sizeof(parsec_task_class_t *));
    tp->super.task_classes_array[0] = &parsec_map_operator;
    tp->super.devices_index_mask = PARSEC_DEVICES_ALL;
    tp->super.update_nb_runtime_task = parsec_add_fetch_runtime_task;
    (void)parsec_taskpool_reserve_id((parsec_taskpool_t *)tp);
    return (parsec_taskpool_t*)tp;
}
