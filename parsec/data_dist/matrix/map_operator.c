/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/utils/debug.h"
#include "parsec/remote_dep.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/parsec_prof_grapher.h"
#include "parsec/scheduling.h"
#include "parsec/datarepo.h"
#include "parsec/devices/device.h"
#include "parsec/vpmap.h"
#include "parsec/data_internal.h"
#include "parsec/interfaces/interface.h"

#if defined(PARSEC_PROF_TRACE)
int parsec_map_operator_profiling_array[2] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \
   parsec_profile_data_collection_info_t info;                         \
   info.desc = (parsec_data_collection_t*)refdesc;                     \
   info.id = refid;                                         \
   PARSEC_PROFILING_TRACE(context->es_profile,               \
                         __tp->super.super.profiling_array[(key)],\
                         eid, __tp->super.super.taskpool_id, (void*)&info);  \
  } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif

typedef struct parsec_map_operator_taskpool {
    parsec_taskpool_t          super;
    const parsec_tiled_matrix_dc_t* src;
          parsec_tiled_matrix_dc_t* dest;
    volatile uint32_t          next_k;
    parsec_operator_t           op;
    void*                      op_data;
} parsec_map_operator_taskpool_t;

typedef struct __parsec_map_operator_taskpool {
    parsec_map_operator_taskpool_t super;
} __parsec_map_operator_taskpool_t;

static const parsec_flow_t flow_of_map_operator;
static const parsec_task_class_t parsec_map_operator;

#define src(k,n)  (((parsec_data_collection_t*)__tp->super.src)->data_of((parsec_data_collection_t*)__tp->super.src, (k), (n)))
#define dest(k,n)  (((parsec_data_collection_t*)__tp->super.dest)->data_of((parsec_data_collection_t*)__tp->super.dest, (k), (n)))

#if defined(PARSEC_PROF_TRACE)
static inline uint32_t map_operator_op_hash(const __parsec_map_operator_taskpool_t *tp, int k, int n )
{
    return tp->super.src->mt * k + n;
}
#endif  /* defined(PARSEC_PROF_TRACE) */

static inline int minexpr_of_row_fct(const parsec_taskpool_t *tp, const assignment_t *assignments)
{
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)tp;
    (void)assignments;
    return __tp->super.src->i;
}
static const expr_t minexpr_of_row = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = minexpr_of_row_fct }
};
static inline int maxexpr_of_row_fct(const parsec_taskpool_t *tp, const assignment_t *assignments)
{
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)tp;

    (void)__tp;
    (void)assignments;
    return __tp->super.src->mt;
}
static const expr_t maxexpr_of_row = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = maxexpr_of_row_fct }
};
static const symbol_t symb_row = {
    .min = &minexpr_of_row,
    .max = &maxexpr_of_row,
    .flags = PARSEC_SYMBOL_IS_STANDALONE
};

static inline int minexpr_of_column_fct(const parsec_taskpool_t *tp, const assignment_t *assignments)
{
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)tp;
    (void)assignments;
    return __tp->super.src->j;
}

static const expr_t minexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = minexpr_of_column_fct }
};

static inline int maxexpr_of_column_fct(const parsec_taskpool_t *tp, const assignment_t *assignments)
{
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)tp;

    (void)__tp;
    (void)assignments;
    return __tp->super.src->nt;
}
static const expr_t maxexpr_of_column = {
    .op = EXPR_OP_INLINE,
    .u_expr = { .inline_func_int32 = maxexpr_of_column_fct }
};
static const symbol_t symb_column = {
    .min = &minexpr_of_column,
    .max = &maxexpr_of_column,
    .flags = PARSEC_SYMBOL_IS_STANDALONE
};

static inline int affinity_of_map_operator(parsec_task_t *this_task,
                                           parsec_data_ref_t *ref)
{
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    ref->dc = (parsec_data_collection_t*)__tp->super.src;
    ref->key = ref->dc->data_key(ref->dc, k, n);
    return 1;
}

static inline int initial_data_of_map_operator(parsec_task_t *this_task,
                                               parsec_data_ref_t *refs)
{
    int __flow_nb = 0;
    parsec_data_collection_t *__d;
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;

    __d = (parsec_data_collection_t*)__tp->super.src;
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
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;

    __d = (parsec_data_collection_t*)__tp->super.dest;
    data_refs[__flow_nb].dc = __d;
    data_refs[__flow_nb].key = __d->data_key(__d, k, n);
    __flow_nb++;

    return __flow_nb;
}

static const dep_t flow_of_map_operator_dep_in = {
    .cond = NULL,
    .task_class_id = 0,  /* parsec_map_operator.task_class_id */
    .flow = &flow_of_map_operator,
};

static const dep_t flow_of_map_operator_dep_out = {
    .cond = NULL,
    .task_class_id = 0,  /* parsec_map_operator.task_class_id */
    .dep_index = 1,
    .flow = &flow_of_map_operator,
};

static const parsec_flow_t flow_of_map_operator = {
    .name = "I",
    .sym_type = SYM_INOUT,
    .flow_flags = FLOW_ACCESS_RW,
    .flow_index = 0,
    .dep_in  = { &flow_of_map_operator_dep_in },
    .dep_out = { &flow_of_map_operator_dep_out }
};

static parsec_ontask_iterate_t
add_task_to_list(parsec_execution_stream_t *es,
                 const parsec_task_t *newcontext,
                 const parsec_task_t *oldcontext,
                 const dep_t* dep,
                 parsec_dep_data_description_t* data,
                 int rank_src, int rank_dst,
                 int vpid_dst,
                 void *_ready_lists)
{
    parsec_task_t** pready_list = (parsec_task_t**)_ready_lists;
    parsec_task_t* new_context = (parsec_task_t*)parsec_thread_mempool_allocate( es->context_mempool );

    new_context->status = PARSEC_TASK_STATUS_NONE;
    PARSEC_COPY_EXECUTION_CONTEXT(new_context, newcontext);
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
    __parsec_map_operator_taskpool_t *__tp = (__parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value+1;
    parsec_task_t nt;

    nt.priority = 0;
    nt.chore_id = 0;
    nt.data[0].data_repo = NULL;
    nt.data[1].data_repo = NULL;
    /* If this is the last n, try to move to the next k */
    for( ; k < (int)__tp->super.src->nt; n = 0) {
        for( ; n < (int)__tp->super.src->mt; n++ ) {
            if( __tp->super.src->super.myrank !=
                ((parsec_data_collection_t*)__tp->super.src)->rank_of((parsec_data_collection_t*)__tp->super.src,
                                                                     k, n) )
                continue;
            int vpid =  ((parsec_data_collection_t*)__tp->super.src)->vpid_of((parsec_data_collection_t*)__tp->super.src,
                                                                             k, n);
            /* Here we go, one ready local task */
            nt.locals[0].value = k;
            nt.locals[1].value = n;
            nt.task_class = &parsec_map_operator /*this*/;
            nt.taskpool = this_task->taskpool;
            nt.data[0].data_in = this_task->data[0].data_out;
            nt.data[1].data_in = this_task->data[1].data_out;

            ontask(es, &nt, this_task, &flow_of_map_operator_dep_out, NULL,
                   __tp->super.src->super.myrank,
                   __tp->super.src->super.myrank,
                   vpid,
                   ontask_arg);
            return;
        }
        /* Go to the next row ... atomically */
        k = parsec_atomic_inc_32b( &__tp->super.next_k );
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

    ready_list = (parsec_task_t **)calloc(sizeof(parsec_task_t *),
                                                      vpmap_get_nb_vp());

    iterate_successors(es, this_task, action_mask, add_task_to_list, ready_list);

    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
        for(i = 0; i < vpmap_get_nb_vp(); i++) {
            if( NULL != ready_list[i] ) {
                if( i == es->virtual_process->vp_id )
                    __parsec_schedule(es, ready_list[i], 0);
                else
                    __parsec_schedule(es->virtual_process->parsec_context->virtual_processes[i]->execution_streams[0],
                                     ready_list[i], 0);
            }
        }
    }

    if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS) {
        /**
         * There is no repo to be release in this instance, so instead just release the
         * reference of the data copy.
         *
         * data_repo_entry_used_once( eu, this_task->data[0].data_repo, this_task->data[0].data_repo->key );
         */
        PARSEC_DATA_COPY_RELEASE(this_task->data[0].data_in);
    }

    free(ready_list);

    (void)deps;
    return 1;
}

static int data_lookup(parsec_execution_stream_t *es,
                       parsec_task_t *this_task)
{
    const __parsec_map_operator_taskpool_t *__tp = (__parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;

    (void)es;

    if( NULL != __tp->super.src ) {
        this_task->data[0].data_in   = parsec_data_get_copy(src(k,n), 0);
        this_task->data[0].data_repo = NULL;
        this_task->data[0].data_out  = NULL;
    }
    if( NULL != __tp->super.dest ) {
        this_task->data[1].data_in   = parsec_data_get_copy(dest(k,n), 0);
        this_task->data[1].data_repo = NULL;
        this_task->data[1].data_out  = this_task->data[1].data_in;
    }
    return 0;
}

static int hook_of(parsec_execution_stream_t *es,
                   parsec_task_t *this_task)
{
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t*)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    const void* src_data = NULL;
    void* dest_data = NULL;

    if( NULL != __tp->super.src ) {
        src_data = PARSEC_DATA_COPY_GET_PTR(this_task->data[0].data_in);
    }
    if( NULL != __tp->super.dest ) {
        dest_data = PARSEC_DATA_COPY_GET_PTR(this_task->data[1].data_in);
    }

#if !defined(PARSEC_PROF_DRY_BODY)
    TAKE_TIME(es, 2*this_task->task_class->task_class_id,
              map_operator_op_hash( __tp, k, n ), __tp->super.src,
              ((parsec_data_collection_t*)(__tp->super.src))->data_key((parsec_data_collection_t*)__tp->super.src, k, n) );
    __tp->super.op( es, src_data, dest_data, __tp->super.op_data, k, n );
#endif
    (void)es;
    return 0;
}

static int complete_hook(parsec_execution_stream_t *es,
                         parsec_task_t *this_task)
{
    const __parsec_map_operator_taskpool_t *__tp = (const __parsec_map_operator_taskpool_t *)this_task->taskpool;
    int k = this_task->locals[0].value;
    int n = this_task->locals[1].value;
    (void)k; (void)n; (void)__tp;

    TAKE_TIME(es, 2*this_task->task_class->task_class_id+1, map_operator_op_hash( __tp, k, n ), NULL, 0);

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

static __parsec_chore_t __parsec_map_chores[] = {
    { .type     = PARSEC_DEV_CPU,
      .evaluate = NULL,
      .hook     = hook_of },
    { .type     = PARSEC_DEV_NONE,
      .evaluate = NULL,
      .hook     = NULL },
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
    .key = NULL,
    .prepare_input = data_lookup,
    .incarnations = __parsec_map_chores,
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
    __parsec_map_operator_taskpool_t *__tp = (__parsec_map_operator_taskpool_t*)tp;
    parsec_task_t fake_context;
    parsec_task_t *ready_list;
    int k = 0, n = 0, count = 0, vpid = 0;
    parsec_execution_stream_t* es;

    *startup_list = NULL;
    fake_context.task_class = &parsec_map_operator;
    fake_context.taskpool = tp;
    fake_context.priority = 0;
    fake_context.data[0].data_repo = NULL;
    fake_context.data[0].data_in   = NULL;
    fake_context.data[1].data_repo = NULL;
    fake_context.data[1].data_in   = NULL;
    for( vpid = 0; vpid < vpmap_get_nb_vp(); vpid++ ) {
        /* If this is the last n, try to move to the next k */
        count = 0;
        for( ; k < (int)__tp->super.src->nt; n = 0) {
            for( ; n < (int)__tp->super.src->mt; n++ ) {
                if (__tp->super.src->super.myrank !=
                    ((parsec_data_collection_t*)__tp->super.src)->rank_of((parsec_data_collection_t*)__tp->super.src,
                                                                         k, n) )
                    continue;

                if( vpid != ((parsec_data_collection_t*)__tp->super.src)->vpid_of((parsec_data_collection_t*)__tp->super.src,
                                                                                 k, n) )
                    continue;
                /* Here we go, one ready local task */
                ready_list = NULL;
                es = context->virtual_processes[vpid]->execution_streams[count];
                fake_context.locals[0].value = k;
                fake_context.locals[1].value = n;
                add_task_to_list(es, &fake_context, NULL, &flow_of_map_operator_dep_out, NULL,
                                 __tp->super.src->super.myrank, -1,
                                 0, (void*)&ready_list);
                __parsec_schedule( es, ready_list, 0 );
                count++;
                if( count == context->virtual_processes[vpid]->nb_cores )
                    goto done;
                break;
            }
            /* Go to the next row ... atomically */
            k = parsec_atomic_inc_32b( &__tp->super.next_k );
        }
    done:  continue;
    }
}

/**
 * Apply the operator op on all tiles of the src matrix. The src matrix is const, the
 * result is supposed to be pushed on the dest matrix. However, any of the two matrices
 * can be NULL, and then the data is reported as NULL in the corresponding op
 * floweter.
 */
parsec_taskpool_t*
parsec_map_operator_New(const parsec_tiled_matrix_dc_t* src,
                       parsec_tiled_matrix_dc_t* dest,
                       parsec_operator_t op,
                       void* op_data)
{
    __parsec_map_operator_taskpool_t *res;

    if( (NULL == src) && (NULL == dest) )
        return NULL;
    /* src and dest should have similar distributions */
    /* TODO */
    res =  (__parsec_map_operator_taskpool_t*)calloc(1, sizeof(__parsec_map_operator_taskpool_t));
    res->super.src     = src;
    res->super.dest    = dest;
    res->super.op      = op;
    res->super.op_data = op_data;

#  if defined(PARSEC_PROF_TRACE)
    res->super.super.profiling_array = parsec_map_operator_profiling_array;
    if( -1 == parsec_map_operator_profiling_array[0] ) {
        parsec_profiling_add_dictionary_keyword("operator", "fill:CC2828",
                                               sizeof(parsec_profile_data_collection_info_t), PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR,
                                               (int*)&res->super.super.profiling_array[0 + 2 * parsec_map_operator.task_class_id],
                                               (int*)&res->super.super.profiling_array[1 + 2 * parsec_map_operator.task_class_id]);
    }
#  endif /* defined(PARSEC_PROF_TRACE) */

    res->super.super.taskpool_id = 1111;
    res->super.super.nb_tasks = src->nb_local_tiles;
    res->super.super.nb_pending_actions = 1;  /* for all local tasks */
    res->super.super.startup_hook = parsec_map_operator_startup_fn;
    (void)parsec_taskpool_reserve_id((parsec_taskpool_t *)res);
    return (parsec_taskpool_t*)res;
}

void parsec_map_operator_Destruct( parsec_taskpool_t* o )
{
    free(o);
}
