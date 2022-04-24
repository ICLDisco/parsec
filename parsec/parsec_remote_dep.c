#include "parsec/parsec_config.h"

#include <mpi.h>
#include "profiling.h"
#include "parsec/class/list.h"
#include "parsec/utils/output.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/debug.h"
#include "parsec/debug_marks.h"
#include "parsec/data.h"
#include "parsec/papi_sde.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/remote_dep.h"
#include "parsec/class/dequeue.h"

#include "parsec/parsec_binary_profile.h"

#define PARSEC_DTD_SKIP_SAVING -1

int parsec_comm_gets_max  = DEP_NB_CONCURENT * MAX_PARAM_COUNT;
int parsec_comm_gets      = 0;
int parsec_comm_puts_max  = DEP_NB_CONCURENT * MAX_PARAM_COUNT;
int parsec_comm_puts      = 0;

/**
 * Number of data movements to be extracted at each step. Bigger the number
 * larger the amount spent in ordering the tasks, but greater the potential
 * benefits of doing things in the right order.
 */
static void remote_dep_mpi_params(parsec_context_t* context);
static int parsec_param_nb_tasks_extracted = 20;
/* For the meaning of aggregate, short and eager, refer to the
 * param register help text for comm_aggregate,
 * comm_short_limit and comm_eager_limit respectively.
 */
static size_t parsec_param_short_limit = RDEP_MSG_SHORT_LIMIT;
#if RDEP_MSG_EAGER_LIMIT != 0
/* Disable this by default as it is currently broken
 static size_t parsec_param_eager_limit = RDEP_MSG_EAGER_LIMIT;
*/
static size_t parsec_param_eager_limit = 0;
#endif  /* RDEP_MSG_EAGER_LIMIT != 0 */
static int parsec_param_enable_aggregate = 0;
#if defined(PARSEC_HAVE_MPI_OVERTAKE)
int parsec_param_enable_mpi_overtake = 1;
#endif

parsec_mempool_t *parsec_remote_dep_cb_data_mempool;

typedef struct remote_dep_cb_data_s {
    parsec_list_item_t        super;
    parsec_thread_mempool_t *mempool_owner;
    parsec_remote_deps_t *deps; /* always local */
    parsec_ce_mem_reg_handle_t memory_handle;
    int k;
} remote_dep_cb_data_t;

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(remote_dep_cb_data_t);

PARSEC_OBJ_CLASS_INSTANCE(remote_dep_cb_data_t, parsec_list_item_t,
                   NULL, NULL);

char*
remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin,
                         char* str,
                         size_t len)
{
    parsec_task_t task;

    task.taskpool = parsec_taskpool_lookup( origin->taskpool_id );
    if( NULL == task.taskpool ) return snprintf(str, len, "UNKNOWN_of_TASKPOOL_%d", origin->taskpool_id), str;
    task.task_class   = task.taskpool->task_classes_array[origin->task_class_id];
    if( NULL == task.task_class ) return snprintf(str, len, "UNKNOWN_of_TASKCLASS_%d", origin->task_class_id), str;
    memcpy(&task.locals, origin->locals, sizeof(parsec_assignment_t) * task.task_class->nb_locals);
    task.priority     = 0xFFFFFFFF;
    return parsec_task_snprintf(str, len, &task);
}

/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(remote_dep_wire_activate_t)
#define dep_extent dep_count
#define DEP_EAGER_BUFFER_SIZE (dep_extent+RDEP_MSG_EAGER_LIMIT)
#define datakey_dtt MPI_LONG
#define datakey_count 3

static pthread_t dep_thread_id;
parsec_dequeue_t dep_cmd_queue;
parsec_list_t    dep_cmd_fifo;             /* ordered non threaded fifo */
parsec_list_t    dep_activates_fifo;       /* ordered non threaded fifo */
parsec_list_t    dep_activates_noobj_fifo; /* non threaded fifo */
parsec_list_t    dep_put_fifo;             /* ordered non threaded fifo */

/* help manage the messages in the same category, where a category is either messages
 * to the same destination, or with the same action key.
 */
static dep_cmd_item_t** parsec_mpi_same_pos_items;
static int parsec_mpi_same_pos_items_size = 0;

static int mpi_initialized = 0;
#if defined(PARSEC_REMOTE_DEP_USE_THREADS)
static pthread_mutex_t mpi_thread_mutex;
static pthread_cond_t mpi_thread_condition;
#endif

parsec_execution_stream_t parsec_comm_es = {
    .th_id = 0,
    .core_id = -1,
    .socket_id = -1,
#if defined(PARSEC_PROF_TRACE)
    .es_profile = NULL,
#endif /* PARSEC_PROF_TRACE */
    .scheduler_object = NULL,
#if defined(PARSEC_SIM)
    .largest_simulation_date = 0,
#endif
#if defined(PARSEC_PROF_PINS)
    .pins_events_cb = {{0}},
#endif  /* defined(PARSEC_PROF_PINS) */
#if defined(PARSEC_PROF_RUSAGE_EU)
#if defined(PARSEC_HAVE_GETRUSAGE) || !defined(__bgp__)
    ._es_rusage = {{0}},
#endif /* PARSEC_HAVE_GETRUSAGE */
#endif
    .virtual_process = NULL,
    .context_mempool = NULL,
    .datarepo_mempools = {0}
};

static void remote_dep_mpi_put_start(parsec_execution_stream_t* es, dep_cmd_item_t* item);
static void remote_dep_mpi_get_start(parsec_execution_stream_t* es, parsec_remote_deps_t* deps);

static int
remote_dep_mpi_get_end_cb(parsec_comm_engine_t *ce,
                          parsec_ce_tag_t tag,
                          void *msg,
                          size_t msg_size,
                          int src,
                          void *cb_data);

static int
remote_dep_mpi_put_end_cb(parsec_comm_engine_t *ce,
                       parsec_ce_mem_reg_handle_t lreg,
                       ptrdiff_t ldispl,
                       parsec_ce_mem_reg_handle_t rreg,
                       ptrdiff_t rdispl,
                       size_t size,
                       int remote,
                       void *cb_data);

static parsec_remote_deps_t*
remote_dep_release_incoming(parsec_execution_stream_t* es,
                            parsec_remote_deps_t* origin,
                            remote_dep_datakey_t complete_mask);



static int remote_dep_nothread_send(parsec_execution_stream_t* es,
                                    dep_cmd_item_t **head_item);
static int remote_dep_ce_init(parsec_context_t* context);
static int remote_dep_ce_fini(parsec_context_t* context);

#ifdef PARSEC_PROF_TRACE
parsec_thread_profiling_t* MPIctl_prof;
parsec_thread_profiling_t* MPIsnd_prof;
parsec_thread_profiling_t* MPIrcv_prof;
int MPI_Activate_sk, MPI_Activate_ek;
int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
int MPI_Data_plds_sk, MPI_Data_plds_ek;
int MPI_Data_pldr_sk, MPI_Data_pldr_ek;
int activate_cb_trace_sk, activate_cb_trace_ek;
int put_cb_trace_sk, put_cb_trace_ek;

/**
 * The structure describe the MPI events saves into the profiling stream. The following
 * string represent it's description so that an external package can decrypt the
 * binary format of the stream.
 */

static char parsec_profile_remote_dep_mpi_info_to_string[] = "src{int32_t};dst{int32_t};tid{int64_t};tpid{int32_t};tcid{int32_t}";

static void remote_dep_mpi_profiling_init(void)
{
    parsec_profiling_add_dictionary_keyword( "MPI_ACTIVATE", "fill:#FF0000",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Activate_sk, &MPI_Activate_ek);
    parsec_profiling_add_dictionary_keyword( "MPI_DATA_CTL", "fill:#000077",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_ctl_sk, &MPI_Data_ctl_ek);
    parsec_profiling_add_dictionary_keyword( "MPI_DATA_PLD_SND", "fill:#B08080",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_plds_sk, &MPI_Data_plds_ek);
    parsec_profiling_add_dictionary_keyword( "MPI_DATA_PLD_RCV", "fill:#80B080",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_pldr_sk, &MPI_Data_pldr_ek);

    parsec_profiling_add_dictionary_keyword( "ACTIVATE_CB", "fill:#FF0000",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &activate_cb_trace_sk, &activate_cb_trace_ek);
    parsec_profiling_add_dictionary_keyword( "PUT_CB", "fill:#FF0000",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &put_cb_trace_sk, &put_cb_trace_ek);

    MPIctl_prof = parsec_profiling_thread_init( 2*1024*1024, "MPI ctl");
    MPIsnd_prof = parsec_profiling_thread_init( 2*1024*1024, "MPI isend");
    MPIrcv_prof = parsec_profiling_thread_init( 2*1024*1024, "MPI irecv");
    parsec_comm_es.es_profile = MPIctl_prof;
}

static void remote_dep_mpi_profiling_fini(void)
{
    /* Nothing to do, the thread_profiling structures will be automatically
     * released when the master profiling system is shut down.
     */
    MPIsnd_prof = NULL;
    MPIrcv_prof = NULL;
    MPIctl_prof = NULL;
}
#else

#define remote_dep_mpi_profiling_init() do {} while(0)
#define remote_dep_mpi_profiling_fini() do {} while(0)

#endif  /* PARSEC_PROF_TRACE */

/**
 * Store the user provided communicator in the PaRSEC context. We need to make a
 * copy to make sure the communicator does not dissapear before the communication
 * engine starts up.
 */
int remote_dep_set_ctx(parsec_context_t* context, intptr_t opaque_comm_ctx )
{
    MPI_Comm comm;
    int rc;

    /* We can only change the communicator if the communication engine is not active */
    if( 1 < parsec_communication_engine_up ) {
        parsec_warning("Cannot change PaRSEC's MPI communicator while the engine is running [ignored]");
        return PARSEC_ERROR;
    }
    /* Are we trying to set a congruent communicator a second time? */
    assert(-1 != opaque_comm_ctx /* -1 reserved for non-initialized */);
    if( -1 != context->comm_ctx ) {
        MPI_Comm_compare((MPI_Comm)context->comm_ctx, (MPI_Comm)opaque_comm_ctx, &rc);
        if( (MPI_IDENT == rc) || (MPI_CONGRUENT == rc) ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "Set the same or a congruent communicator. Nothing to do");
            return PARSEC_SUCCESS;
        }
        MPI_Comm_free((MPI_Comm*)&context->comm_ctx);
    }
    rc = MPI_Comm_dup((MPI_Comm)opaque_comm_ctx, &comm);
    context->comm_ctx = (intptr_t)comm;
    return (MPI_SUCCESS == rc) ? PARSEC_SUCCESS : PARSEC_ERROR;
}

static void remote_dep_mpi_params(parsec_context_t* context) {
    (void)context;
#if RDEP_MSG_SHORT_LIMIT != 0
    parsec_mca_param_reg_sizet_name("runtime", "comm_short_limit", "Controls the maximum size of a short message. Short messages contain both the control message notifying the completion of a task and the associated data that fit completely in that buffer length. The maximum size of a short message should be lower than the network MTU.",
                                  false, false, parsec_param_short_limit, &parsec_param_short_limit);
    if(parsec_param_short_limit > RDEP_MSG_SHORT_LIMIT) {
        parsec_warning("User requested a short message limit of %d which is greater than compiled in limit %d; value reset to compiled limit %d", parsec_param_short_limit, RDEP_MSG_SHORT_LIMIT, RDEP_MSG_SHORT_LIMIT);
        parsec_param_short_limit = RDEP_MSG_SHORT_LIMIT;
    }
#endif
#if RDEP_MSG_EAGER_LIMIT != 0
    parsec_mca_param_reg_sizet_name("runtime", "comm_eager_limit", "Controls the maximum size of a message that uses the eager protocol. Eager messages are sent eagerly before a 2-sided synchronization and may cause flow control and memory contentions at the receiver, but have a better latency.",
                                  false, false, parsec_param_eager_limit, &parsec_param_eager_limit);
#endif
    parsec_mca_param_reg_int_name("runtime", "comm_aggregate", "Aggregate multiple dependencies in the same short message (1=true,0=false).",
                                  false, false, parsec_param_enable_aggregate, &parsec_param_enable_aggregate);
}

void
remote_dep_mpi_initialize_execution_stream(parsec_context_t *context)
{
    memcpy(&parsec_comm_es, context->virtual_processes[0]->execution_streams[0], sizeof(parsec_execution_stream_t));
}

int remote_dep_dequeue_new_taskpool(parsec_taskpool_t* tp)
{
    if(!mpi_initialized) return 0;
    remote_dep_inc_flying_messages(tp);
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_NEW_TASKPOOL;
    item->priority = 0;
    item->cmd.new_taskpool.tp = tp;
    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*)item);
    return 1;
}

int
remote_dep_dequeue_delayed_dep_release(parsec_remote_deps_t *deps)
{
    if(!mpi_initialized) return 0;
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_DTD_DELAYED_RELEASE;
    item->priority = 0;
    item->cmd.release.deps = deps;
    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*)item);
    return 1;
}

int
remote_dep_dequeue_send(parsec_execution_stream_t* es, int rank,
                        parsec_remote_deps_t* deps)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action   = DEP_ACTIVATE;
    item->priority = deps->max_priority;
    item->cmd.activate.peer             = rank;
    item->cmd.activate.task.source_deps = (remote_dep_datakey_t)deps;
    item->cmd.activate.task.output_mask = 0;
    item->cmd.activate.task.callback_fn = 0;
    item->cmd.activate.task.remote_memory_handle = NULL; /* we don't have it yet */
    item->cmd.activate.task.remote_callback_data = (remote_dep_datakey_t)NULL;

    /* if MPI is multithreaded do not thread-shift the send activate */
    if( parsec_comm_es.virtual_process->parsec_context->flags & PARSEC_CONTEXT_FLAG_COMM_MT ) {
        parsec_list_item_singleton(&item->pos_list); /* NOTE: this disables aggregation in MT cases. */
        remote_dep_nothread_send(es, &item);
    }
    else {
        parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*)item);
    }
    return 1;
}

void parsec_remote_dep_memcpy(parsec_execution_stream_t* es,
                              parsec_taskpool_t* tp,
                              parsec_data_copy_t *dst,
                              parsec_data_copy_t *src,
                              parsec_dep_data_description_t* data)
{
    assert( dst );
    /* if the communication engine supports multithreads do the reshaping in place */
    if( parsec_ce.parsec_context->flags & PARSEC_CONTEXT_FLAG_COMM_MT ) {
        if( parsec_ce.reshape(&parsec_ce, es, dst, src,
                              data->layout, data->displ, 0, data->count) ) {
            return;
        }
    }

    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream,
                         "create MEMCPY request from %p to %p count %d",
                         PARSEC_DATA_COPY_GET_PTR(src) + data->displ,
                         PARSEC_DATA_COPY_GET_PTR(dst) + 0,
                         data->count);
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_MEMCPY;
    item->priority = 0;
    item->cmd.memcpy.taskpool     = tp;
    item->cmd.memcpy.source       = src;
    item->cmd.memcpy.destination  = dst;
    item->cmd.memcpy.layout       = data->layout;
    item->cmd.memcpy.displ_src    = data->displ;
    item->cmd.memcpy.displ_dst    = 0;
    item->cmd.memcpy.count        = data->count;

    PARSEC_OBJ_RETAIN(src);
    remote_dep_inc_flying_messages(tp);

    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*) item);
}

static inline parsec_data_copy_t*
remote_dep_copy_allocate(parsec_dep_data_description_t* data)
{
    parsec_data_copy_t* dc;
    if( NULL == data->arena ) {
        assert(0 == data->count);
        return NULL;
    }
    dc = parsec_arena_get_copy(data->arena, data->count, 0);
    dc->coherency_state = PARSEC_DATA_COHERENCY_EXCLUSIVE;
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tMalloc new remote tile %p size %" PRIu64 " count = %" PRIu64 " displ = %" PRIi64 "",
            dc, data->arena->elem_size, data->count, data->displ);
    return dc;
}
#define is_inplace(ctx,dep) NULL
#define is_read_only(ctx,dep) NULL

/**
 * This function is called from the task successors iterator. It exists for a
 * single purpose: to retrieve the datatype involved with the operation. Thus,
 * once a datatype has been succesfully retrieved it must cancel the iterator
 * progress in order to return ASAP the datatype to the communication engine.
 */
parsec_ontask_iterate_t
remote_dep_mpi_retrieve_datatype(parsec_execution_stream_t *eu,
                                 const parsec_task_t *newcontext,
                                 const parsec_task_t *oldcontext,
                                 const parsec_dep_t* dep,
                                 parsec_dep_data_description_t* out_data,
                                 int src_rank, int dst_rank, int dst_vpid,
                                 void *param)
{
    (void)eu; (void)oldcontext; (void)dst_vpid; (void)newcontext; (void)out_data;
    if( dst_rank != eu->virtual_process->parsec_context->my_rank )
        return PARSEC_ITERATE_CONTINUE;

    parsec_remote_deps_t *deps               = (parsec_remote_deps_t*)param;
    struct remote_dep_output_param_s* output = &deps->output[dep->dep_datatype_index];
    const parsec_task_class_t* fct           = newcontext->task_class;
    uint32_t flow_mask                       = (1U << dep->flow->flow_index) | 0x80000000;  /* in flow */
    /* Extract the datatype, count and displacement from the target task */
    if( PARSEC_HOOK_RETURN_DONE == fct->get_datatype(eu, newcontext, &flow_mask, &output->data) ) {
        /* something is wrong, we are unable to extract the expected datatype
         from the receiver task. At this point it is difficult to stop the
         algorithm, so let's assume the send datatype is to be used instead.*/
        output->data = *out_data;
    }

    parsec_data_t* data_arena = is_read_only(oldcontext, dep);
    if(NULL == data_arena) {
        output->deps_mask &= ~(1U << dep->dep_index); /* unmark all data that are RO we already hold from previous tasks */
    } else {
        output->deps_mask |= (1U << dep->dep_index); /* mark all data that are not RO */
        data_arena = is_inplace(oldcontext, dep);  /* Can we do it inplace */
    }
    output->data.data = NULL;

    if( deps->max_priority < newcontext->priority ) deps->max_priority = newcontext->priority;
    deps->incoming_mask |= (1U << dep->dep_datatype_index);
    deps->root           = src_rank;
    return PARSEC_ITERATE_STOP;
}

/**
 * Retrieve the datatypes involved in this communication. In addition the flag
 * PARSEC_ACTION_RECV_INIT_REMOTE_DEPS set the priority to the maximum priority
 * of all the children.
 */
static int
remote_dep_get_datatypes(parsec_execution_stream_t* es,
                         parsec_remote_deps_t* origin,
                         int storage_id, int *position)
{
    parsec_task_t task;
    uint32_t i, j, k, local_mask = 0;

    assert(NULL == origin->taskpool);
    origin->taskpool = parsec_taskpool_lookup(origin->msg.taskpool_id);
    if( NULL == origin->taskpool )
        return -1; /* the parsec taskpool doesn't exist yet */

    task.taskpool   = origin->taskpool;
    /* Do not set the task.task_class here, because it might trigger a race condition in DTD */

    task.priority = 0;  /* unknown yet */

    /* This function is divided into DTD and PTG's logic */
    if( PARSEC_TASKPOOL_TYPE_DTD == origin->taskpool->taskpool_type ) {
        parsec_dtd_taskpool_t *dtd_tp = NULL;
        parsec_dtd_task_t *dtd_task = NULL;

        dtd_tp = (parsec_dtd_taskpool_t *)origin->taskpool;

        /* if( NULL == task.task_class ), this case will be taken care of automatically */

        /* We need to convert from a dep_datatype_index mask into a dep_index
         * mask. However, in order to be able to use the above iterator we need to
         * be able to identify the dep_index for each particular datatype index, and
         * call the iterate_successors on each of the dep_index sets.
         */
        int return_defer = 0;
        for(k = 0; origin->msg.output_mask>>k; k++) {
            if(!(origin->msg.output_mask & (1U<<k))) continue;

            uint64_t key = (uint64_t)origin->msg.locals[0].value<<32 | (1U<<k);
            local_mask = 0;
            local_mask |= (1U<<k);

            parsec_hash_table_lock_bucket(dtd_tp->task_hash_table, (parsec_key_t)key);
            dtd_task = parsec_dtd_find_task( dtd_tp, key );

            if( NULL == dtd_task ) {
                return_defer = 1;

                /* AM buffers are reused by the comm engine once the activation
                 * has been conveyed to upper layer. In case of DTD we might receive msg to
                 * activate a task that the local node (the recipient of the activation)
                 * have not discovered yet. In that case we need to store the buffer,
                 * but note, we only need to store it the first time we are receiving this
                 * activation. PARSEC_DTD_SKIP_SAVING indicates whether this is the first
                 * time or not. Since, this function is called from other places (when
                 * we later try to activate a task for which we have already received
                 * an activation for) we do not need to store the buffer and we send
                 * PARSEC_DTD_SKIP_SAVING as an indicaton of that.
                 */
                if( storage_id != PARSEC_DTD_SKIP_SAVING) {
                    char* packed_buffer;
                    /* Copy the eager data to some temp storage */
                    packed_buffer = malloc(origin->msg.length);
                    memcpy(packed_buffer, origin->eager_msg + *position, origin->msg.length);
                    *position += origin->msg.length;  /* move to the next order */
                    origin->taskpool = (parsec_taskpool_t*)packed_buffer;  /* temporary storage */
                }
                parsec_dtd_track_remote_dep( dtd_tp, key, origin );
            }

            parsec_hash_table_unlock_bucket(dtd_tp->task_hash_table, (parsec_key_t)key);

            if(return_defer) {
                return -2;
            }

            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tRetrieve datatype with mask 0x%x (remote_dep_get_datatypes)", local_mask);
            task.locals[k] = origin->msg.locals[k];
            task.task_class = dtd_task->super.task_class;
            origin->msg.task_class_id = dtd_task->super.task_class->task_class_id;
            task.task_class->iterate_successors(es, (parsec_task_t *)dtd_task,
                                               local_mask,
                                               remote_dep_mpi_retrieve_datatype,
                                               origin);
        }
    } else {
        task.task_class = task.taskpool->task_classes_array[origin->msg.task_class_id];

        for(i = 0; i < task.task_class->nb_locals; i++)
            task.locals[i] = origin->msg.locals[i];

        /* We need to convert from a dep_datatype_index mask into a dep_index
         * mask. However, in order to be able to use the above iterator we need to
         * be able to identify the dep_index for each particular datatype index, and
         * call the iterate_successors on each of the dep_index sets.
         */
        for(k = 0; origin->msg.output_mask>>k; k++) {
            if(!(origin->msg.output_mask & (1U<<k))) continue;
            for(local_mask = i = 0; NULL != task.task_class->out[i]; i++ ) {
                if(!(task.task_class->out[i]->flow_datatype_mask & (1U<<k))) continue;
                for(j = 0; NULL != task.task_class->out[i]->dep_out[j]; j++ )
                    if(k == task.task_class->out[i]->dep_out[j]->dep_datatype_index)
                        local_mask |= (1U << task.task_class->out[i]->dep_out[j]->dep_index);
                if( 0 != local_mask ) break;  /* we have our local mask, go get the datatype */
            }

            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tRetrieve datatype with mask 0x%x (remote_dep_get_datatypes)", local_mask);
            task.task_class->iterate_successors(es, &task,
                                                local_mask,
                                                remote_dep_mpi_retrieve_datatype,
                                                origin);
        }
    }

    /**
     * At this point the msg->output_mask contains the root mask, and should be
     * keep as is and be propagated down the communication pattern. On the
     * origin->incoming_mask we have the mask of all local data to be retrieved
     * from the predecessor.
     */
    origin->outgoing_mask = origin->incoming_mask;  /* safekeeper */
    return 0;
}

/**
 * An activation message has been received, and the remote_dep_wire_activate_t
 * part has already been extracted into the deps->msg. This function handles the
 * rest of the receiver logic, extract the possible eager and control data from
 * the buffer, post all the short protocol receives and all other local
 * cleanups.
 */
static void remote_dep_mpi_recv_activate(parsec_execution_stream_t* es,
                                         parsec_remote_deps_t* deps,
                                         char* packed_buffer,
                                         int length,
                                         int* position)
{
    (void) length; (void) position;
    (void) packed_buffer;
    remote_dep_datakey_t complete_mask = 0;
    int k;
#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN);
#endif

#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    parsec_debug_verbose(6, parsec_debug_output, "MPI:\tFROM\t%d\tActivate\t% -8s\n"
          "\twith datakey %lx\tparams %lx length %d (pack buf %d/%d) prio %d",
           deps->from, tmp, deps->msg.deps, deps->incoming_mask,
           deps->msg.length, *position, length, deps->max_priority);
#endif
    for(k = 0; deps->incoming_mask>>k; k++) {
        if(!(deps->incoming_mask & (1U<<k))) continue;
        /* Check for CTL and data that do not carry payload */
        if((NULL == deps->output[k].data.arena) || (0 == deps->output[k].data.count)) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MPI:\tHERE\t%d\tGet NONE\t% -8s\tk=%d\twith datakey %lx at <NA> type CONTROL",
                    deps->from, tmp, k, deps->msg.deps);
            deps->output[k].data.data = NULL;
            complete_mask |= (1U<<k);
            continue;
        }
    }
    assert(length == *position);

    /* Release all the already satisfied deps without posting the RDV */
    if(complete_mask) {
#if defined(PARSEC_DEBUG_NOISIER)
        for(int k = 0; complete_mask>>k; k++)
            if((1U<<k) & complete_mask)
                PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MPI:\tHERE\t%d\tGet PREEND\t% -8s\tk=%d\twith datakey %lx at %p ALREADY SATISFIED\t(tag=%d)",
                        deps->from, tmp, k, deps->msg.deps, deps->output[k].data.data, k );
#endif
        /* If this is the only call then force the remote deps propagation */
        deps = remote_dep_release_incoming(es, deps, complete_mask);
    }

    /* Store the request in the rdv queue if any unsatisfied dep exist at this point */
    if(NULL != deps) {
        assert(0 != deps->incoming_mask);
        assert(0 != deps->msg.output_mask);
        parsec_list_nolock_push_sorted(&dep_activates_fifo, (parsec_list_item_t*)deps, rdep_prio);
    }

    /* Check if we have any pending GET orders */
    if(parsec_ce.can_serve(&parsec_ce) && !parsec_list_nolock_is_empty(&dep_activates_fifo)) {
        deps = (parsec_remote_deps_t*)parsec_list_nolock_pop_front(&dep_activates_fifo);
        remote_dep_mpi_get_start(es, deps);
    }
}

static int
remote_dep_mpi_save_activate_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
                                void *msg, size_t msg_size, int src,
                                void *cb_data)
{
    (void) tag; (void) cb_data;
    parsec_execution_stream_t* es = &parsec_comm_es;

    PARSEC_PINS(es, ACTIVATE_CB_BEGIN, NULL);
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    int position = 0, length = msg_size, rc;
    parsec_remote_deps_t* deps = NULL;

    while(position < length) {
        deps = remote_deps_allocate(&parsec_remote_dep_context.freelist);

        ce->unpack(ce, msg, length, &position, &deps->msg, dep_count);
        deps->from = src;
        deps->eager_msg = msg;

        /* Retrieve the data arenas and update the msg.incoming_mask to reflect
         * the data we should be receiving from the predecessor.
         */
        rc = remote_dep_get_datatypes(es, deps, 0, &position);

        if( -1 == rc ) {
            /* the corresponding tp doesn't exist, yet. Put it in unexpected */
            char* packed_buffer;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MPI:\tFROM\t%d\tActivate NoTPool\t% -8s\tk=%d\twith datakey %lx\tparams %lx",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    0, deps->msg.deps, deps->msg.output_mask);
            /* Copy the eager data to some temp storage */
            packed_buffer = malloc(deps->msg.length);
            memcpy(packed_buffer, msg + position, deps->msg.length);
            position += deps->msg.length;  /* move to the next order */
            deps->taskpool = (parsec_taskpool_t*)packed_buffer;  /* temporary storage */
            parsec_list_nolock_push_back(&dep_activates_noobj_fifo, (parsec_list_item_t*)deps);
            continue;
        } else {
            assert(deps->taskpool != NULL);
            if( -2 == rc ) { /* DTD problems, defer activating this remote dep */
                assert(deps->incoming_mask != deps->msg.output_mask);
                continue;
            }
        }

        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "MPI:\tFROM\t%d\tActivate\t% -8s\tk=%d\twith datakey %lx\tparams %lx",
               src, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
               0, deps->msg.deps, deps->msg.output_mask);
        /* Import the activation message and prepare for the reception */
        remote_dep_mpi_recv_activate(es, deps, msg,
                                     position + deps->msg.length, &position);
        assert( parsec_param_enable_aggregate || (position == length));
    }
    assert(position == length);
    PARSEC_PINS(es, ACTIVATE_CB_END, NULL);
    return 1;
}

static int
remote_dep_mpi_save_put_cb(parsec_comm_engine_t *ce,
                           parsec_ce_tag_t tag,
                           void *msg,
                           size_t msg_size,
                           int src,
                           void *cb_data)
{
    (void) ce; (void) tag; (void) cb_data; (void) msg_size;
    remote_dep_wire_get_t* task;
    parsec_remote_deps_t *deps;
    dep_cmd_item_t* item;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    parsec_execution_stream_t* es = &parsec_comm_es;

    item = (dep_cmd_item_t*) malloc(sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(&item->super, parsec_list_item_t);
    item->action = DEP_GET_DATA;
    item->cmd.activate.peer = src;

    task = &(item->cmd.activate.task);
    /* copy the static part of the message, the part after this contains the memory_handle
     * of the other side.
     */
    memcpy(task, msg, sizeof(remote_dep_wire_get_t));

    /* we are expecting exactly one wire_get_t + remote memory handle */
    assert(msg_size == sizeof(remote_dep_wire_get_t) + ce->get_mem_handle_size());

    item->cmd.activate.remote_memory_handle = malloc(ce->get_mem_handle_size());
    memcpy( item->cmd.activate.remote_memory_handle,
            ((char*)msg) + sizeof(remote_dep_wire_get_t),
            ce->get_mem_handle_size() );

    deps = (parsec_remote_deps_t*)(remote_dep_datakey_t)task->source_deps; /* get our deps back */
    assert(0 != deps->pending_ack);
    assert(0 != deps->outgoing_mask);
    item->priority = deps->max_priority;

    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MPI: Put cb_received for %s from %d tag %u which 0x%x (deps %p)",
                remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), item->cmd.activate.peer,
                -1, task->output_mask, (void*)deps);

    /* Get the highest priority PUT operation */
    parsec_list_nolock_push_sorted(&dep_put_fifo, (parsec_list_item_t*)item, dep_cmd_prio);
    if( parsec_ce.can_serve(&parsec_ce) ) {
        item = (dep_cmd_item_t*)parsec_list_nolock_pop_front(&dep_put_fifo);
        remote_dep_mpi_put_start(es, item);
    } else {
        PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MPI: Put DELAYED for %s from %d tag %u which 0x%x (deps %p)",
                remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), item->cmd.activate.peer,
                -1, task->output_mask, (void*)deps);
    }
    return 1;
}

int
remote_dep_dequeue_init(parsec_context_t* context)
{
    pthread_attr_t thread_attr;
    int is_mpi_up = 0;
    int thread_level_support;

    assert(mpi_initialized == 0);

    remote_dep_mpi_params(context);

    MPI_Initialized(&is_mpi_up);
    if( 0 == is_mpi_up ) {
        /**
         * MPI is not up, so we will consider this as a single node run. Fall
         * back to the no-MPI case.
         */
        context->nb_nodes = 1;
        parsec_communication_engine_up = -1;  /* No communications supported */
        /*TODO: restore the original behavior when modular datatype engine is
         * available */
        parsec_fatal("MPI was not initialized. This version of PaRSEC was compiled with MPI datatype supports and *needs* MPI to execute.\n"
                     "\t* Please initialized MPI in the application (MPI_Init/MPI_Init_thread) prior to initializing PaRSEC.\n"
                     "\t* Alternatively, compile a version of PaRSEC without MPI (-DPARSEC_DIST_WITH_MPI=OFF in ccmake)\n");
        return 1;
    }
    parsec_communication_engine_up = 0;  /* we have communication capabilities */

    MPI_Query_thread( &thread_level_support );
    if( thread_level_support == MPI_THREAD_SINGLE ||
        thread_level_support == MPI_THREAD_FUNNELED ) {
        parsec_warning("MPI was not initialized with the appropriate level of thread support.\n"
                      "\t* Current level is %s, while MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE is needed\n"
                      "\t* to guarantee correctness of the PaRSEC runtime.\n",
                thread_level_support == MPI_THREAD_SINGLE ? "MPI_THREAD_SINGLE" : "MPI_THREAD_FUNNELED" );
    }
    if( -1 == context->comm_ctx ) {
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        context->comm_ctx = (intptr_t)comm;
        assert(-1 != context->comm_ctx /* -1 reserved for non-initialized */);
    }
    MPI_Comm_size( (MPI_Comm)context->comm_ctx, (int*)&(context->nb_nodes));

    if(parsec_param_comm_thread_multiple) {
        if( thread_level_support >= MPI_THREAD_MULTIPLE ) {
            context->flags |= PARSEC_CONTEXT_FLAG_COMM_MT;
        }
        else if(parsec_param_comm_thread_multiple != -1) {
            parsec_warning("Requested multithreaded access to the communication engine, but MPI is not initialized with MPI_THREAD_MULTIPLE.\n"
                        "\t* PaRSEC will continue with the funneled thread communication engine model.\n");
        }
#if RDEP_MSG_EAGER_LIMIT != 0
        if( (context->flags & PARSEC_CONTEXT_FLAG_COMM_MT) && parsec_param_eager_limit ) {
            parsec_warning("Using eager and thread multiple MPI messaging is not implemented yet. Disabling Eager.");
            parsec_param_eager_limit = 0;
        }
#endif
    }
#if defined(PARSEC_HAVE_MPI_OVERTAKE)
    parsec_mca_param_reg_int_name("runtime", "comm_mpi_overtake", "Lets MPI allow overtaking of messages (if applicable). (0: no, 1: yes)",
                                  false, false, parsec_param_enable_mpi_overtake, &parsec_param_enable_mpi_overtake);
#endif

    /**
     * Finalize the initialization of the upper level structures
     * Worst case: one of the DAGs is going to use up to
     * MAX_PARAM_COUNT times nb_nodes dependencies.
     */
    remote_deps_allocation_init(context->nb_nodes, MAX_PARAM_COUNT);

    PARSEC_OBJ_CONSTRUCT(&dep_cmd_queue, parsec_dequeue_t);
    PARSEC_OBJ_CONSTRUCT(&dep_cmd_fifo, parsec_list_t);

    /* From now on the communication capabilities are enabled */
    parsec_communication_engine_up = 1;
    if(context->nb_nodes == 1) {
        /* We're all by ourselves. In case we need to use comm engin to handle data copies
         * between different formats let's setup it up.
         */
        remote_dep_ce_init(context);

        goto up_and_running;
    }

    /* Build the condition used to drive the MPI thread */
    pthread_mutex_init( &mpi_thread_mutex, NULL );
    pthread_cond_init( &mpi_thread_condition, NULL );

    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);

   /**
    * We need to synchronize with the newly spawned thread. We will use the
    * condition for this. If we lock the mutex prior to spawning the MPI thread,
    * and then go in a condition wait, the MPI thread can lock the mutex, and
    * then call condition signal. This insure proper synchronization. Similar
    * mechanism will be used to turn on and off the MPI thread.
    */
    pthread_mutex_lock(&mpi_thread_mutex);

    pthread_create(&dep_thread_id,
                   &thread_attr,
                   (void* (*)(void*))remote_dep_dequeue_main,
                   (void*)context);

    /* Wait until the MPI thread signals it's awakening */
    pthread_cond_wait( &mpi_thread_condition, &mpi_thread_mutex );
  up_and_running:
    mpi_initialized = 1;  /* up and running */

    return context->nb_nodes;
}

int
remote_dep_dequeue_fini(parsec_context_t* context)
{
    if( 0 == mpi_initialized ) return 0;
    (void)context;

    /**
     * We suppose the off function was called before. Then we will append a
     * shutdown command in the MPI thread queue, and wake the MPI thread. Upon
     * processing of the pending command the MPI thread will exit, we will be
     * able to catch this by locking the mutex.  Once we know the MPI thread is
     * gone, cleaning up will be straighforward.
     */
    if( 1 < parsec_communication_engine_up ) {
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
        void *ret;

        item->action = DEP_CTL;
        item->cmd.ctl.enable = -1;  /* turn off and return from the MPI thread */
        item->priority = 0;
        parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*) item);

        /* I am supposed to own the lock. Wake the MPI thread */
        pthread_cond_signal(&mpi_thread_condition);
        pthread_mutex_unlock(&mpi_thread_mutex);
        pthread_join(dep_thread_id, &ret);
        assert((parsec_context_t*)ret == context);
    }
    else if ( parsec_communication_engine_up == 1 ) {
        remote_dep_ce_fini(context);
    }

    assert(NULL == parsec_dequeue_pop_front(&dep_cmd_queue));
    PARSEC_OBJ_DESTRUCT(&dep_cmd_queue);
    assert(NULL == parsec_dequeue_pop_front(&dep_cmd_fifo));
    PARSEC_OBJ_DESTRUCT(&dep_cmd_fifo);
    mpi_initialized = 0;

    return 0;
}

static int
remote_dep_ce_init(parsec_context_t* context)
{
    int rc;
    /* Do this first to give a chance to the communication engine to define
     * who this process is by setting the corresponding info in the
     * parsec_context.
     */
    if( NULL == parsec_comm_engine_init(context) ) {
        parsec_warning("Communication engine failed to start. Additional information might be available in the corresponding error message");
        return PARSEC_ERR_NOT_FOUND;
    }

    PARSEC_OBJ_CONSTRUCT(&dep_activates_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&dep_activates_noobj_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&dep_put_fifo, parsec_list_t);

    parsec_mpi_same_pos_items_size = context->nb_nodes + (int)DEP_LAST;
    parsec_mpi_same_pos_items = (dep_cmd_item_t**)calloc(parsec_mpi_same_pos_items_size,
                                                        sizeof(dep_cmd_item_t*));

    /* Register Persistant requests */
    rc = parsec_ce.tag_register(REMOTE_DEP_ACTIVATE_TAG, remote_dep_mpi_save_activate_cb, context,
                                DEP_EAGER_BUFFER_SIZE * sizeof(char));
    if( PARSEC_SUCCESS != rc ) {
        parsec_warning("[CE] Failed to register communication tag REMOTE_DEP_ACTIVATE_TAG (error %d)\n", rc);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }
    rc = parsec_ce.tag_register(REMOTE_DEP_GET_DATA_TAG, remote_dep_mpi_save_put_cb, context,
                                4096);
    if( PARSEC_SUCCESS != rc ) {
        parsec_warning("[CE] Failed to register communication tag REMOTE_DEP_GET_DATA_TAG (error %d)\n", rc);
        parsec_ce.tag_unregister(REMOTE_DEP_ACTIVATE_TAG);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }

    parsec_remote_dep_cb_data_mempool = (parsec_mempool_t*) malloc (sizeof(parsec_mempool_t));
    parsec_mempool_construct(parsec_remote_dep_cb_data_mempool,
                             PARSEC_OBJ_CLASS(remote_dep_cb_data_t), sizeof(remote_dep_cb_data_t),
                             offsetof(remote_dep_cb_data_t, mempool_owner),
                             1);

    remote_dep_mpi_profiling_init();
    return 0;
}

static int
remote_dep_ce_fini(parsec_context_t* context)
{
    remote_dep_mpi_profiling_fini();

    // Unregister tags
    parsec_ce.tag_unregister(REMOTE_DEP_ACTIVATE_TAG);
    parsec_ce.tag_unregister(REMOTE_DEP_GET_DATA_TAG);
    //parsec_ce.tag_unregister(REMOTE_DEP_PUT_END_TAG);

    parsec_mempool_destruct(parsec_remote_dep_cb_data_mempool);
    free(parsec_remote_dep_cb_data_mempool);

    free(parsec_mpi_same_pos_items); parsec_mpi_same_pos_items = NULL;
    parsec_mpi_same_pos_items_size = 0;

    PARSEC_OBJ_DESTRUCT(&dep_activates_fifo);
    PARSEC_OBJ_DESTRUCT(&dep_activates_noobj_fifo);
    PARSEC_OBJ_DESTRUCT(&dep_put_fifo);

    parsec_comm_engine_fini(&parsec_ce);

    (void)context;
    return 0;
}

/* The possible values for parsec_communication_engine_up are: 0 if no
 * communication capabilities are enabled, 1 if we are in a single node scenario
 * and the main thread will check the communications on a regular basis, 2 if
 * the order is enqueued but the thread is not yet on, and 3 if the thread is
 * running.
 */
int
remote_dep_dequeue_on(parsec_context_t* context)
{
    /* If we are the only participant in this execution, we should not have to
     * communicate with any other process. However, we might have to execute all
     * local data copies, which requires MPI.
     */
    if( 0 >= parsec_communication_engine_up ) return -1;
    if( context->nb_nodes == 1 ) return 1;

    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine signalled ON on process %d/%d",
                         context->my_rank, context->nb_nodes);

    /* At this point I am supposed to own the mutex */
    parsec_communication_engine_up = 2;
    pthread_cond_signal(&mpi_thread_condition);
    pthread_mutex_unlock(&mpi_thread_mutex);

    /* The waking up of the communication thread happen asynchronously, once the thread
     * receives the signal. At that point it acquires the mpi_thread_mutex and set the
     * global variable parsec_communication_engine_up to 3.
     */

    /**
     * We need to wait for the communication thread to perform the mpi_setup
     * as it will fill-up my_rank on the context.
     */
    while( 3 != parsec_communication_engine_up ) sched_yield();

    (void)context;
    return 1;
}

int
remote_dep_dequeue_off(parsec_context_t* context)
{
    if(parsec_communication_engine_up < 2) return -1;  /* The start order has not been issued */

    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_CTL;
    item->cmd.ctl.enable = 0;  /* turn OFF the MPI thread */
    item->priority = 0;
    /* wait until the communication thread is up and running */
    while( 3 != parsec_communication_engine_up ) sched_yield();
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine signalled OFF on process %d/%d",
                         context->my_rank, context->nb_nodes);
    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*) item);

    /* wait until we own the PaRSEC MPI synchronization mutex */
    pthread_mutex_lock(&mpi_thread_mutex);
    assert( 1 == parsec_communication_engine_up );
    (void)context;  /* silence warning */
    return 0;
}

static int remote_dep_mpi_on(parsec_context_t* context)
{
#ifdef PARSEC_PROF_TRACE
    /* put a start marker for each type of event */
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, 0);
    TAKE_TIME(MPIsnd_prof, MPI_Activate_sk, 0);
    TAKE_TIME(MPIrcv_prof, MPI_Activate_sk, 0);
    parsec_ce.sync(&parsec_ce);   /* TODO GB: do we still need this here ? */
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, 0);
    TAKE_TIME(MPIsnd_prof, MPI_Activate_ek, 0);
    TAKE_TIME(MPIrcv_prof, MPI_Activate_ek, 0);
#endif
    (void)context;
    return 0;
}

void* remote_dep_dequeue_main(parsec_context_t* context)
{
    int whatsup;

    remote_dep_bind_thread(context);
    PARSEC_PAPI_SDE_THREAD_INIT();

    remote_dep_ce_init(context);

    /* Now synchronize with the main thread */
    pthread_mutex_lock(&mpi_thread_mutex);
    pthread_cond_signal(&mpi_thread_condition);

    /* This is the main loop. Wait until being woken up by the main thread, do
     * the MPI stuff until we get the OFF or FINI commands. Then react the them.
     * However, the first time do the delayed initialization that could not have
     * been done before due to the lack of other component initialization.
     */

    /* Let's wait until we are awaken */
    pthread_cond_wait(&mpi_thread_condition, &mpi_thread_mutex);
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine ON on process %d/%d",
                         context->my_rank, context->nb_nodes);
    /* The MPI thread is owning the lock */
    assert( parsec_communication_engine_up == 2 );

    /* Lazy or delayed initializations */
    remote_dep_mpi_initialize_execution_stream(context);

    remote_dep_mpi_on(context);
    /* acknoledge the activation */
    parsec_communication_engine_up = 3;
    whatsup = remote_dep_dequeue_nothread_progress(&parsec_comm_es, -1 /* loop till explicitly asked to return */);
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine OFF on process %d/%d",
                         context->my_rank, context->nb_nodes);
    parsec_communication_engine_up = 1;  /* went to sleep */

    while( -1 != whatsup ) {
        /* Let's wait until we are awaken */
        pthread_cond_wait(&mpi_thread_condition, &mpi_thread_mutex);
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine ON on process %d/%d",
                             context->my_rank, context->nb_nodes);
        /* The MPI thread is owning the lock */
        assert( parsec_communication_engine_up == 2 );
        remote_dep_mpi_on(context);
        /* acknoledge the activation */
        parsec_communication_engine_up = 3;
        whatsup = remote_dep_dequeue_nothread_progress(&parsec_comm_es, -1 /* loop till explicitly asked to return */);
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine OFF on process %d/%d",
                             context->my_rank, context->nb_nodes);
        parsec_communication_engine_up = 1;  /* went to sleep */
    }

    /* Release all resources */
    remote_dep_ce_fini(context);
    PARSEC_PAPI_SDE_THREAD_FINI();

    return (void*)context;
}

/**
 * Perform a memcopy with datatypes by doing a local sendrecv.
 */
int
remote_dep_nothread_memcpy(parsec_execution_stream_t* es,
                           dep_cmd_item_t *item)
{
    dep_cmd_t* cmd = &item->cmd;
    int rc = parsec_ce.reshape(&parsec_ce, es, cmd->memcpy.destination, cmd->memcpy.source,
                               cmd->memcpy.layout, cmd->memcpy.displ_src, cmd->memcpy.displ_dst, cmd->memcpy.count);

    PARSEC_DATA_COPY_RELEASE(cmd->memcpy.source);
    remote_dep_dec_flying_messages(item->cmd.memcpy.taskpool);
    (void)es;
    return (MPI_SUCCESS == rc ? 0 : -1);
}

static int
remote_dep_mpi_put_end_cb(parsec_comm_engine_t *ce,
                       parsec_ce_mem_reg_handle_t lreg,
                       ptrdiff_t ldispl,
                       parsec_ce_mem_reg_handle_t rreg,
                       ptrdiff_t rdispl,
                       size_t size,
                       int remote,
                       void *cb_data)
{
    (void) ldispl; (void) rdispl; (void) size; (void) remote; (void) rreg;
    /* Retreive deps from callback_data */
    parsec_remote_deps_t* deps = ((remote_dep_cb_data_t *)cb_data)->deps;

    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MPI:\tTO\tna\tPut END  \tunknown \tk=%d\twith deps %p\tparams bla\t(tag=bla) data ptr bla",
            ((remote_dep_cb_data_t *)cb_data)->k, deps);


    TAKE_TIME(MPIsnd_prof, MPI_Data_plds_ek, ((remote_dep_cb_data_t *)cb_data)->k);

    remote_dep_complete_and_cleanup(&deps, 1);

    ce->mem_unregister(&lreg);
    parsec_thread_mempool_free(parsec_remote_dep_cb_data_mempool->thread_mempools, cb_data);

    parsec_comm_puts--;
    return 1;
}

static void
remote_dep_mpi_put_start(parsec_execution_stream_t* es,
                         dep_cmd_item_t* item)
{
    remote_dep_wire_get_t* task = &(item->cmd.activate.task);
#if !defined(PARSEC_PROF_DRY_DEP)
    parsec_remote_deps_t* deps = (parsec_remote_deps_t*) (uintptr_t) task->source_deps;
    int k, nbdtt;
    void* dataptr;
    MPI_Datatype dtt;
#endif  /* !defined(PARSEC_PROF_DRY_DEP) */
#if defined(PARSEC_DEBUG_NOISIER)
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    (void)es;
    DEBUG_MARK_CTL_MSG_GET_RECV(item->cmd.activate.peer, (void*)task, task);

#if !defined(PARSEC_PROF_DRY_DEP)
    assert(task->output_mask);
    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MPI:\tPUT mask=%lx deps 0x%lx", task->output_mask, task->source_deps);

    for(k = 0; task->output_mask>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if(!((1U<<k) & task->output_mask)) continue;

        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "MPI:\t[idx %d mask(0x%x / 0x%x)] %p, %p", k, (1U<<k), task->output_mask,
                deps->output[k].data.data, PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data));
        dataptr = PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data);
        dtt     = deps->output[k].data.layout;
        nbdtt   = deps->output[k].data.count;
        (void) nbdtt;

        task->output_mask ^= (1U<<k);

        parsec_ce_mem_reg_handle_t source_memory_handle;
        size_t source_memory_handle_size;

        if(parsec_ce.capabilites.supports_noncontiguous_datatype) {
            parsec_ce.mem_register(dataptr, PARSEC_MEM_TYPE_NONCONTIGUOUS,
                                   nbdtt, dtt,
                                   -1,
                                   &source_memory_handle, &source_memory_handle_size);
        } else {
            /* TODO: Implement converter to pack and unpack */
            int dtt_size;
            parsec_type_size(dtt, &dtt_size);
            parsec_ce.mem_register(dataptr, PARSEC_MEM_TYPE_CONTIGUOUS,
                                   -1, NULL,
                                   dtt_size,
                                   &source_memory_handle, &source_memory_handle_size);

        }

        parsec_ce_mem_reg_handle_t remote_memory_handle = item->cmd.activate.remote_memory_handle;

#if defined(PARSEC_DEBUG_NOISIER)
        MPI_Type_get_name(dtt, type_name, &len);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MPI:\tTO\t%d\tPut START\tunknown \tk=%d\twith deps 0x%lx at %p type %s\t(tag=bla displ = %ld)",
               item->cmd.activate.peer, k, task->source_deps, dataptr, type_name, deps->output[k].data.displ);
#endif

        remote_dep_cb_data_t *cb_data = (remote_dep_cb_data_t *) parsec_thread_mempool_allocate
                                            (parsec_remote_dep_cb_data_mempool->thread_mempools);
        cb_data->deps  = deps;
        cb_data->k     = k;

        TAKE_TIME_WITH_INFO(MPIsnd_prof, MPI_Data_plds_sk, k,
                            es->virtual_process->parsec_context->my_rank,
                            item->cmd.activate.peer, deps->msg);

        /* the remote side should send us 8 bytes as the callback data to be passed back to them */
        parsec_ce.put(&parsec_ce, source_memory_handle, 0,
                      remote_memory_handle, 0,
                      0, item->cmd.activate.peer,
                      remote_dep_mpi_put_end_cb, cb_data,
                      (parsec_ce_tag_t)task->callback_fn, &task->remote_callback_data, sizeof(uintptr_t));

        parsec_comm_puts++;
    }
#endif  /* !defined(PARSEC_PROF_DRY_DEP) */
    if(0 == task->output_mask) {
        if(NULL != item->cmd.activate.remote_memory_handle) {
            free(item->cmd.activate.remote_memory_handle);
            item->cmd.activate.remote_memory_handle = NULL;
        }
        free(item);
    }
}

void
remote_dep_mpi_new_taskpool(parsec_execution_stream_t* es,
                            dep_cmd_item_t *dep_cmd_item)
{
    parsec_list_item_t *item;
    parsec_taskpool_t* obj = dep_cmd_item->cmd.new_taskpool.tp;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    for(item = PARSEC_LIST_ITERATOR_FIRST(&dep_activates_noobj_fifo);
        item != PARSEC_LIST_ITERATOR_END(&dep_activates_noobj_fifo);
        item = PARSEC_LIST_ITERATOR_NEXT(item) ) {
        parsec_remote_deps_t* deps = (parsec_remote_deps_t*)item;
        if( deps->msg.taskpool_id == obj->taskpool_id ) {
            char* buffer = (char*)deps->taskpool;  /* get back the buffer from the "temporary" storage */
            int rc, position = 0;
            deps->taskpool = NULL;
            rc = remote_dep_get_datatypes(es, deps, PARSEC_DTD_SKIP_SAVING, &position); assert( -1 != rc );
            assert(deps->taskpool != NULL);
            PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tFROM\t%d\tActivate NEWOBJ\t% -8s\twith datakey %lx\tparams %lx",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    deps->msg.deps, deps->msg.output_mask);

            item = parsec_list_nolock_remove(&dep_activates_noobj_fifo, item);

            /* In case of DTD execution, receiving rank might not have discovered
             * the task responsible for this message. So we have to put this message
             * in a hash table so that we can activate it, when this rank discovers it.
             */
            if( -2 == rc ) { /* DTD problems, defer activating this remote dep */
                deps->taskpool = (parsec_taskpool_t*) buffer;
                continue;
            }

            remote_dep_mpi_recv_activate(es, deps, buffer, deps->msg.length, &position);
            free(buffer);
            (void)rc;
        }
    }
    remote_dep_dec_flying_messages(obj);
}

/* In DTD runs, remote nodes might ask us to activate tasks that has not been
 * discovered in the local node yet. We delay activation of those tasks and
 * push the dep in a hash table. As soon as we discover the remote task, for
 * which an activation is already pending, we issue a command to activate that
 * dep, This function does the necessary steps to continue the activation of
 * the remote task.
 */
void
remote_dep_mpi_release_delayed_deps(parsec_execution_stream_t* es,
                                    dep_cmd_item_t *item)
{
    PARSEC_PINS(es, ACTIVATE_CB_BEGIN, NULL);
    parsec_remote_deps_t *deps = item->cmd.release.deps;
    int rc, position = 0;
    char* buffer = (char*)deps->taskpool;  /* get back the buffer from the "temporary" storage */
    deps->taskpool = NULL;

    rc = remote_dep_get_datatypes(es, deps, 1, &position);

    assert(rc != -2);
    (void)rc;

    assert(deps != NULL);
    remote_dep_mpi_recv_activate(es, deps, buffer, deps->msg.length, &position);
    free(buffer);
    PARSEC_PINS(es, ACTIVATE_CB_END, NULL);
}

static void
remote_dep_mpi_get_start(parsec_execution_stream_t* es,
                         parsec_remote_deps_t* deps)
{
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from, k, count, nbdtt;
    remote_dep_wire_get_t msg;
    MPI_Datatype dtt;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN], type_name[MPI_MAX_OBJECT_NAME];
    int len;
    remote_dep_cmd_to_string(task, tmp, MAX_TASK_STRLEN);
#endif
#ifdef PARSEC_PROF_TRACE
    int32_t save_get = 0;
    int32_t event_id = parsec_atomic_fetch_inc_int32(&save_get);
#endif  /* PARSEC_PROF_TRACE */
    for(k = count = 0; deps->incoming_mask >> k; k++)
        if( ((1U<<k) & deps->incoming_mask) ) count++;

    (void)es;
    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.source_deps = task->deps; /* the deps copied from activate message from source */
    msg.callback_fn = (uintptr_t)remote_dep_mpi_get_end_cb; /* We let the source know to call this
                                                             * function when the PUT is over, in a true
                                                             * one sided case the (integer) value of this
                                                             * function pointer will be registered as the
                                                             * TAG to receive the same notification. */

    for(k = 0; deps->incoming_mask >> k; k++) {
        if( !((1U<<k) & deps->incoming_mask) ) continue;
        msg.output_mask = 0;  /* Only get what I need */
        msg.output_mask |= (1U<<k);

        /* We pack the callback data that should be passed to us when the other side
         * notifies us to invoke the callback_fn we have assigned above
         */
        remote_dep_cb_data_t *callback_data = (remote_dep_cb_data_t *) parsec_thread_mempool_allocate
                                                    (parsec_remote_dep_cb_data_mempool->thread_mempools);
        callback_data->deps = deps;
        callback_data->k    = k;

        /* prepare the local receiving data */
        assert(NULL == deps->output[k].data.data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
        if(NULL == deps->output[k].data.data) {
            deps->output[k].data.data = remote_dep_copy_allocate(&deps->output[k].data);
        }
        dtt   = deps->output[k].data.layout;
        nbdtt = deps->output[k].data.count;

#  if defined(PARSEC_DEBUG_NOISIER)
        MPI_Type_get_name(dtt, type_name, &len);
        int _size;
        MPI_Type_size(dtt, &_size);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MPI:\tTO\t%d\tGet START\t% -8s\tk=%d\twith datakey %lx at %p type %s count %d displ %ld extent %d\t(tag=%d)",
                from, tmp, k, task->deps, PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), type_name, nbdtt,
                deps->output[k].data.displ, deps->output[k].data.arena->elem_size * nbdtt, k);
#  endif

        /* We have the remote mem_handle.
         * Let's allocate our mem_reg_handle
         * and let the source know.
         */
        parsec_ce_mem_reg_handle_t receiver_memory_handle;
        size_t receiver_memory_handle_size;

        if(parsec_ce.capabilites.supports_noncontiguous_datatype) {
            parsec_ce.mem_register(PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), PARSEC_MEM_TYPE_NONCONTIGUOUS,
                                   nbdtt, dtt,
                                   -1,
                                   &receiver_memory_handle, &receiver_memory_handle_size);
        } else {
            /* TODO: Implement converter to pack and unpack */
            int dtt_size;
            parsec_type_size(dtt, &dtt_size);
            parsec_ce.mem_register(PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), PARSEC_MEM_TYPE_CONTIGUOUS,
                                   -1, NULL,
                                   dtt_size,
                                   &receiver_memory_handle, &receiver_memory_handle_size);

        }

        callback_data->memory_handle = receiver_memory_handle;

        /* We need multiple information to be passed to the callback_fn we have assigned above.
         * We pack the pointer to this callback_data and pass to the other side so we can complete
         * cleanup and take necessary action when the data is available on our side */
        msg.remote_callback_data = (remote_dep_datakey_t)callback_data;

        /* We pack the static message(remote_dep_wire_get_t) and our memory_handle and send this message
         * to the source. Source is anticipating this exact configuration.
         */
        int buf_size = sizeof(remote_dep_wire_get_t) + receiver_memory_handle_size;
        void *buf = malloc(buf_size);
        memcpy( buf,
                &msg,
                sizeof(remote_dep_wire_get_t) );
        memcpy( ((char*)buf) +  sizeof(remote_dep_wire_get_t),
                receiver_memory_handle,
                receiver_memory_handle_size );

        TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Data_ctl_sk, event_id,
                            from, es->virtual_process->parsec_context->my_rank, (*task));

        /* Send AM */
        parsec_ce.send_am(&parsec_ce, REMOTE_DEP_GET_DATA_TAG, from, buf, buf_size);
        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, event_id);

        TAKE_TIME_WITH_INFO(MPIrcv_prof, MPI_Data_pldr_sk, k, from,
                            es->virtual_process->parsec_context->my_rank, deps->msg);

        free(buf);

        parsec_comm_gets++;
    }
}

void remote_dep_mpi_get_end(parsec_execution_stream_t* es,
                            int idx,
                            parsec_remote_deps_t* deps)
{
    /* The ref on the data will be released below */
    remote_dep_release_incoming(es, deps, (1U<<idx));
}

static int
remote_dep_mpi_get_end_cb(parsec_comm_engine_t *ce,
                          parsec_ce_tag_t tag,
                          void *msg,
                          size_t msg_size,
                          int src,
                          void *cb_data)
{
    (void) ce; (void) tag; (void) msg_size; (void) cb_data; (void) src;
    parsec_execution_stream_t* es = &parsec_comm_es;

    /* We send 8 bytes to the source to give it back to us when the PUT is completed,
     * let's retrieve that
     */
    uintptr_t *retrieve_pointer_to_callback = (uintptr_t *)msg;
    remote_dep_cb_data_t *callback_data = (remote_dep_cb_data_t *)*retrieve_pointer_to_callback;
    parsec_remote_deps_t *deps = (parsec_remote_deps_t *)callback_data->deps;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MPI:\tFROM\t%d\tGet END  \t% -8s\tk=%d\twith datakey na        \tparams %lx\t(tag=%d)",
            src, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
            callback_data->k, deps->incoming_mask, src);


    TAKE_TIME(MPIrcv_prof, MPI_Data_pldr_ek, callback_data->k);
    remote_dep_mpi_get_end(es, callback_data->k, deps);

    parsec_ce.mem_unregister(&callback_data->memory_handle);
    parsec_thread_mempool_free(parsec_remote_dep_cb_data_mempool->thread_mempools, callback_data);

    parsec_comm_gets--;

    return 1;
}

/**
 * Trigger the local reception of a remote task data. Upon completion of all
 * pending receives related to a remote task completion, we call the
 * release_deps to enable all local tasks and then start the activation
 * propagation.
 */
static parsec_remote_deps_t*
remote_dep_release_incoming(parsec_execution_stream_t* es,
                            parsec_remote_deps_t* origin,
                            remote_dep_datakey_t complete_mask)
{
    parsec_task_t task;
    const parsec_flow_t* target;
    int i, pidx;
    uint32_t action_mask = 0;

    /* Update the mask of remaining dependencies to avoid releasing the same outputs twice */
    assert((origin->incoming_mask & complete_mask) == complete_mask);
    origin->incoming_mask ^= complete_mask;

    task.taskpool = origin->taskpool;
    task.task_class = task.taskpool->task_classes_array[origin->msg.task_class_id];
    task.priority = origin->priority;
    for(i = 0; i < task.task_class->nb_locals;
        task.locals[i] = origin->msg.locals[i], i++);
    for(i = 0; i < task.task_class->nb_flows;
        task.data[i].data_in = task.data[i].data_out = NULL, task.data[i].data_repo = NULL, i++);

    for(i = 0; complete_mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1U<<i) & complete_mask) ) continue;
        pidx = 0;
        target = task.task_class->out[pidx];
        while( !((1U<<i) & target->flow_datatype_mask) ) {
            target = task.task_class->out[++pidx];
            assert(NULL != target);
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "MPI:\tDATA %p(%s) released from %p[%d] flow idx %d",
                origin->output[i].data.data, target->name, origin, i, target->flow_index);
        task.data[target->flow_index].data_repo = NULL;
        task.data[target->flow_index].data_in   = origin->output[i].data.data;
        task.data[target->flow_index].data_out  = origin->output[i].data.data;
    }

#ifdef PARSEC_DIST_COLLECTIVES
    /* Corresponding comment below on the propagation part */
    if(0 == origin->incoming_mask && PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type) {
        remote_dep_inc_flying_messages(task.taskpool);
        (void)parsec_atomic_fetch_inc_int32(&origin->pending_ack);
    }
#endif  /* PARSEC_DIST_COLLECTIVES */

    if(PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type) {
        /* We need to convert from a dep_datatype_index mask into a dep_index mask */
        for(int i = 0; NULL != task.task_class->out[i]; i++ ) {
            target = task.task_class->out[i];
            if( !(complete_mask & target->flow_datatype_mask) ) continue;
            for(int j = 0; NULL != target->dep_out[j]; j++ )
                if(complete_mask & (1U << target->dep_out[j]->dep_datatype_index))
                    action_mask |= (1U << target->dep_out[j]->dep_index);
        }
    } else if(PARSEC_TASKPOOL_TYPE_DTD == origin->taskpool->taskpool_type) {
        action_mask = complete_mask;
    } else {
        assert(0);
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "MPI:\tTranslate mask from 0x%lx to 0x%x (remote_dep_release_incoming)",
            complete_mask, action_mask);
    (void)task.task_class->release_deps(es, &task,
                                        action_mask | PARSEC_ACTION_RELEASE_LOCAL_DEPS,
                                        NULL);
    assert(0 == (origin->incoming_mask & complete_mask));

    if(0 != origin->incoming_mask)  /* not done receiving */
        return origin;

    /**
     * All incoming data are now received, start the propagation. We first
     * release the local dependencies, thus we must ensure the communication
     * engine is not prevented from completing the propagation (the code few
     * lines above). Once the propagation is started we can release the
     * references on the allocated data and on the dependency.
     */
    uint32_t mask = origin->outgoing_mask;
    origin->outgoing_mask = 0;

#if defined(PARSEC_DIST_COLLECTIVES)
    if( PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type ) /* indicates it is a PTG taskpool */
        parsec_remote_dep_propagate(es, &task, origin);
#endif  /* PARSEC_DIST_COLLECTIVES */
    /**
     * Release the dependency owned by the communication engine for all data
     * internally allocated by the engine.
     */
    for(i = 0; mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1U<<i) & mask) ) continue;
        if( NULL != origin->output[i].data.data )  /* except CONTROLs */
            PARSEC_DATA_COPY_RELEASE(origin->output[i].data.data);
    }
#if defined(PARSEC_DIST_COLLECTIVES)
    if(PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type) {
        remote_dep_complete_and_cleanup(&origin, 1);
    } else {
        remote_deps_free(origin);
    }
#else
    remote_deps_free(origin);
#endif  /* PARSEC_DIST_COLLECTIVES */

    return NULL;
}

/**
 * Given a remote_dep_wire_activate message it packs as much as possible
 * into the provided buffer. If possible (eager allowed and enough room
 * in the buffer) some of the arguments will also be packed. Beware, the
 * remote_dep_wire_activate message itself must be updated with the
 * correct length before packing.
 *
 * @returns 1 if the message can't be packed due to lack of space, or 0
 * otherwise.
 */
static int remote_dep_mpi_pack_dep(int peer,
                                   dep_cmd_item_t* item,
                                   char* packed_buffer,
                                   int length,
                                   int* position)
{
    parsec_remote_deps_t *deps = (parsec_remote_deps_t*)item->cmd.activate.task.source_deps;
    remote_dep_wire_activate_t* msg = &deps->msg;
    int k, dsize, saved_position = *position;
    uint32_t peer_bank, peer_mask, expected = 0;
#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, 128);
#endif

    peer_bank = peer / (sizeof(uint32_t) * 8);
    peer_mask = 1U << (peer % (sizeof(uint32_t) * 8));

    //MPI_Pack_size(dep_count, dep_dtt, dep_comm, &dsize);
    dsize = dep_count;
    if( (length - (*position)) < dsize ) {  /* no room. bail out */
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Can't pack at %d/%d. Bail out!", *position, length);
        return 1;
    }
    /* Don't pack yet, we need to update the length field before packing */
    *position += dsize;
    assert((0 != msg->output_mask) &&   /* this should be preset */
           (msg->output_mask & deps->outgoing_mask) == deps->outgoing_mask);
    msg->length = 0;
    item->cmd.activate.task.output_mask = 0;  /* clean start */
    /* Treat for special cases: CTL, Eager, etc... */
    for(k = 0; deps->outgoing_mask >> k; k++) {
        if( !((1U << k) & deps->outgoing_mask )) continue;
        if( !(deps->output[k].rank_bits[peer_bank] & peer_mask) ) continue;

        /* Remove CTL from the message we expect to send */
#if defined(PARSEC_PROF_DRY_DEP)
        deps->output[k].data.arena = NULL; /* make all data a control */
#endif
        if(NULL == deps->output[k].data.arena) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, " CTL\t%s\tparam %d\tdemoted to be a control", tmp, k);
            continue;
        }
        assert(deps->output[k].data.count > 0);

        expected++;
        item->cmd.activate.task.output_mask |= (1U<<k);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "DATA\t%s\tparam %d\tdeps %p send on demand (increase deps counter by %d [%d])",
                tmp, k, deps, expected, deps->pending_ack);
    }
    if(expected)
        (void)parsec_atomic_fetch_add_int32(&deps->pending_ack, expected);  /* Keep track of the inflight data */

#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    parsec_debug_verbose(6, parsec_debug_output, "MPI:\tTO\t%d\tActivate\t% -8s\n"
          "    \t\t\twith datakey %lx\tmask %lx\t(tag=%d) eager mask %lu length %d",
          peer, tmp, msg->deps, msg->output_mask, -1,
          msg->output_mask ^ item->cmd.activate.task.output_mask, msg->length);
#endif
    /* And now pack the updated message (msg->length and msg->output_mask) itself. */
    //MPI_Pack(msg, dep_count, dep_dtt, packed_buffer, length, &saved_position, dep_comm);
    parsec_ce.pack(&parsec_ce, msg, dep_count, packed_buffer, length, &saved_position);
    msg->length = dsize;
    return 0;
}

/**
 * Starting with a particular item pack as many remote_dep_wire_activate
 * messages with the same destination (from the item ring associated with
 * pos_list) into a buffer. Upon completion the entire buffer is send to the
 * remote peer, the completed messages are released and the header is updated to
 * the next unsent message.
 */
static int
remote_dep_nothread_send(parsec_execution_stream_t* es,
                         dep_cmd_item_t **head_item)
{
    (void)es;
    parsec_remote_deps_t *deps;
    dep_cmd_item_t *item = *head_item;
    parsec_list_item_t* ring = NULL;
    char packed_buffer[DEP_EAGER_BUFFER_SIZE];
    int peer, position = 0;
#ifdef PARSEC_PROF_TRACE
    static int save_act = 0;
    int event_id = parsec_atomic_fetch_inc_int32(&save_act);
#endif  /* PARSEC_PROF_TRACE */

    peer = item->cmd.activate.peer;  /* this doesn't change */
    deps = (parsec_remote_deps_t*)item->cmd.activate.task.source_deps;

    TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Activate_sk, event_id,
                        es->virtual_process->parsec_context->my_rank, peer, deps->msg);

  pack_more:
    assert(peer == item->cmd.activate.peer);
    deps = (parsec_remote_deps_t*)item->cmd.activate.task.source_deps;

    parsec_list_item_singleton((parsec_list_item_t*)item);
    if( 0 == remote_dep_mpi_pack_dep(peer, item, packed_buffer,
                                     DEP_EAGER_BUFFER_SIZE, &position) ) {
        /* space left on the buffer. Move to the next item with the same destination */
        dep_cmd_item_t* next = (dep_cmd_item_t*)parsec_list_item_ring_chop(&item->pos_list);
        if( NULL == ring ) ring = (parsec_list_item_t*)item;
        else parsec_list_item_ring_push(ring, (parsec_list_item_t*)item);
        if( NULL != next ) {
            item = container_of(next, dep_cmd_item_t, pos_list);
            assert(DEP_ACTIVATE == item->action);
            if( parsec_param_enable_aggregate )
                goto pack_more;
        } else item = NULL;
    }
    *head_item = item;
    assert(NULL != ring);

    parsec_ce.send_am(&parsec_ce, REMOTE_DEP_ACTIVATE_TAG, peer, packed_buffer, position);

    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, event_id);
    DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(peer, (void*)&deps->msg, &deps->msg);

    do {
        item = (dep_cmd_item_t*)ring;
        ring = parsec_list_item_ring_chop(ring);
        deps = (parsec_remote_deps_t*)item->cmd.activate.task.source_deps;

        free(item);  /* only large messages are left */

        remote_dep_complete_and_cleanup(&deps, 1);
    } while( NULL != ring );
    return 0;
}

static int
remote_dep_mpi_progress(parsec_execution_stream_t* es)
{
    int ret = 0;

    if( !PARSEC_THREAD_IS_MASTER(es) ) return 0;

    ret = parsec_ce.progress(&parsec_ce);

    if(parsec_ce.can_serve(&parsec_ce) && !parsec_list_nolock_is_empty(&dep_activates_fifo)) {
            parsec_remote_deps_t* deps = (parsec_remote_deps_t*)parsec_list_nolock_pop_front(&dep_activates_fifo);
            remote_dep_mpi_get_start(es, deps);
    }
    if(parsec_ce.can_serve(&parsec_ce) && !parsec_list_nolock_is_empty(&dep_put_fifo)) {
            dep_cmd_item_t* item = (dep_cmd_item_t*)parsec_list_nolock_pop_front(&dep_put_fifo);
            remote_dep_mpi_put_start(es, item);
    }

    return ret;
}

int
remote_dep_dequeue_nothread_progress(parsec_execution_stream_t* es,
                                     int cycles)
{
    parsec_context_t* context = es->virtual_process->parsec_context;
    parsec_list_item_t *items;
    dep_cmd_item_t *item, *same_pos = NULL;
    parsec_list_t temp_list;
    int ret = 0, how_many, position, executed_tasks = 0;

    PARSEC_OBJ_CONSTRUCT(&temp_list, parsec_list_t);
 check_pending_queues:
    if( cycles >= 0 )
        if( 0 == cycles--) return executed_tasks;  /* report how many events were progressed */

    /* Move a number of transfers from the shared dequeue into our ordered lifo. */
    how_many = 0;
    while( NULL != (item = (dep_cmd_item_t*) parsec_dequeue_try_pop_front(&dep_cmd_queue)) ) {
        if( DEP_CTL == item->action ) {
            /* A DEP_CTL is a barrier that must not be crossed, flush the
             * ordered fifo and don't add anything until it is consumed */
            if( parsec_list_nolock_is_empty(&dep_cmd_fifo) && parsec_list_nolock_is_empty(&temp_list) )
                goto handle_now;
            parsec_dequeue_push_front(&dep_cmd_queue, (parsec_list_item_t*)item);
            break;
        }
        how_many++;
        same_pos = NULL;
        /* Find the position in the array of the first possible item in the same category */
        position = (DEP_ACTIVATE == item->action) ? item->cmd.activate.peer : (context->nb_nodes + item->action);

        parsec_list_item_singleton(&item->pos_list);
        same_pos = parsec_mpi_same_pos_items[position];
        if((NULL != same_pos) && (same_pos->priority >= item->priority)) {
            /* insert the item in the peer list */
            parsec_list_item_ring_push_sorted(&same_pos->pos_list, &item->pos_list, dep_mpi_pos_list);
        } else {
            if(NULL != same_pos) {
                /* this is the new head of the list. */
                parsec_list_item_ring_push(&same_pos->pos_list, &item->pos_list);
                /* Remove previous elem from the priority list. The element
                 might be either in the dep_cmd_fifo if it is old enough to be
                 pushed there, or in the temp_list waiting to be moved
                 upstream. Pay attention from which queue it is removed. */
#if defined(PARSEC_DEBUG_PARANOID)
                parsec_list_nolock_remove((struct parsec_list_t*)same_pos->super.belong_to, (parsec_list_item_t*)same_pos);
#else
                parsec_list_nolock_remove(NULL, (parsec_list_item_t*)same_pos);
#endif
                parsec_list_item_singleton((parsec_list_item_t*)same_pos);
            }
            parsec_mpi_same_pos_items[position] = item;
            /* And add ourselves in the temp list */
            parsec_list_nolock_push_front(&temp_list, (parsec_list_item_t*)item);
        }
        if(how_many > parsec_param_nb_tasks_extracted)
            break;
    }
    if( !parsec_list_nolock_is_empty(&temp_list) ) {
        /* Sort the temporary list */
        parsec_list_nolock_sort(&temp_list, dep_cmd_prio);
        /* Remove the ordered items from the list, and clean the list */
        items = parsec_list_nolock_unchain(&temp_list);
        /* Insert them into the locally ordered cmd_fifo */
        parsec_list_nolock_chain_sorted(&dep_cmd_fifo, items, dep_cmd_prio);
    }
    /* Extract the head of the list and point the array to the correct value */
    if(NULL == (item = (dep_cmd_item_t*)parsec_list_nolock_pop_front(&dep_cmd_fifo)) ) {
        do {
            ret = remote_dep_mpi_progress(es);
        } while(ret);

        if( !ret
         && ((comm_yield == 2)
          || (comm_yield == 1
           && !parsec_list_nolock_is_empty(&dep_activates_fifo)
           && !parsec_list_nolock_is_empty(&dep_put_fifo))) ) {
            struct timespec ts;
            ts.tv_sec = 0; ts.tv_nsec = comm_yield_ns;
            nanosleep(&ts, NULL);
        }
        goto check_pending_queues;
    }
    position = (DEP_ACTIVATE == item->action) ? item->cmd.activate.peer : (context->nb_nodes + item->action);
    assert(DEP_CTL != item->action);
    executed_tasks++;  /* count all the tasks executed during this call */
  handle_now:
    switch(item->action) {
    case DEP_CTL:
        ret = item->cmd.ctl.enable;
        PARSEC_OBJ_DESTRUCT(&temp_list);
        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "rank %d DISABLE MPI communication engine", parsec_debug_rank);
        free(item);
        return ret;  /* FINI or OFF */
    case DEP_NEW_TASKPOOL:
        remote_dep_mpi_new_taskpool(es, item);
        break;
    case DEP_DTD_DELAYED_RELEASE:
        remote_dep_mpi_release_delayed_deps(es, item);
        break;
    case DEP_ACTIVATE:
        remote_dep_nothread_send(es, &item);
        same_pos = item;
        goto have_same_pos;
#if 0
    case DEP_PUT_DATA:
#if 0 != RDEP_MSG_EAGER_LIMIT
        remote_dep_mpi_put_eager(es, item);
        same_pos = NULL;
#else
        assert("This should never be called!");
#endif  /* 0 != RDEP_MSG_EAGER_LIMIT */
        goto have_same_pos;
#endif
    case DEP_MEMCPY:
        remote_dep_nothread_memcpy(es, item);
        break;
    default:
        assert(0 && item->action); /* Not a valid action */
        break;
    }

    /* Correct the other structures */
    same_pos = (dep_cmd_item_t*)parsec_list_item_ring_chop(&item->pos_list);
    if( NULL != same_pos)
        same_pos = container_of(same_pos, dep_cmd_item_t, pos_list);
    free(item);
  have_same_pos:
    if( NULL != same_pos) {
        parsec_list_nolock_push_front(&temp_list, (parsec_list_item_t*)same_pos);
        /* if we still have pending messages of the same type, stay here for an extra loop */
        if( cycles >= 0 ) cycles++;
    }
    parsec_mpi_same_pos_items[position] = same_pos;

    goto check_pending_queues;
}
