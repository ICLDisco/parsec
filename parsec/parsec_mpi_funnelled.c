/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <mpi.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "parsec/parsec_mpi_funnelled.h"
#include "parsec/remote_dep.h"
#include "parsec/class/parsec_hash_table.h"
#include "parsec/class/dequeue.h"
#include "parsec/class/list.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/mca_param.h"

static int
mpi_no_thread_push_posted_req(parsec_comm_engine_t *ce);

// TODO put all the active ones(for debug) in a table and create a mempool
parsec_mempool_t *mpi_funnelled_mem_reg_handle_mempool = NULL;

/* Memory handles, opaque to upper layers */
typedef struct mpi_funnelled_mem_reg_handle_s {
    parsec_list_item_t        super;
    parsec_thread_mempool_t *mempool_owner;
    void *self;
    void *mem;
    parsec_datatype_t datatype;
    int count;
} mpi_funnelled_mem_reg_handle_t;

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(mpi_funnelled_mem_reg_handle_t);

/* To create object of class mpi_funnelled_mem_reg_handle_t that inherits
 * parsec_list_item_t class
 */
PARSEC_OBJ_CLASS_INSTANCE(mpi_funnelled_mem_reg_handle_t, parsec_list_item_t,
                   NULL, NULL);

/* Pointers are converted to long to be used as keys to fetch data in the get
 * rdv protocol. Make sure we can carry pointers correctly.
 */
#ifdef PARSEC_HAVE_LIMITS_H
#include <limits.h>
#endif
#if ULONG_MAX < UINTPTR_MAX
#error "unsigned long is not large enough to hold a pointer!"
#endif

/* note: tags are necessary to order communication between pairs. They are used to
 * correctly handle data transfers, as each data provider will provide a tag which
 * combined with the source ensure message matching consistency. As MPI requires the
 * max tag to be positive, initializing it to a negative value allows us to check
 * if the layer has been initialized or not.
 */
static int MAX_MPI_TAG = -1, mca_tag_ub = -1;
static volatile int __VAL_NEXT_TAG = 0;
#if INT_MAX == INT32_MAX
#define next_tag_cas(t, o, n) parsec_atomic_cas_int32(t, o, n)
#elif INT_MAX == INT64_MAX
#define next_tag_cas(t, o, n) parsec_atomic_cas_int64(t, o, n)
#else
#error "next_tag_cas written to support sizeof(int) of 4 or 8"
#endif
static inline int next_tag(int k) {
    int __tag, __tag_o, __next_tag;
reread:
    __tag = __tag_o = __VAL_NEXT_TAG;
    if( __tag > (MAX_MPI_TAG-k) ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "rank %d tag rollover: min %d < %d (+%d) < max %d", parsec_debug_rank,
                0, __tag, k, MAX_MPI_TAG);
        __tag = 0;
    }
    __next_tag = __tag+k;

    if( parsec_comm_es.virtual_process->parsec_context->flags & PARSEC_CONTEXT_FLAG_COMM_MT ) {
        if(!next_tag_cas(&__VAL_NEXT_TAG, __tag_o, __next_tag)) {
            goto reread;
        }
    }
    else {
        __VAL_NEXT_TAG = __next_tag;
    }
    return __tag;
}

/* Count and protect the internal building of the arrays of AM */
static int parsec_ce_am_design_version = 0;
static int parsec_ce_am_build_version = 0;
static parsec_atomic_lock_t parsec_ce_am_build_lock = PARSEC_ATOMIC_UNLOCKED;
static int mpi_funnelled_tag_unregister_unsafe_internal(parsec_ce_tag_t tag);

/* Range of index allowed for each type of request.
 * For registered tags, each will get EACH_STATIC_REQ_RANGE spots in the array of requests.
 * For dynamic tags, there will be a total of MAX_DYNAMIC_REQ_RANGE
 * spots in the request array.
 */
#define MAX_DYNAMIC_REQ_RANGE 30 /* according to current implementation */
#define MAX_NUM_RECV_REQ_IN_ARRAY 15
#define EACH_STATIC_REQ_RANGE 5 /* for each registered tag */

typedef enum parsec_ce_tag_status_e {
    PARSEC_CE_TAG_STATUS_INACTIVE = 1,
    PARSEC_CE_TAG_STATUS_ENABLE,
    PARSEC_CE_TAG_STATUS_ACTIVE,
    PARSEC_CE_TAG_STATUS_DISABLE
} parsec_ce_tag_status_t;

typedef struct mpi_funnelled_tag_s {
    parsec_ce_tag_t tag; /* tag user wants to register */
    int16_t start_idx; /* Records the starting index for every TAG
                        * to unregister from the array_of_[requests/indices/statuses]
                        */
    parsec_ce_tag_status_t status;  /* The current status of this tag (inactive/active, enable/disable) */
    size_t  msg_length; /* Maximum length allowed to send for this TAG */
    parsec_ce_am_callback_t callback;  /* callback to call upon reception of the
                                          associated AM */
    void*  cb_data;  /* upper-level data to pass back to the callback */
    char*  am_backend_memory; /* Buffer where we will receive msg for each TAG. This area is
                               * allocated to host the number of AM messages for this tag times
                               * the msg_length. It is better if the msg_length is a multiple of the
                               * cache lines size.
                               */
} mpi_funnelled_tag_t;

static struct mpi_funnelled_tag_s parsec_mpi_funnelled_array_of_registered_tags[PARSEC_MAX_REGISTERED_TAGS];

typedef enum mpi_funnelled_callback_type_e {
    MPI_FUNNELLED_TYPE_AM       = 0, /* indicating active message */
    MPI_FUNNELLED_TYPE_ONESIDED = 1,  /* indicating one sided */
    MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM = 2  /* indicating one sided with am callback type */
} mpi_funnelled_callback_type_t;

/* Structure to hold information about callbacks,
 * since we have multiple type of callbacks (active message and one-sided,
 * we store a type to know which to call.
 */
typedef struct mpi_funnelled_callback_s {
    long storage1; /* callback data */
    long storage2; /* callback data */
    void *cb_data; /* callback data */
    mpi_funnelled_callback_type_t type;
    mpi_funnelled_tag_t *tag_reg;
    bool is_dynamic_recv;

    union {
        struct {
            parsec_ce_am_callback_t fct;
        } am;
        struct {
            parsec_ce_am_callback_t fct;
            void *msg;
        } onesided_mimic_am;
    } cb_type;
    struct {
        parsec_ce_onesided_callback_t fct;
        parsec_ce_mem_reg_handle_t lreg; /* local memory handle */
        ptrdiff_t ldispl; /* displacement for local memory handle */
        parsec_ce_mem_reg_handle_t rreg; /* remote memory handle */
        ptrdiff_t rdispl; /* displacement for remote memory handle */
        size_t size; /* size of data */
        int remote; /* remote process id */
        int tag;
    } onesided;
} mpi_funnelled_callback_t;

/* The internal communicators used by the CE to host its requests and
 * other operations. Some are copies of the context->comm_ctx (which is
 * a duplicate of whatever the user provided upon initialization). We
 * use two communicators one for handling the AM and the other dedicated
 * to moving data around.
 */
static MPI_Comm parsec_ce_mpi_am_comm[PARSEC_MAX_REGISTERED_TAGS] = {MPI_COMM_NULL};  /* Active message communicator */
static MPI_Comm parsec_ce_mpi_comm = MPI_COMM_NULL;     /* Data moving communicator */
/* The internal communicator for all intra-node communications */
static MPI_Comm parsec_ce_mpi_self_comm = MPI_COMM_NULL;

static mpi_funnelled_callback_t *array_of_callbacks;
static MPI_Request              *array_of_requests;
static int                      *array_of_indices;
static MPI_Status               *array_of_statuses;

/** The expected size of the next version of the arrays of requests */
static int size_of_total_reqs = 0;
/** The current size of the arrays of requests. */
static int current_size_of_total_reqs = 0;
static int mpi_funnelled_last_active_req = 0;
static int mpi_funnelled_static_req_idx = 0;
static int mpi_funnelled_num_recv_req_in_arr = 0;

#if defined(PARSEC_HAVE_MPI_OVERTAKE)
static int parsec_param_enable_mpi_overtake = 1;
#else
static int parsec_param_enable_mpi_overtake = 0;  /* Default to 0 if not supported to avoid complaints about the MCA */
#endif

/* List to hold pending requests */
parsec_list_t mpi_funnelled_dynamic_sendreq_fifo; /* ordered non threaded fifo */
parsec_list_t mpi_funnelled_dynamic_recvreq_fifo; /* ordered non threaded fifo */
parsec_mempool_t *mpi_funnelled_dynamic_req_mempool = NULL;

/* This structure is used to save all the information necessary to
 * invoke a callback after a MPI_Request is satisfied
 */
typedef struct mpi_funnelled_dynamic_req_s {
    parsec_list_item_t super;
    parsec_thread_mempool_t *mempool_owner;
    int post_isend;
    MPI_Request request;
    mpi_funnelled_callback_t cb;
} mpi_funnelled_dynamic_req_t;

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(mpi_funnelled_dynamic_req_t);

/* To create object of class mpi_funnelled_dynamic_req_t that inherits
 * parsec_list_item_t class
 */
PARSEC_OBJ_CLASS_INSTANCE(mpi_funnelled_dynamic_req_t, parsec_list_item_t,
                   NULL, NULL);

/* Data we pass internally inside GET and PUT for handshake and other
 * synchronizations.
 */
typedef struct get_am_data_s {
    int tag;
    parsec_ce_mem_reg_handle_t lreg;
    parsec_ce_mem_reg_handle_t rreg;
    uintptr_t cb_fn;
    uintptr_t deps;
} get_am_data_t;

typedef struct mpi_funnelled_handshake_info_s {
    int tag;
    parsec_ce_mem_reg_handle_t source_memory_handle;
    parsec_ce_mem_reg_handle_t remote_memory_handle;
    uintptr_t cb_fn;

} mpi_funnelled_handshake_info_t;

/* This is the callback that is triggered on the sender side for a
 * GET. In this function we get the TAG on which the receiver has
 * posted an Irecv and using which the sender should post an Isend
 */
static int
mpi_funnelled_internal_get_am_callback(parsec_comm_engine_t *ce,
                                       parsec_ce_tag_t tag,
                                       void *msg,
                                       size_t msg_size,
                                       int src,
                                       void *cb_data)
{
    (void) ce; (void) tag; (void) cb_data;
    assert(mpi_funnelled_last_active_req <= current_size_of_total_reqs);

    mpi_funnelled_handshake_info_t *handshake_info = (mpi_funnelled_handshake_info_t *) msg;
    mpi_funnelled_callback_t *cb;

    /* This rank sent it's mem_reg in the activation msg, which is being
     * sent back as rreg of the msg */
    mpi_funnelled_mem_reg_handle_t *remote_memory_handle = (mpi_funnelled_mem_reg_handle_t *) (handshake_info->remote_memory_handle); /* This is the memory handle of the remote(our) side */

    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);

    int post_in_static_array = mpi_funnelled_last_active_req < current_size_of_total_reqs;
    mpi_funnelled_dynamic_req_t *item;

    if(post_in_static_array) {
        cb = &array_of_callbacks[mpi_funnelled_last_active_req];
        MPI_Isend(remote_memory_handle->mem, remote_memory_handle->count, remote_memory_handle->datatype,
                  src, handshake_info->tag, parsec_ce_mpi_comm,
                  &array_of_requests[mpi_funnelled_last_active_req]);
    } else {
        item = (mpi_funnelled_dynamic_req_t *)parsec_thread_mempool_allocate(mpi_funnelled_dynamic_req_mempool->thread_mempools);
        item->post_isend = 1;
        cb = &item->cb;
    }

    /* we(the remote side) requested the source to forward us callback data that will be passed
     * to the callback function to notify upper level that the data has reached. We are copying
     * the callback data sent from the source.
     */
    void *callback_data = malloc(msg_size - sizeof(mpi_funnelled_handshake_info_t));
    memcpy( callback_data,
            ((char*)msg) + sizeof(mpi_funnelled_handshake_info_t),
            msg_size - sizeof(mpi_funnelled_handshake_info_t) );

    cb->cb_type.onesided_mimic_am.fct = (parsec_ce_am_callback_t) handshake_info->cb_fn;
    cb->cb_type.onesided_mimic_am.msg = callback_data;
    cb->storage1 = mpi_funnelled_last_active_req;
    cb->storage2 = src;
    cb->cb_data  = remote_memory_handle;
    cb->tag_reg  = NULL;
    cb->type     = MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM;
    cb->is_dynamic_recv = false;

    cb->onesided.fct = NULL;
    cb->onesided.lreg = remote_memory_handle;
    cb->onesided.ldispl = 0;
    cb->onesided.remote = src;
    cb->onesided.tag = handshake_info->tag;

    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_sendreq_fifo,
                                     (parsec_list_item_t *)item);
    }

    return 1;
}

/* This is the callback that is triggered on the receiver side for a
 * PUT. This is where we know the TAG to post the Irecv on.
 */
static int
mpi_funnelled_internal_put_am_callback(parsec_comm_engine_t *ce,
                                       parsec_ce_tag_t tag,
                                       void *msg,
                                       size_t msg_size,
                                       int src,
                                       void *cb_data)
{
    (void) ce; (void) tag; (void)msg_size; (void)cb_data;

    mpi_funnelled_callback_t *cb;
    MPI_Request *request;

    mpi_funnelled_handshake_info_t *handshake_info = (mpi_funnelled_handshake_info_t *) msg;

    /* Get the local memory handle from the peer (it was originally sent with the request) */
    mpi_funnelled_mem_reg_handle_t *remote_memory_handle = (mpi_funnelled_mem_reg_handle_t*)handshake_info->remote_memory_handle;

    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);

    mpi_funnelled_dynamic_req_t *item = NULL;
    int post_in_static_array = mpi_funnelled_last_active_req < current_size_of_total_reqs;
    if (MAX_NUM_RECV_REQ_IN_ARRAY >= mpi_funnelled_num_recv_req_in_arr) {
        post_in_static_array = 0;
    } else if (post_in_static_array) {
        mpi_funnelled_num_recv_req_in_arr++;
    }

    if(post_in_static_array) {
        request = &array_of_requests[mpi_funnelled_last_active_req];
        cb = &array_of_callbacks[mpi_funnelled_last_active_req];
    } else {
        /* we are not delaying posting the Irecv as the other side will post the Isend as soon
         * as it get an acknowledgement of the completion of the active message it sent for handshake.
         * This ensures we are not generating MPI unexpected and all the sends and receives are in order.
         */
        item = (mpi_funnelled_dynamic_req_t *)parsec_thread_mempool_allocate(mpi_funnelled_dynamic_req_mempool->thread_mempools);
        item->post_isend = 0;
        request = &item->request;
        cb = &item->cb;
    }

    MPI_Irecv(remote_memory_handle->mem, remote_memory_handle->count, remote_memory_handle->datatype,
              src, handshake_info->tag, parsec_ce_mpi_comm, request);

    /* we(the remote side) requested the source to forward us callback data that will be passed
     * to the callback function to notify upper level that the data has reached. We are copying
     * the callback data sent from the source.
     */
    void *callback_data = malloc(msg_size - sizeof(mpi_funnelled_handshake_info_t));
    memcpy( callback_data,
            ((char*)msg) + sizeof(mpi_funnelled_handshake_info_t),
            msg_size - sizeof(mpi_funnelled_handshake_info_t) );

    /* We sent the pointer to the call back function for PUT over notification.
     * For a TRUE one sided this would be accomplished by an active message at
     * the tag of the integer value of the function pointer we trigger as callback.
     */
    cb->cb_type.onesided_mimic_am.fct = (parsec_ce_am_callback_t) handshake_info->cb_fn;
    cb->cb_type.onesided_mimic_am.msg = callback_data;
    cb->storage1 = mpi_funnelled_last_active_req;
    cb->storage2 = src;
    cb->cb_data  = cb->cb_data;
    cb->tag_reg  = NULL;
    cb->type     = MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM;
    cb->is_dynamic_recv = true;

    /* we don't need to initialize anything in the onesided part, we will never send
     * a message to the peer but instead will only complete the local receive and
     * trigger the local AM callback.
     */
    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_recvreq_fifo,
                                     (parsec_list_item_t *)item);
    }

    return 1;
}

int parsec_mpi_sendrecv(parsec_comm_engine_t *ce,
                        parsec_execution_stream_t* es,
                        parsec_data_copy_t *dst,
                        int64_t displ_dst,
                        parsec_datatype_t layout_dst,
                        uint64_t count_dst,
                        parsec_data_copy_t *src,
                        int64_t displ_src,
                        parsec_datatype_t layout_src,
                        uint64_t count_src)
{
    int rc;
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream,
                         "COPY LOCAL DATA from %p (%d elements of dtt %p) to %p (%d elements of dtt %p)",
                         PARSEC_DATA_COPY_GET_PTR(src) + displ_src, count_src, layout_src,
                         PARSEC_DATA_COPY_GET_PTR(dst) + displ_dst, count_dst, layout_dst);
    rc = MPI_Sendrecv((char*)PARSEC_DATA_COPY_GET_PTR(src) + displ_src,
                      count_src, layout_src, 0, es->th_id,
                      (char*)PARSEC_DATA_COPY_GET_PTR(dst) + displ_dst,
                      count_dst, layout_dst, 0, es->th_id,
                      parsec_ce_mpi_self_comm, MPI_STATUS_IGNORE);
    (void)ce;
    return (MPI_SUCCESS == rc ? 0 : -1);
}

/**
 * Store the user provided communicator in the PaRSEC context. We need to make a
 * copy to make sure the communicator does not disappear before the communication
 * engine starts up.
 * 
 * This function is collective, all processes in the current and the new communicator
 * should call it in same time.
 */
static int parsec_mpi_set_ctx(parsec_comm_engine_t* ce, intptr_t opaque_comm_ctx )
{
    parsec_context_t* context = ce->parsec_context;
    MPI_Comm comm;
    int rc;

    /* We can only change the communicator if the communication engine is not active */
    if( 1 < parsec_communication_engine_up ) {
        parsec_warning("Cannot change PaRSEC's MPI communicator while the engine is running [ignored]");
        return PARSEC_ERROR;
    }

    /* The engine was never yet started, so there is nothing to clean */
    if( NULL != ce->sync ) {
        ce->disable(ce);
        ce->sync(ce);
        assert( -1 != context->comm_ctx );
        MPI_Comm_free((MPI_Comm*)&context->comm_ctx);
    }
    rc = MPI_Comm_dup((MPI_Comm)opaque_comm_ctx, &comm);
    context->comm_ctx = (intptr_t)comm;
    parsec_ce_am_design_version++;  /* signal need for update */
    /* We need to know who we are and how many others are there, in order to
     * correctly initialize the communication engine at the next start. */
    MPI_Comm_size( (MPI_Comm)context->comm_ctx, (int*)&(context->nb_nodes));
    MPI_Comm_rank( (MPI_Comm)context->comm_ctx, (int*)&(context->my_rank));

    /* We might not need a barrier on the new communicator, the communicator
     * creation is a strong enough synchronization.
     */
    parsec_taskpool_sync_ids_context(context->comm_ctx);

    return (MPI_SUCCESS == rc) ? PARSEC_SUCCESS : PARSEC_ERROR;
}

/**
 * The following function take care of all the steps necessary to initialize the
 * invariable part of the communication engine such as the const dependencies
 * to MPI (max tag and other global info), or local objects.
 */
static int mpi_funneled_init_once(parsec_context_t* context)
{
    int mpi_tag_ub_exists, *ub;

    assert(-1 == MAX_MPI_TAG);

    assert(MPI_COMM_NULL == parsec_ce_mpi_self_comm);
    MPI_Comm_dup(MPI_COMM_SELF, &parsec_ce_mpi_self_comm);
    assert(MPI_COMM_NULL == parsec_ce_mpi_comm);
    assert(MPI_COMM_NULL == parsec_ce_mpi_am_comm[0]);

    /*
     * Based on MPI 1.1 the MPI_TAG_UB should only be defined
     * on MPI_COMM_WORLD.
     */
#if defined(PARSEC_HAVE_MPI_20)
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &mpi_tag_ub_exists);
#else
    MPI_Attr_get(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &mpi_tag_ub_exists);
#endif  /* defined(PARSEC_HAVE_MPI_20) */

    parsec_mca_param_reg_int_name("mpi", "tag_ub",
                                  "The upper bound of the TAG used by the MPI communication engine. Bounded by the MPI_TAG_UB attribute on the MPI implementation MPI_COMM_WORLD. (-1 for MPI default)",
                                  false, false, -1, &mca_tag_ub);

    if( !mpi_tag_ub_exists ) {
        MAX_MPI_TAG = (-1 == mca_tag_ub) ? INT_MAX : mca_tag_ub;
        parsec_warning("Your MPI implementation does not define MPI_TAG_UB and thus violates the standard (MPI-2.2, page 29, line 30). The max tag is therefore set using the MCA mpi_tag_ub (current value %d).\n", MAX_MPI_TAG);
    } else {
        MAX_MPI_TAG = ((-1 == mca_tag_ub) || (mca_tag_ub > *ub)) ? *ub : mca_tag_ub;
    }
    if( MAX_MPI_TAG < INT_MAX ) {
        parsec_debug_verbose(3, parsec_comm_output_stream,
                             "MPI:\tYour MPI implementation defines the maximal TAG value to %d (0x%08x),"
                             " which might be too small should you have more than %d pending remote dependencies",
                             MAX_MPI_TAG, (unsigned int)MAX_MPI_TAG, MAX_MPI_TAG / MAX_DEP_OUT_COUNT);
    }

    parsec_mca_param_reg_int_name("runtime", "comm_mpi_overtake",
#if defined(PARSEC_HAVE_MPI_OVERTAKE)
                                  "Enable MPI allow overtaking of messages (if applicable). (0: no, 1: yes)",
#else
                                  "Not supported by the current MPI library (forced to zero)",
#endif  /* defined(PARSEC_HAVE_MPI_OVERTAKE) */
                                  false, false, parsec_param_enable_mpi_overtake, &parsec_param_enable_mpi_overtake);
#if !defined(PARSEC_HAVE_MPI_OVERTAKE)
    parsec_param_enable_mpi_overtake = 0;  /* Don't allow to be changed */
#endif  /* !defined(PARSEC_HAVE_MPI_OVERTAKE) */

    (void)context;
    return 0;
}

parsec_comm_engine_t *
mpi_funnelled_init(parsec_context_t *context)
{
    int i, rc;

    if( -1 == MAX_MPI_TAG )
        if( 0 != (rc = mpi_funneled_init_once(context)) ) {
            parsec_debug_verbose(3, parsec_comm_output_stream, "MPI: Failed to correctly retrieve the max TAG."
                                 " PaRSEC cannot continue using MPI\n");
            return NULL;
        }

    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "rank %d ENABLE MPI communication engine",
                         parsec_debug_rank);

    for(i = 0; i < PARSEC_MAX_REGISTERED_TAGS; i++) {
        parsec_mpi_funnelled_array_of_registered_tags[i].am_backend_memory = NULL;
        parsec_mpi_funnelled_array_of_registered_tags[i].callback = NULL;
        parsec_mpi_funnelled_array_of_registered_tags[i].status = PARSEC_CE_TAG_STATUS_INACTIVE;
    }

    size_of_total_reqs = MAX_DYNAMIC_REQ_RANGE;

     /* Make all the fn pointers point to this component's function */
    parsec_ce.enable              = mpi_no_thread_enable;
    parsec_ce.disable             = mpi_no_thread_disable;
    parsec_ce.set_ctx             = parsec_mpi_set_ctx;
    parsec_ce.fini                = mpi_funnelled_fini;
    parsec_ce.tag_register        = mpi_no_thread_tag_register;
    parsec_ce.tag_unregister      = mpi_no_thread_tag_unregister;
    parsec_ce.mem_register        = NULL;
    parsec_ce.mem_unregister      = NULL;
    parsec_ce.get_mem_handle_size = NULL;
    parsec_ce.mem_retrieve        = NULL;
    parsec_ce.put                 = NULL;
    parsec_ce.get                 = NULL;
    parsec_ce.progress            = NULL;
    parsec_ce.pack                = NULL;
    parsec_ce.pack_size           = NULL;
    parsec_ce.unpack              = NULL;
    parsec_ce.sync                = NULL;
    parsec_ce.reshape             = NULL;
    parsec_ce.can_serve           = NULL;
    parsec_ce.send_am             = NULL;

    parsec_ce.parsec_context      = context;
    parsec_ce.capabilites.sided   = 2;
    parsec_ce.capabilites.supports_noncontiguous_datatype = 1;

    /* Define some sensible values. We assume the application will initialize PaRSEC using
     * the entire MPI_COMM_WORLD, but we need to prepare some decent default values. */
    if( -1 == context->comm_ctx ) {
        MPI_Comm_size( MPI_COMM_WORLD, (int*)&(context->nb_nodes));
        MPI_Comm_rank( MPI_COMM_WORLD, (int*)&(context->my_rank));
        context->comm_ctx = (intptr_t)MPI_COMM_WORLD;
    }
    /* Register for internal GET and PUT AMs */
    parsec_ce.tag_register(PARSEC_CE_MPI_FUNNELLED_GET_TAG_INTERNAL,
                           mpi_funnelled_internal_get_am_callback,
                           context,
                           4096);

    parsec_ce.tag_register(PARSEC_CE_MPI_FUNNELLED_PUT_TAG_INTERNAL,
                           mpi_funnelled_internal_put_am_callback,
                           context,
                           4096);

    return &parsec_ce;
}

/**
 * The communication engine is now completely disabled. All internal resources
 * are released, and no future communications are possible.
 * Anything initialized in init_once must be disposed off here
 */
int
mpi_funnelled_fini(parsec_comm_engine_t *ce)
{
    assert( -1 != MAX_MPI_TAG );

    /* TODO: GO through all registered tags and unregister them */
    ce->tag_unregister(PARSEC_CE_MPI_FUNNELLED_GET_TAG_INTERNAL);
    ce->tag_unregister(PARSEC_CE_MPI_FUNNELLED_PUT_TAG_INTERNAL);
    parsec_atomic_lock(&parsec_ce_am_build_lock);
    for(int tag = 0; tag < PARSEC_MAX_REGISTERED_TAGS; tag++) {
        mpi_funnelled_tag_t *tag_struct = &parsec_mpi_funnelled_array_of_registered_tags[tag];
        if( tag_struct->status != PARSEC_CE_TAG_STATUS_INACTIVE)
            mpi_funnelled_tag_unregister_unsafe_internal(tag);
    }
    /* By now the entire AM management is clean, reset the design and build versions.*/
    parsec_ce_am_design_version = 0;
    parsec_ce_am_build_version = 0;
    parsec_atomic_unlock(&parsec_ce_am_build_lock);

    free(array_of_callbacks); array_of_callbacks = NULL;
    free(array_of_requests);  array_of_requests  = NULL;
    free(array_of_indices);   array_of_indices   = NULL;
    free(array_of_statuses);  array_of_statuses  = NULL;

    if( NULL != mpi_funnelled_mem_reg_handle_mempool ) {
        PARSEC_OBJ_DESTRUCT(&mpi_funnelled_dynamic_sendreq_fifo);
        PARSEC_OBJ_DESTRUCT(&mpi_funnelled_dynamic_recvreq_fifo);

        parsec_mempool_destruct(mpi_funnelled_mem_reg_handle_mempool);
        free(mpi_funnelled_mem_reg_handle_mempool); mpi_funnelled_mem_reg_handle_mempool = NULL;

        parsec_mempool_destruct(mpi_funnelled_dynamic_req_mempool);
        free(mpi_funnelled_dynamic_req_mempool); mpi_funnelled_dynamic_req_mempool = NULL;
    }
    /* Remove the static handles */
    MPI_Comm_free(&parsec_ce_mpi_self_comm); /* parsec_ce_mpi_self_comm becomes MPI_COMM_NULL */

    /* Release the context communicators if any */
    if( MPI_COMM_NULL != parsec_ce_mpi_comm) {
        MPI_Comm_free(&parsec_ce_mpi_comm);
        for(int i = 0; i < PARSEC_MAX_REGISTERED_TAGS; i++) {
            MPI_Comm_free(&parsec_ce_mpi_am_comm[i]);
        }
        ce->parsec_context->comm_ctx = -1; /* We use -1 for the opaque comm_ctx, rather than the MPI specific MPI_COMM_NULL */
    }
    assert(MPI_COMM_NULL == parsec_ce_mpi_comm );  /* no communicator */
    assert(MPI_COMM_NULL == parsec_ce_mpi_am_comm[0] );  /* no communicator */
    MAX_MPI_TAG = -1;  /* mark the layer as uninitialized */
    size_of_total_reqs = 0;
    mpi_funnelled_last_active_req = 0;
    mpi_funnelled_static_req_idx = 0;

    return 1;
}

/**
 * @brief Register AM tags. Classical API, a tag a callback function and a callback data
 *        for the message arrival and a maximum length of the AM message for preallocating
 *        all necessary buffers. The registration can happen at any moment, before or after
 *        the communication engine was started. If the communication engine is inactive,
 *        the tags will be stored but anything related to the communication registration of
 *        such AM will be delayed until the engine starts. If the engine is running, we
 *        notify it that the AM infrastructure needs to be rebuilt, and it will do it at
 *        the next progress cycle.
 */
int
mpi_no_thread_tag_register(parsec_ce_tag_t tag,
                           parsec_ce_am_callback_t callback,
                           void *cb_data,
                           size_t msg_length)
{
    /* All internal tags have been registered */
    if(tag >= PARSEC_MAX_REGISTERED_TAGS) {
        parsec_warning("Tag is out of range, it has to be between %d - %d\n", 0, PARSEC_MAX_REGISTERED_TAGS);
        return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
    }
    mpi_funnelled_tag_t *tag_struct = &parsec_mpi_funnelled_array_of_registered_tags[tag];
    if(NULL != parsec_mpi_funnelled_array_of_registered_tags[tag].callback ) {
        parsec_warning("Tag: %ld is already registered (callback %p, callback data %p, msg length %d)\n",
                       tag, tag_struct->callback, tag_struct->cb_data, (int)tag_struct->msg_length);
        return PARSEC_ERR_EXISTS;
    }
    parsec_atomic_lock(&parsec_ce_am_build_lock);
    tag_struct->tag = tag;
    tag_struct->msg_length = (msg_length + 15) & ~0xF;  /* align to 16 bytes */
    tag_struct->callback = callback;
    tag_struct->cb_data = cb_data;
    tag_struct->status = PARSEC_CE_TAG_STATUS_ENABLE;

    /* Update the total number of requests we know about */
    size_of_total_reqs += EACH_STATIC_REQ_RANGE;
    /* Make sure the AM infrastructure is rebuilt at the next progress cycle */
    parsec_ce_am_design_version++;
    parsec_atomic_unlock(&parsec_ce_am_build_lock);
    return PARSEC_SUCCESS;
}

static int parsec_ce_rebuild_am_requests(void)
{
    mpi_funnelled_callback_t* cb;

    if(parsec_ce_am_build_version == parsec_ce_am_design_version) {
        /* There is nothing to update, the engine is ready to rock */
        return PARSEC_SUCCESS;
    }

    parsec_atomic_lock(&parsec_ce_am_build_lock);
    parsec_ce_am_build_version = parsec_ce_am_design_version;
    /* Reallocate the management arrays. Indices and statuses can be simply shifted around,
     * while the array of callbacks and statuses need a little extra work.
     */
    array_of_indices = realloc(array_of_indices, size_of_total_reqs * sizeof(int));
    array_of_statuses = realloc(array_of_statuses, size_of_total_reqs * sizeof(MPI_Status));
    mpi_funnelled_callback_t *tmp_array_cb = malloc(sizeof(mpi_funnelled_callback_t) * size_of_total_reqs);
    MPI_Request *tmp_array_req = malloc(sizeof(MPI_Request) * size_of_total_reqs);

    int idx = 0, old_idx = 0;
    for( int tag = 0; tag < PARSEC_MAX_REGISTERED_TAGS; tag++ ) {
        mpi_funnelled_tag_t *tag_struct = &parsec_mpi_funnelled_array_of_registered_tags[tag];
        if( (NULL == tag_struct->callback) ||
            (tag_struct->status == PARSEC_CE_TAG_STATUS_INACTIVE) )  /* No changes for this tag */
            continue;
        if( tag_struct->status == PARSEC_CE_TAG_STATUS_DISABLE ) {
            mpi_funnelled_tag_unregister_unsafe_internal(tag);
            old_idx += EACH_STATIC_REQ_RANGE;
            continue;
        }
        if( tag_struct->status == PARSEC_CE_TAG_STATUS_ACTIVE ) {
            memcpy(&tmp_array_cb[idx], &array_of_callbacks[old_idx],
                   sizeof(mpi_funnelled_callback_t) * EACH_STATIC_REQ_RANGE);
            memcpy(&tmp_array_req[idx], &array_of_requests[old_idx],
                   sizeof(MPI_Request) * EACH_STATIC_REQ_RANGE);
            idx     += EACH_STATIC_REQ_RANGE;
            old_idx += EACH_STATIC_REQ_RANGE;
            continue;
        }
        assert(PARSEC_CE_TAG_STATUS_ENABLE == tag_struct->status);

        char *buf = (char *) calloc(EACH_STATIC_REQ_RANGE, tag_struct->msg_length * sizeof(char));

        tag_struct->am_backend_memory = buf;
        tag_struct->start_idx  = idx;
        tag_struct->status = PARSEC_CE_TAG_STATUS_ACTIVE;

        for(int i = 0; i < EACH_STATIC_REQ_RANGE; i++) {
            buf = tag_struct->am_backend_memory + i * tag_struct->msg_length * sizeof(char);

            /* Even though the address of array_of_requests changes after every
             * new registration of tags, the initialization of the requests will
             * still work as the memory is copied after initialization.
             */
            MPI_Recv_init(buf, tag_struct->msg_length, MPI_BYTE,
                          MPI_ANY_SOURCE, tag, parsec_ce_mpi_am_comm[tag],
                          &tmp_array_req[idx]);

            cb = &tmp_array_cb[idx];
            cb->cb_type.am.fct = tag_struct->callback;
            cb->cb_data        = tag_struct->cb_data;
            cb->storage1       = idx;
            cb->storage2       = i;
            cb->tag_reg        = tag_struct;
            cb->type           = MPI_FUNNELLED_TYPE_AM;
            cb->is_dynamic_recv = false;
            idx++;
        }
        /* Tag ready to receive data, start all persistent receives */
        MPI_Startall(EACH_STATIC_REQ_RANGE, &tmp_array_req[idx - EACH_STATIC_REQ_RANGE]);
    }
    /* Replace the arrays of callbacks and requests with the newly populated ones */
    free(array_of_callbacks);
    array_of_callbacks = tmp_array_cb;
    free(array_of_requests);
    array_of_requests = tmp_array_req;

    mpi_funnelled_last_active_req = idx;

    current_size_of_total_reqs = size_of_total_reqs;
    assert((idx + MAX_DYNAMIC_REQ_RANGE) == current_size_of_total_reqs);
    parsec_atomic_unlock(&parsec_ce_am_build_lock);
    return PARSEC_SUCCESS;
}

static int
mpi_funnelled_tag_unregister_unsafe_internal(parsec_ce_tag_t tag)
{
    /* remove this tag from the arrays */
    mpi_funnelled_tag_t *tag_struct = &parsec_mpi_funnelled_array_of_registered_tags[tag];

    if( (PARSEC_CE_TAG_STATUS_ACTIVE == tag_struct->status) ||
        (PARSEC_CE_TAG_STATUS_DISABLE == tag_struct->status) ) {
        MPI_Status status;

        for(int flag, i = tag_struct->start_idx; i < tag_struct->start_idx + EACH_STATIC_REQ_RANGE; i++) {
#if !defined(CRAY_MPICH_VERSION)
            // MPI Cancel broken on Cray
            MPI_Cancel(&array_of_requests[i]);
            MPI_Test(&array_of_requests[i], &flag, &status);
#endif
            MPI_Request_free(&array_of_requests[i]);
            assert( MPI_REQUEST_NULL == array_of_requests[i] );
        }
        free(tag_struct->am_backend_memory);
        tag_struct->am_backend_memory = NULL;
    }
    tag_struct->callback = NULL;
    tag_struct->cb_data = NULL;
    tag_struct->start_idx = -1;
    tag_struct->status = PARSEC_CE_TAG_STATUS_INACTIVE;

    return PARSEC_SUCCESS;
}

int
mpi_no_thread_tag_unregister(parsec_ce_tag_t tag)
{
    mpi_funnelled_tag_t *tag_struct = &parsec_mpi_funnelled_array_of_registered_tags[tag];
    if( (PARSEC_CE_TAG_STATUS_INACTIVE == tag_struct->status) ||
        (PARSEC_CE_TAG_STATUS_DISABLE == tag_struct->status) ) {
        if(parsec_ce.parsec_context->nb_nodes > 1) {
            parsec_inform("Tag %ld is not registered\n", tag);
        }
        return PARSEC_SUCCESS;
    }
    parsec_atomic_lock(&parsec_ce_am_build_lock);
    if( PARSEC_CE_TAG_STATUS_ENABLE == tag_struct->status ) {
        /* requests not yet create, change the status and return */
        tag_struct->status = PARSEC_CE_TAG_STATUS_INACTIVE;
        parsec_atomic_unlock(&parsec_ce_am_build_lock);
        return PARSEC_SUCCESS;
    }
    tag_struct->status = PARSEC_CE_TAG_STATUS_DISABLE;
    /* if the engine is active, notify it to rebuild the arrays of requests at the next cycle */
    parsec_ce_am_design_version++;
    parsec_atomic_unlock(&parsec_ce_am_build_lock);

    return PARSEC_SUCCESS;
}

int
mpi_no_thread_mem_register(void *mem, parsec_mem_type_t mem_type,
                           size_t count, parsec_datatype_t datatype,
                           size_t mem_size,
                           parsec_ce_mem_reg_handle_t *lreg,
                           size_t *lreg_size)
{
    /* For now we only expect non-contiguous data or a layout and count */
    assert(mem_type == PARSEC_MEM_TYPE_NONCONTIGUOUS);
    (void) mem_type; (void) mem_size;

    /* This is mpi two_sided, the type can be of noncontiguous */
    *lreg = (void *)parsec_thread_mempool_allocate(mpi_funnelled_mem_reg_handle_mempool->thread_mempools);

    mpi_funnelled_mem_reg_handle_t *handle = (mpi_funnelled_mem_reg_handle_t *) *lreg;
    *lreg_size = sizeof(mpi_funnelled_mem_reg_handle_t);

    handle->self = handle;
    handle->mem  = mem;
    handle->datatype = datatype;
    handle->count = count;

    // Push in a table

    return 1;
}

int
mpi_no_thread_mem_unregister(parsec_ce_mem_reg_handle_t *lreg)
{
    //remove from table

    mpi_funnelled_mem_reg_handle_t *handle = (mpi_funnelled_mem_reg_handle_t *) *lreg;
    parsec_thread_mempool_free(mpi_funnelled_mem_reg_handle_mempool->thread_mempools, handle->self);
    return 1;
}

/* Returns the size of the memory handle that is opaque to the upper level */
int mpi_no_thread_get_mem_reg_handle_size(void)
{
    return sizeof(mpi_funnelled_mem_reg_handle_t);
}

/* Return the address of memory and the size that was registered
 * with a mem_reg_handle
 */
int
mpi_no_thread_mem_retrieve(parsec_ce_mem_reg_handle_t lreg,
                           void **mem, parsec_datatype_t *datatype, int *count)
{
    mpi_funnelled_mem_reg_handle_t *handle = (mpi_funnelled_mem_reg_handle_t *) lreg;
    *mem = handle->mem;
    *datatype = handle->datatype;
    *count = handle->count;

    return 1;
}

int
mpi_no_thread_put(parsec_comm_engine_t *ce,
                  parsec_ce_mem_reg_handle_t lreg,
                  ptrdiff_t ldispl,
                  parsec_ce_mem_reg_handle_t rreg,
                  ptrdiff_t rdispl,
                  size_t size,
                  int remote,
                  parsec_ce_onesided_callback_t l_cb, void *l_cb_data,
                  parsec_ce_tag_t r_tag, void *r_cb_data, size_t r_cb_data_size)
{
    assert(mpi_funnelled_last_active_req < current_size_of_total_reqs);

    (void)r_cb_data; (void) size;

    mpi_funnelled_callback_t *cb;

    int tag = next_tag(1);

    mpi_funnelled_mem_reg_handle_t *source_memory_handle = (mpi_funnelled_mem_reg_handle_t *) lreg;
    mpi_funnelled_mem_reg_handle_t *remote_memory_handle = (mpi_funnelled_mem_reg_handle_t *) rreg;


    mpi_funnelled_handshake_info_t handshake_info;

    handshake_info.tag = tag;
    handshake_info.source_memory_handle = source_memory_handle;
    handshake_info.remote_memory_handle = remote_memory_handle->self; /* pass the actual pointer
                                                                         instead of copying the whole
                                                                         memory_handle */
    handshake_info.cb_fn = (uintptr_t) r_tag;

    /* We pack the static message(handshake_info) and the callback data
     * the other side have sent us, to be forwarded.
     */
    int buf_size = sizeof(mpi_funnelled_handshake_info_t) + r_cb_data_size;
    void *buf = malloc(buf_size);
    memcpy( buf,
            &handshake_info,
            sizeof(mpi_funnelled_handshake_info_t) );
    memcpy( ((char *)buf) + sizeof(mpi_funnelled_handshake_info_t),
            r_cb_data,
            r_cb_data_size );

    /* Send AM to src to post Isend on this tag */
    /* this is blocking, so using data on stack is not a problem */
    ce->send_am(ce, PARSEC_CE_MPI_FUNNELLED_PUT_TAG_INTERNAL, remote, buf, buf_size);

    free(buf);

    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);
    /* Now we can post the Isend on the lreg */
    /*MPI_Isend((char *)ldata->mem + ldispl, ldata->size, MPI_BYTE, remote, tag, comm,
              &array_of_requests[mpi_funnelled_last_active_req]);*/

    int post_in_static_array = mpi_funnelled_last_active_req < current_size_of_total_reqs;
    mpi_funnelled_dynamic_req_t *item;

    if(post_in_static_array) {
        cb = &array_of_callbacks[mpi_funnelled_last_active_req];
        MPI_Isend((char *)source_memory_handle->mem + ldispl, source_memory_handle->count,
                  source_memory_handle->datatype, remote, tag, parsec_ce_mpi_comm,
                  &array_of_requests[mpi_funnelled_last_active_req]);
    } else {
        item = (mpi_funnelled_dynamic_req_t *)parsec_thread_mempool_allocate(mpi_funnelled_dynamic_req_mempool->thread_mempools);
        item->post_isend = 1;
        cb = &item->cb;
    }

    cb->storage1 = mpi_funnelled_last_active_req;
    cb->storage2 = remote;
    cb->cb_data  = l_cb_data;
    cb->tag_reg = NULL;
    cb->type = MPI_FUNNELLED_TYPE_ONESIDED;
    cb->is_dynamic_recv = false;

    cb->onesided.fct = l_cb;
    cb->onesided.lreg = source_memory_handle->self;
    cb->onesided.ldispl = ldispl;
    cb->onesided.rreg = remote_memory_handle;
    cb->onesided.rdispl = rdispl;
    cb->onesided.size = source_memory_handle->count;
    cb->onesided.remote = remote;
    cb->onesided.tag = tag;

    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_sendreq_fifo,
                                     (parsec_list_item_t *)item);
    }

    return 1;
}

int
mpi_no_thread_get(parsec_comm_engine_t *ce,
                  parsec_ce_mem_reg_handle_t lreg,
                  ptrdiff_t ldispl,
                  parsec_ce_mem_reg_handle_t rreg,
                  ptrdiff_t rdispl,
                  size_t size,
                  int remote,
                  parsec_ce_onesided_callback_t l_cb, void *l_cb_data,
                  parsec_ce_tag_t r_tag, void *r_cb_data, size_t r_cb_data_size)
{
    (void)r_tag; (void)r_cb_data;

    mpi_funnelled_callback_t *cb;
    MPI_Request *request;

    int tag = next_tag(1);

    mpi_funnelled_mem_reg_handle_t *source_memory_handle = (mpi_funnelled_mem_reg_handle_t *) lreg;
    mpi_funnelled_mem_reg_handle_t *remote_memory_handle = (mpi_funnelled_mem_reg_handle_t *) rreg;

    mpi_funnelled_handshake_info_t handshake_info;

    handshake_info.tag = tag;
    handshake_info.source_memory_handle = source_memory_handle;
    handshake_info.remote_memory_handle = remote_memory_handle->self; /* we store the actual pointer, as we
                                                                         do not pass the while handle */
    handshake_info.cb_fn = r_tag; /* This is what the other side has passed to us to invoke when the GET is done */

    /* Packing the callback data the other side has sent us and sending it back to them */
    int buf_size = sizeof(mpi_funnelled_handshake_info_t) + r_cb_data_size;

    void *buf = malloc(buf_size);
    memcpy( buf,
            &handshake_info,
            sizeof(mpi_funnelled_handshake_info_t) );
    memcpy( ((char *)buf) + sizeof(mpi_funnelled_handshake_info_t),
            r_cb_data,
            r_cb_data_size );


    /* Send AM to src to post Isend on this tag */
    /* this is blocking, so using data on stack is not a problem */
    ce->send_am(ce, PARSEC_CE_MPI_FUNNELLED_GET_TAG_INTERNAL, remote, buf, buf_size);

    free(buf);

    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);

    int post_in_static_array = mpi_funnelled_last_active_req < current_size_of_total_reqs;
    if (MAX_NUM_RECV_REQ_IN_ARRAY >= mpi_funnelled_num_recv_req_in_arr) {
        post_in_static_array = 0;
    } else if (post_in_static_array) {
        mpi_funnelled_num_recv_req_in_arr++;
    }

    mpi_funnelled_dynamic_req_t *item;

    if(post_in_static_array) {
        request = &array_of_requests[mpi_funnelled_last_active_req];
        cb = &array_of_callbacks[mpi_funnelled_last_active_req];
    } else {
        item = (mpi_funnelled_dynamic_req_t *)parsec_thread_mempool_allocate(mpi_funnelled_dynamic_req_mempool->thread_mempools);
        item->post_isend = 0;
        request = &item->request;
        cb = &item->cb;
    }

    MPI_Irecv((char*)source_memory_handle->mem + ldispl, source_memory_handle->count, source_memory_handle->datatype,
              remote, tag, parsec_ce_mpi_comm,
              request);

    cb->storage1 = mpi_funnelled_last_active_req;
    cb->storage2 = remote;
    cb->cb_data  = l_cb_data;
    cb->tag_reg = NULL;
    cb->type     = MPI_FUNNELLED_TYPE_ONESIDED;
    cb->is_dynamic_recv = true;

    cb->onesided.fct = l_cb;
    cb->onesided.lreg = source_memory_handle;
    cb->onesided.ldispl = ldispl;
    cb->onesided.rreg = remote_memory_handle;
    cb->onesided.rdispl = rdispl;
    cb->onesided.size = size;
    cb->onesided.remote = remote;
    cb->onesided.tag = tag;

    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_recvreq_fifo,
                                     (parsec_list_item_t *)item);
    }

    return 1;
}

int
mpi_no_thread_send_active_message(parsec_comm_engine_t *ce,
                                  parsec_ce_tag_t tag,
                                  int remote,
                                  void *addr, size_t size)
{
    (void) ce;
    mpi_funnelled_tag_t *tag_struct = &parsec_mpi_funnelled_array_of_registered_tags[tag];
    assert(tag_struct->msg_length >= size);
    (void) tag_struct;

    MPI_Send(addr, size, MPI_BYTE, remote, tag, parsec_ce_mpi_am_comm[tag]);

    return 1;
}

/* Common function to serve callbacks of completed request */
int
mpi_no_thread_serve_cb(parsec_comm_engine_t *ce, mpi_funnelled_callback_t *cb,
                       int mpi_tag, int mpi_source, int length, void *buf)
{
    int ret = 0;
    if(cb->type == MPI_FUNNELLED_TYPE_AM) {
        if(cb->cb_type.am.fct != NULL) {
            ret = cb->cb_type.am.fct(ce, mpi_tag, buf, length,
                                     mpi_source, cb->cb_data);
        }
        /* this is a persistent request, let's reset it */
        MPI_Start(&array_of_requests[cb->storage1]);
    } else if(cb->type == MPI_FUNNELLED_TYPE_ONESIDED) {
        if(NULL != cb->onesided.fct) {
            ret = cb->onesided.fct(ce, cb->onesided.lreg,
                                   cb->onesided.ldispl,
                                   cb->onesided.rreg,
                                   cb->onesided.rdispl,
                                   cb->onesided.size,
                                   cb->onesided.remote,
                                   cb->cb_data);
        }
    } else if (cb->type == MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM) {
        if(NULL != cb->cb_type.onesided_mimic_am.fct) {
            ret = cb->cb_type.onesided_mimic_am.fct(ce, mpi_tag, cb->cb_type.onesided_mimic_am.msg,
                                     length, mpi_source, cb->cb_data);
            free(cb->cb_type.onesided_mimic_am.msg);
        }
    } else {
        /* We only have three types */
        assert(0);
    }

    return ret;
}

static int
mpi_no_thread_push_posted_req(parsec_comm_engine_t *ce)
{
    (void) ce;
    assert(mpi_funnelled_last_active_req < current_size_of_total_reqs);
    assert(MAX_NUM_RECV_REQ_IN_ARRAY >= mpi_funnelled_num_recv_req_in_arr);

    mpi_funnelled_dynamic_req_t *item = NULL;
    if (MAX_NUM_RECV_REQ_IN_ARRAY > mpi_funnelled_num_recv_req_in_arr) {
        item = (mpi_funnelled_dynamic_req_t *) parsec_list_nolock_pop_front(&mpi_funnelled_dynamic_recvreq_fifo);
        if (NULL != item) {
            mpi_funnelled_num_recv_req_in_arr++;
            item->cb.is_dynamic_recv = true;
        }
    }
    if (NULL == item) {
        item = (mpi_funnelled_dynamic_req_t *) parsec_list_nolock_pop_front(&mpi_funnelled_dynamic_sendreq_fifo);
    }
    if (NULL == item) {
        return 0;
    }

    array_of_requests[mpi_funnelled_last_active_req] = item->request;
    item->request = MPI_REQUEST_NULL;

    array_of_callbacks[mpi_funnelled_last_active_req].storage1 = item->cb.storage1;
    array_of_callbacks[mpi_funnelled_last_active_req].storage2 = item->cb.storage2;
    array_of_callbacks[mpi_funnelled_last_active_req].cb_data = item->cb.cb_data;
    array_of_callbacks[mpi_funnelled_last_active_req].type = item->cb.type;
    array_of_callbacks[mpi_funnelled_last_active_req].tag_reg = item->cb.tag_reg;
    array_of_callbacks[mpi_funnelled_last_active_req].is_dynamic_recv = item->cb.is_dynamic_recv;

    if(item->cb.type == MPI_FUNNELLED_TYPE_ONESIDED) {
        array_of_callbacks[mpi_funnelled_last_active_req].onesided = item->cb.onesided;
    } else if (item->cb.type == MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM) {
        array_of_callbacks[mpi_funnelled_last_active_req].onesided = item->cb.onesided;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided_mimic_am.fct =
            item->cb.cb_type.onesided_mimic_am.fct;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided_mimic_am.msg =
            item->cb.cb_type.onesided_mimic_am.msg;
    } else {
        /* No other types of callbacks should be postponed */
        assert(0);
    }

    if(item->post_isend) {
        mpi_funnelled_mem_reg_handle_t *ldata = (mpi_funnelled_mem_reg_handle_t *) item->cb.onesided.lreg;
        MPI_Isend((char *)ldata->mem + item->cb.onesided.ldispl, ldata->count,
                  ldata->datatype, item->cb.onesided.remote, item->cb.onesided.tag, parsec_ce_mpi_comm,
                  &array_of_requests[mpi_funnelled_last_active_req]);
    }

    mpi_funnelled_last_active_req++;

    parsec_thread_mempool_free(mpi_funnelled_dynamic_req_mempool->thread_mempools, item);

    return 1;
}

int
mpi_no_thread_progress(parsec_comm_engine_t *ce)
{
    MPI_Status *status;
    int ret = 0, idx, outcount, pos;
    mpi_funnelled_callback_t *cb;
    int length;

    do {
        MPI_Testsome(mpi_funnelled_last_active_req, array_of_requests,
                     &outcount, array_of_indices, array_of_statuses);

        if(0 == outcount) goto feed_more_work;  /* can we push some more work? */

        /* Trigger the callbacks */
        for( idx = 0; idx < outcount; idx++ ) {
            cb = &array_of_callbacks[array_of_indices[idx]];
            if (cb->is_dynamic_recv) {
                mpi_funnelled_num_recv_req_in_arr--;
            }
            status = &(array_of_statuses[idx]);

            MPI_Get_count(status, MPI_PACKED, &length);

            /* Serve the callback and comeback */
            mpi_no_thread_serve_cb(ce, cb, status->MPI_TAG,
                                   status->MPI_SOURCE, length,
                                   MPI_FUNNELLED_TYPE_AM == cb->type ? (cb->tag_reg->am_backend_memory + cb->tag_reg->msg_length * cb->storage2) : NULL);
            ret++;
        }
        for( idx = outcount-1; idx >= 0; idx-- ) {
            pos = array_of_indices[idx];
            if(MPI_REQUEST_NULL != array_of_requests[pos])
                continue;  /* The callback replaced the completed request, keep going */
            assert(pos >= mpi_funnelled_static_req_idx);
            /* Get the last active callback to replace the empty one */
            mpi_funnelled_last_active_req--;
            if(mpi_funnelled_last_active_req > pos) {
                array_of_requests[pos]  = array_of_requests[mpi_funnelled_last_active_req];
                array_of_callbacks[pos] = array_of_callbacks[mpi_funnelled_last_active_req];
            }
            array_of_requests[mpi_funnelled_last_active_req] = MPI_REQUEST_NULL;
        }

      feed_more_work:
        /* check completion of posted requests */
        while(mpi_funnelled_last_active_req < current_size_of_total_reqs &&
              (!parsec_list_nolock_is_empty(&mpi_funnelled_dynamic_sendreq_fifo) ||
               !parsec_list_nolock_is_empty(&mpi_funnelled_dynamic_recvreq_fifo))) {
            assert(mpi_funnelled_last_active_req < current_size_of_total_reqs);
            if (0 == mpi_no_thread_push_posted_req(ce)) {
                break;
            }
        }
        if(0 == outcount) return ret;
    } while(1);
}

/**
 * @brief Check that the binding is correct. However, this operation is extremely expensive
 *        and highly not scalable so we should only do this operation when really necessary.
 * 
 * @param context 
 * @return int SUCCESS if the global bindings are OK, error otherwise.
 */
static int
parsec_check_overlapping_binding(parsec_context_t *context)
{
#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI) && defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    if( context->nb_nodes <= parsec_slow_bind_warning ) {
        MPI_Comm comml = MPI_COMM_NULL; int i, nl = 0, rl = MPI_PROC_NULL;
        MPI_Comm commw = (MPI_Comm)context->comm_ctx;
        assert(-1 != context->comm_ctx);
        MPI_Comm_split_type(commw, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comml);
        MPI_Comm_size(comml, &nl);
        if( 1 < nl ) {
            /* Hu-ho, double check that our binding is not conflicting with other
             * local procs */
            MPI_Comm_rank(comml, &rl);
            char *myset = NULL, *allsets = NULL;

            if( 0 != hwloc_bitmap_list_asprintf(&myset, context->cpuset_allowed_mask) ) {
            }
            int setlen = strlen(myset);
            int *setlens = NULL;
            if( 0 == rl ) {
                setlens = calloc(nl, sizeof(int));
            }
            MPI_Gather(&setlen, 1, MPI_INT, setlens, 1, MPI_INT, 0, comml);

            int *displs = NULL;
            if( 0 == rl ) {
                displs = calloc(nl, sizeof(int));
                displs[0] = 0;
                for( i = 1; i < nl; i++ ) {
                    displs[i] = displs[i-1]+setlens[i-1];
                }
                allsets = calloc(displs[nl-1]+setlens[nl-1], sizeof(char));
            }
            MPI_Gatherv(myset, setlen, MPI_CHAR, allsets, setlens, displs, MPI_CHAR, 0, comml);
            free(myset);

            if( 0 == rl ) {
                int notgood = false;
                for( i = 1; i < nl; i++ ) {
                    hwloc_bitmap_t other = hwloc_bitmap_alloc();
                    hwloc_bitmap_list_sscanf(other, &allsets[displs[i]]);
                    if(hwloc_bitmap_intersects(context->cpuset_allowed_mask, other)) {
                        notgood = true;
                    }
                    hwloc_bitmap_free(other);
                }
                if( notgood ) {
                    parsec_warning("/!\\ PERFORMANCE MIGHT BE REDUCED /!\\: "
                                   "Multiple PaRSEC processes on the same node may share the same physical core(s);\n"
                                    "\tThis is often unintentional, and will perform poorly.\n"
                                   "\tNote that in managed environments (e.g., ALPS, jsrun), the launcher may set `cgroups`\n"
                                   "\tand hide the real binding from PaRSEC; if you verified that the binding is correct,\n"
                                   "\tthis message can be silenced using the MCA argument `runtime_warn_slow_binding`.\n");
                }
                free(setlens);
                free(allsets);
                free(displs);
            }
        }
    }
    return PARSEC_SUCCESS;
#else
    (void)context;
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif
}

int
mpi_no_thread_enable(parsec_comm_engine_t *ce)
{
    parsec_context_t *context = ce->parsec_context;
    int i;

    /* Did anything changed that would require a reconstruction of the management structures? */
    assert(-1 != context->comm_ctx);
    if(parsec_ce_mpi_comm == (MPI_Comm)context->comm_ctx) {
        return PARSEC_SUCCESS;
    }
    /* Finish the initialization of the communication engine */
    parsec_ce.mem_register        = mpi_no_thread_mem_register;
    parsec_ce.mem_unregister      = mpi_no_thread_mem_unregister;
    parsec_ce.get_mem_handle_size = mpi_no_thread_get_mem_reg_handle_size;
    parsec_ce.mem_retrieve        = mpi_no_thread_mem_retrieve;
    parsec_ce.put                 = mpi_no_thread_put;
    parsec_ce.get                 = mpi_no_thread_get;
    parsec_ce.progress            = mpi_no_thread_progress;
    parsec_ce.pack                = mpi_no_thread_pack;
    parsec_ce.pack_size           = mpi_no_thread_pack_size;
    parsec_ce.unpack              = mpi_no_thread_unpack;
    parsec_ce.sync                = mpi_no_thread_sync;
    parsec_ce.reshape             = parsec_mpi_sendrecv;
    parsec_ce.can_serve           = mpi_no_thread_can_push_more;
    parsec_ce.send_am             = mpi_no_thread_send_active_message;

    /* Initialize the arrays */
    array_of_callbacks = (mpi_funnelled_callback_t *) calloc(MAX_DYNAMIC_REQ_RANGE,
                            sizeof(mpi_funnelled_callback_t));
    array_of_requests  = (MPI_Request *) calloc(MAX_DYNAMIC_REQ_RANGE, sizeof(MPI_Request));
    array_of_indices   = (int *) calloc(MAX_DYNAMIC_REQ_RANGE, sizeof(int));
    array_of_statuses  = (MPI_Status *) calloc(MAX_DYNAMIC_REQ_RANGE, sizeof(MPI_Status));

    for(i = 0; i < MAX_DYNAMIC_REQ_RANGE; i++) {
        array_of_requests[i] = MPI_REQUEST_NULL;
    }

    PARSEC_OBJ_CONSTRUCT(&mpi_funnelled_dynamic_sendreq_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&mpi_funnelled_dynamic_recvreq_fifo, parsec_list_t);

    mpi_funnelled_mem_reg_handle_mempool = (parsec_mempool_t*) malloc (sizeof(parsec_mempool_t));
    parsec_mempool_construct(mpi_funnelled_mem_reg_handle_mempool,
                             PARSEC_OBJ_CLASS(mpi_funnelled_mem_reg_handle_t), sizeof(mpi_funnelled_mem_reg_handle_t),
                             offsetof(mpi_funnelled_mem_reg_handle_t, mempool_owner),
                             1);

    mpi_funnelled_dynamic_req_mempool = (parsec_mempool_t*) malloc (sizeof(parsec_mempool_t));
    parsec_mempool_construct(mpi_funnelled_dynamic_req_mempool,
                             PARSEC_OBJ_CLASS(mpi_funnelled_dynamic_req_t), sizeof(mpi_funnelled_dynamic_req_t),
                             offsetof(mpi_funnelled_dynamic_req_t, mempool_owner),
                             1);

    MPI_Info info = MPI_INFO_NULL;
    MPI_Info_create(&info);
#if defined(PARSEC_HAVE_MPI_OVERTAKE)
    if( parsec_param_enable_mpi_overtake ) {
        MPI_Info_set(info, "mpi_assert_allow_overtaking", "true");
    }
#endif /* defined(PARSEC_HAVE_MPI_OVERTAKE) */
    /* There is no need to enable overtake for the AM communicator */
    MPI_Comm_dup_with_info((MPI_Comm) context->comm_ctx, info, &parsec_ce_mpi_comm);
    MPI_Info_free(&info);
    /* Replace the provided communicator with a pointer to the PaRSEC duplicate */
    context->comm_ctx = (uintptr_t)parsec_ce_mpi_comm;

    MPI_Comm_size(parsec_ce_mpi_comm, &(context->nb_nodes));
    MPI_Comm_rank(parsec_ce_mpi_comm, &(context->my_rank));

    for(i = 0; i < PARSEC_MAX_REGISTERED_TAGS; i++) {
        MPI_Comm_dup((MPI_Comm) context->comm_ctx, &parsec_ce_mpi_am_comm[i]);
    }

    parsec_check_overlapping_binding(context);

    parsec_ce_rebuild_am_requests();
    return 1;
}

int
mpi_no_thread_disable(parsec_comm_engine_t *ce)
{
    (void) ce;
    return 1;
}

int
mpi_no_thread_pack(parsec_comm_engine_t *ce,
                   void *inbuf, int incount, parsec_datatype_t type,
                   void *outbuf, int outsize,
                   int *positionA)
{
    (void) ce;
    return MPI_Pack(inbuf, incount, type, outbuf, outsize, positionA, parsec_ce_mpi_comm);

}

int
mpi_no_thread_pack_size(parsec_comm_engine_t *ce,
                        int incount, parsec_datatype_t type,
                        int* size)
{
    (void) ce;
    return MPI_Pack_size(incount, type, parsec_ce_mpi_comm, size);
}
int
mpi_no_thread_unpack(parsec_comm_engine_t *ce,
                     void *inbuf, int insize, int *position,
                     void *outbuf, int outcount, parsec_datatype_t type)
{
    (void) ce;
    return MPI_Unpack(inbuf, insize, position, outbuf, outcount, type, parsec_ce_mpi_comm);
}

/* Mechanism to post global synchronization from upper layer */
int
mpi_no_thread_sync(parsec_comm_engine_t *ce)
{
    (void) ce;
    MPI_Barrier(parsec_ce_mpi_comm);
    return 0;
}

/* The upper layer will query the bottom layer before pushing
 * additional one-sided messages.
 */
int
mpi_no_thread_can_push_more(parsec_comm_engine_t *ce)
{
    (void) ce;
#if 0
    /* Here we first push pending work before we decide and let the
     * upper layer know if they should push more work or not
     */
    /* Push saved requests first */
    while(mpi_funnelled_last_active_req < current_size_of_total_reqs &&
          !parsec_list_nolock_is_empty(&mpi_funnelled_dynamic_req_fifo)) {
        assert(mpi_funnelled_last_active_req < current_size_of_total_reqs);
        mpi_no_thread_progress_saved_req(ce);
    }
#endif

    /* Do we have room to post more requests? */
    return mpi_funnelled_last_active_req < current_size_of_total_reqs;
}
