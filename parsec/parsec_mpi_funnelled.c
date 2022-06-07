/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
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

/* Range between which tags are allowed to be registered.
 * For now we allow 10 tags to be registered
 */
#define MPI_FUNNELLED_MIN_TAG 2
#define MPI_FUNNELLED_MAX_TAG (MPI_FUNNELLED_MIN_TAG + 10)

/* Internal TAG for GET and PUT activation message,
 * for two sides to agree on a "TAG" to post Irecv and Isend on
 */
#define MPI_FUNNELLED_GET_TAG_INTERNAL 0
#define MPI_FUNNELLED_PUT_TAG_INTERNAL 1

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
#define MIN_MPI_TAG (REMOTE_DEP_MAX_CTRL_TAG+1)
static int MAX_MPI_TAG = -1, mca_tag_ub = -1;
static volatile int __VAL_NEXT_TAG = MIN_MPI_TAG;
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
                MIN_MPI_TAG, __tag, k, MAX_MPI_TAG);
        __tag = MIN_MPI_TAG;
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

/* Range of index allowed for each type of request.
 * For registered tags, each will get 5 spots in the array of requests.
 * For dynamic tags, there will be a total of MAX_DYNAMIC_REQ_RANGE
 * spots in the same array.
 */
#define MAX_DYNAMIC_REQ_RANGE 30 /* according to current implementation */
#define EACH_STATIC_REQ_RANGE 5 /* for each registered tag */

/* Hash table for tag_structure. Each registered tags will be book-kept
 * using this structure.
 */
static int tag_hash_table_size = 1<<MPI_FUNNELLED_MAX_TAG; /**< Default tag hash table size */
static parsec_hash_table_t *tag_hash_table;

static parsec_key_fn_t tag_key_fns = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

typedef struct mpi_funnelled_tag_s {
    parsec_hash_table_item_t  ht_item;
    parsec_ce_tag_t tag; /* tag user wants to register */
    char **buf; /* Buffer where we will receive msg for each TAG
                 * there will be EACH_STATIC_REQ_RANGE buffers
                 * each of size msg_length.
                 */
    int start_idx; /* Records the starting index for every TAG
                    * to unregister from the array_of_[requests/indices/statuses]
                    */
    size_t msg_length; /* Maximum length allowed to send for each
                        * registered TAG.
                        */
} mpi_funnelled_tag_t;

typedef enum {
    MPI_FUNNELLED_TYPE_AM       = 0, /* indicating active message */
    MPI_FUNNELLED_TYPE_ONESIDED = 1,  /* indicating one sided */
    MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM = 2  /* indicating one sided with am callback type */
} mpi_funnelled_callback_type;

/* Structure to hold information about callbacks,
 * since we have multiple type of callbacks (active message and one-sided,
 * we store a type to know which to call.
 */
typedef struct mpi_funnelled_callback_s {
    long storage1; /* callback data */
    long storage2; /* callback data */
    void *cb_data; /* callback data */
    mpi_funnelled_callback_type type;
    mpi_funnelled_tag_t *tag;

    union {
        struct {
            parsec_ce_am_callback_t fct;
        } am;
        struct {
            parsec_ce_onesided_callback_t fct;
            parsec_ce_mem_reg_handle_t lreg; /* local memory handle */
            ptrdiff_t ldispl; /* displacement for local memory handle */
            parsec_ce_mem_reg_handle_t rreg; /* remote memory handle */
            ptrdiff_t rdispl; /* displacement for remote memory handle */
            size_t size; /* size of data */
            int remote; /* remote process id */
        } onesided;
        struct {
            parsec_ce_am_callback_t fct;
            void *msg;
        } onesided_mimic_am;
    } cb_type;
} mpi_funnelled_callback_t;

/* The internal communicator used by the communication engine to host its requests and
 * other operations. It is a copy of the context->comm_ctx (which is a duplicate of
 * whatever the user provides).
 */
static MPI_Comm dep_comm = MPI_COMM_NULL;
/* The internal communicator for all intra-node communications */
static MPI_Comm dep_self = MPI_COMM_NULL;

static mpi_funnelled_callback_t *array_of_callbacks;
static MPI_Request           *array_of_requests;
static int                   *array_of_indices;
static MPI_Status            *array_of_statuses;

static int size_of_total_reqs = 0;
static int mpi_funnelled_last_active_req = 0;
static int mpi_funnelled_static_req_idx = 0;

static int nb_internal_tag = 0;
static int count_internal_tag = 0;

#if defined(PARSEC_HAVE_MPI_OVERTAKE)
static int parsec_param_enable_mpi_overtake;
#endif

/* List to hold pending requests */
parsec_list_t mpi_funnelled_dynamic_req_fifo; /* ordered non threaded fifo */
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
    (void) ce; (void) tag; (void) msg_size; (void) cb_data;
    assert(mpi_funnelled_last_active_req < size_of_total_reqs);

    mpi_funnelled_callback_t *cb;
    MPI_Request *request;

    mpi_funnelled_handshake_info_t *handshake_info = (mpi_funnelled_handshake_info_t *) msg;


    /* This rank sent it's mem_reg in the activation msg, which is being
     * sent back as rreg of the msg */
    mpi_funnelled_mem_reg_handle_t *remote_memory_handle = (mpi_funnelled_mem_reg_handle_t *) (handshake_info->remote_memory_handle); /* This is the memory handle of the remote(our) side */

    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);

    int post_in_static_array = mpi_funnelled_last_active_req < size_of_total_reqs;
    mpi_funnelled_dynamic_req_t *item;

    if(post_in_static_array) {
        request = &array_of_requests[mpi_funnelled_last_active_req];
        cb = &array_of_callbacks[mpi_funnelled_last_active_req];
        MPI_Isend(remote_memory_handle->mem, remote_memory_handle->count, remote_memory_handle->datatype,
                  src, handshake_info->tag, dep_comm,
                  request);
    } else {
        item = (mpi_funnelled_dynamic_req_t *)parsec_thread_mempool_allocate(mpi_funnelled_dynamic_req_mempool->thread_mempools);
        item->post_isend = 1;
        request = &item->request;
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
    cb->cb_data  = cb->cb_data;
    cb->tag      = NULL;
    cb->type     = MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM;

    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_req_fifo,
                                     (parsec_list_item_t *)item);
        /*if(mpi_funnelled_last_active_req < size_of_total_reqs) {
            assert(mpi_funnelled_last_active_req < size_of_total_reqs);
            mpi_no_thread_push_posted_req(ce);
        }*/
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

    mpi_funnelled_mem_reg_handle_t *remote_memory_handle = (mpi_funnelled_mem_reg_handle_t *) (handshake_info->remote_memory_handle); /* This is the memory handle of the remote(our) side */

    assert(handshake_info->tag >= MIN_MPI_TAG);
    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);

    int _size;
    MPI_Type_size(remote_memory_handle->datatype, &_size);

    int post_in_static_array = 1;
    mpi_funnelled_dynamic_req_t *item;
    if(!(mpi_funnelled_last_active_req < size_of_total_reqs)) {
        post_in_static_array = 0;
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
              src, handshake_info->tag, dep_comm, request);

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
    cb->tag      = NULL;
    cb->type     = MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM;

    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_req_fifo,
                                     (parsec_list_item_t *)item);
        /*if(mpi_funnelled_last_active_req < size_of_total_reqs) {
            assert(mpi_funnelled_last_active_req < size_of_total_reqs);
            mpi_no_thread_push_posted_req(ce);
        }*/
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
                      dep_self, MPI_STATUS_IGNORE);
    (void)ce;
    return (MPI_SUCCESS == rc ? 0 : -1);
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

    assert(MPI_COMM_NULL == dep_self);
    MPI_Comm_dup(MPI_COMM_SELF, &dep_self);
    assert(MPI_COMM_NULL == dep_comm);

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

    /* Did anything changed that would require a build of the management structures? */
    assert(-1 != context->comm_ctx);
    if(dep_comm == (MPI_Comm)context->comm_ctx) {
        return &parsec_ce;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "rank %d ENABLE MPI communication engine",
                         parsec_debug_rank);

    dep_comm = (MPI_Comm) context->comm_ctx;

#if defined(PARSEC_HAVE_MPI_OVERTAKE)
    if( parsec_param_enable_mpi_overtake ) {
        MPI_Info no_order;
        MPI_Info_create(&no_order);
        MPI_Info_set(no_order, "mpi_assert_allow_overtaking", "true");
        MPI_Comm_set_info(dep_comm, no_order);
        MPI_Info_free(&no_order);
    }
#endif

    MPI_Comm_size(dep_comm, &(context->nb_nodes));
    MPI_Comm_rank(dep_comm, &(context->my_rank));

    /* init hash table for registered tags */
    tag_hash_table = PARSEC_OBJ_NEW(parsec_hash_table_t);
    for(i = 1; i < 16 && (1 << i) < tag_hash_table_size; i++) /* do nothing */;
    parsec_hash_table_init(tag_hash_table,
                           offsetof(mpi_funnelled_tag_t, ht_item),
                           i,
                           tag_key_fns,
                           tag_hash_table);

    /* Initialize the arrays */
    array_of_callbacks = (mpi_funnelled_callback_t *) calloc(MAX_DYNAMIC_REQ_RANGE,
                            sizeof(mpi_funnelled_callback_t));
    array_of_requests  = (MPI_Request *) calloc(MAX_DYNAMIC_REQ_RANGE,
                            sizeof(MPI_Request));
    array_of_indices   = (int *) calloc(MAX_DYNAMIC_REQ_RANGE, sizeof(int));
    array_of_statuses  = (MPI_Status *) calloc(MAX_DYNAMIC_REQ_RANGE,
                            sizeof(MPI_Status));

    for(i = 0; i < MAX_DYNAMIC_REQ_RANGE; i++) {
        array_of_requests[i] = MPI_REQUEST_NULL;
    }

    size_of_total_reqs += MAX_DYNAMIC_REQ_RANGE;

    nb_internal_tag = 2;

    /* Make all the fn pointers point to this component's function */
    parsec_ce.tag_register        = mpi_no_thread_tag_register;
    parsec_ce.tag_unregister      = mpi_no_thread_tag_unregister;
    parsec_ce.mem_register        = mpi_no_thread_mem_register;
    parsec_ce.mem_unregister      = mpi_no_thread_mem_unregister;
    parsec_ce.get_mem_handle_size = mpi_no_thread_get_mem_reg_handle_size;
    parsec_ce.mem_retrieve        = mpi_no_thread_mem_retrieve;
    parsec_ce.put                 = mpi_no_thread_put;
    parsec_ce.get                 = mpi_no_thread_get;
    parsec_ce.progress            = mpi_no_thread_progress;
    parsec_ce.enable              = mpi_no_thread_enable;
    parsec_ce.disable             = mpi_no_thread_disable;
    parsec_ce.pack                = mpi_no_thread_pack;
    parsec_ce.pack_size           = mpi_no_thread_pack_size;
    parsec_ce.unpack              = mpi_no_thread_unpack;
    parsec_ce.sync                = mpi_no_thread_sync;
    parsec_ce.reshape             = parsec_mpi_sendrecv;
    parsec_ce.can_serve           = mpi_no_thread_can_push_more;
    parsec_ce.send_am             = mpi_no_thread_send_active_message;

    parsec_ce.parsec_context      = context;
    parsec_ce.capabilites.sided   = 2;
    parsec_ce.capabilites.supports_noncontiguous_datatype = 1;

    /* Register for internal GET and PUT AMs */
    parsec_ce.tag_register(MPI_FUNNELLED_GET_TAG_INTERNAL,
                           mpi_funnelled_internal_get_am_callback,
                           context,
                           4096);
    count_internal_tag++;

    parsec_ce.tag_register(MPI_FUNNELLED_PUT_TAG_INTERNAL,
                           mpi_funnelled_internal_put_am_callback,
                           context,
                           4096);
    count_internal_tag++;

    PARSEC_OBJ_CONSTRUCT(&mpi_funnelled_dynamic_req_fifo, parsec_list_t);

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
    ce->tag_unregister(MPI_FUNNELLED_GET_TAG_INTERNAL);
    ce->tag_unregister(MPI_FUNNELLED_PUT_TAG_INTERNAL);

    free(array_of_callbacks); array_of_callbacks = NULL;
    free(array_of_requests);  array_of_requests  = NULL;
    free(array_of_indices);   array_of_indices   = NULL;
    free(array_of_statuses);  array_of_statuses  = NULL;

    parsec_hash_table_fini(tag_hash_table);
    PARSEC_OBJ_RELEASE(tag_hash_table);

    PARSEC_OBJ_DESTRUCT(&mpi_funnelled_dynamic_req_fifo);

    parsec_mempool_destruct(mpi_funnelled_mem_reg_handle_mempool);
    free(mpi_funnelled_mem_reg_handle_mempool);

    parsec_mempool_destruct(mpi_funnelled_dynamic_req_mempool);
    free(mpi_funnelled_dynamic_req_mempool);

    /* Remove the static handles */
    MPI_Comm_free(&dep_self); /* dep_self becomes MPI_COMM_NULL */

    /* Release the context communicators if any */
    if( -1 != ce->parsec_context->comm_ctx) {
        MPI_Comm_free((MPI_Comm*)&ce->parsec_context->comm_ctx);
        ce->parsec_context->comm_ctx = -1; /* We use -1 for the opaque comm_ctx, rather than the MPI specific MPI_COMM_NULL */
    }

    MAX_MPI_TAG = -1;  /* mark the layer as uninitialized */

    return 1;
}

/* Users need to register all tags before finalizing the comm
 * engine init.
 * The requested tags should be from 0 up to MPI_FUNNELLED_MAX_TAG,
 * dynamic tags will start from MPI_FUNNELLED_MAX_TAG.
 */
int
mpi_no_thread_tag_register(parsec_ce_tag_t tag,
                           parsec_ce_am_callback_t callback,
                           void *cb_data,
                           size_t msg_length)
{
    mpi_funnelled_callback_t *cb;

    /* All internal tags has been registered */
    if(nb_internal_tag == count_internal_tag) {
        if(tag < MPI_FUNNELLED_MIN_TAG || tag >= MPI_FUNNELLED_MAX_TAG) {
            parsec_warning("Tag is out of range, it has to be between %d - %d\n", MPI_FUNNELLED_MIN_TAG, MPI_FUNNELLED_MAX_TAG);
            return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
        }
        assert( (tag >= MPI_FUNNELLED_MIN_TAG) && (tag < MPI_FUNNELLED_MAX_TAG) );
    }

    if(NULL != parsec_hash_table_nolock_find(tag_hash_table, (parsec_key_t)tag)) {
        parsec_warning("Tag: %d is already registered\n", (int)tag);
        return PARSEC_ERR_EXISTS;
    }

    size_of_total_reqs += EACH_STATIC_REQ_RANGE;

    array_of_indices = realloc(array_of_indices, size_of_total_reqs * sizeof(int));
    array_of_statuses = realloc(array_of_statuses, size_of_total_reqs * sizeof(MPI_Status));

    /* Packing persistent tags in the beginning of the array */
    /* Allocate a new array that is "EACH_STATIC_REQ_RANGE" size bigger
     * than the previous allocation.
     */
    mpi_funnelled_callback_t *tmp_array_cb = malloc(sizeof(mpi_funnelled_callback_t) * size_of_total_reqs);
    /* Copy any previous persistent message info in the beginning */
    memcpy(tmp_array_cb, array_of_callbacks, sizeof(mpi_funnelled_callback_t) * mpi_funnelled_static_req_idx);
    /* Leaving "EACH_STATIC_REQ_RANGE" number elements in the middle and copying
     * the rest for the dynamic tag messages.
     */
    memcpy(tmp_array_cb + mpi_funnelled_static_req_idx + EACH_STATIC_REQ_RANGE, array_of_callbacks + mpi_funnelled_static_req_idx, sizeof(mpi_funnelled_callback_t) * MAX_DYNAMIC_REQ_RANGE);
    free(array_of_callbacks);
    array_of_callbacks = tmp_array_cb;

    /* Same procedure followed as array_of_callbacks. */
    MPI_Request *tmp_array_req = malloc(sizeof(MPI_Request) * size_of_total_reqs);
    memcpy(tmp_array_req, array_of_requests, sizeof(MPI_Request) * mpi_funnelled_static_req_idx);
    memcpy(tmp_array_req + mpi_funnelled_static_req_idx +  EACH_STATIC_REQ_RANGE, array_of_requests + mpi_funnelled_static_req_idx, sizeof(MPI_Request) * MAX_DYNAMIC_REQ_RANGE);
    free(array_of_requests);
    array_of_requests = tmp_array_req;

    char **buf = (char **) calloc(EACH_STATIC_REQ_RANGE, sizeof(char *));
    buf[0] = (char*)calloc(EACH_STATIC_REQ_RANGE, msg_length * sizeof(char));

    mpi_funnelled_tag_t *tag_struct = malloc(sizeof(mpi_funnelled_tag_t));
    tag_struct->tag = tag;
    tag_struct->buf = buf;
    tag_struct->start_idx  = mpi_funnelled_static_req_idx;
    tag_struct->msg_length = msg_length;

    for(int i = 0; i < EACH_STATIC_REQ_RANGE; i++) {
        buf[i] = buf[0] + i * msg_length * sizeof(char);

        /* Even though the address of array_of_requests changes after every
         * new registration of tags, the initialization of the requests will
         * still work as the memory is copied after initialization.
         */
        MPI_Recv_init(buf[i], msg_length, MPI_BYTE,
                      MPI_ANY_SOURCE, tag, dep_comm,
                      &array_of_requests[mpi_funnelled_static_req_idx]);

        cb = &array_of_callbacks[mpi_funnelled_static_req_idx];
        cb->cb_type.am.fct = callback;
        cb->storage1 = mpi_funnelled_static_req_idx;
        cb->storage2 = i;
        cb->cb_data  = cb_data;
        cb->tag      = tag_struct;
        cb->type     = MPI_FUNNELLED_TYPE_AM;
        MPI_Start(&array_of_requests[mpi_funnelled_static_req_idx]);
        mpi_funnelled_static_req_idx++;
    }

    /* insert in ht for bookkeeping */
    tag_struct->ht_item.key = (parsec_key_t)tag;
    parsec_hash_table_nolock_insert(tag_hash_table, &tag_struct->ht_item );

    assert((mpi_funnelled_static_req_idx + MAX_DYNAMIC_REQ_RANGE) == size_of_total_reqs);

    mpi_funnelled_last_active_req += EACH_STATIC_REQ_RANGE;

    return PARSEC_SUCCESS;
}

int
mpi_no_thread_tag_unregister(parsec_ce_tag_t tag)
{
    mpi_funnelled_tag_t *tag_struct = parsec_hash_table_nolock_find(tag_hash_table, (parsec_key_t)tag);
    if(NULL == tag_struct) {
        parsec_inform("Tag %ld is not registered\n", (int)tag);
        return 0;
    }

    /* remove this tag from the arrays */
    /* WARNING: Assumed after this no wait or test will be called on
     * array_of_requests
     */
    int i, flag;
    MPI_Status status;

    for(i = tag_struct->start_idx; i < tag_struct->start_idx + EACH_STATIC_REQ_RANGE; i++) {
        MPI_Cancel(&array_of_requests[i]);
        MPI_Test(&array_of_requests[i], &flag, &status);
        MPI_Request_free(&array_of_requests[i]);
        assert( MPI_REQUEST_NULL == array_of_requests[i] );
    }

    parsec_hash_table_remove(tag_hash_table, (parsec_key_t)tag);

    free(tag_struct->buf[0]);
    free(tag_struct->buf);

    free(tag_struct);

    return 1;
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
    assert(mpi_funnelled_last_active_req < size_of_total_reqs);

    (void)r_cb_data; (void) size;

    mpi_funnelled_callback_t *cb;
    MPI_Request *request;

    int tag = next_tag(1);
    assert(tag >= MIN_MPI_TAG);

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
    ce->send_am(ce, MPI_FUNNELLED_PUT_TAG_INTERNAL, remote, buf, buf_size);

    free(buf);

    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);
    /* Now we can post the Isend on the lreg */
    /*MPI_Isend((char *)ldata->mem + ldispl, ldata->size, MPI_BYTE, remote, tag, comm,
              &array_of_requests[mpi_funnelled_last_active_req]);*/

    int post_in_static_array = 1;
    mpi_funnelled_dynamic_req_t *item;
    if(!(mpi_funnelled_last_active_req < size_of_total_reqs)) {
        post_in_static_array = 0;
    }

    if(post_in_static_array) {
        request = &array_of_requests[mpi_funnelled_last_active_req];
        cb = &array_of_callbacks[mpi_funnelled_last_active_req];
        MPI_Isend((char *)source_memory_handle->mem + ldispl, source_memory_handle->count,
                  source_memory_handle->datatype, remote, tag, dep_comm,
                  request);
    } else {
        item = (mpi_funnelled_dynamic_req_t *)parsec_thread_mempool_allocate(mpi_funnelled_dynamic_req_mempool->thread_mempools);
        item->post_isend = 1;
        request = &item->request;
        cb = &item->cb;
    }

    cb->cb_type.onesided.fct = l_cb;
    cb->storage1 = mpi_funnelled_last_active_req;
    cb->storage2 = remote;
    cb->cb_data  = l_cb_data;
    cb->cb_type.onesided.lreg = source_memory_handle->self;
    cb->cb_type.onesided.ldispl = ldispl;
    cb->cb_type.onesided.rreg = remote_memory_handle;
    cb->cb_type.onesided.rdispl = rdispl;
    cb->cb_type.onesided.size = tag; /* This should be taken care of */
    cb->cb_type.onesided.remote = remote;
    cb->tag  = NULL;
    cb->type = MPI_FUNNELLED_TYPE_ONESIDED;

    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_req_fifo,
                                     (parsec_list_item_t *)item);
        /*if(mpi_funnelled_last_active_req < size_of_total_reqs) {
            assert(mpi_funnelled_last_active_req < size_of_total_reqs);
            mpi_no_thread_push_posted_req(ce);
        } */
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
    ce->send_am(ce, MPI_FUNNELLED_GET_TAG_INTERNAL, remote, buf, buf_size);

    free(buf);


    assert(mpi_funnelled_last_active_req >= mpi_funnelled_static_req_idx);

    int post_in_static_array = 1;
    mpi_funnelled_dynamic_req_t *item;
    if(!(mpi_funnelled_last_active_req < size_of_total_reqs)) {
        post_in_static_array = 0;
    }

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
              remote, tag, dep_comm,
              request);

    cb->cb_type.onesided.fct = l_cb;
    cb->storage1 = mpi_funnelled_last_active_req;
    cb->storage2 = remote;
    cb->cb_data  = l_cb_data;
    cb->cb_type.onesided.lreg = source_memory_handle;
    cb->cb_type.onesided.ldispl = ldispl;
    cb->cb_type.onesided.rreg = remote_memory_handle;
    cb->cb_type.onesided.rdispl = rdispl;
    cb->cb_type.onesided.size = size;
    cb->cb_type.onesided.remote = remote;
    cb->tag      = NULL;
    cb->type     = MPI_FUNNELLED_TYPE_ONESIDED;

    if(post_in_static_array) {
        mpi_funnelled_last_active_req++;
    } else {
        parsec_list_nolock_push_back(&mpi_funnelled_dynamic_req_fifo,
                                     (parsec_list_item_t *)item);
        /*if(mpi_funnelled_last_active_req < size_of_total_reqs) {
            assert(mpi_funnelled_last_active_req < size_of_total_reqs);
            mpi_no_thread_push_posted_req(ce);
        }*/
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
    parsec_key_t key = 0 | tag ;
    mpi_funnelled_tag_t *tag_struct = parsec_hash_table_nolock_find(tag_hash_table, key);
    assert(tag_struct->msg_length >= size);
    (void) tag_struct;

    MPI_Send(addr, size, MPI_BYTE, remote, tag, dep_comm);

    return 1;
}

/* Common function to serve callbacks of completed request */
int
mpi_no_thread_serve_cb(parsec_comm_engine_t *ce, mpi_funnelled_callback_t *cb,
                       int mpi_tag, int mpi_source, int length, void *buf,
                       int reset)
{
    int ret = 0;
    if(cb->type == MPI_FUNNELLED_TYPE_AM) {
        if(cb->cb_type.am.fct != NULL) {
            ret = cb->cb_type.am.fct(ce, mpi_tag, buf, length,
                                     mpi_source, cb->cb_data);
        }
        /* this is a persistent request, let's reset it if reset variable is ON */
        if(reset) {
            /* Let's re-enable the pending request in the same position */
            MPI_Start(&array_of_requests[cb->storage1]);
        }
    } else if(cb->type == MPI_FUNNELLED_TYPE_ONESIDED) {
        if(cb->cb_type.onesided.fct != NULL) {
            ret = cb->cb_type.onesided.fct(ce, cb->cb_type.onesided.lreg,
                                           cb->cb_type.onesided.ldispl,
                                           cb->cb_type.onesided.rreg,
                                           cb->cb_type.onesided.rdispl,
                                           cb->cb_type.onesided.size,
                                           cb->cb_type.onesided.remote,
                                           cb->cb_data);
        }
    } else if (cb->type == MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM) {
        if(cb->cb_type.onesided_mimic_am.fct != NULL) {
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
    assert(mpi_funnelled_last_active_req < size_of_total_reqs);

    mpi_funnelled_dynamic_req_t *item;
    item = (mpi_funnelled_dynamic_req_t *) parsec_list_nolock_pop_front(&mpi_funnelled_dynamic_req_fifo);

    MPI_Request tmp = array_of_requests[mpi_funnelled_last_active_req];
    array_of_requests[mpi_funnelled_last_active_req] = item->request;
    item->request = tmp;
    item->request = MPI_REQUEST_NULL;

    array_of_callbacks[mpi_funnelled_last_active_req].storage1 = item->cb.storage1;
    array_of_callbacks[mpi_funnelled_last_active_req].storage2 = item->cb.storage2;
    array_of_callbacks[mpi_funnelled_last_active_req].cb_data = item->cb.cb_data;
    array_of_callbacks[mpi_funnelled_last_active_req].type = item->cb.type;
    array_of_callbacks[mpi_funnelled_last_active_req].tag = item->cb.tag;

    if(item->cb.type == MPI_FUNNELLED_TYPE_ONESIDED) {
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided.fct    = item->cb.cb_type.onesided.fct;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided.lreg   = item->cb.cb_type.onesided.lreg;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided.ldispl = item->cb.cb_type.onesided.ldispl;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided.rreg   = item->cb.cb_type.onesided.rreg;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided.rdispl = item->cb.cb_type.onesided.rdispl;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided.size   = item->cb.cb_type.onesided.size;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided.remote = item->cb.cb_type.onesided.remote;
    } else if (item->cb.type == MPI_FUNNELLED_TYPE_ONESIDED_MIMIC_AM) {
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided_mimic_am.fct =
            item->cb.cb_type.onesided_mimic_am.fct;
        array_of_callbacks[mpi_funnelled_last_active_req].cb_type.onesided_mimic_am.msg =
            item->cb.cb_type.onesided_mimic_am.msg;
    } else {
        /* No other types of callbacks should be postponed */
        assert(0);
    }

    if(item->post_isend) {
        mpi_funnelled_mem_reg_handle_t *ldata = (mpi_funnelled_mem_reg_handle_t *) item->cb.cb_type.onesided.lreg;
        MPI_Isend((char *)ldata->mem + item->cb.cb_type.onesided.ldispl, ldata->count,
                  ldata->datatype, item->cb.cb_type.onesided.remote, item->cb.cb_type.onesided.size, dep_comm,
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
            status = &(array_of_statuses[idx]);

            MPI_Get_count(status, MPI_PACKED, &length);

            /* Serve the callback and comeback */
            mpi_no_thread_serve_cb(ce, cb, status->MPI_TAG,
                                   status->MPI_SOURCE, length,
                                   MPI_FUNNELLED_TYPE_AM == cb->type ? (void *)cb->tag->buf[cb->storage2] : NULL,
                                   1);
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
        while(mpi_funnelled_last_active_req < size_of_total_reqs &&
              !parsec_list_nolock_is_empty(&mpi_funnelled_dynamic_req_fifo)) {
            assert(mpi_funnelled_last_active_req < size_of_total_reqs);
            mpi_no_thread_push_posted_req(ce);
        }
        if(0 == outcount) return ret;
    } while(1);
}

int
mpi_no_thread_enable(parsec_comm_engine_t *ce)
{
    (void) ce;
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
    return MPI_Pack(inbuf, incount, type, outbuf, outsize, positionA, dep_comm);

}

int
mpi_no_thread_pack_size(parsec_comm_engine_t *ce,
                        int incount, parsec_datatype_t type,
                        int* size)
{
    (void) ce;
    return MPI_Pack_size(incount, type, dep_comm, size);
}
int
mpi_no_thread_unpack(parsec_comm_engine_t *ce,
                     void *inbuf, int insize, int *position,
                     void *outbuf, int outcount, parsec_datatype_t type)
{
    (void) ce;
    return MPI_Unpack(inbuf, insize, position, outbuf, outcount, type, dep_comm);
}

/* Mechanism to post global synchronization from upper layer */
int
mpi_no_thread_sync(parsec_comm_engine_t *ce)
{
    (void) ce;
    MPI_Barrier(dep_comm);
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
    while(mpi_funnelled_last_active_req < size_of_total_reqs &&
          !parsec_list_nolock_is_empty(&mpi_funnelled_dynamic_req_fifo)) {
        assert(mpi_funnelled_last_active_req < size_of_total_reqs);
        mpi_no_thread_progress_saved_req(ce);
    }
#endif

    /* Do we have room to post more requests? */
    return mpi_funnelled_last_active_req < size_of_total_reqs;
}
