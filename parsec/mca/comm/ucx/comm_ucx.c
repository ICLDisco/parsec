/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
/**
 * @file
 *
 * UCX communication engine backend.
 *
 * This first UCX backend uses PMIx for bootstrap and UCX active messages plus
 * CPU-memory RMA.  It intentionally advertises only contiguous CPU datatype
 * support; callers that need sparse datatype movement must pack before handing
 * memory to this backend.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/comm/ucx/comm_ucx.h"
#include "parsec/parsec_comm_engine.h"
#include "parsec/remote_dep.h"
#include "parsec/utils/debug.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <pmix.h>
#include <ucp/api/ucp.h>

#define PARSEC_UCX_WORKER_ADDRESS_KEY "parsec.ucx.worker.address"
#define PARSEC_UCX_MAX_RKEY_SIZE 512

typedef struct parsec_ucx_am_header_s {
    int32_t source;
} parsec_ucx_am_header_t;

typedef struct parsec_ucx_callback_am_header_s {
    int32_t source;
    uintptr_t callback;
} parsec_ucx_callback_am_header_t;

typedef struct parsec_ucx_mem_handle_wire_s {
    uint64_t remote_addr;
    uint64_t mem_size;
    uint32_t rkey_size;
    unsigned char rkey[PARSEC_UCX_MAX_RKEY_SIZE];
} parsec_ucx_mem_handle_wire_t;

typedef struct parsec_ucx_mem_handle_s {
    parsec_ucx_mem_handle_wire_t wire;
    void *mem;
    size_t mem_size;
    parsec_datatype_t datatype;
    int count;
    ucp_mem_h memh;
} parsec_ucx_mem_handle_t;

typedef struct parsec_ucx_am_registration_s {
    parsec_ce_tag_t tag;
    parsec_ce_am_callback_t callback;
    void *cb_data;
    size_t max_msg_length;
} parsec_ucx_am_registration_t;

typedef struct parsec_ucx_state_s {
    pmix_proc_t pmix_proc;
    int pmix_initialized;
    int rank;
    int size;

    ucp_context_h context;
    ucp_worker_h worker;
    ucp_address_t *worker_address;
    size_t worker_address_length;
    ucp_ep_h *eps;
    int owns_context;
    int owns_worker;

    parsec_ucx_am_registration_t tags[PARSEC_MAX_REGISTERED_TAGS];
} parsec_ucx_state_t;

static parsec_ucx_state_t parsec_ucx_state;

static int comm_ucx_enable(parsec_comm_engine_t *comm_engine);
static int comm_ucx_disable(parsec_comm_engine_t *comm_engine);
static int comm_ucx_set_ctx(parsec_comm_engine_t *comm_engine, intptr_t ctx);
static int comm_ucx_fini(parsec_comm_engine_t *comm_engine);
static int comm_ucx_tag_register(parsec_ce_tag_t tag,
                                 parsec_ce_am_callback_t cb,
                                 void *cb_data,
                                 size_t msg_length);
static int comm_ucx_tag_unregister(parsec_ce_tag_t tag);
static int comm_ucx_mem_register(void *mem,
                                 parsec_mem_type_t mem_type,
                                 size_t count,
                                 parsec_datatype_t datatype,
                                 size_t mem_size,
                                 parsec_ce_mem_reg_handle_t *lreg,
                                 size_t *lreg_size);
static int comm_ucx_mem_unregister(parsec_ce_mem_reg_handle_t *lreg);
static int comm_ucx_get_mem_reg_handle_size(void);
static int comm_ucx_mem_retrieve(parsec_ce_mem_reg_handle_t lreg,
                                 void **mem,
                                 parsec_datatype_t *datatype,
                                 int *count);
static int comm_ucx_put(parsec_comm_engine_t *comm_engine,
                        parsec_ce_mem_reg_handle_t lreg,
                        ptrdiff_t ldispl,
                        parsec_ce_mem_reg_handle_t rreg,
                        ptrdiff_t rdispl,
                        size_t size,
                        int remote,
                        parsec_ce_onesided_callback_t l_cb,
                        void *l_cb_data,
                        parsec_ce_tag_t r_tag,
                        void *r_cb_data,
                        size_t r_cb_data_size);
static int comm_ucx_get(parsec_comm_engine_t *comm_engine,
                        parsec_ce_mem_reg_handle_t lreg,
                        ptrdiff_t ldispl,
                        parsec_ce_mem_reg_handle_t rreg,
                        ptrdiff_t rdispl,
                        size_t size,
                        int remote,
                        parsec_ce_onesided_callback_t l_cb,
                        void *l_cb_data,
                        parsec_ce_tag_t r_tag,
                        void *r_cb_data,
                        size_t r_cb_data_size);
static int comm_ucx_send_am(parsec_comm_engine_t *comm_engine,
                            parsec_ce_tag_t tag,
                            int remote,
                            void *addr,
                            size_t size);
static int comm_ucx_progress(parsec_comm_engine_t *comm_engine);
static int comm_ucx_pack(parsec_comm_engine_t *ce,
                         void *inbuf,
                         int incount,
                         parsec_datatype_t type,
                         void *outbuf,
                         int outsize,
                         int *position);
static int comm_ucx_pack_size(parsec_comm_engine_t *ce,
                              int incount,
                              parsec_datatype_t type,
                              int *size);
static int comm_ucx_unpack(parsec_comm_engine_t *ce,
                           void *inbuf,
                           int insize,
                           int *position,
                           void *outbuf,
                           int outcount,
                           parsec_datatype_t type);
static int comm_ucx_sync(parsec_comm_engine_t *comm_engine);
static int comm_ucx_can_serve(parsec_comm_engine_t *comm_engine);
static int comm_ucx_taskpool_sync_ids(parsec_comm_engine_t *comm_engine,
                                      intptr_t comm_ctx,
                                      uint32_t *next_taskpool_id);
static int comm_ucx_reshape(parsec_comm_engine_t *ce,
                            parsec_execution_stream_t *es,
                            parsec_data_copy_t *dst,
                            int64_t displ_dst,
                            parsec_datatype_t layout_dst,
                            uint64_t count_dst,
                            parsec_data_copy_t *src,
                            int64_t displ_src,
                            parsec_datatype_t layout_src,
                            uint64_t count_src);
static int comm_ucx_install_callback_am_handler(parsec_ucx_state_t *state);
static int comm_ucx_send_callback_am(parsec_comm_engine_t *comm_engine,
                                     int remote,
                                     parsec_ce_tag_t callback,
                                     void *cb_data,
                                     size_t cb_data_size);
static int comm_ucx_late_init(parsec_context_t *context,
                              parsec_ucx_state_t *state);

static int
comm_ucx_status_to_parsec(ucs_status_t status, const char *what)
{
    if( UCS_OK == status ) {
        return PARSEC_SUCCESS;
    }
    parsec_warning("UCX %s failed: %s", what, ucs_status_string(status));
    return PARSEC_ERROR;
}

static int
comm_ucx_wait_request(parsec_ucx_state_t *state, void *request, const char *what)
{
    ucs_status_t status;

    if( NULL == request ) {
        return PARSEC_SUCCESS;
    }
    if( UCS_PTR_IS_ERR(request) ) {
        return comm_ucx_status_to_parsec(UCS_PTR_STATUS(request), what);
    }

    do {
        status = ucp_request_check_status(request);
        if( UCS_INPROGRESS == status ) {
            ucp_worker_progress(state->worker);
        }
    } while( UCS_INPROGRESS == status );

    ucp_request_free(request);
    return comm_ucx_status_to_parsec(status, what);
}

static int
comm_ucx_direct_am(parsec_comm_engine_t *ce,
                   parsec_ucx_am_registration_t *registration,
                   void *addr,
                   size_t size,
                   int source)
{
    void *buffer = NULL;
    int rc;

    if( NULL == registration->callback ) {
        return PARSEC_ERR_NOT_FOUND;
    }
    if( registration->max_msg_length < size ) {
        return PARSEC_ERR_BAD_PARAM;
    }

    if( 0 != size ) {
        buffer = malloc(size);
        if( NULL == buffer ) {
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
        memcpy(buffer, addr, size);
    }
    rc = registration->callback(ce, registration->tag, buffer, size,
                                source, registration->cb_data);
    free(buffer);
    return rc;
}

static ucs_status_t
comm_ucx_am_callback(void *arg,
                     const void *header,
                     size_t header_length,
                     void *data,
                     size_t length,
                     const ucp_am_recv_param_t *param)
{
    parsec_ucx_am_registration_t *registration = (parsec_ucx_am_registration_t *)arg;
    parsec_ucx_am_header_t am_header;
    void *buffer = NULL;

    if( sizeof(am_header) != header_length ) {
        return UCS_ERR_INVALID_PARAM;
    }
    memcpy(&am_header, header, sizeof(am_header));

    if( NULL == registration->callback ) {
        return UCS_OK;
    }
    if( param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV ) {
        return UCS_ERR_UNSUPPORTED;
    }
    if( registration->max_msg_length < length ) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    if( 0 != length ) {
        buffer = malloc(length);
        if( NULL == buffer ) {
            return UCS_ERR_NO_MEMORY;
        }
        memcpy(buffer, data, length);
    }

    registration->callback(&parsec_ce, registration->tag, buffer, length,
                           am_header.source, registration->cb_data);
    free(buffer);
    return UCS_OK;
}

static ucs_status_t
comm_ucx_callback_am_callback(void *arg,
                              const void *header,
                              size_t header_length,
                              void *data,
                              size_t length,
                              const ucp_am_recv_param_t *param)
{
    parsec_ucx_callback_am_header_t callback_header;
    parsec_ce_am_callback_t callback;
    void *buffer = NULL;

    (void)arg;
    if( sizeof(callback_header) != header_length ) {
        return UCS_ERR_INVALID_PARAM;
    }
    memcpy(&callback_header, header, sizeof(callback_header));
    callback = (parsec_ce_am_callback_t)callback_header.callback;
    if( NULL == callback ) {
        return UCS_OK;
    }
    if( param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV ) {
        return UCS_ERR_UNSUPPORTED;
    }

    if( 0 != length ) {
        buffer = malloc(length);
        if( NULL == buffer ) {
            return UCS_ERR_NO_MEMORY;
        }
        memcpy(buffer, data, length);
    }

    callback(&parsec_ce, PARSEC_CE_REMOTE_DEP_PUT_END_TAG, buffer, length,
             callback_header.source, NULL);
    free(buffer);
    return UCS_OK;
}

static int
comm_ucx_install_am_handler(parsec_ucx_state_t *state, parsec_ce_tag_t tag)
{
    ucp_am_handler_param_t params;
    ucs_status_t status;

    if( tag >= PARSEC_MAX_REGISTERED_TAGS ) {
        return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
    }

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                        UCP_AM_HANDLER_PARAM_FIELD_CB |
                        UCP_AM_HANDLER_PARAM_FIELD_ARG |
                        UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    params.id = (uint16_t)tag;
    params.cb = comm_ucx_am_callback;
    params.arg = &state->tags[tag];
    params.flags = UCP_AM_FLAG_WHOLE_MSG;
    status = ucp_worker_set_am_recv_handler(state->worker, &params);
    return comm_ucx_status_to_parsec(status, "AM handler registration");
}

static int
comm_ucx_install_callback_am_handler(parsec_ucx_state_t *state)
{
    ucp_am_handler_param_t params;
    ucs_status_t status;

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                        UCP_AM_HANDLER_PARAM_FIELD_CB |
                        UCP_AM_HANDLER_PARAM_FIELD_ARG |
                        UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    params.id = PARSEC_CE_REMOTE_DEP_PUT_END_TAG;
    params.cb = comm_ucx_callback_am_callback;
    params.arg = state;
    params.flags = UCP_AM_FLAG_WHOLE_MSG;
    status = ucp_worker_set_am_recv_handler(state->worker, &params);
    return comm_ucx_status_to_parsec(status, "callback AM handler registration");
}

static int
comm_ucx_pmix_get_job_size(parsec_ucx_state_t *state)
{
    pmix_value_t *value = NULL;
    pmix_status_t prc;

    prc = PMIx_Get(&state->pmix_proc, PMIX_JOB_SIZE, NULL, 0, &value);
    if( PMIX_SUCCESS != prc ) {
        parsec_warning("PMIx failed to retrieve %s: %d", PMIX_JOB_SIZE, prc);
        return PARSEC_ERROR;
    }

    switch(value->type) {
    case PMIX_UINT32:
        state->size = (int)value->data.uint32;
        break;
    case PMIX_UINT64:
        state->size = (int)value->data.uint64;
        break;
    case PMIX_SIZE:
        state->size = (int)value->data.size;
        break;
    case PMIX_INT:
        state->size = value->data.integer;
        break;
    default:
        PMIX_VALUE_RELEASE(value);
        parsec_warning("PMIx returned unsupported %s type", PMIX_JOB_SIZE);
        return PARSEC_ERROR;
    }
    PMIX_VALUE_RELEASE(value);
    return (state->size > 0) ? PARSEC_SUCCESS : PARSEC_ERROR;
}

static int
comm_ucx_pmix_bootstrap(parsec_ucx_state_t *state)
{
    pmix_value_t value;
    pmix_status_t prc;

    prc = PMIx_Init(&state->pmix_proc, NULL, 0);
    if( PMIX_SUCCESS != prc ) {
        parsec_warning("PMIx_Init failed: %d", prc);
        return PARSEC_ERROR;
    }
    state->pmix_initialized = 1;
    state->rank = (int)state->pmix_proc.rank;
    if( PARSEC_SUCCESS != comm_ucx_pmix_get_job_size(state) ) {
        return PARSEC_ERROR;
    }

    memset(&value, 0, sizeof(value));
    value.type = PMIX_BYTE_OBJECT;
    value.data.bo.bytes = (char *)state->worker_address;
    value.data.bo.size = state->worker_address_length;

    prc = PMIx_Put(PMIX_GLOBAL, PARSEC_UCX_WORKER_ADDRESS_KEY, &value);
    if( PMIX_SUCCESS != prc ) {
        parsec_warning("PMIx_Put failed while publishing UCX worker address: %d", prc);
        return PARSEC_ERROR;
    }
    prc = PMIx_Commit();
    if( PMIX_SUCCESS != prc ) {
        parsec_warning("PMIx_Commit failed while publishing UCX worker address: %d", prc);
        return PARSEC_ERROR;
    }
    prc = PMIx_Fence(NULL, 0, NULL, 0);
    if( PMIX_SUCCESS != prc ) {
        parsec_warning("PMIx_Fence failed during UCX bootstrap: %d", prc);
        return PARSEC_ERROR;
    }
    return PARSEC_SUCCESS;
}

static int
comm_ucx_connect_endpoints(parsec_ucx_state_t *state)
{
    for(int peer_rank = 0; peer_rank < state->size; peer_rank++) {
        pmix_proc_t peer;
        pmix_value_t *value = NULL;
        pmix_status_t prc;
        ucp_ep_params_t ep_params;
        ucs_status_t status;

        if( peer_rank == state->rank ) {
            continue;
        }

        PMIX_LOAD_PROCID(&peer, state->pmix_proc.nspace, peer_rank);
        prc = PMIx_Get(&peer, PARSEC_UCX_WORKER_ADDRESS_KEY, NULL, 0, &value);
        if( PMIX_SUCCESS != prc ) {
            parsec_warning("PMIx_Get failed for UCX worker address of rank %d: %d",
                           peer_rank, prc);
            return PARSEC_ERROR;
        }
        if( (PMIX_BYTE_OBJECT != value->type) ||
            (NULL == value->data.bo.bytes) ||
            (0 == value->data.bo.size) ) {
            PMIX_VALUE_RELEASE(value);
            parsec_warning("PMIx returned an invalid UCX worker address for rank %d",
                           peer_rank);
            return PARSEC_ERROR;
        }

        memset(&ep_params, 0, sizeof(ep_params));
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                               UCP_EP_PARAM_FIELD_ERR_MODE;
        ep_params.address = (ucp_address_t *)value->data.bo.bytes;
        ep_params.err_mode = UCP_ERR_HANDLING_MODE_NONE;
        status = ucp_ep_create(state->worker, &ep_params, &state->eps[peer_rank]);
        PMIX_VALUE_RELEASE(value);
        if( UCS_OK != status ) {
            return comm_ucx_status_to_parsec(status, "endpoint creation");
        }
    }
    return PARSEC_SUCCESS;
}

static int
comm_ucx_init_context(parsec_ucx_state_t *state)
{
    ucp_config_t *config;
    ucp_params_t params;
    ucp_worker_params_t worker_params;
    ucs_status_t status;

    status = ucp_config_read(NULL, NULL, &config);
    if( UCS_OK != status ) {
        return comm_ucx_status_to_parsec(status, "config read");
    }

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_AM | UCP_FEATURE_RMA;
    status = ucp_init(&params, config, &state->context);
    ucp_config_release(config);
    if( UCS_OK != status ) {
        return comm_ucx_status_to_parsec(status, "context initialization");
    }
    state->owns_context = 1;

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(state->context, &worker_params, &state->worker);
    if( UCS_OK != status ) {
        return comm_ucx_status_to_parsec(status, "worker creation");
    }
    state->owns_worker = 1;

    status = ucp_worker_get_address(state->worker,
                                    &state->worker_address,
                                    &state->worker_address_length);
    return comm_ucx_status_to_parsec(status, "worker address retrieval");
}

static int
comm_ucx_attach_external_worker(parsec_ucx_state_t *state,
                                const parsec_comm_ucx_external_worker_t *external)
{
    ucs_status_t status;

    if( (NULL == external) ||
        (NULL == external->context) ||
        (NULL == external->worker) ) {
        return PARSEC_ERR_BAD_PARAM;
    }

    state->context = external->context;
    state->worker = external->worker;
    state->owns_context = 0;
    state->owns_worker = 0;

    status = ucp_worker_get_address(state->worker,
                                    &state->worker_address,
                                    &state->worker_address_length);
    return comm_ucx_status_to_parsec(status, "external worker address retrieval");
}

static void
comm_ucx_init_tags(parsec_ucx_state_t *state)
{
    for(parsec_ce_tag_t tag = 0; tag < PARSEC_MAX_REGISTERED_TAGS; tag++) {
        state->tags[tag].tag = tag;
        state->tags[tag].callback = NULL;
        state->tags[tag].cb_data = NULL;
        state->tags[tag].max_msg_length = 0;
    }
}

static void
comm_ucx_install_engine(parsec_context_t *context, parsec_ucx_state_t *state)
{
    parsec_ce.parsec_context = context;
    parsec_ce.capabilites.sided = 2;
    parsec_ce.capabilites.supports_noncontiguous_datatype = 0;
    parsec_ce.capabilites.multithreaded = 0;
    parsec_ce.enable = comm_ucx_enable;
    parsec_ce.disable = comm_ucx_disable;
    parsec_ce.set_ctx = comm_ucx_set_ctx;
    parsec_ce.fini = comm_ucx_fini;
    parsec_ce.tag_register = comm_ucx_tag_register;
    parsec_ce.tag_unregister = comm_ucx_tag_unregister;
    parsec_ce.mem_register = comm_ucx_mem_register;
    parsec_ce.mem_unregister = comm_ucx_mem_unregister;
    parsec_ce.get_mem_handle_size = comm_ucx_get_mem_reg_handle_size;
    parsec_ce.mem_retrieve = comm_ucx_mem_retrieve;
    parsec_ce.put = comm_ucx_put;
    parsec_ce.get = comm_ucx_get;
    parsec_ce.progress = comm_ucx_progress;
    parsec_ce.pack = comm_ucx_pack;
    parsec_ce.pack_size = comm_ucx_pack_size;
    parsec_ce.unpack = comm_ucx_unpack;
    parsec_ce.reshape = comm_ucx_reshape;
    parsec_ce.sync = comm_ucx_sync;
    parsec_ce.can_serve = comm_ucx_can_serve;
    parsec_ce.send_am = comm_ucx_send_am;
    parsec_ce.taskpool_sync_ids = comm_ucx_taskpool_sync_ids;

    context->my_rank = state->rank;
    context->nb_nodes = state->size;
    context->comm_ctx = (intptr_t)state;
}

static int
comm_ucx_late_init(parsec_context_t *context, parsec_ucx_state_t *state)
{
    if( PARSEC_SUCCESS != comm_ucx_pmix_bootstrap(state) ) {
        return PARSEC_ERROR;
    }

    state->eps = (ucp_ep_h *)calloc((size_t)state->size, sizeof(*state->eps));
    if( NULL == state->eps ) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    if( PARSEC_SUCCESS != comm_ucx_connect_endpoints(state) ) {
        return PARSEC_ERROR;
    }
    if( PARSEC_SUCCESS != comm_ucx_install_callback_am_handler(state) ) {
        return PARSEC_ERROR;
    }

    comm_ucx_init_tags(state);
    comm_ucx_install_engine(context, state);
    return PARSEC_SUCCESS;
}

parsec_comm_engine_t *
comm_ucx_init(parsec_context_t *context)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;
    intptr_t external_ctx = context->comm_ctx;

    memset(state, 0, sizeof(*state));
    state->rank = -1;
    state->size = -1;

    if( -1 != external_ctx ) {
        if( PARSEC_SUCCESS != comm_ucx_attach_external_worker(state,
                (const parsec_comm_ucx_external_worker_t *)external_ctx) ) {
            comm_ucx_fini(&parsec_ce);
            return NULL;
        }
    } else if( PARSEC_SUCCESS != comm_ucx_init_context(state) ) {
        comm_ucx_fini(&parsec_ce);
        return NULL;
    }

    if( PARSEC_SUCCESS != comm_ucx_late_init(context, state) ) {
        comm_ucx_fini(&parsec_ce);
        return NULL;
    }

    parsec_debug_verbose(4, parsec_debug_output,
                         "UCX communication engine initialized rank %d/%d",
                         context->my_rank, context->nb_nodes);
    return &parsec_ce;
}

static int
comm_ucx_enable(parsec_comm_engine_t *comm_engine)
{
    (void)comm_engine;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_disable(parsec_comm_engine_t *comm_engine)
{
    (void)comm_engine;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_set_ctx(parsec_comm_engine_t *comm_engine, intptr_t ctx)
{
    parsec_context_t *context = comm_engine->parsec_context;
    parsec_ucx_state_t *state = &parsec_ucx_state;
    int rc;

    if( 1 < parsec_communication_engine_up ) {
        parsec_warning("Cannot change PaRSEC's UCX worker while the communication engine is running [ignored]");
        return PARSEC_ERROR;
    }
    if( -1 == ctx ) {
        return PARSEC_ERR_BAD_PARAM;
    }

    /*
     * set_ctx hands PaRSEC an application-owned UCX worker. PaRSEC releases
     * only the resources it creates around that worker: worker address,
     * endpoints, AM handlers, and PMIx publication.
     */
    comm_ucx_fini(comm_engine);
    memset(state, 0, sizeof(*state));
    state->rank = -1;
    state->size = -1;

    rc = comm_ucx_attach_external_worker(state,
            (const parsec_comm_ucx_external_worker_t *)ctx);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    rc = comm_ucx_late_init(context, state);
    if( PARSEC_SUCCESS != rc ) {
        comm_ucx_fini(comm_engine);
    }
    return rc;
}

static int
comm_ucx_fini(parsec_comm_engine_t *comm_engine)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;
    ucp_request_param_t close_params;

    memset(&close_params, 0, sizeof(close_params));

    if( NULL != state->eps ) {
        for(int peer_rank = 0; peer_rank < state->size; peer_rank++) {
            if( NULL != state->eps[peer_rank] ) {
                void *request = ucp_ep_close_nbx(state->eps[peer_rank], &close_params);
                (void)comm_ucx_wait_request(state, request, "endpoint close");
                state->eps[peer_rank] = NULL;
            }
        }
        free(state->eps);
        state->eps = NULL;
    }
    if( NULL != state->worker_address ) {
        ucp_worker_release_address(state->worker, state->worker_address);
        state->worker_address = NULL;
        state->worker_address_length = 0;
    }
    if( NULL != state->worker ) {
        if( state->owns_worker ) {
            ucp_worker_destroy(state->worker);
        }
        state->worker = NULL;
    }
    if( NULL != state->context ) {
        if( state->owns_context ) {
            ucp_cleanup(state->context);
        }
        state->context = NULL;
    }
    if( state->pmix_initialized ) {
        PMIx_Finalize(NULL, 0);
        state->pmix_initialized = 0;
    }
    memset(state, 0, sizeof(*state));
    if( (NULL != comm_engine) && (NULL != comm_engine->parsec_context) ) {
        comm_engine->parsec_context->comm_ctx = -1;
    }
    return PARSEC_SUCCESS;
}

static int
comm_ucx_tag_register(parsec_ce_tag_t tag,
                      parsec_ce_am_callback_t cb,
                      void *cb_data,
                      size_t msg_length)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;

    if( tag >= PARSEC_MAX_REGISTERED_TAGS ) {
        return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
    }
    if( PARSEC_CE_REMOTE_DEP_PUT_END_TAG == tag ) {
        return PARSEC_ERR_EXISTS;
    }

    state->tags[tag].tag = tag;
    state->tags[tag].callback = cb;
    state->tags[tag].cb_data = cb_data;
    state->tags[tag].max_msg_length = msg_length;
    return comm_ucx_install_am_handler(state, tag);
}

static int
comm_ucx_tag_unregister(parsec_ce_tag_t tag)
{
    if( tag >= PARSEC_MAX_REGISTERED_TAGS ) {
        return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
    }
    if( PARSEC_CE_REMOTE_DEP_PUT_END_TAG == tag ) {
        return PARSEC_SUCCESS;
    }
    parsec_ucx_state.tags[tag].callback = NULL;
    parsec_ucx_state.tags[tag].cb_data = NULL;
    parsec_ucx_state.tags[tag].max_msg_length = 0;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_mem_register(void *mem,
                      parsec_mem_type_t mem_type,
                      size_t count,
                      parsec_datatype_t datatype,
                      size_t mem_size,
                      parsec_ce_mem_reg_handle_t *lreg,
                      size_t *lreg_size)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;
    parsec_ucx_mem_handle_t *handle;
    ucp_mem_map_params_t params;
    void *rkey_buffer = NULL;
    size_t rkey_size = 0;
    ucs_status_t status;

    if( (PARSEC_MEM_TYPE_CONTIGUOUS != mem_type) ||
        (NULL == mem) ||
        ((size_t)-1 == mem_size) ||
        (0 == mem_size) ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    handle = (parsec_ucx_mem_handle_t *)calloc(1, sizeof(*handle));
    if( NULL == handle ) {
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = mem;
    params.length = mem_size;
    status = ucp_mem_map(state->context, &params, &handle->memh);
    if( UCS_OK != status ) {
        free(handle);
        return comm_ucx_status_to_parsec(status, "memory registration");
    }

    status = ucp_rkey_pack(state->context, handle->memh, &rkey_buffer, &rkey_size);
    if( UCS_OK != status ) {
        ucp_mem_unmap(state->context, handle->memh);
        free(handle);
        return comm_ucx_status_to_parsec(status, "rkey packing");
    }
    if( rkey_size > PARSEC_UCX_MAX_RKEY_SIZE ) {
        ucp_rkey_buffer_release(rkey_buffer);
        ucp_mem_unmap(state->context, handle->memh);
        free(handle);
        parsec_warning("UCX rkey size %zu exceeds PaRSEC wire limit %d",
                       rkey_size, PARSEC_UCX_MAX_RKEY_SIZE);
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    handle->mem = mem;
    handle->mem_size = mem_size;
    handle->datatype = datatype;
    handle->count = (count > (size_t)INT_MAX) ? INT_MAX : (int)count;
    handle->wire.remote_addr = (uint64_t)(uintptr_t)mem;
    handle->wire.mem_size = (uint64_t)mem_size;
    handle->wire.rkey_size = (uint32_t)rkey_size;
    memcpy(handle->wire.rkey, rkey_buffer, rkey_size);
    ucp_rkey_buffer_release(rkey_buffer);

    *lreg = handle;
    *lreg_size = sizeof(handle->wire);
    return PARSEC_SUCCESS;
}

static int
comm_ucx_mem_unregister(parsec_ce_mem_reg_handle_t *lreg)
{
    parsec_ucx_mem_handle_t *handle;

    if( (NULL == lreg) || (NULL == *lreg) ) {
        return PARSEC_SUCCESS;
    }
    handle = (parsec_ucx_mem_handle_t *)*lreg;
    if( NULL != handle->memh ) {
        ucp_mem_unmap(parsec_ucx_state.context, handle->memh);
    }
    free(handle);
    *lreg = NULL;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_get_mem_reg_handle_size(void)
{
    return sizeof(parsec_ucx_mem_handle_wire_t);
}

static int
comm_ucx_mem_retrieve(parsec_ce_mem_reg_handle_t lreg,
                      void **mem,
                      parsec_datatype_t *datatype,
                      int *count)
{
    parsec_ucx_mem_handle_t *handle = (parsec_ucx_mem_handle_t *)lreg;

    *mem = handle->mem;
    *datatype = handle->datatype;
    *count = handle->count;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_rkey_unpack(parsec_ucx_state_t *state,
                     int remote,
                     parsec_ucx_mem_handle_wire_t *remote_wire,
                     ucp_rkey_h *rkey)
{
    ucs_status_t status;

    if( (remote < 0) || (remote >= state->size) ||
        (remote == state->rank) ||
        (NULL == state->eps[remote]) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    status = ucp_ep_rkey_unpack(state->eps[remote],
                                remote_wire->rkey,
                                rkey);
    return comm_ucx_status_to_parsec(status, "rkey unpack");
}

static int
comm_ucx_put(parsec_comm_engine_t *comm_engine,
             parsec_ce_mem_reg_handle_t lreg,
             ptrdiff_t ldispl,
             parsec_ce_mem_reg_handle_t rreg,
             ptrdiff_t rdispl,
             size_t size,
             int remote,
             parsec_ce_onesided_callback_t l_cb,
             void *l_cb_data,
             parsec_ce_tag_t r_tag,
             void *r_cb_data,
             size_t r_cb_data_size)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;
    parsec_ucx_mem_handle_t *local = (parsec_ucx_mem_handle_t *)lreg;
    parsec_ucx_mem_handle_wire_t *remote_wire = (parsec_ucx_mem_handle_wire_t *)rreg;
    size_t transfer_size = (0 == size) ? local->mem_size : size;
    char *local_addr = (char *)local->mem + ldispl;
    int rc;

    if( remote == state->rank ) {
        memcpy((void *)(uintptr_t)(remote_wire->remote_addr + rdispl),
               local_addr, transfer_size);
    } else {
        ucp_rkey_h rkey = NULL;
        ucp_request_param_t params;
        void *request;

        rc = comm_ucx_rkey_unpack(state, remote, remote_wire, &rkey);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
        memset(&params, 0, sizeof(params));
        request = ucp_put_nbx(state->eps[remote], local_addr, transfer_size,
                              remote_wire->remote_addr + rdispl, rkey, &params);
        rc = comm_ucx_wait_request(state, request, "PUT");
        ucp_rkey_destroy(rkey);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
    }

    if( 0 != r_tag ) {
        rc = comm_ucx_send_callback_am(comm_engine, remote, r_tag,
                                       r_cb_data, r_cb_data_size);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
    }
    if( NULL != l_cb ) {
        return l_cb(comm_engine, lreg, ldispl, rreg, rdispl,
                    transfer_size, remote, l_cb_data);
    }
    return PARSEC_SUCCESS;
}

static int
comm_ucx_get(parsec_comm_engine_t *comm_engine,
             parsec_ce_mem_reg_handle_t lreg,
             ptrdiff_t ldispl,
             parsec_ce_mem_reg_handle_t rreg,
             ptrdiff_t rdispl,
             size_t size,
             int remote,
             parsec_ce_onesided_callback_t l_cb,
             void *l_cb_data,
             parsec_ce_tag_t r_tag,
             void *r_cb_data,
             size_t r_cb_data_size)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;
    parsec_ucx_mem_handle_t *local = (parsec_ucx_mem_handle_t *)lreg;
    parsec_ucx_mem_handle_wire_t *remote_wire = (parsec_ucx_mem_handle_wire_t *)rreg;
    size_t transfer_size = (0 == size) ? local->mem_size : size;
    char *local_addr = (char *)local->mem + ldispl;
    int rc;

    if( remote == state->rank ) {
        memcpy(local_addr, (void *)(uintptr_t)(remote_wire->remote_addr + rdispl),
               transfer_size);
    } else {
        ucp_rkey_h rkey = NULL;
        ucp_request_param_t params;
        void *request;

        rc = comm_ucx_rkey_unpack(state, remote, remote_wire, &rkey);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
        memset(&params, 0, sizeof(params));
        request = ucp_get_nbx(state->eps[remote], local_addr, transfer_size,
                              remote_wire->remote_addr + rdispl, rkey, &params);
        rc = comm_ucx_wait_request(state, request, "GET");
        ucp_rkey_destroy(rkey);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
    }

    if( NULL != l_cb ) {
        rc = l_cb(comm_engine, lreg, ldispl, rreg, rdispl,
                  transfer_size, remote, l_cb_data);
        if( PARSEC_SUCCESS != rc ) {
            return rc;
        }
    }
    if( 0 != r_tag ) {
        /*
         * The comm-engine API carries the remote completion callback as a
         * function pointer in r_tag.  UCX AM ids cannot be those pointers, so
         * use the reserved internal AM id and carry the callback pointer in the
         * AM header.
         */
        return comm_ucx_send_callback_am(comm_engine, remote, r_tag,
                                         r_cb_data, r_cb_data_size);
    }
    return PARSEC_SUCCESS;
}

static int
comm_ucx_send_callback_am(parsec_comm_engine_t *comm_engine,
                          int remote,
                          parsec_ce_tag_t callback,
                          void *cb_data,
                          size_t cb_data_size)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;
    parsec_ucx_callback_am_header_t header;
    ucp_request_param_t params;
    void *request;

    if( (remote < 0) || (remote >= state->size) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    if( remote == state->rank ) {
        parsec_ce_am_callback_t cb = (parsec_ce_am_callback_t)(uintptr_t)callback;
        return cb(comm_engine, PARSEC_CE_REMOTE_DEP_PUT_END_TAG,
                  cb_data, cb_data_size, state->rank, NULL);
    }

    header.source = state->rank;
    header.callback = (uintptr_t)callback;
    memset(&params, 0, sizeof(params));
    request = ucp_am_send_nbx(state->eps[remote],
                              PARSEC_CE_REMOTE_DEP_PUT_END_TAG,
                              &header, sizeof(header),
                              cb_data, cb_data_size, &params);
    return comm_ucx_wait_request(state, request, "callback active message send");
}

static int
comm_ucx_send_am(parsec_comm_engine_t *comm_engine,
                 parsec_ce_tag_t tag,
                 int remote,
                 void *addr,
                 size_t size)
{
    parsec_ucx_state_t *state = &parsec_ucx_state;
    parsec_ucx_am_header_t header;
    ucp_request_param_t params;
    void *request;

    if( tag >= PARSEC_MAX_REGISTERED_TAGS ) {
        return PARSEC_ERR_VALUE_OUT_OF_BOUNDS;
    }
    if( (remote < 0) || (remote >= state->size) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    if( remote == state->rank ) {
        return comm_ucx_direct_am(comm_engine, &state->tags[tag],
                                  addr, size, state->rank);
    }
    if( state->tags[tag].max_msg_length < size ) {
        return PARSEC_ERR_BAD_PARAM;
    }

    header.source = state->rank;
    memset(&params, 0, sizeof(params));
    request = ucp_am_send_nbx(state->eps[remote], (unsigned)tag,
                              &header, sizeof(header),
                              addr, size, &params);
    return comm_ucx_wait_request(state, request, "active message send");
}

static int
comm_ucx_progress(parsec_comm_engine_t *comm_engine)
{
    int count = 0;

    (void)comm_engine;
    for(int i = 0; i < 16; i++) {
        int rc = ucp_worker_progress(parsec_ucx_state.worker);
        count += rc;
        if( 0 == rc ) {
            break;
        }
    }
    return count;
}

static int
comm_ucx_pack_size(parsec_comm_engine_t *ce,
                   int incount,
                   parsec_datatype_t type,
                   int *size)
{
    int dtt_size, rc;

    (void)ce;
    if( PARSEC_SUCCESS != parsec_type_contiguous(type) ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }
    rc = parsec_type_size(type, &dtt_size);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    *size = incount * dtt_size;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_pack(parsec_comm_engine_t *ce,
              void *inbuf,
              int incount,
              parsec_datatype_t type,
              void *outbuf,
              int outsize,
              int *position)
{
    int size, rc;

    rc = comm_ucx_pack_size(ce, incount, type, &size);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    if( (*position < 0) || ((*position + size) > outsize) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    memcpy((char *)outbuf + *position, inbuf, (size_t)size);
    *position += size;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_unpack(parsec_comm_engine_t *ce,
                void *inbuf,
                int insize,
                int *position,
                void *outbuf,
                int outcount,
                parsec_datatype_t type)
{
    int size, rc;

    rc = comm_ucx_pack_size(ce, outcount, type, &size);
    if( PARSEC_SUCCESS != rc ) {
        return rc;
    }
    if( (*position < 0) || ((*position + size) > insize) ) {
        return PARSEC_ERR_BAD_PARAM;
    }
    memcpy(outbuf, (char *)inbuf + *position, (size_t)size);
    *position += size;
    return PARSEC_SUCCESS;
}

static int
comm_ucx_sync(parsec_comm_engine_t *comm_engine)
{
    pmix_status_t prc;

    (void)comm_engine;
    prc = PMIx_Fence(NULL, 0, NULL, 0);
    return (PMIX_SUCCESS == prc) ? PARSEC_SUCCESS : PARSEC_ERROR;
}

static int
comm_ucx_can_serve(parsec_comm_engine_t *comm_engine)
{
    (void)comm_engine;
    return 1;
}

static int
comm_ucx_taskpool_sync_ids(parsec_comm_engine_t *comm_engine,
                           intptr_t comm_ctx,
                           uint32_t *next_taskpool_id)
{
    (void)comm_engine;
    (void)comm_ctx;
    (void)next_taskpool_id;
    /*
     * UCX will need a backend-specific collective, likely through the PMIx
     * bootstrap path, to replace MPI_Allreduce for taskpool-id convergence.
     */
    return PARSEC_ERR_NOT_IMPLEMENTED;
}

static int
comm_ucx_reshape(parsec_comm_engine_t *ce,
                 parsec_execution_stream_t *es,
                 parsec_data_copy_t *dst,
                 int64_t displ_dst,
                 parsec_datatype_t layout_dst,
                 uint64_t count_dst,
                 parsec_data_copy_t *src,
                 int64_t displ_src,
                 parsec_datatype_t layout_src,
                 uint64_t count_src)
{
    (void)ce; (void)es; (void)dst; (void)displ_dst; (void)layout_dst;
    (void)count_dst; (void)src; (void)displ_src; (void)layout_src;
    (void)count_src;
    return PARSEC_ERR_NOT_SUPPORTED;
}
