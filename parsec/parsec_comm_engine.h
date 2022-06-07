/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_PARSEC_COMM_ENGINE_H__
#define __USE_PARSEC_COMM_ENGINE_H__

#include <stddef.h>
#include <stdint.h>
#include "parsec/runtime.h"
#include "parsec/datatype.h"

typedef enum {
    PARSEC_MEM_TYPE_CONTIGUOUS = 0,
    PARSEC_MEM_TYPE_NONCONTIGUOUS = 1
} parsec_mem_type_t;

typedef void* parsec_ce_mem_reg_handle_t;

typedef struct parsec_comm_engine_capabilites_s parsec_comm_engine_capabilites_t;

typedef struct parsec_comm_engine_s parsec_comm_engine_t;

typedef int (*parsec_ce_callback_t)(void *cb_data);

typedef uint64_t parsec_ce_tag_t;

typedef int (*parsec_ce_am_callback_t)(parsec_comm_engine_t *ce,
                                       parsec_ce_tag_t tag,
                                       void *msg,
                                       size_t msg_size,
                                       int src,
                                       void *cb_data);

typedef int (*parsec_ce_tag_register_fn_t)(parsec_ce_tag_t tag,
                                           parsec_ce_am_callback_t cb,
                                           void *cb_data,
                                           size_t msg_length/*bytes*/);

typedef int (*parsec_ce_tag_unregister_fn_t)(parsec_ce_tag_t tag);

/* PaRSEC will try to use non-contiguous type for lower layer capable of
 * supporting it.
 * For non-contiguous type the lower layer will expect layout and count and for
 * contiguous only size will be provided.
 * Please indicate the mem type using PARSEC_MEM_TYPE_CONTIGUOUS and
 * PARSEC_MEM_TYPE_NONCONTIGUOUS.
 */
typedef int (*parsec_ce_mem_register_fn_t)(void *mem, parsec_mem_type_t mem_type,
                                           size_t count, parsec_datatype_t datatype,
                                           size_t mem_size,
                                           parsec_ce_mem_reg_handle_t *lreg,
                                           size_t *lreg_size);

typedef int (*parsec_ce_mem_unregister_fn_t)(parsec_ce_mem_reg_handle_t *lreg);

typedef int (*parsec_ce_get_mem_reg_handle_size_fn_t)(void);

typedef int (*parsec_ce_mem_retrieve_fn_t)(parsec_ce_mem_reg_handle_t lreg, void **mem, parsec_datatype_t *datatype, int *count);

typedef int (*parsec_ce_onesided_callback_t)(parsec_comm_engine_t *comm_engine,
                             parsec_ce_mem_reg_handle_t lreg,
                             ptrdiff_t ldispl,
                             parsec_ce_mem_reg_handle_t rreg,
                             ptrdiff_t rdispl,
                             size_t size,
                             int remote,
                             void *cb_data);

typedef int (*parsec_ce_put_fn_t)(parsec_comm_engine_t *comm_engine,
                                  parsec_ce_mem_reg_handle_t lreg,
                                  ptrdiff_t ldispl,
                                  parsec_ce_mem_reg_handle_t rreg,
                                  ptrdiff_t rdispl,
                                  size_t size,
                                  int remote,
                                  parsec_ce_onesided_callback_t l_cb, void *l_cb_data,
                                  parsec_ce_tag_t r_tag, void *r_cb_data, size_t r_cb_data_size);

typedef int (*parsec_ce_get_fn_t)(parsec_comm_engine_t *comm_engine,
                                  parsec_ce_mem_reg_handle_t lreg,
                                  ptrdiff_t ldispl,
                                  parsec_ce_mem_reg_handle_t rreg,
                                  ptrdiff_t rdispl,
                                  size_t size,
                                  int remote,
                                  parsec_ce_onesided_callback_t l_cb, void *l_cb_data,
                                  parsec_ce_tag_t r_tag, void *r_cb_data, size_t r_cb_data_size);

typedef int (*parsec_ce_send_active_message_fn_t)(parsec_comm_engine_t *comm_engine,
                                             parsec_ce_tag_t tag,
                                             int remote,
                                             void *addr, size_t size);

typedef int (*parsec_ce_progress_fn_t)(parsec_comm_engine_t *comm_engine);

typedef int (*parsec_ce_enable_fn_t)(parsec_comm_engine_t *comm_engine);
typedef int (*parsec_ce_disable_fn_t)(parsec_comm_engine_t *comm_engine);

typedef int (*parsec_ce_pack_fn_t)(parsec_comm_engine_t *ce,
                                   void *inbuf, int incount, parsec_datatype_t type,
                                   void *outbuf, int outsize,
                                   int *positionA);

typedef int (*parsec_ce_pack_size_fn_t)(parsec_comm_engine_t *ce,
                                        int incount, parsec_datatype_t type,
                                        int *size);

typedef int (*parsec_ce_unpack_fn_t)(parsec_comm_engine_t *ce,
                                     void *inbuf, int insize, int *position,
                                     void *outbuf, int outcount, parsec_datatype_t type);

typedef int (*parsec_ce_sync_fn_t)(parsec_comm_engine_t *comm_engine);
typedef int (*parsec_ce_can_serve_fn_t)(parsec_comm_engine_t *comm_engine);

/**
 * This function realize a data reshaping, by conceptually packing the dst
 * into src.
 * TODO: need to distinguish between src_layout and dst_layout
 */
typedef int (*parsec_ce_reshape_fn_t)(parsec_comm_engine_t* ce,
                                      parsec_execution_stream_t* es,
                                      parsec_data_copy_t *dst,
                                      int64_t displ_dst,
                                      parsec_datatype_t layout_dst,
                                      uint64_t count_dst,
                                      parsec_data_copy_t *src,
                                      int64_t displ_src,
                                      parsec_datatype_t layout_src,
                                      uint64_t count_src);

struct parsec_comm_engine_capabilites_s {
    unsigned int sided : 2; /* Valid values are 1 and 2 */
    unsigned int supports_noncontiguous_datatype : 1;
    unsigned int multithreaded : 1;
};

struct parsec_comm_engine_s {
    parsec_context_t                      *parsec_context;
    parsec_comm_engine_capabilites_t       capabilites;
    parsec_ce_tag_register_fn_t            tag_register;
    parsec_ce_tag_unregister_fn_t          tag_unregister;
    parsec_ce_mem_register_fn_t            mem_register;
    parsec_ce_mem_unregister_fn_t          mem_unregister;
    parsec_ce_get_mem_reg_handle_size_fn_t get_mem_handle_size;
    parsec_ce_mem_retrieve_fn_t            mem_retrieve;
    parsec_ce_put_fn_t                     put;
    parsec_ce_get_fn_t                     get;
    parsec_ce_progress_fn_t                progress;
    parsec_ce_enable_fn_t                  enable;
    parsec_ce_disable_fn_t                 disable;
    parsec_ce_pack_fn_t                    pack;
    parsec_ce_pack_size_fn_t               pack_size;
    parsec_ce_unpack_fn_t                  unpack;
    parsec_ce_reshape_fn_t                 reshape;
    parsec_ce_sync_fn_t                    sync;
    parsec_ce_can_serve_fn_t               can_serve;
    parsec_ce_send_active_message_fn_t     send_am;
};

/* global comm_engine */
PARSEC_DECLSPEC extern parsec_comm_engine_t parsec_ce;

parsec_comm_engine_t * parsec_comm_engine_init(parsec_context_t *parsec_context);
int parsec_comm_engine_fini(parsec_comm_engine_t *comm_engine);

#endif /* __USE_PARSEC_COMM_ENGINE_H__ */
