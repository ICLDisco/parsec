/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
/**
 * @file
 *
 * Backend-local interface for the funnelled MPI communication engine.
 *
 * These entry points populate the transport function table exposed through
 * parsec_comm_engine_t.  They remain MPI-specific and should not be called
 * directly by generic remote-dependency code; generic code should use the
 * parsec_comm_engine_t callbacks instead.
 */
#ifndef __USE_PARSEC_MPI_FUNNELLED_H__
#define __USE_PARSEC_MPI_FUNNELLED_H__

#include "parsec/parsec_comm_engine.h"

/** Initialize the funnelled MPI communication engine for a PaRSEC context. */
parsec_comm_engine_t * mpi_funnelled_init(parsec_context_t *parsec_context);

/** Finalize the funnelled MPI communication engine instance. */
int mpi_funnelled_fini(parsec_comm_engine_t *comm_engine);

/** Register an active-message tag and receive callback in the MPI backend. */
int mpi_no_thread_tag_register(parsec_ce_tag_t tag,
                               parsec_ce_am_callback_t cb,
                               void *cb_data,
                               size_t msg_length);

/** Unregister a previously registered active-message tag. */
int mpi_no_thread_tag_unregister(parsec_ce_tag_t tag);

/** Register a local memory region and return the backend memory handle. */
int
mpi_no_thread_mem_register(void *mem, parsec_mem_type_t mem_type,
                           size_t count, parsec_datatype_t datatype,
                           size_t mem_size,
                           parsec_ce_mem_reg_handle_t *lreg,
                           size_t *lreg_size);

/** Release a memory handle returned by mpi_no_thread_mem_register(). */
int mpi_no_thread_mem_unregister(parsec_ce_mem_reg_handle_t *lreg);

/** Return the fixed wire size used for MPI memory-registration handles. */
int mpi_no_thread_get_mem_reg_handle_size(void);

/** Decode a local MPI memory-registration handle. */
int mpi_no_thread_mem_retrieve(parsec_ce_mem_reg_handle_t lreg, void **mem, parsec_datatype_t *datatype, int *count);

/** Start a remote PUT through the funnelled MPI backend. */
int mpi_no_thread_put(parsec_comm_engine_t *comm_engine,
                      parsec_ce_mem_reg_handle_t lreg,
                      ptrdiff_t ldispl,
                      parsec_ce_mem_reg_handle_t rreg,
                      ptrdiff_t rdispl,
                      size_t size,
                      int remote,
                      parsec_ce_onesided_callback_t l_cb, void *l_cb_data,
                      parsec_ce_tag_t r_tag, void *r_cb_data, size_t r_cb_data_size);

/** Start a remote GET through the funnelled MPI backend. */
int mpi_no_thread_get(parsec_comm_engine_t *comm_engine,
                      parsec_ce_mem_reg_handle_t lreg,
                      ptrdiff_t ldispl,
                      parsec_ce_mem_reg_handle_t rreg,
                      ptrdiff_t rdispl,
                      size_t size,
                      int remote,
                      parsec_ce_onesided_callback_t l_cb, void *l_cb_data,
                      parsec_ce_tag_t r_tag, void *r_cb_data, size_t r_cb_data_size);

/** Send an active message to a remote rank. */
int mpi_no_thread_send_active_message(parsec_comm_engine_t *comm_engine,
                                      parsec_ce_tag_t tag,
                                      int remote,
                                      void *addr, size_t size);

/** Make progress on pending MPI communication requests. */
int mpi_no_thread_progress(parsec_comm_engine_t *comm_engine);

/** Enable active-message receives for the MPI backend. */
int mpi_no_thread_enable(parsec_comm_engine_t *comm_engine);

/** Disable active-message receives for the MPI backend. */
int mpi_no_thread_disable(parsec_comm_engine_t *comm_engine);

/** Pack data using MPI datatype semantics. */
int mpi_no_thread_pack(parsec_comm_engine_t *ce,
                       void *inbuf, int incount, parsec_datatype_t type,
                       void *outbuf, int outsize,
                       int *positionA);

/** Compute the size needed to pack data with MPI datatype semantics. */
int mpi_no_thread_pack_size(parsec_comm_engine_t *ce,
                            int incount, parsec_datatype_t type,
                            int *size);

/** Unpack data using MPI datatype semantics. */
int mpi_no_thread_unpack(parsec_comm_engine_t *ce,
                         void *inbuf, int insize, int *position,
                         void *outbuf, int outcount, parsec_datatype_t type);

/** Synchronize all outstanding MPI communication operations. */
int mpi_no_thread_sync(parsec_comm_engine_t *comm_engine);

/** Report whether the MPI backend can accept more pending work. */
int
mpi_no_thread_can_push_more(parsec_comm_engine_t *c_e);

#endif /* __USE_PARSEC_MPI_FUNNELLED_H__ */
