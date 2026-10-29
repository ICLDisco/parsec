/*
 * PaRSEC4Julia - Public C Interface Header
 * 
 * Provides C function declarations for Julia FFI bindings
 */

#ifndef PARSEC4JULIA_H
#define PARSEC4JULIA_H

#include <stdio.h>
#include <stdint.h>
#include "parsec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Context Management */
/* ========================================================================== */

parsec_context_t* jl_parsec_init(int nb_cores);
int jl_parsec_context_start(parsec_context_t* ctx);
int jl_parsec_context_wait(parsec_context_t* ctx);
int jl_parsec_context_add_taskpool(parsec_context_t* ctx, parsec_taskpool_t* tp);
int jl_parsec_fini(parsec_context_t* ctx);

/* ========================================================================== */
/* Taskpool Management */
/* ========================================================================== */

parsec_taskpool_t* jl_parsec_dtd_taskpool_new(void);
int jl_parsec_taskpool_wait(parsec_taskpool_t* tp);
void jl_parsec_taskpool_free(parsec_taskpool_t* tp);

/* ========================================================================== */
/* Task Class Creation & Chores */
/* ========================================================================== */

parsec_task_class_t* jl_parsec_dtd_create_task_class(
    parsec_taskpool_t* tp,
    const char* name,
    int nargs,
    const int* types,
    const int* flags);

int jl_parsec_dtd_task_class_add_chore(
    parsec_taskpool_t* tp,
    parsec_task_class_t* tc,
    int device,
    void* kernel_func);

void jl_parsec_dtd_task_class_release(parsec_taskpool_t* tp, parsec_task_class_t* tc);

/* ========================================================================== */
/* Task Insertion */
/* ========================================================================== */

int jl_parsec_dtd_insert_task(
    parsec_taskpool_t* tp,
    parsec_task_class_t* tc,
    int priority,
    int device,
    int nargs,
    const int* ins_flags,
    void** args);

/* ========================================================================== */
/* Arena & Datatype Management */
/* ========================================================================== */

int jl_create_tile_full_arena(
    void* ctx, int mb, int nb, int* arena_id_out);

void jl_destroy_arena_datatype(void* ctx, int arena_id);

/* ========================================================================== */
/* Matrix Management */
/* ========================================================================== */

void* jl_matrix_bc_alloc(void);

int jl_matrix_bc_init(
    void* dc,
    const char* name,
    int mtype,
    int storage,
    int myrank,
    int mb, int nb,
    int m, int n,
    int i, int j,
    int m_global, int n_global,
    int p, int q,
    int kp, int kq);

void jl_matrix_bc_destroy(void* dc);

void* jl_dtd_tile_of(void* dc, int m, int n);

void* jl_matrix_bc_tiled_ptr(void* dc);
int jl_matrix_bc_nb_local_tiles(void* dc);
int jl_matrix_bc_bsiz(void* dc);
uintptr_t jl_matrix_bc_mat_ptr(void* dc);

void jl_dtd_data_collection_init(void* dc);

void jl_dtd_data_flush_all(void* tp, void* dc);

/* ========================================================================== */
/* Redistribute API (PTG + DTD) */
/* ========================================================================== */

int jl_parsec_redistribute_dtd(void* ctx, void* src_dc, void* dst_dc,
                               int size_row, int size_col,
                               int disi_Y, int disj_Y,
                               int disi_T, int disj_T);

int jl_parsec_redistribute(void* ctx, void* src_dc, void* dst_dc,
                           int size_row, int size_col,
                           int disi_Y, int disj_Y,
                           int disi_T, int disj_T);

/* ========================================================================== */
/* Predefined CPU Kernels */
/* ========================================================================== */

int jl_chore_zero_tile(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_chore_init_tile(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_chore_noop(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_chore_gemm_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_callback_signal_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_callback_signal_gpu(parsec_execution_stream_t *es, parsec_task_t *this_task);

/* ========================================================================== */
/* Kernel Lookup */
/* ========================================================================== */

void* jl_get_kernel_by_name(const char *kernel_name, int device_type);

/* ========================================================================== */
/* Julia Bridge: Request Queue + Proxy Kernel API */
/* ========================================================================== */

/* Request structure (opaque pointer for Julia side) */
typedef struct julia_req_t julia_req_t;

/* Initialize Julia bridge with request slots */
void parsec_julia_bridge_init(int nslots);

/* Wait for next request (blocking, called by Julia worker) */
julia_req_t* parsec_julia_wait_request(void);

/* Mark request as completed (called by Julia worker) */
void parsec_julia_complete_request(julia_req_t* req, int status);

/* Shutdown Julia bridge (signal workers to exit) */
void parsec_julia_bridge_shutdown(void);

/* Proxy kernels (registered with task classes) */
int jl_proxy_gemm_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_proxy_init_tile(parsec_execution_stream_t *es, parsec_task_t *this_task);

#ifdef __cplusplus
}
#endif

#endif /* PARSEC4JULIA_H */
