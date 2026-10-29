/*
 * DTD Wrapper for PaRSEC4Julia
 * 
 * Provides C interface for Julia to access PaRSEC DTD API
 * Handles varargs wrapping for task class creation and insertion
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#include "parsec.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/redistribute/redistribute_internal.h"
#include "parsec/data_internal.h"
#include "parsec/interfaces/dtd/insert_function.h"

/* CBLAS interface for BLAS operations */
#ifdef HAVE_BLAS
typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef long long CBLAS_INDEX;

extern void cblas_dgemm64_(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
                           const CBLAS_INDEX K, const double alpha, const double  *A,
                           const CBLAS_INDEX lda, const double  *B, const CBLAS_INDEX ldb,
                           const double beta, double  *C, const CBLAS_INDEX ldc);
#endif

/* ========================================================================== */
/* Context Management */
/* ========================================================================== */

parsec_context_t* jl_parsec_init(int nb_cores)
{
    return parsec_init(nb_cores, NULL, NULL);
}

int jl_parsec_context_start(parsec_context_t* ctx)
{
    if (ctx == NULL) return -1;
    return parsec_context_start(ctx);
}

int jl_parsec_context_wait(parsec_context_t* ctx)
{
    if (ctx == NULL) return -1;
    return parsec_context_wait(ctx);
}

int jl_parsec_context_add_taskpool(parsec_context_t* ctx, parsec_taskpool_t* tp)
{
    if (ctx == NULL || tp == NULL) return -1;
    return parsec_context_add_taskpool(ctx, tp);
}

int jl_parsec_fini(parsec_context_t* ctx)
{
    if (ctx == NULL) return -1;
    return parsec_fini(&ctx);
}

/* ========================================================================== */
/* Taskpool Management */
/* ========================================================================== */

parsec_taskpool_t* jl_parsec_dtd_taskpool_new(void)
{
    return parsec_dtd_taskpool_new();
}

int jl_parsec_taskpool_wait(parsec_taskpool_t* tp)
{
    if (tp == NULL) return -1;
    return parsec_taskpool_wait(tp);
}

void jl_parsec_taskpool_free(parsec_taskpool_t* tp)
{
    if (tp != NULL) {
        parsec_taskpool_free(tp);
    }
}

/* ========================================================================== */
/* Task Class Creation (varargs wrapper) */
/* ========================================================================== */

parsec_task_class_t* jl_parsec_dtd_create_task_class(
    parsec_taskpool_t* tp,
    const char* name,
    int nargs,
    const int* types,
    const int* flags)
{
    if (tp == NULL || name == NULL) return NULL;
    if (nargs < 0 || nargs > 12) return NULL;

    /* Handle up to 12 args using varargs */
    switch (nargs) {
        case 0:
            return parsec_dtd_create_task_class(tp, name, PARSEC_DTD_ARG_END);
        
        case 1:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                PARSEC_DTD_ARG_END);
        
        case 2:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                PARSEC_DTD_ARG_END);
        
        case 3:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                PARSEC_DTD_ARG_END);
        
        case 4:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                PARSEC_DTD_ARG_END);
        
        case 5:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                PARSEC_DTD_ARG_END);
        
        case 6:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                types[5], flags[5],
                PARSEC_DTD_ARG_END);
        
        case 7:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                types[5], flags[5],
                types[6], flags[6],
                PARSEC_DTD_ARG_END);
        
        case 8:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                types[5], flags[5],
                types[6], flags[6],
                types[7], flags[7],
                PARSEC_DTD_ARG_END);
        
        case 9:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                types[5], flags[5],
                types[6], flags[6],
                types[7], flags[7],
                types[8], flags[8],
                PARSEC_DTD_ARG_END);
        
        case 10:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                types[5], flags[5],
                types[6], flags[6],
                types[7], flags[7],
                types[8], flags[8],
                types[9], flags[9],
                PARSEC_DTD_ARG_END);
        
        case 11:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                types[5], flags[5],
                types[6], flags[6],
                types[7], flags[7],
                types[8], flags[8],
                types[9], flags[9],
                types[10], flags[10],
                PARSEC_DTD_ARG_END);
        
        case 12:
            return parsec_dtd_create_task_class(tp, name,
                types[0], flags[0],
                types[1], flags[1],
                types[2], flags[2],
                types[3], flags[3],
                types[4], flags[4],
                types[5], flags[5],
                types[6], flags[6],
                types[7], flags[7],
                types[8], flags[8],
                types[9], flags[9],
                types[10], flags[10],
                types[11], flags[11],
                PARSEC_DTD_ARG_END);
        
        default:
            return NULL;
    }
}

/* ========================================================================== */
/* Chore Management */
/* ========================================================================== */

int jl_parsec_dtd_task_class_add_chore(
    parsec_taskpool_t* tp,
    parsec_task_class_t* tc,
    int device_type,
    void* fn)
{
    if (tp == NULL || tc == NULL) return -1;
    
    int ret = parsec_dtd_task_class_add_chore(tp, tc, device_type, fn);
    return ret;
}

void jl_parsec_dtd_task_class_release(
    parsec_taskpool_t* tp,
    parsec_task_class_t* tc)
{
    if (tp != NULL && tc != NULL) {
        parsec_dtd_task_class_release(tp, tc);
    }
}

/* ========================================================================== */
/* Task Insertion (varargs wrapper) */
/* ========================================================================== */

int jl_parsec_dtd_insert_task(
    parsec_taskpool_t* tp,
    parsec_task_class_t* tc,
    int priority,
    int device,
    int nargs,
    const int* ins_flags,
    void** args)
{
    if (tp == NULL || tc == NULL) return -1;
    if (nargs < 0 || nargs > 12) return -1;
    
    switch (nargs) {
        case 0:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 1:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 2:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 3:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 4:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 5:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 6:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                ins_flags[5], args[5],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 7:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                ins_flags[5], args[5],
                ins_flags[6], args[6],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 8:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                ins_flags[5], args[5],
                ins_flags[6], args[6],
                ins_flags[7], args[7],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 9:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                ins_flags[5], args[5],
                ins_flags[6], args[6],
                ins_flags[7], args[7],
                ins_flags[8], args[8],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 10:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                ins_flags[5], args[5],
                ins_flags[6], args[6],
                ins_flags[7], args[7],
                ins_flags[8], args[8],
                ins_flags[9], args[9],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 11:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                ins_flags[5], args[5],
                ins_flags[6], args[6],
                ins_flags[7], args[7],
                ins_flags[8], args[8],
                ins_flags[9], args[9],
                ins_flags[10], args[10],
                PARSEC_DTD_ARG_END);
            return 0;
        
        case 12:
            parsec_dtd_insert_task_with_task_class(tp, tc, priority, device,
                ins_flags[0], args[0],
                ins_flags[1], args[1],
                ins_flags[2], args[2],
                ins_flags[3], args[3],
                ins_flags[4], args[4],
                ins_flags[5], args[5],
                ins_flags[6], args[6],
                ins_flags[7], args[7],
                ins_flags[8], args[8],
                ins_flags[9], args[9],
                ins_flags[10], args[10],
                ins_flags[11], args[11],
                PARSEC_DTD_ARG_END);
            return 0;
        
        default:
            return -1;
    }
}

/* ========================================================================== */
/* Matrix Block-Cyclic */
/* ========================================================================== */

void* jl_matrix_bc_alloc(void)
{
    return (void*)calloc(1, sizeof(parsec_matrix_block_cyclic_t));
}

int jl_matrix_bc_init(
    void* dc,
    const char* name,
    int mtype, int storage, int myrank,
    int mb, int nb,
    int lm, int ln, int i0, int j0,
    int m, int n, int P, int Q,
    int kp, int kq)
{
    if (dc == NULL) {
        fprintf(stderr, "ERROR: jl_matrix_bc_init called with NULL dc\n");
        return -1;
    }
    
    parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
    
    /* Call official parsec_matrix_block_cyclic_init */
    parsec_matrix_block_cyclic_init(d, mtype, storage, myrank,
                                    mb, nb,
                                    lm, ln, i0, j0, m, n,
                                    P, Q, kp, kq, 0, 0);
    
    if (name != NULL) {
        parsec_data_collection_set_key((parsec_data_collection_t*)&d->super.super, name);
    }
    
    /* Allocate contiguous buffer */
    size_t buf_size = (size_t)d->super.nb_local_tiles * (size_t)d->super.bsiz * sizeof(double);
    d->mat = parsec_data_allocate(buf_size);
    
    if (d->mat == NULL) {
        fprintf(stderr, "ERROR: parsec_data_allocate failed\n");
        return -1;
    }
    
    /* DO NOT initialize DTD data collection here - let Julia do it explicitly */
    /* parsec_dtd_data_collection_init((parsec_data_collection_t*)&d->super.super); */
    
    return 0;
}

void jl_matrix_bc_destroy(void* dc)
{
    if (dc == NULL) return;
    
    parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
    parsec_data_collection_t* A = &d->super.super;
    
    /* Only call fini if hash table present */
    if (A->tile_h_table != NULL) {
        parsec_dtd_data_collection_fini(A);
    }
    
    if (d->mat != NULL) {
        parsec_data_free(d->mat);
    }
    
    parsec_tiled_matrix_destroy_data(&d->super);
    parsec_data_collection_destroy(A);
    free(d);
}

void* jl_dtd_tile_of(void* dc, int m, int n)
{
    if (dc == NULL) return NULL;
    
    parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
    if (d->super.super.tile_h_table == NULL) return NULL;
    
    parsec_data_key_t key = d->super.super.data_key(&d->super.super, m, n);
    return (void*)PARSEC_DTD_TILE_OF_KEY(&d->super.super, key);
}

void* jl_matrix_bc_tiled_ptr(void* dc)
{
    if (dc == NULL) return NULL;
    return (void*)&((parsec_matrix_block_cyclic_t*)dc)->super;
}

int jl_matrix_bc_nb_local_tiles(void* dc)
{
    if (dc == NULL) return -1;
    return ((parsec_matrix_block_cyclic_t*)dc)->super.nb_local_tiles;
}

int jl_matrix_bc_bsiz(void* dc)
{
    if (dc == NULL) return -1;
    return ((parsec_matrix_block_cyclic_t*)dc)->super.bsiz;
}

uintptr_t jl_matrix_bc_mat_ptr(void* dc)
{
    if (dc == NULL) return (uintptr_t)0;
    return (uintptr_t)((parsec_matrix_block_cyclic_t*)dc)->mat;
}

void jl_dtd_data_collection_init(void* dc)
{
    if (dc != NULL) {
        parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
        parsec_dtd_data_collection_init((parsec_data_collection_t*)&d->super.super);
    }
}

void jl_dtd_data_flush_all(void* tp, void* dc)
{
    if (tp != NULL && dc != NULL) {
        parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
        parsec_dtd_data_flush_all((parsec_taskpool_t*)tp,
                                  (parsec_data_collection_t*)&d->super.super);
    }
}

/* ========================================================================== */
/* Redistribute API (PTG + DTD)
 *
 * Julia stores parsec_matrix_block_cyclic_t* in ParsecMatrixBlockCyclic.dc.
 * PaRSEC redistribute expects parsec_tiled_matrix_t* (the `super` member),
 * matching Python's py_matrix_bc_tiled_ptr().
 */
/* ========================================================================== */

int jl_parsec_redistribute_dtd(void* ctx, void* src_dc, void* dst_dc,
                               int size_row, int size_col,
                               int disi_Y, int disj_Y,
                               int disi_T, int disj_T)
{
    if (ctx == NULL || src_dc == NULL || dst_dc == NULL) return -1;

    parsec_context_t* parsec = (parsec_context_t*)ctx;
    parsec_tiled_matrix_t* dcY = (parsec_tiled_matrix_t*)jl_matrix_bc_tiled_ptr(src_dc);
    parsec_tiled_matrix_t* dcT = (parsec_tiled_matrix_t*)jl_matrix_bc_tiled_ptr(dst_dc);
    if (dcY == NULL || dcT == NULL) return -1;

    return parsec_redistribute_dtd(parsec, dcY, dcT,
                                   size_row, size_col,
                                   disi_Y, disj_Y,
                                   disi_T, disj_T);
}

int jl_parsec_redistribute(void* ctx, void* src_dc, void* dst_dc,
                           int size_row, int size_col,
                           int disi_Y, int disj_Y,
                           int disi_T, int disj_T)
{
    if (ctx == NULL || src_dc == NULL || dst_dc == NULL) return -1;

    parsec_context_t* parsec = (parsec_context_t*)ctx;
    parsec_tiled_matrix_t* dcY = (parsec_tiled_matrix_t*)jl_matrix_bc_tiled_ptr(src_dc);
    parsec_tiled_matrix_t* dcT = (parsec_tiled_matrix_t*)jl_matrix_bc_tiled_ptr(dst_dc);
    if (dcY == NULL || dcT == NULL) return -1;

    return parsec_redistribute(parsec, dcY, dcT,
                               size_row, size_col,
                               disi_Y, disj_Y,
                               disi_T, disj_T);
}

/* ========================================================================== */
/* Arena Datatype */
/* ========================================================================== */

int jl_create_tile_full_arena(void* ctx, int mb, int nb, int* tile_full_dt)
{
    if (ctx == NULL || tile_full_dt == NULL) return -1;
    
    parsec_context_t* c = (parsec_context_t*)ctx;
    
    parsec_arena_datatype_t* adt = parsec_dtd_create_arena_datatype(c, tile_full_dt);
    if (adt == NULL) return -1;
    
    /* Add arena rect: column-major with ld=mb */
    parsec_add2arena_rect(adt, parsec_datatype_double_t, mb, nb, mb);
    
    return 0;
}

void jl_destroy_arena_datatype(void* ctx, int arena_id)
{
    if (ctx != NULL) {
        parsec_context_t* c = (parsec_context_t*)ctx;
        parsec_dtd_destroy_arena_datatype(c, arena_id);
    }
}
/* ========================================================================== */
/* Predefined Kernels for Julia */
/* ========================================================================== */

/**
 * Initialize tile with zero values
 * unpack_args: (double *data, int m, int n, int mb, int nb, int seed)
 */
int jl_chore_zero_tile(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    double *data;
    int m, n, mb, nb, seed;
    
    parsec_dtd_unpack_args(this_task, &data, &m, &n, &mb, &nb, &seed);
    
    if (data == NULL) return -1;
    
    /* Zero out tile */
    for (int i = 0; i < mb * nb; i++) {
        data[i] = 0.0;
    }
    
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Initialize tile with random values using LCG jump-ahead algorithm
 * Matches C reference implementation exactly
 * unpack_args: (double *data, int m, int n, int mb, int nb, unsigned int seed)
 */
#define RND64_A 6364136223846793005ULL
#define RND64_C 1ULL
#define RND_MUL 5.4210108624275222e-20

static unsigned long long jl_rnd64_jump(unsigned long long n, unsigned long long seed)
{
    unsigned long long a_k = RND64_A;
    unsigned long long c_k = RND64_C;
    unsigned long long ran = seed;
    
    while (n > 0) {
        if (n & 1) {
            ran = a_k * ran + c_k;
        }
        c_k *= (a_k + 1);
        a_k *= a_k;
        n >>= 1;
    }
    return ran;
}

/* ========================================================================== */
/* Basic Kernel Implementations */
/* ========================================================================== */

/**
 * No-op kernel (does nothing, returns success)
 */
int jl_chore_noop(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    (void)this_task;
    return PARSEC_HOOK_RETURN_DONE;
}

/* Callback signal kernel is implemented near the end of the file as
 * jl_callback_signal_cpu/jl_callback_signal_gpu.
 */

int jl_chore_init_tile(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    double *data;
    int m, n, mb, nb;
    unsigned int seed;
    
    parsec_dtd_unpack_args(this_task, &data, &m, &n, &mb, &nb, &seed);
    
    if (data == NULL) return -1;
    
    /* Initialize tile using LCG jump-ahead algorithm (matching C reference)
     * This produces the same random sequence as the official C implementation
     */
    unsigned long long jump = (unsigned long long)m + (unsigned long long)n * 1000000ULL;
    unsigned long long ran;
    
    for (int j = 0; j < nb; j++) {
        ran = jl_rnd64_jump(mb * jump, (unsigned long long)seed);
        for (int i = 0; i < mb; i++) {
            /* Store in column-major order (Fortran/PaRSEC convention) */
            data[i + j * mb] = 0.5 - ran * RND_MUL;
            ran = RND64_A * ran + RND64_C;
        }
        jump += 1000000ULL;
    }
    
    return PARSEC_HOOK_RETURN_DONE;
}

#ifdef HAVE_BLAS
/* CBLAS GEMM is declared in cblas.h */

/**
 * CPU BLAS GEMM kernel using cblas_dgemm
 * Computes: C = A*B + C
 * Layout: Column-major (Fortran order, matching PaRSEC arena)
 * unpack_args: (double *A, double *B, double *C, int m, int n, int k, 
 *               int mb, int nb, int kb)
 */
int jl_chore_gemm_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    double *A, *B, *C;
    int m, n, k, mb, nb, kb;
    
    parsec_dtd_unpack_args(this_task, &A, &B, &C, &m, &n, &k, &mb, &nb, &kb);
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "ERROR: GEMM kernel received NULL pointer\n");
        return -1;
    }
    
    /* Use CBLAS DGEMM: C = alpha*A*B + beta*C
     * Layout: Column-Major (Fortran order) to match PaRSEC arena
     * A: mb x kb matrix, leading dimension = mb
     * B: kb x nb matrix, leading dimension = kb  
     * C: mb x nb matrix, leading dimension = mb
     */
    cblas_dgemm64_(CblasColMajor,      /* Column-major layout (Fortran) */
                   CblasNoTrans,       /* A not transposed */
                   CblasNoTrans,       /* B not transposed */
                   (CBLAS_INDEX)mb, (CBLAS_INDEX)nb, (CBLAS_INDEX)kb,  /* M, N, K */
                   1.0,                /* alpha = 1.0 */
                   A, (CBLAS_INDEX)mb, /* A, lda = mb */
                   B, (CBLAS_INDEX)kb, /* B, ldb = kb */
                   1.0,                /* beta = 1.0 (accumulate) */
                   C, (CBLAS_INDEX)mb);/* C, ldc = mb */
    
    return PARSEC_HOOK_RETURN_DONE;
}
#endif

/* ========================================================================== */
/* Helper: Get kernel function pointer by name */
/* ========================================================================== */

/* Forward declarations for proxy kernels */
#ifdef HAVE_BLAS
int jl_proxy_gemm_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_proxy_init_tile(parsec_execution_stream_t *es, parsec_task_t *this_task);
#endif

/* Forward declarations for callback kernels */
int jl_callback_signal_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task);
int jl_callback_signal_gpu(parsec_execution_stream_t *es, parsec_task_t *this_task);

/**
 * Get predefined kernel function pointer by name
 * Returns pointer to kernel function or NULL if not found
 */
void* jl_get_kernel_by_name(const char *kernel_name, int device_type)
{
    if (kernel_name == NULL) return NULL;

    /* Callback signal kernel (available on all devices) */
    if (strcmp(kernel_name, "callback_signal") == 0) {
        if (device_type == PARSEC_DEV_CUDA) {
            return (void*)jl_callback_signal_gpu;
        } else {
            return (void*)jl_callback_signal_cpu;
        }
    }
    
    if (device_type == PARSEC_DEV_CPU) {
        if (strcmp(kernel_name, "zero_tile") == 0) {
            return (void*)jl_chore_zero_tile;
        } else if (strcmp(kernel_name, "init_tile") == 0) {
            return (void*)jl_chore_init_tile;
        } else if (strcmp(kernel_name, "noop") == 0) {
            return (void*)jl_chore_noop;
        }
#ifdef HAVE_BLAS
        else if (strcmp(kernel_name, "gemm_cpu") == 0) {
            return (void*)jl_chore_gemm_cpu;
        } else if (strcmp(kernel_name, "julia_proxy_gemm") == 0) {
            return (void*)jl_proxy_gemm_cpu;
        } else if (strcmp(kernel_name, "julia_proxy_init") == 0) {
            return (void*)jl_proxy_init_tile;
        }
#endif
    }
    
    return NULL;
}

/* ========================================================================== */
/* Julia Bridge: Request Queue + Proxy Kernel */
/* ========================================================================== */

#include <pthread.h>

/* Request structure passed from C proxy to Julia worker */
typedef struct {
    /* Request kind */
    int kind;  /* 1=callback, 2=gemm, 3=init */

    /* Kernel identification */
    int kernel_id;

    /* Optional signal pointer for callback */
    void *signal_ptr;
    
    /* Tile data pointers */
    void *A;
    void *B;
    void *C;
    
    /* Tile dimensions */
    int mb, nb, kb;
    
    /* Leading dimensions (for stride) */
    int lda, ldb, ldc;
    
    /* Scalar parameters */
    double alpha;
    double beta;
    
    /* Synchronization fields */
    pthread_mutex_t lock;
    pthread_cond_t cv;
    int state;  /* 0=empty, 1=ready (C->Julia), 2=done (Julia->C) */
} julia_req_t;

/* Request queue (simple ring buffer) */
#define MAX_REQUESTS 64
static julia_req_t g_requests[MAX_REQUESTS];
static int g_req_head = 0;  /* Where C pushes new requests */
static int g_req_tail = 0;  /* Where Julia pops requests */
static pthread_mutex_t g_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_queue_cv = PTHREAD_COND_INITIALIZER;
static int g_queue_shutdown = 0;

/* Limit outstanding compute requests to avoid blocking all PaRSEC workers */
#define MAX_OUTSTANDING 2
static int g_outstanding = 0;
static pthread_mutex_t g_out_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_out_cv = PTHREAD_COND_INITIALIZER;

/* Kernel IDs (must match Julia side) */
#define KERNEL_ID_INIT_TILE  1
#define KERNEL_ID_GEMM       2

/* Request kinds */
#define REQ_KIND_CALLBACK 1
#define REQ_KIND_GEMM     2
#define REQ_KIND_INIT     3

/**
 * Initialize Julia bridge (allocate request structures)
 */
void parsec_julia_bridge_init(int nslots)
{
    (void)nslots; /* We use fixed size ring buffer */
    
    for (int i = 0; i < MAX_REQUESTS; i++) {
        pthread_mutex_init(&g_requests[i].lock, NULL);
        pthread_cond_init(&g_requests[i].cv, NULL);
        g_requests[i].state = 0;  /* empty */
    }
    
    g_req_head = 0;
    g_req_tail = 0;
    g_queue_shutdown = 0;
    
    printf("[Julia Bridge] Initialized with %d request slots\n", MAX_REQUESTS);
}

/**
 * Push a request to the queue (called by C proxy kernel)
 * Returns pointer to the request slot
 */
static julia_req_t* push_request(void)
{
    pthread_mutex_lock(&g_queue_lock);
    
    /* Find next free slot (circular) */
    int next_head = (g_req_head + 1) % MAX_REQUESTS;
    if (next_head == g_req_tail) {
        /* Queue full - should not happen with proper sizing */
        pthread_mutex_unlock(&g_queue_lock);
        fprintf(stderr, "[Julia Bridge] ERROR: Request queue full!\n");
        return NULL;
    }
    
    julia_req_t *req = &g_requests[g_req_head];
    g_req_head = next_head;
    
    /* Signal Julia workers that new request is available */
    pthread_cond_signal(&g_queue_cv);
    pthread_mutex_unlock(&g_queue_lock);
    
    return req;
}

/**
 * Wait for a request from the queue (called by Julia worker via ccall)
 * Blocks until request is available or shutdown
 * Returns: pointer to request, or NULL if shutdown
 */
julia_req_t* jl_parsec_pop_req(void)
{
    pthread_mutex_lock(&g_queue_lock);

    while (g_req_tail == g_req_head && !g_queue_shutdown) {
        pthread_cond_wait(&g_queue_cv, &g_queue_lock);
    }

    if (g_queue_shutdown) {
        pthread_mutex_unlock(&g_queue_lock);
        return NULL;
    }

    /* Pop request from tail */
    julia_req_t *req = &g_requests[g_req_tail];
    g_req_tail = (g_req_tail + 1) % MAX_REQUESTS;

    pthread_mutex_unlock(&g_queue_lock);

    /* Wait until C proxy fills request (state == 1) */
    pthread_mutex_lock(&req->lock);
    while (req->state != 1) {
        pthread_cond_wait(&req->cv, &req->lock);
    }
    pthread_mutex_unlock(&req->lock);

    return req;
}

/**
 * Mark request as completed (called by Julia worker via ccall)
 */
void jl_parsec_mark_done(julia_req_t *req, int status)
{
    if (req == NULL) return;
    
    pthread_mutex_lock(&req->lock);
    if (req->kind == REQ_KIND_CALLBACK) {
        req->state = 0;  /* Reset immediately */
    } else {
        req->state = 2;  /* Done */
        pthread_cond_signal(&req->cv);  /* Wake up waiting C proxy */
    }
    pthread_mutex_unlock(&req->lock);
}

/**
 * Shutdown Julia bridge (signal workers to exit)
 */
void parsec_julia_bridge_shutdown(void)
{
    pthread_mutex_lock(&g_queue_lock);
    g_queue_shutdown = 1;
    pthread_cond_broadcast(&g_queue_cv);  /* Wake all workers */
    pthread_mutex_unlock(&g_queue_lock);
    
    printf("[Julia Bridge] Shutdown initiated\n");
}

#ifdef HAVE_BLAS
/**
 * Proxy kernel for Julia GEMM
 * Unpacks args, pushes request to Julia, waits for completion
 */
int jl_proxy_gemm_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    
    double *A, *B, *C;
    int m, n, k, mb, nb, kb;
    
    parsec_dtd_unpack_args(this_task, &A, &B, &C, &m, &n, &k, &mb, &nb, &kb);
    
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "ERROR: Proxy GEMM received NULL pointer\n");
        return -1;
    }

    static int dbg = -1;
    static int dbg_count = 0;
    if (dbg == -1) {
        dbg = (getenv("PARSEC_JULIA_DEBUG") != NULL);
    }
    
    /* Limit outstanding compute requests */
    pthread_mutex_lock(&g_out_lock);
    while (g_outstanding >= MAX_OUTSTANDING) {
        pthread_cond_wait(&g_out_cv, &g_out_lock);
    }
    g_outstanding++;
    pthread_mutex_unlock(&g_out_lock);

    /* Get a request slot */
    julia_req_t *req = push_request();
    if (req == NULL) {
        pthread_mutex_lock(&g_out_lock);
        g_outstanding--;
        pthread_cond_signal(&g_out_cv);
        pthread_mutex_unlock(&g_out_lock);
        return -1;
    }
    
    /* Fill request */
    pthread_mutex_lock(&req->lock);
    req->kind = REQ_KIND_GEMM;
    req->kernel_id = KERNEL_ID_GEMM;
    req->signal_ptr = NULL;
    req->A = A;
    req->B = B;
    req->C = C;
    req->mb = mb;
    req->nb = nb;
    req->kb = kb;
    req->lda = mb;  /* Column-major */
    req->ldb = kb;
    req->ldc = mb;
    req->alpha = 1.0;
    req->beta = 1.0;
    req->state = 1;  /* Ready */
    pthread_cond_signal(&req->cv);  /* Notify Julia worker */
    pthread_mutex_unlock(&req->lock);

    if (dbg && dbg_count < 5) {
        fprintf(stderr, "[Proxy GEMM] req=%p A=%p B=%p C=%p mb=%d nb=%d kb=%d\n",
                (void*)req, A, B, C, mb, nb, kb);
        dbg_count++;
    }
    
    /* Wait for Julia to complete */
    pthread_mutex_lock(&req->lock);
    while (req->state != 2) {  /* Wait for done */
        pthread_cond_wait(&req->cv, &req->lock);
    }
    req->state = 0;  /* Reset to empty for reuse */
    pthread_mutex_unlock(&req->lock);

    pthread_mutex_lock(&g_out_lock);
    g_outstanding--;
    pthread_cond_signal(&g_out_cv);
    pthread_mutex_unlock(&g_out_lock);
    
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Proxy kernel for Julia tile initialization
 */
int jl_proxy_init_tile(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    
    double *A;
    int m, n, mb, nb, seed_offset;
    
    parsec_dtd_unpack_args(this_task, &A, &m, &n, &mb, &nb, &seed_offset);
    
    if (A == NULL) {
        fprintf(stderr, "ERROR: Proxy init received NULL pointer\n");
        return -1;
    }
    
    /* Limit outstanding compute requests */
    pthread_mutex_lock(&g_out_lock);
    while (g_outstanding >= MAX_OUTSTANDING) {
        pthread_cond_wait(&g_out_cv, &g_out_lock);
    }
    g_outstanding++;
    pthread_mutex_unlock(&g_out_lock);

    /* Get a request slot */
    julia_req_t *req = push_request();
    if (req == NULL) {
        pthread_mutex_lock(&g_out_lock);
        g_outstanding--;
        pthread_cond_signal(&g_out_cv);
        pthread_mutex_unlock(&g_out_lock);
        return -1;
    }
    
    /* Fill request */
    pthread_mutex_lock(&req->lock);
    req->kind = REQ_KIND_INIT;
    req->kernel_id = KERNEL_ID_INIT_TILE;
    req->signal_ptr = NULL;
    req->A = A;
    req->B = NULL;
    req->C = NULL;
    req->mb = mb;
    req->nb = nb;
    req->kb = seed_offset;  /* Reuse kb field for seed */
    req->lda = mb;
    req->ldb = 0;
    req->ldc = 0;
    req->alpha = 0.0;
    req->beta = 0.0;
    req->state = 1;  /* Ready */
    pthread_cond_signal(&req->cv);  /* Notify Julia worker */
    pthread_mutex_unlock(&req->lock);
    
    /* Wait for Julia to complete */
    pthread_mutex_lock(&req->lock);
    while (req->state != 2) {
        pthread_cond_wait(&req->cv, &req->lock);
    }
    req->state = 0;  /* Reset */
    pthread_mutex_unlock(&req->lock);

    pthread_mutex_lock(&g_out_lock);
    g_outstanding--;
    pthread_cond_signal(&g_out_cv);
    pthread_mutex_unlock(&g_out_lock);
    
    return PARSEC_HOOK_RETURN_DONE;
}
#endif /* HAVE_BLAS */

/**
 * Callback signal kernel (路线 A)
 * 
 * Purpose: Signal completion to Julia after task finishes
 * This kernel is scheduled as a dependency of the output tile,
 * ensuring it runs after all GEMM updates are complete.
 * 
 * Input:
 *   - signal_ptr (VALUE): pointer to volatile int signal variable
 * 
 * Action: Set *(volatile int*)signal_ptr = 1
 */
int jl_callback_signal_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;  /* Unused */
    
    void *tile;
    volatile int *signal_ptr;
    
    /* Unpack dependency tile (unused) + signal pointer */
    parsec_dtd_unpack_args(this_task, &tile, &signal_ptr);
    (void)tile;
    
    /* Enqueue callback request for Julia side */
    julia_req_t *req = push_request();
    if (req == NULL) return PARSEC_HOOK_RETURN_DONE;

    pthread_mutex_lock(&req->lock);
    req->kind = REQ_KIND_CALLBACK;
    req->kernel_id = 0;
    req->signal_ptr = (void*)signal_ptr;
    req->A = NULL;
    req->B = NULL;
    req->C = NULL;
    req->mb = 0;
    req->nb = 0;
    req->kb = 0;
    req->lda = 0;
    req->ldb = 0;
    req->ldc = 0;
    req->alpha = 0.0;
    req->beta = 0.0;
    req->state = 1;  /* Ready */
    pthread_cond_signal(&req->cv);
    pthread_mutex_unlock(&req->lock);
    
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * GPU stub for callback signal (does nothing on GPU)
 * 
 * This is a no-op on GPU devices since callback signaling is CPU-only.
 */
int jl_callback_signal_gpu(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    (void)this_task;
    
    /* GPU doesn't need to do anything for signals */
    return PARSEC_HOOK_RETURN_DONE;
}