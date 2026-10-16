# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdint cimport uintptr_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport cython
import sys

# Global dictionary for Python kernels
_python_kernels = {}


# -----------------------------------------------------------------------------
# C / PaRSEC externs
# -----------------------------------------------------------------------------
cdef extern from "parsec.h":
    ctypedef struct parsec_context_t
    ctypedef struct parsec_taskpool_t

    parsec_context_t* parsec_init(int nb_cores, int *argc, char ***argv)
    int parsec_fini(parsec_context_t **ctx)

    int parsec_context_start(parsec_context_t* ctx)
    int parsec_context_wait(parsec_context_t* ctx) nogil
    int parsec_context_add_taskpool(parsec_context_t* ctx, parsec_taskpool_t* tp)

    int parsec_taskpool_wait(parsec_taskpool_t* tp) nogil
    void parsec_taskpool_free(parsec_taskpool_t* tp)


cdef extern from "parsec/interfaces/dtd/insert_function.h":
    ctypedef struct parsec_task_class_t
    ctypedef struct parsec_task_t
    ctypedef struct parsec_execution_stream_t

    ctypedef int parsec_dtd_funcptr_t(parsec_execution_stream_t*, parsec_task_t*) nogil

    parsec_taskpool_t* parsec_dtd_taskpool_new()

    # varargs (public API signatures)
    parsec_task_class_t* parsec_dtd_create_task_class(parsec_taskpool_t* tp,
                                                      const char* name, ...) nogil
    void parsec_dtd_insert_task_with_task_class(parsec_taskpool_t* tp,
                                                parsec_task_class_t* tc,
                                                int priority,
                                                int device_type, ...) nogil

    int parsec_dtd_task_class_add_chore(parsec_taskpool_t* tp,
                                        parsec_task_class_t* tc,
                                        int device_type,
                                        void* fn) nogil

    void parsec_dtd_task_class_release(parsec_taskpool_t* tp,
                                       parsec_task_class_t* tc) nogil

    void parsec_dtd_unpack_args(parsec_task_t* task, ...) nogil

    void parsec_dtd_data_flush_all(parsec_taskpool_t* tp, void* dc) nogil
    void parsec_dtd_data_collection_init(void* dc) nogil
    void parsec_dtd_data_collection_fini(void* dc) nogil

    # arena
    ctypedef struct parsec_arena_datatype_t
    parsec_arena_datatype_t* parsec_dtd_create_arena_datatype(parsec_context_t* ctx, int* arena_id) nogil
    void parsec_dtd_destroy_arena_datatype(parsec_context_t* ctx, int arena_id) nogil


cdef extern from "parsec/data_dist/matrix/matrix.h":
    ctypedef struct parsec_tiled_matrix_t


# -----------------------------------------------------------------------------
# C helper block: macros/constants + matrix_bc wrapper + varargs “dynamic” wrapper
# -----------------------------------------------------------------------------
cdef extern from *:
    r"""
    #include <stdint.h>
    #include <stdlib.h>
    #include <sys/time.h>
    #include "parsec.h"
    #include "parsec/data_dist/matrix/matrix.h"
    #include "parsec/data_internal.h"
    #include "parsec/arena.h"
    #include "parsec/interfaces/dtd/insert_function_internal.h"

    /* Time measurement (same as stencil_core.pyx) */
    #ifdef PARSEC_HAVE_MPI
    #include <mpi.h>
    #endif

    static inline double py_get_cur_time(void) {
        #ifdef PARSEC_HAVE_MPI
        return MPI_Wtime();
        #else
        struct timeval tv;
        double t;
        gettimeofday(&tv, NULL);
        t = tv.tv_sec + tv.tv_usec / 1e6;
        return t;
        #endif
    }

    /* Global sync_time_elapsed for timing (same as stencil_core.pyx) */
    double dtd_sync_time_elapsed = 0.0;

    #ifdef PARSEC_HAVE_MPI
    #define DTD_SYNC_TIME_START() do { \
            MPI_Barrier(MPI_COMM_WORLD); \
            dtd_sync_time_elapsed = py_get_cur_time(); \
        } while(0)
    #define DTD_SYNC_TIME_STOP() do { \
            MPI_Barrier(MPI_COMM_WORLD); \
            dtd_sync_time_elapsed = py_get_cur_time() - dtd_sync_time_elapsed; \
        } while(0)
    #else
    #define DTD_SYNC_TIME_START() do { \
            dtd_sync_time_elapsed = py_get_cur_time(); \
        } while(0)
    #define DTD_SYNC_TIME_STOP() do { \
            dtd_sync_time_elapsed = py_get_cur_time() - dtd_sync_time_elapsed; \
        } while(0)
    #endif

    /* Helper to access dtd_sync_time_elapsed from Python */
    static inline double* py_get_dtd_sync_time_elapsed_ptr() {
        return &dtd_sync_time_elapsed;
    }

    static inline void* py_get_task_class_from_task(parsec_task_t* task) {
        return (void*)task->task_class;
    }

    /* Ensure access to the global DTD tile mempool used internally by PaRSEC's DTD
     * implementation. The symbol is defined in PaRSEC's DTD source; declare it
     * here as extern so we can check it at runtime to avoid dereferencing NULL. */
    extern parsec_mempool_t *parsec_dtd_tile_mempool;

    // ---- constants (macros -> functions) ----
    static inline int py_PARSEc_INPUT(void)    { return PARSEC_INPUT; }
    static inline int py_PARSEc_INOUT(void)    { return PARSEC_INOUT; }
    static inline int py_PARSEc_AFFINITY(void) { return PARSEC_AFFINITY; }
    static inline int py_PARSEc_VALUE(void)    { return PARSEC_VALUE; }
    static inline int py_PASSED_BY_REF(void)   { return PASSED_BY_REF; }

    static inline int py_PARSEC_DEV_CPU(void)  { return PARSEC_DEV_CPU; }
    static inline int py_PARSEC_DEV_CUDA(void) {
    #if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_CUDA)
        return PARSEC_DEV_CUDA;
    #else
        return -1;
    #endif
    }

    static inline int py_PARSEC_MATRIX_DOUBLE(void) { return PARSEC_MATRIX_DOUBLE; }
    static inline int py_PARSEC_MATRIX_TILE(void)   { return PARSEC_MATRIX_TILE; }

    static inline int py_PARSEC_PUSHOUT(void) { return PARSEC_PUSHOUT; }
    static inline int py_PARSEC_DTD_ARG_END(void) { return PARSEC_DTD_ARG_END; }

    static inline int py_sizeof_int(void)    { return (int)sizeof(int); }
    static inline int py_sizeof_double(void) { return (int)sizeof(double); }

    /* Use the real parsec matrix header for proper layout: */
    #include "parsec/data_dist/matrix/matrix.h"
    #include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
    #include "parsec/data_dist/matrix/redistribute/redistribute_internal.h"

    static inline void* py_alloc_matrix_bc(void)
    {
        return (void*)calloc(1, sizeof(parsec_matrix_block_cyclic_t));
    }

    static inline int py_init_matrix_bc(void* dc,
                                        const char* key,
                                        int mtype, int storage,
                                        int myrank,
                                        int mb, int nb,
                                        int lm, int ln,
                                        int i0, int j0,
                                        int m, int n,
                                        int P, int Q,
                                        int kp, int kq,
                                        int ip, int jq)
    {
        parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
        if(NULL == d) return -1;
        parsec_matrix_block_cyclic_init(d, mtype, storage, myrank,
                                        mb, nb,
                                        lm, ln, i0, j0, m, n,
                                        P, Q, kp, kq, ip, jq);
        if(key) parsec_data_collection_set_key((parsec_data_collection_t*)&d->super.super, key);
        /* Allocate contiguous buffer */
        d->mat = parsec_data_allocate((size_t)d->super.nb_local_tiles * (size_t)d->super.bsiz * (size_t)parsec_datadist_getsizeoftype(d->super.mtype));
        if(NULL == d->mat) return -1;
        /* Require global DTD mempool initialized (i.e., a Parsec DTD taskpool was created) */
        if( NULL == parsec_dtd_tile_mempool ) return -2;
        parsec_dtd_data_collection_init((parsec_data_collection_t*)&d->super.super);
        return 0;
    }

    static inline void py_destroy_matrix_bc(void* dc)
    {
        if(NULL == dc) return;
        parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
        parsec_data_collection_t* A = &d->super.super;
        /* Only call fini if the DTD globals and hash table are present */
        if( NULL != d->super.super.tile_h_table && NULL != parsec_dtd_tile_mempool ) {
            parsec_dtd_data_collection_fini(A);
        }
        if(d->mat) parsec_data_free(d->mat);
        parsec_tiled_matrix_destroy_data(&d->super);
        parsec_data_collection_destroy(A);
        free(d);
    }

    static inline int py_matrix_bc_mt(void* dc) { return ((parsec_matrix_block_cyclic_t*)dc)->super.mt; }
    static inline int py_matrix_bc_nt(void* dc) { return ((parsec_matrix_block_cyclic_t*)dc)->super.nt; }
    static inline int py_matrix_bc_mb(void* dc) { return ((parsec_matrix_block_cyclic_t*)dc)->super.mb; }
    static inline int py_matrix_bc_nb(void* dc) { return ((parsec_matrix_block_cyclic_t*)dc)->super.nb; }

    static inline int py_matrix_bc_nb_local_tiles(void* dc) { return ((parsec_matrix_block_cyclic_t*)dc)->super.nb_local_tiles; }
    static inline int py_matrix_bc_bsiz(void* dc) { return ((parsec_matrix_block_cyclic_t*)dc)->super.bsiz; }
    static inline uintptr_t py_matrix_bc_mat_ptr(void* dc) { return (uintptr_t)((parsec_matrix_block_cyclic_t*)dc)->mat; }
    static inline void* py_matrix_bc_dc_ptr(void* dc) { return (void*)&((parsec_matrix_block_cyclic_t*)dc)->super.super; }

    static inline uintptr_t py_dtd_tile_of(void* dc, int m, int n)
    {
        parsec_matrix_block_cyclic_t* d = (parsec_matrix_block_cyclic_t*)dc;
        if( NULL == d ) return (uintptr_t)0; /* caller checks */
        /* Defensive: ensure data collection hash table initialized */
        if( NULL == d->super.super.tile_h_table ) return (uintptr_t)0;
        /* Defensive: ensure global tile mempool was initialized by creating a DTD taskpool */
        if( NULL == parsec_dtd_tile_mempool ) return (uintptr_t)0;
        parsec_data_key_t key = d->super.super.data_key(&d->super.super, m, n);
        return (uintptr_t)PARSEC_DTD_TILE_OF_KEY(&d->super.super, key);
    }

    static inline parsec_tiled_matrix_t* py_matrix_bc_tiled_ptr(void* dc)
    {
        if(NULL == dc) return NULL;
        return &((parsec_matrix_block_cyclic_t*)dc)->super;
    }

    int parsec_redistribute_dtd(parsec_context_t *parsec,
                                parsec_tiled_matrix_t *dcY,
                                parsec_tiled_matrix_t *dcT,
                                int size_row, int size_col,
                                int disi_Y, int disj_Y,
                                int disi_T, int disj_T);

    // ---- arena helper: create TILE_FULL for double tiles ----
    static inline int py_create_tile_full_arena(parsec_context_t* ctx, int mb, int nb, int* tile_full_dt)
    {
        parsec_arena_datatype_t* adt = parsec_dtd_create_arena_datatype(ctx, tile_full_dt);
        if(NULL == adt) return -1;
        // ld = mb (column-major, Fortran/BLAS standard)
        /* parsec_add2arena_rect now expects 5 args (adt, oldtype, m, n, ld) */
        parsec_add2arena_rect(adt, parsec_datatype_double_t,
                             mb, nb, mb);
        return 0;
    }

    // ---- CUDA / cuBLAS support ----
    #if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    #include "parsec/mca/device/device.h"
    #include "parsec/mca/device/cuda/device_cuda.h"
    #include "cuda_runtime.h"
    #include "cublas_v2.h"

    // Check how many CUDA devices PaRSEC actually sees
    static inline int py_get_nb_cuda_devices(void)
    {
        int nb = 0;
        for(int dev = 0; dev < (int)parsec_nb_devices; dev++) {
            parsec_device_module_t *d = parsec_mca_device_get(dev);
            if(d && d->type == PARSEC_DEV_CUDA) nb++;
        }
        return nb;
    }

    // Global info ID for cuBLAS handle
    static parsec_info_id_t CuHI = -1;  // cuBLAS handle per stream

    // Create cuBLAS handle for a GPU stream
    static void *create_cublas_handle(void *obj, void *p)
    {
        cublasHandle_t handle;
        cublasStatus_t status;
        parsec_cuda_exec_stream_t *stream = (parsec_cuda_exec_stream_t *)obj;
        (void)p;
        status = cublasCreate(&handle);
        if(CUBLAS_STATUS_SUCCESS != status) return NULL;
        status = cublasSetStream(handle, stream->cuda_stream);
        if(CUBLAS_STATUS_SUCCESS != status) {
            cublasDestroy(handle);
            return NULL;
        }
        return (void *)handle;
    }

    static void destroy_cublas_handle(void *_h, void *_n)
    {
        cublasHandle_t handle = (cublasHandle_t)_h;
        if(handle) cublasDestroy(handle);
        (void)_n;
    }

    static int validate_device_ptr(const char* name, const void* ptr)
    {
        struct cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
        if(err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaPointerGetAttributes failed for %s ptr=%p: %s\n",
                    name, ptr, cudaGetErrorString(err));
            fflush(stderr);
            return -1;
        }
    #if CUDART_VERSION >= 10000
        if(attr.type == cudaMemoryTypeHost) {
    #else
        if(attr.memoryType == cudaMemoryTypeHost) {
    #endif
            fprintf(stderr, "ERROR: %s is host pointer, expected device/managed: %p\n", name, ptr);
            fflush(stderr);
            return -1;
        }
        return 0;
    }

    // GPU GEMM kernel using cuBLAS
    static int gemm_kernel_cuda(parsec_device_gpu_module_t *gpu_device,
                                parsec_gpu_task_t *gpu_task,
                                parsec_gpu_exec_stream_t *gpu_stream)
    {
        double *A, *B, *C;
        int m, n, k, mb, nb, kb;
        parsec_task_t *this_task = gpu_task->ec;
        cublasStatus_t status;
        cublasHandle_t handle;
        double *a_gpu, *b_gpu, *c_gpu;

        (void)gpu_device;
        (void)gpu_stream;

        parsec_dtd_unpack_args(this_task,
                               &A, &B, &C,
                               &m, &n, &k,
                               &mb, &nb, &kb);

        // Get device pointers - use this_task (from gpu_task->ec), matching official dtd_test_simple_gemm.c
        a_gpu = (double*)parsec_dtd_get_dev_ptr(this_task, 0);
        b_gpu = (double*)parsec_dtd_get_dev_ptr(this_task, 1);
        c_gpu = (double*)parsec_dtd_get_dev_ptr(this_task, 2);

        // Check for NULL device pointers (can happen if data not on GPU)
        if(NULL == a_gpu || NULL == b_gpu || NULL == c_gpu) {
            fprintf(stderr, "ERROR: NULL device pointer detected: a_gpu=%p b_gpu=%p c_gpu=%p\n", 
                    (void*)a_gpu, (void*)b_gpu, (void*)c_gpu);
            return PARSEC_HOOK_RETURN_ERROR;
        }

        // Validate that pointers are device/managed memory
        if(0 != validate_device_ptr("A", a_gpu) ||
           0 != validate_device_ptr("B", b_gpu) ||
           0 != validate_device_ptr("C", c_gpu)) {
            return PARSEC_HOOK_RETURN_ERROR;
        }

        // Get cuBLAS handle from info system
        handle = parsec_info_get(&gpu_stream->infos, CuHI);
        if(NULL == handle) return PARSEC_HOOK_RETURN_ERROR;
        // Use HOST pointer mode with host scalars for alpha/beta
        status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        if(CUBLAS_STATUS_SUCCESS != status) return PARSEC_HOOK_RETURN_ERROR;

        const double alpha = 1.0;
        const double beta = 1.0;

        // Call cuBLAS DGEMM: C = A*B + C
        status = cublasDgemm_v2(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                mb, nb, kb,
                                &alpha, a_gpu, mb,
                                b_gpu, kb,
                                &beta, c_gpu, mb);

        if(CUBLAS_STATUS_SUCCESS != status)
            return PARSEC_HOOK_RETURN_ERROR;

        return PARSEC_HOOK_RETURN_DONE;
    }

    // Initialize CUDA support
    static int py_cuda_setup(void)
    {
        #if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        if(CuHI == -1) {
            CuHI = parsec_info_register(&parsec_per_stream_infos, "CUBLAS::HANDLE",
                                        destroy_cublas_handle, NULL,
                                        create_cublas_handle, NULL,
                                        NULL);
            if(CuHI == -1) return -1;
        }
        return 0;
        #else
        return -1;
        #endif
    }

    static void py_cuda_teardown(void)
    {
        #if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        if(CuHI != -1) {
            parsec_info_unregister(&parsec_per_stream_infos, CuHI, NULL);
            CuHI = -1;
        }
        #endif
    }
    #else
    // Dummy functions when CUDA not available
    static int py_cuda_setup(void) { return -1; }
    static void py_cuda_teardown(void) {}
    static int gemm_kernel_cuda(void *a, void *b, void *c) { (void)a; (void)b; (void)c; return -1; }
    static inline int py_get_nb_cuda_devices(void) { return 0; }
    #endif

    // ---- varargs dynamic wrappers (support up to 12 args) ----
    static inline parsec_task_class_t*
    py_create_task_class(parsec_taskpool_t* tp, const char* name,
                         int nargs, const int* types, const int* flags)
    {
        switch(nargs) {
            case 0:  return parsec_dtd_create_task_class(tp, name, PARSEC_DTD_ARG_END);
            case 1:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        PARSEC_DTD_ARG_END);
            case 2:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        types[1], flags[1],
                                                        PARSEC_DTD_ARG_END);
            case 3:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        types[1], flags[1],
                                                        types[2], flags[2],
                                                        PARSEC_DTD_ARG_END);
            case 4:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        types[1], flags[1],
                                                        types[2], flags[2],
                                                        types[3], flags[3],
                                                        PARSEC_DTD_ARG_END);
            case 5:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        types[1], flags[1],
                                                        types[2], flags[2],
                                                        types[3], flags[3],
                                                        types[4], flags[4],
                                                        PARSEC_DTD_ARG_END);
            case 6:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        types[1], flags[1],
                                                        types[2], flags[2],
                                                        types[3], flags[3],
                                                        types[4], flags[4],
                                                        types[5], flags[5],
                                                        PARSEC_DTD_ARG_END);
            case 7:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        types[1], flags[1],
                                                        types[2], flags[2],
                                                        types[3], flags[3],
                                                        types[4], flags[4],
                                                        types[5], flags[5],
                                                        types[6], flags[6],
                                                        PARSEC_DTD_ARG_END);
            case 8:  return parsec_dtd_create_task_class(tp, name,
                                                        types[0], flags[0],
                                                        types[1], flags[1],
                                                        types[2], flags[2],
                                                        types[3], flags[3],
                                                        types[4], flags[4],
                                                        types[5], flags[5],
                                                        types[6], flags[6],
                                                        types[7], flags[7],
                                                        PARSEC_DTD_ARG_END);
            case 9:  return parsec_dtd_create_task_class(tp, name,
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
            case 10: return parsec_dtd_create_task_class(tp, name,
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
            case 11: return parsec_dtd_create_task_class(tp, name,
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
            case 12: return parsec_dtd_create_task_class(tp, name,
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
            default: return NULL;
        }
    }

    static inline int
    py_insert_task(parsec_taskpool_t* tp, parsec_task_class_t* tc,
                   int prio, int dev, int nargs, const int* ins_flags, void* const* args)
    {
        switch(nargs) {
            case 0:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev, PARSEC_DTD_ARG_END); return 0;
            case 1:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 2:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 3:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  ins_flags[2], args[2],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 4:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  ins_flags[2], args[2],
                                                                  ins_flags[3], args[3],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 5:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  ins_flags[2], args[2],
                                                                  ins_flags[3], args[3],
                                                                  ins_flags[4], args[4],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 6:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  ins_flags[2], args[2],
                                                                  ins_flags[3], args[3],
                                                                  ins_flags[4], args[4],
                                                                  ins_flags[5], args[5],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 7:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  ins_flags[2], args[2],
                                                                  ins_flags[3], args[3],
                                                                  ins_flags[4], args[4],
                                                                  ins_flags[5], args[5],
                                                                  ins_flags[6], args[6],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 8:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  ins_flags[2], args[2],
                                                                  ins_flags[3], args[3],
                                                                  ins_flags[4], args[4],
                                                                  ins_flags[5], args[5],
                                                                  ins_flags[6], args[6],
                                                                  ins_flags[7], args[7],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 9:  parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
                                                                  ins_flags[0], args[0],
                                                                  ins_flags[1], args[1],
                                                                  ins_flags[2], args[2],
                                                                  ins_flags[3], args[3],
                                                                  ins_flags[4], args[4],
                                                                  ins_flags[5], args[5],
                                                                  ins_flags[6], args[6],
                                                                  ins_flags[7], args[7],
                                                                  ins_flags[8], args[8],
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 10: parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
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
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 11: parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
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
                                                                  PARSEC_DTD_ARG_END); return 0;
            case 12: parsec_dtd_insert_task_with_task_class(tp, tc, prio, dev,
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
                                                                  PARSEC_DTD_ARG_END); return 0;
            default: return -1;
        }
    }
    """
    # Helper to get task_class from task
    void* py_get_task_class_from_task(parsec_task_t* task) nogil
    
    int py_PARSEc_INPUT()
    int py_PARSEc_INOUT()
    int py_PARSEc_AFFINITY()
    int py_PARSEc_VALUE()
    int py_PASSED_BY_REF()
    int py_PARSEC_DEV_CPU()
    int py_PARSEC_DEV_CUDA()
    int py_PARSEC_MATRIX_DOUBLE()
    int py_PARSEC_MATRIX_TILE()
    int py_PARSEC_PUSHOUT()
    int py_PARSEC_DTD_ARG_END()
    int py_sizeof_int()
    int py_sizeof_double()

    void* py_alloc_matrix_bc()
    int py_init_matrix_bc(void* dc, const char* key,
                          int mtype, int storage, int myrank,
                          int mb, int nb, int lm, int ln,
                          int i0, int j0, int m, int n,
                          int P, int Q, int kp, int kq, int ip, int jq)
    void py_destroy_matrix_bc(void* dc)

    int py_matrix_bc_mt(void* dc)
    int py_matrix_bc_nt(void* dc)
    int py_matrix_bc_mb(void* dc)
    int py_matrix_bc_nb(void* dc)

    int py_matrix_bc_nb_local_tiles(void* dc)
    int py_matrix_bc_bsiz(void* dc)
    uintptr_t py_matrix_bc_mat_ptr(void* dc)
    void* py_matrix_bc_dc_ptr(void* dc)

    uintptr_t py_dtd_tile_of(void* dc, int m, int n)

    parsec_tiled_matrix_t* py_matrix_bc_tiled_ptr(void* dc)

    int parsec_redistribute_dtd_c "parsec_redistribute_dtd"(parsec_context_t *parsec,
                                                            parsec_tiled_matrix_t *dcY,
                                                            parsec_tiled_matrix_t *dcT,
                                                            int size_row, int size_col,
                                                            int disi_Y, int disj_Y,
                                                            int disi_T, int disj_T) nogil

    int parsec_redistribute_c "parsec_redistribute"(parsec_context_t *parsec,
                                                    parsec_tiled_matrix_t *dcY,
                                                    parsec_tiled_matrix_t *dcT,
                                                    int size_row, int size_col,
                                                    int disi_Y, int disj_Y,
                                                    int disi_T, int disj_T) nogil

    int py_create_tile_full_arena(parsec_context_t* ctx, int mb, int nb, int* tile_full_dt)

    # CUDA support functions
    int py_cuda_setup()
    void py_cuda_teardown()
    int py_get_nb_cuda_devices()
    int gemm_kernel_cuda(void* gpu_device, void* gpu_task, void* gpu_stream)

    # Time measurement functions
    double* py_get_dtd_sync_time_elapsed_ptr() nogil
    double py_get_cur_time() nogil

    parsec_task_class_t* py_create_task_class(parsec_taskpool_t* tp, const char* name,
                                              int nargs, const int* types, const int* flags) nogil
    int py_insert_task(parsec_taskpool_t* tp, parsec_task_class_t* tc,
                       int prio, int dev, int nargs, const int* ins_flags, void* const* args) nogil


# -----------------------------------------------------------------------------
# Exported constants (like official)
# -----------------------------------------------------------------------------
PASSED_BY_REF = py_PASSED_BY_REF()
PARSEC_INPUT = py_PARSEc_INPUT()
PARSEC_INOUT = py_PARSEc_INOUT()
PARSEC_AFFINITY = py_PARSEc_AFFINITY()
PARSEC_VALUE = py_PARSEc_VALUE()

PARSEC_DEV_CPU = py_PARSEC_DEV_CPU()
PARSEC_DEV_CUDA = py_PARSEC_DEV_CUDA()

PARSEC_MATRIX_DOUBLE = py_PARSEC_MATRIX_DOUBLE()
PARSEC_MATRIX_TILE = py_PARSEC_MATRIX_TILE()
PARSEC_PUSHOUT = py_PARSEC_PUSHOUT()
PARSEC_DTD_ARG_END = py_PARSEC_DTD_ARG_END()
SIZEOF_INT = py_sizeof_int()
SIZEOF_DOUBLE = py_sizeof_double()

PARSEC_DTD_EMPTY_FLAG = 0


# -----------------------------------------------------------------------------
# Built-in chores (pure C, no Python/GIL)
#   - initialize_tile: copy from official sample logic
#   - zero_tile
#   - gemm_tile: naive row-major dgemm tile update
# -----------------------------------------------------------------------------
cdef uint64_t _rnd64_jump(uint64_t n, uint64_t seed) nogil:
    cdef uint64_t a_k = 6364136223846793005
    cdef uint64_t c_k = 1
    cdef uint64_t a = a_k
    cdef uint64_t c = c_k
    cdef uint64_t ran = seed

    while n:
        if n & 1:
            ran = a * ran + c
        c = c * (a + 1)
        a = a * a
        n >>= 1
    return ran


cdef int chore_initialize_tile(parsec_execution_stream_t *es, parsec_task_t *this_task) nogil:
    cdef double *A
    cdef int m, n, mb, nb, seed
    cdef uint64_t jump
    cdef uint64_t ran
    cdef int i, j
    cdef uint64_t LCG_A = <uint64_t>6364136223846793005
    cdef uint64_t LCG_C = <uint64_t>1
    cdef double inv_2p53 = 1.0 / 9007199254740992.0  # 2^53

    parsec_dtd_unpack_args(this_task, &A, &m, &n, &mb, &nb, &seed)

    # Cython 要用 <uint64_t>(...) 这种 cast
    jump = <uint64_t>(n * mb + m * mb * nb)
    ran = _rnd64_jump(jump, <uint64_t>seed)

    for j in range(mb):
        for i in range(nb):
            ran = LCG_A * ran + LCG_C
            A[j*nb + i] = (<double>(ran >> 11)) * inv_2p53

    return 0


cdef int chore_zero_tile(parsec_execution_stream_t *es, parsec_task_t *this_task) nogil:
    cdef double *C
    cdef int mb, nb
    cdef int i
    parsec_dtd_unpack_args(this_task, &C, &mb, &nb)
    for i in range(mb * nb):
        C[i] = 0.0
    return 0


cdef int chore_gemm_tile(parsec_execution_stream_t *es, parsec_task_t *this_task) nogil:
    cdef double *A
    cdef double *B
    cdef double *C
    cdef int m, n, k
    cdef int mb, nb, kb
    cdef int i, j, kk
    cdef double cij

    parsec_dtd_unpack_args(this_task, &A, &B, &C, &m, &n, &k, &mb, &nb, &kb)

    # Row-major: A(mb x kb), B(kb x nb), C(mb x nb)
    for i in range(mb):
        for j in range(nb):
            cij = C[i*nb + j]
            for kk in range(kb):
                cij += A[i*kb + kk] * B[kk*nb + j]
            C[i*nb + j] = cij
    return 0


cdef int chore_python_kernel(parsec_execution_stream_t *es, parsec_task_t *this_task) with gil:
    """Generic Python kernel wrapper"""
    import numpy as np
    
    cdef uintptr_t tc_addr = <uintptr_t>py_get_task_class_from_task(this_task)
    cdef int device_type = py_PARSEC_DEV_CPU()
    key = (tc_addr, device_type)
    
    if key not in _python_kernels:
        return -1
    
    cdef double *data, *A, *B, *C
    cdef int m, n, k, mb, nb, kb, seed, nargs
    
    nargs = _python_kernels[key]['nargs']
    
    try:
        if nargs == 6:  # init
            parsec_dtd_unpack_args(this_task, &data, &m, &n, &mb, &nb, &seed)
            args = [np.asarray(<double[:mb*nb]>data), m, n, mb, nb, seed]
        elif nargs == 9:  # gemm
            parsec_dtd_unpack_args(this_task, &A, &B, &C, &m, &n, &k, &mb, &nb, &kb)
            args = [np.asarray(<double[:mb*kb]>A), np.asarray(<double[:kb*nb]>B),
                    np.asarray(<double[:mb*nb]>C), m, n, k, mb, nb, kb]
        else:
            return -1
        
        _python_kernels[key]['func'](None, args)
        return 0
    except:
        return -1


# -----------------------------------------------------------------------------
# Python-visible wrappers
# -----------------------------------------------------------------------------
cdef class ParsecDTDContext:
    cdef parsec_context_t* _ctx

    def __cinit__(self, int nb_cores=0):
        cdef int argc = 0
        cdef char **argv = NULL
        self._ctx = parsec_init(nb_cores, &argc, &argv)
        if self._ctx == NULL:
            raise RuntimeError("parsec_init failed")

    def start(self):
        if parsec_context_start(self._ctx) != 0:
            raise RuntimeError("parsec_context_start failed")

    def wait(self):
        cdef int rc
        with nogil:
            rc = parsec_context_wait(self._ctx)
        if rc != 0:
            raise RuntimeError("parsec_context_wait failed")

    def add_taskpool(self, ParsecDTDTaskpool tp):
        if tp is None or tp._tp == NULL:
            raise ValueError("invalid taskpool")
        if parsec_context_add_taskpool(self._ctx, tp._tp) != 0:
            raise RuntimeError("parsec_context_add_taskpool failed")

    def create_tile_full_arena(self, int mb, int nb):
        cdef int tile_full_dt = 0
        if py_create_tile_full_arena(self._ctx, mb, nb, &tile_full_dt) != 0:
            raise RuntimeError("create_tile_full_arena failed")
        return tile_full_dt

    def destroy_arena_datatype(self, int arena_id):
        if self._ctx == NULL:
            raise RuntimeError("Context not initialized")
        with nogil:
            parsec_dtd_destroy_arena_datatype(self._ctx, arena_id)

    def cuda_setup(self):
        """Initialize CUDA support (cuBLAS handles, device constants)"""
        rc = py_cuda_setup()
        if rc != 0:
            raise RuntimeError("CUDA setup failed (is CUDA support enabled?)")

    def nb_cuda_devices(self):
        """Get number of CUDA devices PaRSEC can see"""
        return py_get_nb_cuda_devices()

    def cuda_teardown(self):
        """Cleanup CUDA resources"""
        py_cuda_teardown()

    def sync_time_start(self):
        """Start synchronized timing with MPI barrier (like SYNC_TIME_START)"""
        try:
            from mpi4py import MPI
            if MPI.Is_initialized():
                MPI.COMM_WORLD.Barrier()
        except ImportError:
            pass
        
        cdef double* sync_ptr
        with nogil:
            sync_ptr = py_get_dtd_sync_time_elapsed_ptr()
            sync_ptr[0] = py_get_cur_time()
    
    def sync_time_stop(self):
        """Stop synchronized timing with MPI barrier and return elapsed time (like SYNC_TIME_STOP)"""
        try:
            from mpi4py import MPI
            if MPI.Is_initialized():
                MPI.COMM_WORLD.Barrier()
        except ImportError:
            pass
        
        cdef double* sync_ptr
        cdef double elapsed
        with nogil:
            sync_ptr = py_get_dtd_sync_time_elapsed_ptr()
            sync_ptr[0] = py_get_cur_time() - sync_ptr[0]
            elapsed = sync_ptr[0]
        return elapsed

    def fini(self):
        cdef parsec_context_t* tmp = self._ctx
        if tmp != NULL:
            if parsec_fini(&tmp) != 0:
                raise RuntimeError("parsec_fini failed")
            self._ctx = NULL

    def __dealloc__(self):
        try:
            self.fini()
        except Exception:
            pass


cdef class ParsecDTDTaskpool:
    cdef parsec_taskpool_t* _tp
    cdef int tmpi
    cdef double tmpd
    cdef float tmpf
    cdef long tmpl

    def __cinit__(self):
        self._tp = parsec_dtd_taskpool_new()
        if self._tp == NULL:
            raise RuntimeError("parsec_dtd_taskpool_new failed")

    def wait(self):
        cdef int rc
        with nogil:
            rc = parsec_taskpool_wait(self._tp)
        if rc < 0:
            raise RuntimeError(f"parsec_taskpool_wait failed with error code {rc}")

    def free(self):
        if self._tp != NULL:
            parsec_taskpool_free(self._tp)
            self._tp = NULL

    def flush_all(self, ParsecMatrixBlockCyclic mat):
        if mat is None or mat._dc == NULL:
            raise ValueError("invalid matrix")
        parsec_dtd_data_flush_all(self._tp, py_matrix_bc_dc_ptr(<void*>mat._dc))

    def create_task_class(self, name, chore, signature):
        """
        signature: list[(arg_type, arg_flag)]
        chore: "init"/"zero"/"gemm" or None (for manual chore registration)
        """
        cdef void* fn = NULL
        cdef bint auto_register = (chore is not None)
        
        if auto_register:
            if not isinstance(chore, str):
                raise ValueError("chore must be a string or None")
            if chore == "init":
                fn = <void*>chore_initialize_tile
            elif chore == "zero":
                fn = <void*>chore_zero_tile
            elif chore == "gemm":
                fn = <void*>chore_gemm_tile
            else:
                raise ValueError(f"unknown chore '{chore}'")

        if not isinstance(signature, (list, tuple)):
            raise ValueError("signature must be list[(type, flag)]")

        cdef int nargs = len(signature)
        if nargs < 0 or nargs > 12:
            raise ValueError("signature nargs must be <= 12 (current wrapper limit)")

        cdef int* types = <int*>malloc(nargs * sizeof(int))
        cdef int* flags = <int*>malloc(nargs * sizeof(int))
        if (types == NULL) or (flags == NULL):
            if types != NULL: free(types)
            if flags != NULL: free(flags)
            raise MemoryError()

        cdef int i
        for i in range(nargs):
            types[i] = <int>signature[i][0]
            flags[i] = <int>signature[i][1]

        cdef parsec_task_class_t* tc = NULL
        cdef bytes bname = (<str>name).encode('utf-8') if isinstance(name, str) else str(name).encode('utf-8')
        cdef const char* cname = bname

        with nogil:
            tc = py_create_task_class(self._tp, cname, nargs, types, flags)

        free(types); free(flags)

        if tc == NULL:
            raise RuntimeError("create_task_class failed (tc is NULL)")

        # Register chore only if auto_register
        if auto_register:
            if chore == "init" or chore == "zero":
                if parsec_dtd_task_class_add_chore(self._tp, tc, PARSEC_DEV_CPU, fn) != 0:
                    raise RuntimeError("failed to add CPU chore to task class")
            elif chore == "gemm":
                if PARSEC_DEV_CUDA != -1:
                    parsec_dtd_task_class_add_chore(self._tp, tc, PARSEC_DEV_CUDA, fn)
                try:
                    parsec_dtd_task_class_add_chore(self._tp, tc, PARSEC_DEV_CPU, fn)
                except Exception:
                    pass

        cdef ParsecDTDTaskClass obj = ParsecDTDTaskClass.__new__(ParsecDTDTaskClass)
        obj._tc = tc
        obj._nargs = nargs
        obj._types = [signature[i][0] for i in range(nargs)]
        return obj

    def insert_task_with_task_class(self,
                                ParsecDTDTaskClass tc,
                                int priority,
                                int device_type,
                                name,
                                args):
        """
        args: list[(insert_flag, value)]
        - 如果 insert_flag 含 PARSEC_VALUE：value 是 int/float，会按 tc._types[i] (SIZEOF_INT/SIZEOF_DOUBLE) 打包
        - 否则：value 必须是 uintptr（比如 matrix.tile_of(m,n) 返回的 int）
        """
        if tc is None or tc._tc == NULL:
            raise ValueError("invalid task class")
        if not isinstance(args, (list, tuple)):
            raise ValueError("args must be list[(flag, val)]")
        if len(args) != tc._nargs:
            raise ValueError(f"args length {len(args)} != taskclass nargs {tc._nargs}")

        cdef int nargs = tc._nargs
        if nargs > 12:
            raise ValueError("nargs > 12 not supported by this wrapper")

        cdef bytes bname = name.encode('utf-8')
        cdef const char* ctaskname = bname  # 这一步必须在 nogil 外
        # (ctaskname is only for diagnostics; insertion varargs do not accept a name parameter)

        cdef parsec_task_class_t* tc_ptr = tc._tc  # nogil 外取出指针

        # Debug: print insertion diagnostics to help reproduce parsec fatals (disabled)
        # print(f"[py_dtd_debug] insert name={name.decode('utf-8') if isinstance(name, (bytes, bytearray)) else name}, nargs={nargs}")
        # print(f"[py_dtd_debug] tc._types={tc._types}")
        # print(f"[py_dtd_debug] args={args}")
        cdef int* ins_flags = <int*>malloc(nargs * sizeof(int))
        cdef void** cargs   = <void**>malloc(nargs * sizeof(void*))
        if ins_flags == NULL or cargs == NULL:
            if ins_flags != NULL: free(ins_flags)
            if cargs != NULL: free(cargs)
            raise MemoryError()

        # First pass: collect flags and calculate VALUE buffer size
        cdef int i, sz
        cdef int total_value_bytes = 0
        for i in range(nargs):
            # Use user-provided flag for all parameters
            ins_flags[i] = <int>args[i][0]
            # Calculate buffer space for VALUE parameters
            if <int>tc._types[i] == SIZEOF_INT or <int>tc._types[i] == SIZEOF_DOUBLE:
                sz = <int>tc._types[i]
                total_value_bytes += sz

        cdef char* valbuf = NULL
        if total_value_bytes > 0:
            valbuf = <char*>malloc(total_value_bytes)
            if valbuf == NULL:
                free(ins_flags); free(cargs)
                raise MemoryError()

        cdef int off = 0
        cdef int tmpi
        cdef double tmpd

        for i in range(nargs):
            # For VALUE parameters, the tc._types entry is SIZEOF_INT or SIZEOF_DOUBLE
            if <int>tc._types[i] == SIZEOF_INT or <int>tc._types[i] == SIZEOF_DOUBLE:
                sz = <int>tc._types[i]
                cargs[i] = <void*>(valbuf + off)

                if sz == SIZEOF_INT:
                    tmpi = <int>int(args[i][1])
                    memcpy(valbuf + off, &tmpi, SIZEOF_INT)
                elif sz == SIZEOF_DOUBLE:
                    tmpd = <double>float(args[i][1])
                    memcpy(valbuf + off, &tmpd, SIZEOF_DOUBLE)
                else:
                    if valbuf != NULL: free(valbuf)
                    free(ins_flags); free(cargs)
                    raise ValueError(f"unsupported VALUE size {sz} (only int/double supported)")

                off += sz
            else:
                # tile / pointer 参数
                cargs[i] = <void*><uintptr_t>int(args[i][1])

        cdef int rc
        # with nogil:
        rc = py_insert_task(self._tp, tc_ptr, priority, device_type, nargs, ins_flags, cargs)

        # NOTE: DO NOT FREE valbuf, ins_flags, cargs here!
        # PaRSEC stores pointers to these buffers and accesses them when the task executes.
        # The memory must remain valid throughout the program lifetime or until the task completes.
        # This is a potential memory leak, but necessary for PaRSEC to function correctly.
        # TODO: Integrate with taskpool lifecycle to free after tasks complete

        if rc != 0:
            raise RuntimeError(f"insert_task failed rc={rc}")

    def add_chore_to_task_class(self, ParsecDTDTaskClass tc, int device_type, chore_func):
        """Add kernel to task class
        
        For CPU: registers Python wrapper that calls chore_func
        For GPU (CUDA): registers C/cuBLAS gemm_kernel_cuda directly
        """
        if tc is None or tc._tc == NULL:
            raise ValueError("invalid task class")
        
        cdef uintptr_t tc_addr = <uintptr_t>tc._tc
        cdef void* kernel_ptr
        cdef int rc
        
        # For GPU devices, use direct CUDA kernel
        if device_type == py_PARSEC_DEV_CUDA():
            kernel_ptr = <void*>gemm_kernel_cuda
            # No Python kernel to store for GPU
        else:
            # For CPU, use Python wrapper
            if not callable(chore_func):
                raise ValueError("chore_func must be callable")
            _python_kernels[(tc_addr, device_type)] = {
                'func': chore_func, 'nargs': tc._nargs, 'types': tc._types
            }
            kernel_ptr = <void*>chore_python_kernel
        
        rc = parsec_dtd_task_class_add_chore(self._tp, tc._tc, device_type, kernel_ptr)
        if rc != 0:
            if device_type != py_PARSEC_DEV_CUDA() and (tc_addr, device_type) in _python_kernels:
                del _python_kernels[(tc_addr, device_type)]
            raise RuntimeError(f"failed to add chore for device {device_type} (rc={rc})")


    def __dealloc__(self):
        try:
            self.free()
        except Exception:
            pass


cdef class ParsecDTDTaskClass:
    cdef parsec_task_class_t* _tc
    cdef int _nargs
    cdef public object _types  # python list[int]

    def release(self, ParsecDTDTaskpool tp):
        cdef parsec_taskpool_t* tp_ptr
        cdef parsec_task_class_t* tc_ptr

        if tp is None or tp._tp == NULL:
            raise ValueError("invalid taskpool")

        tp_ptr = tp._tp          # 先取出来
        tc_ptr = self._tc        # 先取出来

        if tc_ptr != NULL:
            with nogil:
                parsec_dtd_task_class_release(tp_ptr, tc_ptr)
            self._tc = NULL

    @property
    def nargs(self):
        return self._nargs


cdef class ParsecMatrixBlockCyclic:
    cdef void* _dc
    cdef int _rank

    def __cinit__(self):
        self._dc = NULL
        self._rank = -1

    def init(self, name: str,
             int myrank,
             int mb, int nb,
             int lm, int ln,
             int P, int Q,
             int mtype=PARSEC_MATRIX_DOUBLE,
             int storage=PARSEC_MATRIX_TILE):
        """
        Mirrors parsec_matrix_block_cyclic_init + local-tile allocation + dtd_data_collection_init
        """
        if self._dc != NULL:
            raise RuntimeError("matrix already initialized")
        self._dc = py_alloc_matrix_bc()
        if self._dc == NULL:
            raise MemoryError("alloc_matrix_bc failed")

        self._rank = myrank

        # i0=j0=0, m=lm, n=ln, kp=kq=1, ip=jq=0 (same defaults as official sample)
        cdef int rc = py_init_matrix_bc(self._dc, name.encode('utf-8'),
                                        mtype, storage, myrank,
                                        mb, nb, lm, ln,
                                        0, 0, lm, ln,
                                        P, Q, 1, 1, 0, 0)
        if rc != 0:
            py_destroy_matrix_bc(self._dc)
            self._dc = NULL
            if rc == -2:
                raise RuntimeError("parsec DTD globals not initialized: create a ParsecDTDTaskpool before initializing matrices")
            raise RuntimeError("init_matrix_bc failed")

    def destroy(self):
        if self._dc != NULL:
            py_destroy_matrix_bc(self._dc)
            self._dc = NULL

    @property
    def mt(self): return py_matrix_bc_mt(self._dc)
    @property
    def nt(self): return py_matrix_bc_nt(self._dc)
    @property
    def mb(self): return py_matrix_bc_mb(self._dc)
    @property
    def nb(self): return py_matrix_bc_nb(self._dc)

    def tile_of(self, int m, int n):
        """
        Returns uintptr (int) produced by PARSEC_DTD_TILE_OF_KEY
        """
        if self._dc == NULL:
            raise RuntimeError("matrix not initialized")
        cdef uintptr_t res = py_dtd_tile_of(self._dc, m, n)
        if res == 0:
            raise RuntimeError("matrix DTD hash table not initialized (create and start a ParsecDTDContext and add a ParsecDTDTaskpool before using tile_of)")
        return <uintptr_t>res

    def local_buffer(self):
        """
        Returns numpy view of the local tile storage (1D float64).
        No numpy cimport needed (ctypes view).
        """
        import ctypes
        import numpy as np
        if self._dc == NULL:
            raise RuntimeError("matrix not initialized")
        cdef int ntile = py_matrix_bc_nb_local_tiles(self._dc)
        cdef int bsiz = py_matrix_bc_bsiz(self._dc)
        cdef uintptr_t ptr = py_matrix_bc_mat_ptr(self._dc)
        n = ntile * bsiz
        buf = (ctypes.c_double * n).from_address(ptr)
        return np.ctypeslib.as_array(buf)

    def __dealloc__(self):
        try:
            self.destroy()
        except Exception:
            pass


def parsec_redistribute_dtd(ParsecDTDContext ctx,
                            ParsecMatrixBlockCyclic src,
                            ParsecMatrixBlockCyclic dst,
                            int size_row, int size_col,
                            int disi_Y=0, int disj_Y=0,
                            int disi_T=0, int disj_T=0):
    """Redistribute a submatrix from src to dst using PaRSEC DTD.

    This wraps parsec_redistribute_dtd and starts/waits the context internally.
    """
    if ctx is None or ctx._ctx == NULL:
        raise ValueError("invalid context")
    if src is None or src._dc == NULL:
        raise ValueError("invalid source matrix")
    if dst is None or dst._dc == NULL:
        raise ValueError("invalid target matrix")

    cdef parsec_context_t* ctx_ptr = ctx._ctx
    cdef parsec_tiled_matrix_t* src_ptr = py_matrix_bc_tiled_ptr(src._dc)
    cdef parsec_tiled_matrix_t* dst_ptr = py_matrix_bc_tiled_ptr(dst._dc)
    if src_ptr == NULL or dst_ptr == NULL:
        raise RuntimeError("failed to resolve tiled matrix pointers")

    cdef int rc
    cdef int disi_Y_i = disi_Y
    cdef int disj_Y_i = disj_Y
    cdef int disi_T_i = disi_T
    cdef int disj_T_i = disj_T
    with nogil:
        rc = parsec_redistribute_dtd_c(ctx_ptr, src_ptr, dst_ptr,
                                       size_row, size_col,
                                       disi_Y_i, disj_Y_i,
                                       disi_T_i, disj_T_i)
    if rc != 0:
        raise RuntimeError(f"parsec_redistribute_dtd failed (rc={rc})")


def parsec_redistribute(ParsecDTDContext ctx,
                        ParsecMatrixBlockCyclic src,
                        ParsecMatrixBlockCyclic dst,
                        int size_row, int size_col,
                        int disi_Y=0, int disj_Y=0,
                        int disi_T=0, int disj_T=0):
    """Redistribute a submatrix from src to dst using PaRSEC PTG."""
    if ctx is None or ctx._ctx == NULL:
        raise ValueError("invalid context")
    if src is None or src._dc == NULL:
        raise ValueError("invalid source matrix")
    if dst is None or dst._dc == NULL:
        raise ValueError("invalid target matrix")

    cdef parsec_context_t* ctx_ptr = ctx._ctx
    cdef parsec_tiled_matrix_t* src_ptr = py_matrix_bc_tiled_ptr(src._dc)
    cdef parsec_tiled_matrix_t* dst_ptr = py_matrix_bc_tiled_ptr(dst._dc)
    if src_ptr == NULL or dst_ptr == NULL:
        raise RuntimeError("failed to resolve tiled matrix pointers")

    cdef int rc
    cdef int disi_Y_i = disi_Y
    cdef int disj_Y_i = disj_Y
    cdef int disi_T_i = disi_T
    cdef int disj_T_i = disj_T
    with nogil:
        rc = parsec_redistribute_c(ctx_ptr, src_ptr, dst_ptr,
                                   size_row, size_col,
                                   disi_Y_i, disj_Y_i,
                                   disi_T_i, disj_T_i)
    if rc != 0:
        raise RuntimeError(f"parsec_redistribute failed (rc={rc})")
