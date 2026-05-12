/*
 * Copyright (c) 2021-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024-2026 NVIDIA Corporation.  All rights reserved.
 */

#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/mca/device/device.h"

// The file is not compiled if CUDA is not present or CUBLAS is not found
#include "parsec/mca/device/cuda/device_cuda.h"
#include "cublas_v2.h"

#include <sys/time.h>

#if !defined(timersub)
#define timersub(a, b, result) do {                \
        (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;       \
        (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;    \
        if( (result)->tv_usec < 0 ) {                       \
            --(result)->tv_sec;                             \
            (result)->tv_usec += 1000000;                   \
        }                                                   \
    } while(0)
#endif

#if defined(HAVE_BLAS)
// If our CMake finds a BLAS library, it defines HAVE_BLAS
// BLAS does not guarantee there is a cblas.h, we define our own prototype
typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
#define CBLAS_INDEX int

extern void cblas_dgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
                        const CBLAS_INDEX K, const double alpha, const double  *A,
                        const CBLAS_INDEX lda, const double  *B, const CBLAS_INDEX ldb,
                        const double beta, double  *C, const CBLAS_INDEX ldc);
#endif

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <stdint.h>

static int TILE_FULL = -1;
static parsec_info_id_t CuHI = -1;
static parsec_info_id_t Cu1 = -1;
static int verbose = 0;
static int device = PARSEC_DEV_CUDA;
static int P = -1;
static int Q = -1;
static int cuda_max_batch_size = 32;
static int cuda_max_submitted_batches = 4;

typedef enum {
    GEMM_CUDA_BATCH_NONE,
    GEMM_CUDA_BATCH_ONE_BY_ONE,
    GEMM_CUDA_BATCH_CUBLAS
} gemm_cuda_batch_mode_t;

static gemm_cuda_batch_mode_t cuda_batch_mode = GEMM_CUDA_BATCH_NONE;

typedef struct gemm_cuda_batch_pool_s {
    double **ptrs;
    int max_batch_size;
    int max_submitted_batches;
    int cuda_device_index;
    uint64_t next_submit;
    uint64_t next_complete;
} gemm_cuda_batch_pool_t;

typedef struct gemm_cuda_stream_state_s {
    cublasHandle_t handle;
    gemm_cuda_batch_pool_t batch_pool;
} gemm_cuda_stream_state_t;

typedef struct gemm_cuda_batch_match_data_s {
    int max_batch_size;
    int accepted;
} gemm_cuda_batch_match_data_t;

#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20
#define NBELEM 1
#define GEMM_CUDA_BATCH_LIMIT_REACHED (-1000000)

static int
gemm_cuda_batch_enabled(void)
{
    return GEMM_CUDA_BATCH_NONE != cuda_batch_mode;
}

static void
gemm_cuda_set_batch_mode(const char *mode)
{
    if( 0 == strcmp(mode, "none") ) {
        cuda_batch_mode = GEMM_CUDA_BATCH_NONE;
    } else if( (0 == strcmp(mode, "one-by-one")) ||
               (0 == strcmp(mode, "submit")) ) {
        cuda_batch_mode = GEMM_CUDA_BATCH_ONE_BY_ONE;
    } else if( (0 == strcmp(mode, "cublas")) ||
               (0 == strcmp(mode, "batched")) ) {
        cuda_batch_mode = GEMM_CUDA_BATCH_CUBLAS;
    } else {
        fprintf(stderr, "Error: batch mode should be 'none', 'one-by-one', or 'cublas' (got '%s')\n", mode);
        exit(1);
    }
}

static const char *
gemm_cuda_batch_mode_name(void)
{
    switch(cuda_batch_mode) {
    case GEMM_CUDA_BATCH_ONE_BY_ONE:
        return "one-by-one";
    case GEMM_CUDA_BATCH_CUBLAS:
        return "cublas";
    case GEMM_CUDA_BATCH_NONE:
    default:
        return "none";
    }
}

static int
gemm_parse_positive_int_arg(const char *option, const char *value)
{
    int result = atoi(value);

    if( result <= 0 ) {
        fprintf(stderr, "Error: %s expects a positive integer (got '%s')\n",
                option, value);
        exit(1);
    }
    return result;
}

static unsigned long long int Rnd64_jump(unsigned long long int n, unsigned long long int seed)
{
    unsigned long long int a_k, c_k, ran;
    int i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    ran = seed;
    for( i = 0; n; n >>= 1, ++i ) {
        if( n & 1 )
            ran = a_k * ran + c_k;
        c_k *= (a_k + 1);
        a_k *= a_k;
    }

    return ran;
}

int initialize_tile(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    double *data;
    int i, j, mb, nb, m, n, M, ld;
    unsigned int seed;
    unsigned long long jump, ran;

    parsec_dtd_unpack_args(this_task, &data, &m, &n, &mb, &nb, &M, &ld, &seed);

    jump = (unsigned long long int)m + (unsigned long long int)n * (unsigned long long int)M;

    for( j = 0; j < nb; j++ ) {
        ran = Rnd64_jump(NBELEM * jump, seed);
        for( i = 0; i < mb; i++ ) {
            *data = 0.5f - ran * RndF_Mul;
            ran = Rnd64_A * ran + Rnd64_C;
            data++;
        }
        data += ld - i;
        jump += M;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

int initialize_matrix(parsec_context_t *parsec_context, int rank, parsec_matrix_block_cyclic_t *mat, unsigned int seed,
                      const char *name, int *gpu_device_index, int nb_gpus)
{
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_data_key_t key;
    int perr;

    parsec_task_class_t *init_tc;

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    init_tc = parsec_dtd_create_task_class(tp, "init",
                                           PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                           sizeof(int), PARSEC_VALUE,          /* m    */
                                           sizeof(int), PARSEC_VALUE,          /* n    */
                                           sizeof(int), PARSEC_VALUE,          /* mb   */
                                           sizeof(int), PARSEC_VALUE,          /* nb   */
                                           sizeof(int), PARSEC_VALUE,          /* M    */
                                           sizeof(int), PARSEC_VALUE,          /* ld   */
                                           sizeof(unsigned int), PARSEC_VALUE, /* seed */
                                           PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, init_tc, PARSEC_DEV_CPU, initialize_tile);

    int g = 0;
    for( int i = 0; i < mat->super.mt; i++ ) {
        for( int j = 0; j < mat->super.nt; j++ ) {
            key = mat->super.super.data_key(&mat->super.super, i, j);
            parsec_dtd_insert_task_with_task_class(tp, init_tc, 1, PARSEC_DEV_CPU,
                                                   PARSEC_PUSHOUT, PARSEC_DTD_TILE_OF_KEY(&mat->super.super, key),
                                                   PARSEC_DTD_EMPTY_FLAG, &i,
                                                   PARSEC_DTD_EMPTY_FLAG, &j,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.mb,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.nb,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.m,
                                                   PARSEC_DTD_EMPTY_FLAG, &mat->super.mb,
                                                   PARSEC_DTD_EMPTY_FLAG, &seed,
                                                   PARSEC_DTD_ARG_END);
            if(PARSEC_DEV_CUDA == device &&
               (int)mat->super.super.rank_of_key(&mat->super.super, key) == rank ) {
                if( verbose ) {
                    fprintf(stderr, "Advice %s(%d, %d) to prefer GPU device %d (parsec device %d) of rank %d\n",
                            name, i, j, g, gpu_device_index[g], (int)mat->super.super.rank_of_key(&mat->super.super, key));
                }
                parsec_advise_data_on_device(mat->super.super.data_of_key(&mat->super.super, key),
                                             gpu_device_index[g],
                                             PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);
            }
            g = (g + 1) % nb_gpus;
        }
    }
    parsec_dtd_data_flush_all(tp, &mat->super.super);

    // Wait for task completion
    perr = parsec_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    parsec_dtd_task_class_release(tp, init_tc);

    parsec_taskpool_free(tp);

    return 0;
}

static int
gemm_cuda_batch_match(parsec_gpu_task_t *candidate,
                      parsec_gpu_task_t *batch_head,
                      void *callback_data)
{
    gemm_cuda_batch_match_data_t *batch_data = (gemm_cuda_batch_match_data_t *)callback_data;

    if( (NULL != batch_data) &&
        (batch_data->accepted >= batch_data->max_batch_size - 1) ) {
        return GEMM_CUDA_BATCH_LIMIT_REACHED;
    }

    if( (batch_head->ec->task_class == candidate->ec->task_class) &&
        (batch_head->ec->selected_chore == candidate->ec->selected_chore) &&
        (batch_head->ec->selected_device == candidate->ec->selected_device) ) {
        if( NULL != batch_data ) {
            batch_data->accepted++;
        }
        return 0;
    }
    return 1;
}

static void
gemm_cuda_unpack_task(parsec_gpu_task_t *gpu_task,
                      double **a_gpu, double **b_gpu, double **c_gpu,
                      int *m, int *n, int *k,
                      int *mb, int *nb, int *kb)
{
    double *A, *B, *C;
    parsec_task_t *this_task = gpu_task->ec;

    parsec_dtd_unpack_args(this_task,
                           &A, &B, &C,
                           m, n, k,
                           mb, nb, kb);
    (void)A; (void)B; (void)C;

    *a_gpu = parsec_dtd_get_dev_ptr(this_task, 0);
    *b_gpu = parsec_dtd_get_dev_ptr(this_task, 1);
    *c_gpu = parsec_dtd_get_dev_ptr(this_task, 2);
}

static size_t
gemm_cuda_batch_pool_ptrs_per_slot(gemm_cuda_batch_pool_t *pool)
{
    return (size_t)3 * (size_t)pool->max_batch_size;
}

static double **
gemm_cuda_batch_pool_ptrs(gemm_cuda_batch_pool_t *pool, uint64_t position)
{
    int index = (int)(position % (uint64_t)pool->max_submitted_batches);

    return pool->ptrs + (size_t)index * gemm_cuda_batch_pool_ptrs_per_slot(pool);
}

static int
gemm_cuda_batch_pool_can_submit(gemm_cuda_batch_pool_t *pool)
{
    return (pool->next_submit - pool->next_complete) <
           (uint64_t)pool->max_submitted_batches;
}

static void
gemm_cuda_batch_pool_mark_pending(gemm_cuda_batch_pool_t *pool)
{
    assert(gemm_cuda_batch_pool_can_submit(pool));

    pool->next_submit++;
}

static int
gemm_cuda_batch_pool_release_completed(gemm_cuda_batch_pool_t *pool)
{
    if( pool->next_complete == pool->next_submit ) {
        parsec_warning("Completed GEMM batch has no preallocated batch slot");
        return PARSEC_HOOK_RETURN_ERROR;
    }

    pool->next_complete++;
    return PARSEC_HOOK_RETURN_DONE;
}

static void
gemm_cuda_batch_pool_fini(gemm_cuda_batch_pool_t *pool)
{
    if( NULL == pool ) {
        return;
    }

    if( pool->cuda_device_index >= 0 ) {
        (void)cudaSetDevice(pool->cuda_device_index);
    }

    if( NULL != pool->ptrs ) {
        (void)cudaFree(pool->ptrs);
    }
    memset(pool, 0, sizeof(*pool));
    pool->cuda_device_index = -1;
}

static int
gemm_cuda_batch_pool_init(gemm_cuda_batch_pool_t *pool)
{
    size_t ptrs_per_slot;
    size_t ptrs_size;
    size_t all_ptrs_size;
    cudaError_t status;

    memset(pool, 0, sizeof(*pool));
    pool->max_batch_size = cuda_max_batch_size;
    pool->max_submitted_batches = cuda_max_submitted_batches;
    pool->cuda_device_index = -1;
    ptrs_per_slot = gemm_cuda_batch_pool_ptrs_per_slot(pool);
    ptrs_size = ptrs_per_slot * sizeof(double *);
    all_ptrs_size = (size_t)pool->max_submitted_batches * ptrs_size;

    status = cudaGetDevice(&pool->cuda_device_index);
    PARSEC_CUDA_CHECK_ERROR("cudaGetDevice", status, { goto error; });

    status = cudaMallocManaged((void **)&pool->ptrs, all_ptrs_size, cudaMemAttachGlobal);
    PARSEC_CUDA_CHECK_ERROR("cudaMallocManaged", status, { goto error; });

    return PARSEC_SUCCESS;

  error:
    gemm_cuda_batch_pool_fini(pool);
    return PARSEC_ERROR;
}

static int
gemm_cuda_batch_complete(parsec_device_gpu_module_t *gpu_device,
                         parsec_gpu_task_t **gpu_task,
                         parsec_gpu_exec_stream_t *gpu_stream)
{
    gemm_cuda_stream_state_t *stream_state;
    parsec_gpu_task_t *completed_task = *gpu_task;
    int rc;

    (void)gpu_device;

    stream_state = parsec_info_get(&gpu_stream->infos, CuHI);
    if( NULL == stream_state ) {
        parsec_warning("Completed GEMM batch task %p has no CUDA stream state",
                       (void *)completed_task);
        completed_task->complete_stage = NULL;
        return PARSEC_HOOK_RETURN_ERROR;
    }

    rc = gemm_cuda_batch_pool_release_completed(&stream_state->batch_pool);
    completed_task->complete_stage = NULL;
    return rc;
}

static int
gemm_kernel_cuda_submit_one_by_one(parsec_gpu_task_t *gpu_task,
                                   parsec_gpu_exec_stream_t *gpu_stream,
                                   cublasHandle_t handle,
                                   double *one_device,
                                   int batch_count)
{
    parsec_gpu_task_t *current_gpu_task = gpu_task;
    struct timeval start, end, diff;
    double delta;

    if( verbose ) {
        gettimeofday(&start, NULL);
    }
    do {
        int m, n, k, mb, nb, kb;
        parsec_task_t *this_task = current_gpu_task->ec;
        double *a_gpu, *b_gpu, *c_gpu;
        cublasStatus_t status;

        gemm_cuda_unpack_task(current_gpu_task,
                              &a_gpu, &b_gpu, &c_gpu,
                              &m, &n, &k,
                              &mb, &nb, &kb);

        status = cublasDgemm_v2(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                mb, nb, kb,
                                one_device, a_gpu, mb,
                                b_gpu, kb,
                                one_device, c_gpu, mb);

        if(verbose) {
            fprintf(stderr, "GEMM(%d, %d, %d) with tiles of %dx%d, %dx%d, %dx%d on node %d, GPU %s submitted%s\n",
                    m, n, k, mb, kb, kb, nb, mb, kb,
                    this_task->taskpool->context->my_rank,
                    gpu_stream->name,
                    batch_count > 1 ? " as part of a one-by-one batch" : "");
        }

        PARSEC_CUDA_CHECK_ERROR("cublasDgemm_v2", status,
                                { return PARSEC_HOOK_RETURN_ERROR; });

        current_gpu_task = (parsec_gpu_task_t *)current_gpu_task->list_item.list_next;
    } while( current_gpu_task != gpu_task );

    if( verbose ) {
        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);
        delta = (double)diff.tv_sec + (double)diff.tv_usec/1e6;
        fprintf(stderr, "Submitted %d GEMM task%s one-by-one on GPU stream %s in %g s\n",
                batch_count, batch_count > 1 ? "s" : "", gpu_stream->name, delta);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

static int
gemm_kernel_cuda_submit_cublas_batched(parsec_gpu_task_t *gpu_task,
                                       parsec_gpu_exec_stream_t *gpu_stream,
                                       cublasHandle_t handle,
                                       double *one_device,
                                       gemm_cuda_batch_pool_t *batch_pool,
                                       int batch_count)
{
    parsec_gpu_task_t *current_gpu_task = gpu_task;
    double **ptr_A, **ptr_B, **ptr_C;
    struct timeval start, end, diff;
    double delta;
    int first_m = 0, first_n = 0, first_k = 0;
    int mb = 0, nb = 0, kb = 0;
    int first_mb = 0, first_nb = 0, first_kb = 0;
    int rank = gpu_task->ec->taskpool->context->my_rank;
    cublasStatus_t status;
    uint64_t submit_position = batch_pool->next_submit;

    assert(batch_count <= batch_pool->max_batch_size);

    ptr_A = gemm_cuda_batch_pool_ptrs(batch_pool, submit_position);
    ptr_B = ptr_A + batch_count;
    ptr_C = ptr_B + batch_count;

    for( int i = 0; i < batch_count; i++ ) {
        int m, n, k;

        gemm_cuda_unpack_task(current_gpu_task,
                              &ptr_A[i], &ptr_B[i], &ptr_C[i],
                              &m, &n, &k,
                              &mb, &nb, &kb);
        if( 0 == i ) {
            first_m = m; first_n = n; first_k = k;
            first_mb = mb; first_nb = nb; first_kb = kb;
        } else if( (first_mb != mb) || (first_nb != nb) || (first_kb != kb) ) {
            parsec_warning("cublas batched GEMM requires uniform tile shapes; falling back to one-by-one submission");
            return gemm_kernel_cuda_submit_one_by_one(gpu_task, gpu_stream, handle, one_device, batch_count);
        }
        current_gpu_task = (parsec_gpu_task_t *)current_gpu_task->list_item.list_next;
    }

    if( verbose ) {
        gettimeofday(&start, NULL);
    }
    status = cublasDgemmBatched(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                first_mb, first_nb, first_kb,
                                one_device, (const double * const *)ptr_A, first_mb,
                                (const double * const *)ptr_B, first_kb,
                                one_device, ptr_C, first_mb,
                                batch_count);

    PARSEC_CUDA_CHECK_ERROR("cublasDgemmBatched", status,
                            { return PARSEC_HOOK_RETURN_ERROR; });
    gemm_cuda_batch_pool_mark_pending(batch_pool);
    gpu_task->complete_stage = gemm_cuda_batch_complete;

    if(verbose) {
        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);
        delta = (double)diff.tv_sec + (double)diff.tv_usec/1e6;
        fprintf(stderr, "Batched GEMM(%d, %d, %d) with tiles of %dx%d, %dx%d, %dx%d on node %d, GPU %s submitted in %g s with %d tasks\n",
                first_m, first_n, first_k, first_mb, first_kb, first_kb, first_nb, first_mb, first_kb,
                rank, gpu_stream->name, delta, batch_count);
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int gemm_kernel_cuda(parsec_device_gpu_module_t *gpu_device,
                     parsec_gpu_task_t *gpu_task,
                     parsec_gpu_exec_stream_t *gpu_stream)
{
    gemm_cuda_stream_state_t *stream_state;
    cublasHandle_t handle;
    double *one_device = NULL;
    int batch_count = 1;
    gemm_cuda_batch_pool_t *batch_pool = NULL;

    stream_state = parsec_info_get(&gpu_stream->infos, CuHI);
    if( NULL == stream_state ) {
        return PARSEC_HOOK_RETURN_ERROR;
    }
    handle = stream_state->handle;

    if( GEMM_CUDA_BATCH_CUBLAS == cuda_batch_mode ) {
        batch_pool = &stream_state->batch_pool;
        if( !gemm_cuda_batch_pool_can_submit(batch_pool) ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        }
    }

    if( gemm_cuda_batch_enabled() && (cuda_max_batch_size > 1) ) {
        gemm_cuda_batch_match_data_t batch_data = {
            .max_batch_size = cuda_max_batch_size,
            .accepted = 0
        };
        int nb_batched = parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                                       gemm_cuda_batch_match, &batch_data);
        if( GEMM_CUDA_BATCH_LIMIT_REACHED == nb_batched ) {
            nb_batched = batch_data.accepted;
        } else if( nb_batched < 0 ) {
            return nb_batched;
        }
        batch_count += nb_batched;
    }

    one_device = parsec_info_get(&gpu_device->super.infos, Cu1);
    assert(NULL != one_device);

    if( GEMM_CUDA_BATCH_CUBLAS == cuda_batch_mode ) {
        (void)gpu_device;
        return gemm_kernel_cuda_submit_cublas_batched(gpu_task, gpu_stream,
                                                      handle, one_device,
                                                      batch_pool, batch_count);
    }
    return gemm_kernel_cuda_submit_one_by_one(gpu_task, gpu_stream, handle,
                                              one_device, batch_count);
}

#if defined(HAVE_BLAS)
int gemm_kernel_cpu(parsec_execution_stream_t *es,
                    parsec_task_t *this_task)
{
    double *A, *B, *C;
    int m, n, k, mb, nb, kb;
    double alpha = 1.0;
    double beta = 1.0;
    double delta;
    struct timeval start, end, diff;

    (void)es;

    parsec_dtd_unpack_args(this_task,
                           &A, &B, &C,
                           &m, &n, &k,
                           &mb, &nb, &kb);

    if( verbose ) {
        gettimeofday(&start, NULL);
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mb, nb, kb, alpha, A, mb, B, kb, beta, C, mb);

    if( verbose ) {
        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);
        delta = (double)diff.tv_sec + (double)diff.tv_usec/1e6;
        fprintf(stderr, "GEMM(%d, %d, %d) with tiles of %dx%d, %dx%d, %dx%d on node %d, on core %d: %g s\n",
                m, n, k, mb, kb, kb, nb, mb, kb,
                this_task->taskpool->context->my_rank,
                es->core_id,
                delta);
    }

    return PARSEC_HOOK_RETURN_DONE;
}
#endif

int simple_gemm(parsec_context_t *parsec_context, parsec_matrix_block_cyclic_t *A, parsec_matrix_block_cyclic_t *B, parsec_matrix_block_cyclic_t *C)
{
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_data_key_t keyA, keyB, keyC;
    int perr;

    parsec_task_class_t *gemm_tc;

    perr = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    perr = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(perr, "parsec_context_add_taskpool");

    gemm_tc = parsec_dtd_create_task_class(tp, "GEMM",
                                           PASSED_BY_REF, PARSEC_INPUT | TILE_FULL, /* A  */
                                           PASSED_BY_REF, PARSEC_INPUT | TILE_FULL, /* B  */
                                           PASSED_BY_REF, PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY, /* C  */
                                           sizeof(int), PARSEC_VALUE,               /* m  */
                                           sizeof(int), PARSEC_VALUE,               /* n  */
                                           sizeof(int), PARSEC_VALUE,               /* k  */
                                           sizeof(int), PARSEC_VALUE,               /* mb */
                                           sizeof(int), PARSEC_VALUE,               /* nb */
                                           sizeof(int), PARSEC_VALUE,               /* kb */
                                           PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, gemm_tc,
                                    gemm_cuda_batch_enabled() ? (PARSEC_DEV_CUDA | PARSEC_DEV_CHORE_ALLOW_BATCH) : PARSEC_DEV_CUDA,
                                    gemm_kernel_cuda);
#if defined(HAVE_BLAS)
    parsec_dtd_task_class_add_chore(tp, gemm_tc, PARSEC_DEV_CPU, gemm_kernel_cpu);
#endif

    for( int i = 0; i < C->super.mt; i++ ) {
        for( int j = 0; j < C->super.nt; j++ ) {
            keyC = C->super.super.data_key(&C->super.super, i, j);
            for( int k = 0; k < A->super.nt; k++ ) {
                keyA = A->super.super.data_key(&A->super.super, i, k);
                keyB = B->super.super.data_key(&B->super.super, k, j);
                parsec_dtd_insert_task_with_task_class(tp, gemm_tc, C->super.mt*C->super.nt*A->super.nt - i*C->super.nt + j,
                                                       device,
                                                       PARSEC_INPUT, PARSEC_DTD_TILE_OF_KEY(&A->super.super, keyA),
                                                       PARSEC_INPUT, PARSEC_DTD_TILE_OF_KEY(&B->super.super, keyB),
                                                       k == A->super.nt - 1 ? (PARSEC_INOUT | PARSEC_PUSHOUT) : PARSEC_INOUT,
                                                       PARSEC_DTD_TILE_OF_KEY(&C->super.super, keyC),
                                                       PARSEC_DTD_EMPTY_FLAG, &i,
                                                       PARSEC_DTD_EMPTY_FLAG, &j,
                                                       PARSEC_DTD_EMPTY_FLAG, &k,
                                                       PARSEC_DTD_EMPTY_FLAG, &C->super.mb,
                                                       PARSEC_DTD_EMPTY_FLAG, &C->super.nb,
                                                       PARSEC_DTD_EMPTY_FLAG, &B->super.mb,
                                                       PARSEC_DTD_ARG_END);
            }
        }
    }
    parsec_dtd_data_flush_all(tp, &A->super.super);
    parsec_dtd_data_flush_all(tp, &B->super.super);
    parsec_dtd_data_flush_all(tp, &C->super.super);

    // Wait for task completion
    perr = parsec_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(perr, "parsec_taskpool_wait");

    perr = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(perr, "parsec_context_wait");

    parsec_dtd_task_class_release(tp, gemm_tc);

    parsec_taskpool_free(tp);

    return 0;
}

int get_nb_gpu_devices()
{
    int nb = 0;

    for( int dev = 0; dev < (int)parsec_nb_devices; dev++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA & device->type ) {
            nb++;
        }
    }

    return nb;
}

int *get_gpu_device_index()
{
    int *dev_index = NULL;

    dev_index = (int *)malloc(parsec_nb_devices * sizeof(int));
    int i = 0;
    for( int dev = 0; dev < (int)parsec_nb_devices; dev++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA & device->type ) {
            dev_index[i++] = device->device_index;
        }
    }

    return dev_index;
}

static int preallocate_cuda_stream_states(void)
{
    if( PARSEC_INFO_ID_UNDEFINED == CuHI ) {
        return 0;
    }

    for( int dev = 0; dev < (int)parsec_nb_devices; dev++ ) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        parsec_device_gpu_module_t *gpu_device;

        if( 0 == (PARSEC_DEV_CUDA & device->type) ) {
            continue;
        }

        gpu_device = (parsec_device_gpu_module_t *)device;
        if( PARSEC_SUCCESS != gpu_device->set_device(gpu_device) ) {
            parsec_warning("Failed to select CUDA device %d while preallocating GEMM CUDA stream states",
                           device->device_index);
            return PARSEC_ERROR;
        }

        for( int stream = 0; stream < gpu_device->num_exec_streams; stream++ ) {
            gemm_cuda_stream_state_t *stream_state;

            stream_state = parsec_info_get(&gpu_device->exec_stream[stream]->infos, CuHI);
            if( NULL == stream_state ) {
                parsec_warning("Failed to preallocate GEMM CUDA stream state for CUDA device %d stream %d",
                               device->device_index, stream);
                return PARSEC_ERROR;
            }
        }
    }

    return PARSEC_SUCCESS;
}

static void destroy_cuda_stream_state(void *_state, void *_n)
{
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    gemm_cuda_stream_state_t *stream_state = (gemm_cuda_stream_state_t *)_state;

    if( NULL != stream_state ) {
        if( GEMM_CUDA_BATCH_CUBLAS == cuda_batch_mode ) {
            gemm_cuda_batch_pool_fini(&stream_state->batch_pool);
        }
        cublasDestroy_v2(stream_state->handle);
        free(stream_state);
    }
#endif
    (void)_n;
    (void)_state;
}

static void *create_cuda_stream_state(void *obj, void *p)
{
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    gemm_cuda_stream_state_t *stream_state;
    cublasStatus_t status;
    parsec_cuda_exec_stream_t *stream = (parsec_cuda_exec_stream_t *)obj;
    (void)p;

    stream_state = (gemm_cuda_stream_state_t *)calloc(1, sizeof(gemm_cuda_stream_state_t));
    if( NULL == stream_state ) {
        return NULL;
    }
    stream_state->batch_pool.cuda_device_index = -1;

    /* No need to call cudaSetDevice, as this has been done by PaRSEC before calling the task body */
    status = cublasCreate(&stream_state->handle);
    if( CUBLAS_STATUS_SUCCESS != status ) {
        free(stream_state);
        return NULL;
    }
    status = cublasSetStream(stream_state->handle, stream->cuda_stream);
    if( CUBLAS_STATUS_SUCCESS != status ) {
        cublasDestroy_v2(stream_state->handle);
        free(stream_state);
        return NULL;
    }

    if( GEMM_CUDA_BATCH_CUBLAS == cuda_batch_mode ) {
        if( PARSEC_SUCCESS != gemm_cuda_batch_pool_init(&stream_state->batch_pool) ) {
            cublasDestroy_v2(stream_state->handle);
            free(stream_state);
            return NULL;
        }
    }
    return (void *)stream_state;
#else
    (void)obj;
    (void)p;
    return NULL;
#endif
}

static void destroy_one_on_device(void *_h, void *_n)
{
    (void)_h;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaFree(_h);
#endif
    (void)_n;
}

static void *allocate_one_on_device(void *obj, void *p)
{
     (void)obj;
     (void)p;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
     void *one_device;
     double one_host = 1.0;
     cudaError_t cr;

     cr = cudaMallocManaged(&one_device, sizeof(double), cudaMemAttachGlobal);
     PARSEC_CUDA_CHECK_ERROR("cudaMalloc", cr,
                            { return NULL; });

     cr = cudaMemcpy(one_device, &one_host, sizeof(double), cudaMemcpyHostToDevice);
     PARSEC_CUDA_CHECK_ERROR("cudaMemcpy", cr,
                            { return NULL; });

     return one_device;
#else
    return NULL;
#endif
}

static parsec_matrix_block_cyclic_t *create_initialize_matrix(parsec_context_t *parsec_context, int rank, unsigned int seed, const char *name, int mb, int nb, int M, int N, int *gpu_device_index, int nbgpus)
{
    parsec_matrix_block_cyclic_t *dc;
    dc = calloc(1, sizeof(parsec_matrix_block_cyclic_t));
    parsec_matrix_block_cyclic_init(dc, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE, rank,
                              mb, nb,
                              M, N,
                              0, 0,
                              M, N,
                              P, Q,
                              1, 1,
                              0, 0);
    parsec_data_collection_t *A = &dc->super.super;
    parsec_data_collection_set_key(A, name);
    dc->mat = parsec_data_allocate((size_t)dc->super.nb_local_tiles *
                                   (size_t)dc->super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dc->super.mtype));
    parsec_dtd_data_collection_init(A);
    initialize_matrix(parsec_context, rank, dc, seed, name, gpu_device_index, nbgpus);

    return dc;
}

static void destroy_matrix(parsec_matrix_block_cyclic_t *dc)
{
    parsec_data_collection_t *A = &dc->super.super;
    parsec_dtd_data_collection_fini(A);
    if( NULL != dc->mat ) {
        parsec_data_free(dc->mat);
    }
    parsec_tiled_matrix_destroy_data(&dc->super);
    parsec_data_collection_destroy(&dc->super.super);

    free(dc);
}

int main(int argc, char **argv)
{
    int ret = 0, rc, nbgpus = 0;
    parsec_context_t *parsec_context = NULL;
    int rank, world;
    int mb = 1024, nb = 1024, kb = 1024;
    int M = 16 * mb, N = 16 * nb, K = 16 * kb;
    double min_perf=0.0;
    int runs = 5;
    int debug=-1;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    while( 1 ) {
        int option_index = 0;
        static struct option long_options[] = {
                {"M",       required_argument, 0, 'M'},
                {"N",       required_argument, 0, 'N'},
                {"K",       required_argument, 0, 'K'},
                {"mb",      required_argument, 0, 'm'},
                {"nb",      required_argument, 0, 'n'},
                {"kb",      required_argument, 0, 'k'},
                {"P",       required_argument, 0, 'P'},
                {"Q",       required_argument, 0, 'Q'},
                {"device",  required_argument, 0, 'd'},
                {"nruns",   required_argument, 0, 't'},
                {"verbose", no_argument,       0, 'v'},
                {"batch",   no_argument,       0, 'b'},
                {"batch-mode", required_argument, 0, 'B'},
                {"batch-size", required_argument, 0, 'S'},
                {"batch-slots", required_argument, 0, 'L'},
                {"Debug",   required_argument, 0, 'D'},
                {"Alarm",   required_argument, 0, 'A'},
                {"help",    no_argument,       0, 'h'},
                {0, 0,                         0, 0}
        };

        int c = getopt_long(argc, argv, "M:N:K:m:n:k:P:Q:t:d:B:S:L:D:A:vbh",
                            long_options, &option_index);
        if( c == -1 )
            break;

        switch( c ) {
            case 'M':
                M = atoi(optarg);
                break;
            case 'N':
                N = atoi(optarg);
                break;
            case 'K':
                K = atoi(optarg);
                break;
            case 'm':
                mb = atoi(optarg);
                break;
            case 'n':
                nb = atoi(optarg);
                break;
            case 'k':
                kb = atoi(optarg);
                break;
            case 'P':
                P = atoi(optarg);
                break;
            case 'Q':
                Q = atoi(optarg);
                break;
            case 't':
                runs = atoi(optarg);
                break;
            case 'v':
                verbose = !verbose;
                break;
            case 'b':
                cuda_batch_mode = GEMM_CUDA_BATCH_ONE_BY_ONE;
                break;
            case 'B':
                gemm_cuda_set_batch_mode(optarg);
                break;
            case 'S':
                cuda_max_batch_size = gemm_parse_positive_int_arg("--batch-size", optarg);
                break;
            case 'L':
                cuda_max_submitted_batches = gemm_parse_positive_int_arg("--batch-slots", optarg);
                break;
            case 'd':
                if(strcasecmp(optarg, "GPU") == 0) {
                    device=PARSEC_DEV_CUDA;
                } else if(strcasecmp(optarg, "CPU") == 0) {
#if defined(HAVE_BLAS)
                    device=PARSEC_DEV_CPU;
#else
                    fprintf(stderr, "Error: requested to run on CPU (--device=CPU), but no BLAS library has been found at configure time\n");
                    exit(1);
#endif
                } else {
                    fprintf(stderr, "Error: device parameter should either be 'gpu' or 'cpu' (got '%s')\n", optarg);
                    exit(1);
                }
                break;
            case 'D':
                debug = atoi(optarg);
                break;
            case 'A':
                min_perf = strtod(optarg, NULL);
                break;
            case 'h':
            case '?':
                if( 0 == rank ) {
                    fprintf(stderr,
                            "Usage %s [flags] [-- <parsec options>]\n"
                            " Nota Bene: this test should not be used to evaluate performance of GEMM!\n"
                            "    Use DPLASMA or other linear algebra libraries written on top of PaRSEC to evaluate this.\n"
                            "\n"
                            " Compute pdgemm on a process grid of PxQ, using all available GPUs on each\n"
                            " node (modulo parsec options), using DTD. Compute C += AxB, where A is MxK\n"
                            " tiled in mb x kb, B is KxN tiled in kb x nb, and C is MxN tiled in mb x nb\n"
                            " Executes nruns iterations of the GEMM operation.\n"
                            " flags:\n"
                            "   --M|-M  / --K|-K  / --N|-N:   set M, K and N (resp.)\n"
                            "   --mb|-m / --kb/-k / --nb|-n:  set mb, kb and nb (resp.)\n"
                            "   --nruns|-t:                   set the number of runs to do\n"
                            "   --device|-d:                  which device to use (CPU or GPU)\n"
                            "   --batch|-b:                   enable CUDA batch collection and submit\n"
                            "                                 the collected GEMMs one by one\n"
                            "   --batch-mode|-B:              CUDA batching mode: none, one-by-one,\n"
                            "                                 or cublas (default: %s)\n"
                            "   --batch-size|-S:              maximum number of GEMM tasks per CUDA\n"
                            "                                 batch (default: %d)\n"
                            "   --batch-slots|-L:             maximum number of in-flight cuBLAS\n"
                            "                                 batched submissions per stream (default: %d)\n"
                            "   --verbose|-v:                 display which GEMM runs on which GPU\n"
                            "                                 as execution is unfolding\n"
                            "   --help|-h|-?:                 display this help\n"
                            "   --debug|-D:                   blocks the process passed as parameter and\n"
                            "                                 waits for gdb to connect to it\n"
                            "   --Alarm|-A:                   sets the expected minimum performance for a\n"
                            "                                 single GPU (kills the process if it takes longer\n"
                            "                                 than the time corresponding to the expected\n"
                            "                                 performance to complete the product)\n"
                            "\n"
                            " Nota Bene: this test should not be used to evaluate performance of GEMM!\n"
                            "    Use DPLASMA or other linear algebra libraries written on top of PaRSEC to evaluate this.\n"
                            "\n",
                            argv[0], gemm_cuda_batch_mode_name(),
                            cuda_max_batch_size, cuda_max_submitted_batches);
                }
#if defined(PARSEC_HAVE_MPI)
                MPI_Finalize();
#endif
                exit(0);
        }
    }
    int pargc = argc - optind;
    char **pargv = argv + optind;

    if( -1 == P )
        P = (int)sqrt(world);
    if( -1 == Q )
        Q = world / P;
    while( P * Q != world ) {
        P--;
        Q = world / P;
    }

    if(debug == rank) {
        int loop=1;
        char hostname[64];
        gethostname(hostname, 64);
        fprintf(stderr, "ssh -t %s gdb -p %d\n", hostname, getpid());
        while(loop) { sleep(1); }
    }

    // Number of CPU cores involved
    int ncores = -1; // Use all available cores
    parsec_context = parsec_init(ncores, &pargc, &pargv);

    int *gpu_device_index = NULL;
    if( PARSEC_DEV_CUDA == device ) {
        nbgpus = get_nb_gpu_devices();
        rc = !(nbgpus >= 1);
        if( rc != 0 ) {
            fprintf(stderr, "Rank %d doesn't have CUDA accelerators\n", rank);
#if defined(PARSEC_HAVE_MPI)
            MPI_Abort(MPI_COMM_WORLD, 0);
#endif
            return -1;
        }
        gpu_device_index = get_gpu_device_index();

        // Prepare the CUDA stream state, including the CUBLAS handle.
        CuHI = parsec_info_register(&parsec_per_stream_infos, "DTD_GEMM::CUDA_STREAM_STATE",
                                    destroy_cuda_stream_state, NULL,
                                    create_cuda_stream_state, NULL,
                                    NULL);
        assert(CuHI != -1);
        Cu1 = parsec_info_register(&parsec_per_device_infos, "DEVICE::ONE",
                                   destroy_one_on_device, NULL,
                                   allocate_one_on_device, NULL,
                                   NULL);
        assert(Cu1 != -1);
        rc = preallocate_cuda_stream_states();
        if( PARSEC_SUCCESS != rc ) {
            fprintf(stderr, "Failed to preallocate CUDA GEMM stream states\n");
#if defined(PARSEC_HAVE_MPI)
            MPI_Abort(MPI_COMM_WORLD, rc);
#endif
            return rc;
        }
    }

    // Create datatypes
    parsec_arena_datatype_t *adt = parsec_matrix_adt_new_rect(parsec_datatype_double_t, mb, nb, mb);

    parsec_dtd_attach_arena_datatype(parsec_context, adt, &TILE_FULL);

    // Create and initialize the data
    parsec_matrix_block_cyclic_t *dcA = create_initialize_matrix(parsec_context, rank, 1789, "A", mb, kb, M, K,
                                                           gpu_device_index, nbgpus);
    parsec_matrix_block_cyclic_t *dcB = create_initialize_matrix(parsec_context, rank, 1805, "B", kb, nb, K, N,
                                                           gpu_device_index, nbgpus);
    parsec_matrix_block_cyclic_t *dcC = create_initialize_matrix(parsec_context, rank, 1901, "C", mb, nb, M, N,
                                                           gpu_device_index, nbgpus);

    for( int r = 0; r < runs + 1; r++ ) {
        double gflop = 2.0 * M * N * K / 1e9;
        double maxtime = 0.0;
        if(min_perf > 0.0)
            maxtime = gflop/world/nbgpus/min_perf;
        struct timeval start, end, diff;
        if(maxtime > 0.0 && maxtime < 60.0) maxtime=60.0;
        if(rank == 0 && maxtime > 0.0) fprintf(stderr, "watchdog: %d seconds\n", (int)maxtime);
        if(maxtime > 0.0) alarm((int)maxtime);
        gettimeofday(&start, NULL);
        simple_gemm(parsec_context, dcA, dcB, dcC);
        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);
        double t = (double)diff.tv_sec + (double)diff.tv_usec / 1e6;
        double gflops = gflop / t;
        if( 0 == rank && r > 0 ) {
            fprintf(stderr, "DTD_GEMM PxQxg: %d %d %d M: %d N: %d K: %d mb: %d nb: %d kb: %d batch_mode: %s batch_size: %d batch_slots: %d time: %.6f gflops: %.6f -- done\n",
                    P, Q, nbgpus, M, N, K, mb, nb, kb,
                    gemm_cuda_batch_mode_name(), cuda_max_batch_size,
                    cuda_max_submitted_batches, t, gflops);
        }
    }
    // deactivate the alarm if it was set
    alarm(0);

    if(PARSEC_DEV_CUDA == device) {
        // Cleanup data and parsec data structures
        parsec_info_unregister(&parsec_per_stream_infos, CuHI, NULL);
        parsec_info_unregister(&parsec_per_device_infos, Cu1, NULL);
    }

    parsec_dtd_free_arena_datatype(parsec_context, TILE_FULL);

    destroy_matrix(dcA);
    destroy_matrix(dcB);
    destroy_matrix(dcC);

    parsec_fini(&parsec_context);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    return ret;
}
