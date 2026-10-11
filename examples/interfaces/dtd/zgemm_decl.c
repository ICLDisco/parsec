/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * A real kernel written with PARSEC_DTD_DECLARE_TASK_CLASS.
 *
 * This is DPLASMA's src/dtd_wrappers/zgemm.c, rewritten on the generated
 * interface, kept here so the two can be read side by side. It is shown for
 * reading rather than for running: it needs DPLASMA, a BLAS and cuBLAS, so it
 * is deliberately not listed in this directory's CMakeLists.txt. The
 * buildable examples next to it use the same macro on smaller kernels.
 *
 * Three lists have to be kept in lockstep by hand in the original: the
 * task-class description, the argument list at every insertion site, and the
 * body's parsec_dtd_unpack_args(). Here all three come out of ZGEMM_PARAMS,
 * so they cannot drift and every scalar is type-checked at both ends.
 *
 * The two bodies below read their parameters in different styles on purpose:
 * the CPU one declares a local per parameter, the CUDA one reads the values
 * where the runtime put them. Both are generated from the list above them.
 */

#include "dplasma/config.h"
#include "dplasma_z_dtd.h"
#include "parsec/interfaces/dtd/dtd_task_class_decl.h"

#define ZGEMM_PARAMS(_)                                        \
    _(VALUE, int,                 transA, 0              )     \
    _(VALUE, int,                 transB, 0              )     \
    _(VALUE, int,                 m,      0              )     \
    _(VALUE, int,                 n,      0              )     \
    _(VALUE, int,                 k,      0              )     \
    _(VALUE, dplasma_complex64_t, alpha,  0              )     \
    _(INPUT, dplasma_complex64_t, A,      0              )     \
    _(VALUE, int,                 lda,    0              )     \
    _(INPUT, dplasma_complex64_t, B,      0              )     \
    _(VALUE, int,                 ldb,    0              )     \
    _(VALUE, dplasma_complex64_t, beta,   0              )     \
    _(INOUT, dplasma_complex64_t, C,      PARSEC_AFFINITY)     \
    _(VALUE, int,                 ldc,    0              )

PARSEC_DTD_DECLARE_TASK_CLASS(zgemm, ZGEMM_PARAMS);

/* Terse style: one typed local per parameter, declared by the same list. */
int
parsec_core_zgemm_decl(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    PARSEC_DTD_UNPACK_LOCALS(zgemm, ZGEMM_PARAMS, this_task);

    CORE_zgemm(transA, transB,
               m, n, k,
               alpha, A, lda,
                      B, ldb,
               beta,  C, ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

#if defined(DPLASMA_HAVE_CUDA)

int
parsec_core_zgemm_cuda_decl(parsec_device_gpu_module_t* gpu_device,
                            parsec_gpu_task_t*          gpu_task,
                            parsec_gpu_exec_stream_t*   gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    cublasStatus_t status;
    dplasma_cuda_handles_t* handles;

    /* In-place style, for contrast with the CPU body above: the scalars are
     * read where the runtime already put them, and only the three device
     * pointers are fetched. zgemm_body_t with zgemm_unpack() would give the
     * same values in a private copy. */
    const zgemm_values_t *v = zgemm_values(this_task);
    zgemm_flows_t         g = zgemm_device_flows(this_task);

#if defined(PRECISION_z) || defined(PRECISION_c)
    cuDoubleComplex alphag = make_cuDoubleComplex( creal(v->alpha), cimag(v->alpha));
    cuDoubleComplex betag  = make_cuDoubleComplex( creal(v->beta),  cimag(v->beta));
#else
    double alphag = v->alpha;
    double betag  = v->beta;
#endif

    handles = parsec_info_get(&gpu_stream->infos, dplasma_dtd_cuda_infoid);
    assert(NULL != handles);

    parsec_cuda_exec_stream_t* cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;
    cublasSetStream( handles->cublas_handle, cuda_stream->cuda_stream );
    status = cublasZgemm(handles->cublas_handle,
                         dplasma_cublas_op(v->transA), dplasma_cublas_op(v->transB),
                         v->n, v->m, v->k,
                         &alphag, (cuDoubleComplex*)g.A, v->lda,
                                  (cuDoubleComplex*)g.B, v->ldb,
                         &betag,  (cuDoubleComplex*)g.C, v->ldc );

    DPLASMA_CUBLAS_CHECK_STATUS( "cublasZgemm ", status,
                                 {return PARSEC_HOOK_RETURN_ERROR;} );

    (void)gpu_device;
    return PARSEC_HOOK_RETURN_DONE;
}

#endif /* defined(DPLASMA_HAVE_CUDA) */

parsec_task_class_t*
parsec_dtd_create_zgemm_decl_task_class(parsec_taskpool_t* dtd_tp, int tile_full, int devices)
{
    parsec_task_class_t* zgemm_tc = zgemm_task_class_new(dtd_tp, tile_full);

#if defined(DPLASMA_HAVE_CUDA)
    if( devices & PARSEC_DEV_CUDA )
        parsec_dtd_task_class_add_chore(dtd_tp, zgemm_tc, PARSEC_DEV_CUDA, parsec_core_zgemm_cuda_decl);
#endif

    if( devices & PARSEC_DEV_CPU )
        parsec_dtd_task_class_add_chore(dtd_tp, zgemm_tc, PARSEC_DEV_CPU, parsec_core_zgemm_decl);

    return zgemm_tc;
}

/*
 * What a call site looks like. Compare with the thirteen ordered triplets in
 * tests/testing_zpotrf_dtd_untied.c, where the mapping from position to
 * meaning is carried only by a trailing comment.
 */
void
dplasma_zgemm_decl_insert_example(parsec_taskpool_t *dtd_tp, parsec_task_class_t *zgemm_tc,
                                  parsec_dtd_tile_t *tile_A, parsec_dtd_tile_t *tile_B,
                                  parsec_dtd_tile_t *tile_C,
                                  int priority, int devices, int tempmm, int nb,
                                  int ldam, int ldan, int last_k)
{
    dplasma_complex64_t alpha = -1.0, beta = 1.0;

    zgemm_insert(dtd_tp, zgemm_tc, priority, devices, &(zgemm_args_t){
        .transA = dplasmaNoTrans,
        .transB = dplasmaConjTrans,
        .m      = tempmm,
        .n      = nb,
        .k      = nb,
        .alpha  = alpha,
        .A      = tile_A, .lda = ldan,
        .B      = tile_B, .ldb = ldam,
        .beta   = beta,
        .C      = tile_C, .ldc = ldan,
        /* Per-insertion flow flags, absent above because they default to 0. */
        .C_flags = last_k ? PARSEC_PUSHOUT : PARSEC_DTD_EMPTY_FLAG,
    });
}
