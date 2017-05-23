/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

/*
 *
 * This version of potrf lets any thread insert task in PaRSEC.
 * In the other approach the main thread inserts task and the others
 * execute them. The main thread maintains a window of insertion, reaching
 * which it joins the worker thread. This version lets all thread have
 * the same kind of behavior. Here, we have a task that inserts other tasks.
 * They all insert task depending on which thread acquires the right to execute
 * the task that inserts task.
 *
 */

#include "common.h"
#include "flops.h"
#include "dplasma/lib/dplasmatypes.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

enum regions {
                TILE_FULL,
             };

int
parsec_core_potrf(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    PLASMA_enum *uplo;
    int *m, *lda, *info;
    parsec_complex64_t *A;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &m,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_SCRATCH, &info);

    CORE_zpotrf(*uplo, *m, A, *lda, info);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_trsm(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    PLASMA_enum *side, *uplo, *trans, *diag;
    int  *m, *n, *lda, *ldc;
    parsec_complex64_t *alpha;
    parsec_complex64_t *A, *C;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &side,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &diag,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &alpha,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &C,
                          UNPACK_VALUE, &ldc);

    CORE_ztrsm(*side, *uplo, *trans, *diag,
               *m, *n, *alpha,
               A, *lda,
               C, *ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_herk(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    PLASMA_enum *uplo, *trans;
    int *m, *n, *lda, *ldc;
    parsec_complex64_t *alpha;
    parsec_complex64_t *beta;
    parsec_complex64_t *A;
    parsec_complex64_t *C;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &alpha,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_VALUE, &beta,
                          UNPACK_DATA,  &C,
                          UNPACK_VALUE, &ldc);

    CORE_zherk( *uplo, *trans, *m, *n,
                *alpha, A, *lda,
                *beta,  C, *ldc );

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_gemm(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    PLASMA_enum *transA, *transB;
    int *m, *n, *k, *lda, *ldb, *ldc;
    parsec_complex64_t *alpha, *beta;
    parsec_complex64_t *A;
    parsec_complex64_t *B;
    parsec_complex64_t *C;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &transA,
                          UNPACK_VALUE, &transB,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &k,
                          UNPACK_VALUE, &alpha,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &B,
                          UNPACK_VALUE, &ldb,
                          UNPACK_VALUE, &beta,
                          UNPACK_DATA,  &C,
                          UNPACK_VALUE, &ldc);

    CORE_zgemm(*transA, *transB,
               *m, *n, *k,
               *alpha, A, *lda,
                       B, *ldb,
               *beta,  C, *ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

int
insert_task_lower(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    /* Parameters passed on to Insert_task() */
    int tempkm, tempmm, ldak, ldam, side, transA_p, transA_g, diag, trans, transB, ldan;
    parsec_complex64_t alpha_trsm, alpha_herk, beta;
    sym_two_dim_block_cyclic_t *ddescA;

    int *total, *iteration, *uplo_enum, *info, m, n, k, count = 0;

    parsec_taskpool_t *dtd_tp = (parsec_taskpool_t *)this_task->taskpool;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE,   &total,
                          UNPACK_VALUE,   &iteration,
                          UNPACK_VALUE,   &uplo_enum,
                          UNPACK_SCRATCH, &info,
                          UNPACK_SCRATCH, &ddescA);

    sym_two_dim_block_cyclic_t *__ddescA = ddescA;

    PLASMA_enum uplo = *uplo_enum;
    side = PlasmaRight;
    transA_p = PlasmaConjTrans;
    diag = PlasmaNonUnit;
    alpha_trsm = 1.0;
    trans = PlasmaNoTrans;
    alpha_herk = -1.0;
    beta = 1.0;
    transB = PlasmaConjTrans;
    transA_g = PlasmaNoTrans;

    /* Testing Insert Function */
    for( k = *iteration; k < *total; k++, *iteration += 1, count++ ) {
        if( count > dtd_window_size-1000 ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        }

        tempkm = (k == (ddescA->super.mt - 1)) ? ddescA->super.m - k * ddescA->super.mb : ddescA->super.mb;
        ldak = BLKLDD(&ddescA->super, k);

        parsec_insert_task(dtd_tp, parsec_core_potrf,
                         (*total - k) * (*total-k) * (*total - k)/*priority*/, "Potrf",
                         sizeof(int),      &uplo,              VALUE,
                         sizeof(int),      &tempkm,            VALUE,
                         PASSED_BY_REF,    TILE_OF(A, k, k), INOUT | TILE_FULL | AFFINITY,
                         sizeof(int),      &ldak,              VALUE,
                         sizeof(int *),    &info,              SCRATCH,
                         0);
        for( m = k+1; m < *total; m++, count++ ) {
            tempmm = m == ddescA->super.mt - 1 ? ddescA->super.m - m * ddescA->super.mb : ddescA->super.mb;
            ldam = BLKLDD(&ddescA->super, m);
            parsec_insert_task(dtd_tp, parsec_core_trsm,
                             (*total - m) * (*total-m) * (*total - m) + 3 * ((2 * *total) - k - m - 1) * (m - k)/*priority*/, "Trsm",
                             sizeof(int),      &side,               VALUE,
                             sizeof(int),      &uplo,               VALUE,
                             sizeof(int),      &transA_p,           VALUE,
                             sizeof(int),      &diag,               VALUE,
                             sizeof(int),      &tempmm,             VALUE,
                             sizeof(int),      &ddescA->super.nb,   VALUE,
                             sizeof(parsec_complex64_t),      &alpha_trsm,         VALUE,
                             PASSED_BY_REF,    TILE_OF(A, k, k), INPUT | TILE_FULL,
                             sizeof(int),      &ldak,               VALUE,
                             PASSED_BY_REF,    TILE_OF(A, m, k), INOUT | TILE_FULL | AFFINITY,
                             sizeof(int),      &ldam,               VALUE,
                             0);
        }

        parsec_dtd_data_flush(dtd_tp, TILE_OF(A, k, k));
        for( m = k+1; m < ddescA->super.nt; m++, count++ ){
            tempmm = m == ddescA->super.mt - 1 ? ddescA->super.m - m * ddescA->super.mb : ddescA->super.mb;
            ldam = BLKLDD(&ddescA->super, m);
            parsec_insert_task(dtd_tp, parsec_core_herk,
                            (*total - m) * (*total - m) * (*total - m) + 3 * (m - k)/*priority*/, "Herk",
                            sizeof(int),       &uplo,               VALUE,
                            sizeof(int),       &trans,              VALUE,
                            sizeof(int),       &tempmm,             VALUE,
                            sizeof(int),       &ddescA->super.mb,   VALUE,
                            sizeof(parsec_complex64_t),       &alpha_herk,         VALUE,
                            PASSED_BY_REF,     TILE_OF(A, m, k), INPUT | TILE_FULL,
                            sizeof(int),       &ldam,               VALUE,
                            sizeof(parsec_complex64_t),       &beta,               VALUE,
                            PASSED_BY_REF,     TILE_OF(A, m, m), INOUT | TILE_FULL | AFFINITY,
                            sizeof(int),       &ldam,               VALUE,
                            0);
            for( n = m+1; n < *total; n++, count++ ){
                ldan = BLKLDD(&ddescA->super, n);
                parsec_insert_task(dtd_tp,  parsec_core_gemm,
                               (*total - m) * (*total - m) * (*total - m) + 3 * ((2 * *total) - m - n - 3) * (m - n) + 6 * (m - k) /*priority*/, "Gemm",
                               sizeof(int),        &transA_g,           VALUE,
                               sizeof(int),        &transB,             VALUE,
                               sizeof(int),        &tempmm,             VALUE,
                               sizeof(int),        &ddescA->super.mb,   VALUE,
                               sizeof(int),        &ddescA->super.mb,   VALUE,
                               sizeof(parsec_complex64_t),        &alpha_herk,         VALUE,
                               PASSED_BY_REF,      TILE_OF(A, n, k), INPUT | TILE_FULL,
                               sizeof(int),        &ldan,               VALUE,
                               PASSED_BY_REF,      TILE_OF(A, m, k), INPUT | TILE_FULL,
                               sizeof(int),        &ldam,               VALUE,
                               sizeof(parsec_complex64_t),        &beta,               VALUE,
                               PASSED_BY_REF,      TILE_OF(A, n, m), INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),        &ldan,               VALUE,
                               0);
            }
            parsec_dtd_data_flush(dtd_tp, TILE_OF(A, m, k));
        }
    }

    parsec_dtd_data_flush_all(dtd_tp, (parsec_ddesc_t *)ddescA);
    return PARSEC_HOOK_RETURN_DONE;
}

int
insert_task_upper(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    /* Parameters passed on to Insert_task() */
    int tempkm, tempmm, ldak, ldam, side, transA_p, transA_g, diag, trans, transB, ldan;
    parsec_complex64_t alpha_trsm, alpha_herk, beta;
    sym_two_dim_block_cyclic_t *ddescA;

    int *total, *iteration, *uplo_enum, *info, m, n, k, count = 0;

    parsec_taskpool_t *dtd_tp = (parsec_taskpool_t *)this_task->taskpool;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE,   &total,
                          UNPACK_VALUE,   &iteration,
                          UNPACK_VALUE,   &uplo_enum,
                          UNPACK_SCRATCH, &info,
                          UNPACK_SCRATCH, &ddescA);

    sym_two_dim_block_cyclic_t *__ddescA = ddescA;

    PLASMA_enum uplo = *uplo_enum;

    side = PlasmaLeft;
    transA_p = PlasmaConjTrans;
    diag = PlasmaNonUnit;
    alpha_trsm = 1.0;
    trans = PlasmaConjTrans;
    alpha_herk = -1.0;
    beta = 1.0;
    transB = PlasmaNoTrans;
    transA_g = PlasmaConjTrans;

    /* Testing Insert Function */
    for( k = *iteration; k < *total; k++, *iteration += 1, count++ ) {
        if( count > dtd_window_size-1000 ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        }

        tempkm = k == ddescA->super.nt-1 ? ddescA->super.n-k*ddescA->super.nb : ddescA->super.nb;
        ldak = BLKLDD(&ddescA->super, k);
        parsec_insert_task(dtd_tp, parsec_core_potrf,
                         (*total - k) * (*total-k) * (*total - k)/*priority*/, "Potrf",
                         sizeof(int),      &uplo,              VALUE,
                         sizeof(int),      &tempkm,            VALUE,
                         PASSED_BY_REF,    TILE_OF(A, k, k), INOUT | TILE_FULL | AFFINITY,
                         sizeof(int),      &ldak,              VALUE,
                         sizeof(int *),    info,               SCRATCH,
                         0);
        for (m = k+1; m < *total; m++, count++) {
            tempmm = m == ddescA->super.nt-1 ? ddescA->super.n-m*ddescA->super.nb : ddescA->super.nb;
            parsec_insert_task(dtd_tp, parsec_core_trsm,
                             (*total - m) * (*total-m) * (*total - m) + 3 * ((2 * *total) - k - m - 1) * (m - k)/*priority*/, "Trsm",
                             sizeof(int),      &side,               VALUE,
                             sizeof(int),      &uplo,               VALUE,
                             sizeof(int),      &transA_p,           VALUE,
                             sizeof(int),      &diag,               VALUE,
                             sizeof(int),      &ddescA->super.nb,   VALUE,
                             sizeof(int),      &tempmm,             VALUE,
                             sizeof(parsec_complex64_t),      &alpha_trsm,         VALUE,
                             PASSED_BY_REF,    TILE_OF(A, k, k), INPUT | TILE_FULL,
                             sizeof(int),      &ldak,               VALUE,
                             PASSED_BY_REF,    TILE_OF(A, k, m), INOUT | TILE_FULL | AFFINITY,
                             sizeof(int),      &ldak,               VALUE,
                             0);
        }
        parsec_dtd_data_flush(dtd_tp, TILE_OF(A, k, k));

        for (m = k+1; m < ddescA->super.mt; m++, count++) {
            tempmm = m == ddescA->super.nt-1 ? ddescA->super.n-m*ddescA->super.nb : ddescA->super.nb;
            ldam = BLKLDD(&ddescA->super, m);
            parsec_insert_task(dtd_tp, parsec_core_herk,
                            (*total - m) * (*total - m) * (*total - m) + 3 * (m - k)/*priority*/, "Herk",
                            sizeof(int),       &uplo,               VALUE,
                            sizeof(int),       &trans,              VALUE,
                            sizeof(int),       &tempmm,             VALUE,
                            sizeof(int),       &ddescA->super.mb,   VALUE,
                            sizeof(parsec_complex64_t),       &alpha_herk,         VALUE,
                            PASSED_BY_REF,     TILE_OF(A, k, m), INPUT | TILE_FULL,
                            sizeof(int),       &ldak,               VALUE,
                            sizeof(parsec_complex64_t),    &beta,                  VALUE,
                            PASSED_BY_REF,     TILE_OF(A, m, m), INOUT | TILE_FULL | AFFINITY,
                            sizeof(int),       &ldam,               VALUE,
                            0);
            for (n = m+1; n < *total; n++, count++) {
               ldan = BLKLDD(&ddescA->super, n);
               parsec_insert_task(dtd_tp,  parsec_core_gemm,
                               (*total - m) * (*total - m) * (*total - m) + 3 * ((2 * *total) - m - n - 3) * (m - n) + 6 * (m - k) /*priority*/, "Gemm",
                               sizeof(int),        &transA_g,           VALUE,
                               sizeof(int),        &transB,             VALUE,
                               sizeof(int),        &ddescA->super.mb,   VALUE,
                               sizeof(int),        &tempmm,             VALUE,
                               sizeof(int),        &ddescA->super.mb,   VALUE,
                               sizeof(parsec_complex64_t),        &alpha_herk,         VALUE,
                               PASSED_BY_REF,      TILE_OF(A, k, m), INPUT | TILE_FULL,
                               sizeof(int),        &ldak,               VALUE,
                               PASSED_BY_REF,      TILE_OF(A, k, n), INPUT | TILE_FULL,
                               sizeof(int),        &ldak,               VALUE,
                               sizeof(parsec_complex64_t),        &beta,               VALUE,
                               PASSED_BY_REF,      TILE_OF(A, m, n), INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),        &ldan,               VALUE,
                               0);
            }
            parsec_dtd_data_flush(dtd_tp, TILE_OF(A, k, m));
        }
    }

    parsec_dtd_data_flush_all(dtd_tp, (parsec_ddesc_t *)ddescA);
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaUpper;
    int info = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(PARSEC_HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));

    /* Initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    LDB = dplasma_imax( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    parsec_dtd_ddesc_init((parsec_ddesc_t *)&ddescA);

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( parsec, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, random_seed);
    if(loud > 3) printf("Done\n");

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new( );

    /* Default type */
    dplasma_add2arena_tile( parsec_dtd_arenas[0],
                            ddescA.super.mb*ddescA.super.nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, ddescA.super.mb );

    /* Registering the handle with parsec context */
    parsec_enqueue( parsec, dtd_tp );

    SYNC_TIME_START();

    /* #### PaRSEC context starting #### */

    /* Start parsec context */
    parsec_context_start(parsec);

    int iteration = 0, total;
    if( PlasmaLower == uplo ) {
        total = ddescA.super.mt;
        parsec_insert_task( dtd_tp,       insert_task_lower, 0, "insert_task_lower",
                           sizeof(int),           &total,             VALUE,
                           sizeof(int),           &iteration,         VALUE,
                           sizeof(int),           &uplo,              VALUE,
                           sizeof(int *),         &info,              SCRATCH,
                           sizeof(sym_two_dim_block_cyclic_t *), &ddescA, SCRATCH,
                           0 );

    } else {
        total = ddescA.super.nt;
        parsec_insert_task( dtd_tp,       insert_task_upper, 0, "insert_task_upper",
                           sizeof(int),           &total,             VALUE,
                           sizeof(int),           &iteration,         VALUE,
                           sizeof(int),           &uplo,              VALUE,
                           sizeof(int *),         &info,              SCRATCH,
                           sizeof(sym_two_dim_block_cyclic_t *), &ddescA, SCRATCH,
                           0 );

    }

    /* Finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_taskpool_wait( parsec, dtd_tp );

    /* Waiting on all handle and turning everything off for this context */
    parsec_context_wait(parsec);

    /* #### PaRSEC context is done #### */

    SYNC_TIME_PRINT(rank, ("\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

    /* Cleaning up the parsec handle */
    parsec_taskpool_free( dtd_tp );

    if( 0 == rank && info != 0 ) {
        printf("-- Factorization is suspicious (info = %d) ! \n", info);
        ret |= 1;
    }
    if( !info && check ) {
        /* Check the factorization */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
            sym_two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble,
                                       nodes, rank, MB, NB, LDA, N, 0, 0,
                                       N, N, P, uplo));
        dplasma_zplghe( parsec, (double)(N), uplo,
                        (tiled_matrix_desc_t *)&ddescA0, random_seed);

        ret |= check_zpotrf( parsec, (rank == 0) ? loud : 0, uplo,
                             (tiled_matrix_desc_t *)&ddescA,
                             (tiled_matrix_desc_t *)&ddescA0);

        /* Check the solution */
        PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescB, random_seed+1);

        PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
            two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );

        dplasma_zpotrs(parsec, uplo,
                       (tiled_matrix_desc_t *)&ddescA,
                       (tiled_matrix_desc_t *)&ddescX );

        ret |= check_zaxmb( parsec, (rank == 0) ? loud : 0, uplo,
                            (tiled_matrix_desc_t *)&ddescA0,
                            (tiled_matrix_desc_t *)&ddescB,
                            (tiled_matrix_desc_t *)&ddescX);

        /* Cleanup */
        parsec_data_free(ddescA0.mat); ddescA0.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0 );
        parsec_data_free(ddescB.mat); ddescB.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB );
        parsec_data_free(ddescX.mat); ddescX.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX );
    }

    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_FULL] );
    parsec_dtd_ddesc_fini( (parsec_ddesc_t *)&ddescA );
    parsec_data_free(ddescA.mat); ddescA.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);

    cleanup_parsec(parsec, iparam);
    return ret;
}
