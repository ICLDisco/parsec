/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
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

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaUpper;
    int info = 0;
    int ret = 0;

    int m, n, k, total; /* loop counter */
    /* Parameters passed on to Insert_task() */
    int tempkm, tempmm, ldak, ldam, side, transA_p, transA_g, diag, trans, transB, ldan;
    parsec_complex64_t alpha_trsm, alpha_herk, beta;

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

    /* initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    LDB = dplasma_imax( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        sym_two_dim_block_cyclic, (&dcA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    /* Initializing dc for dtd */
    sym_two_dim_block_cyclic_t *__dcA = &dcA;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( parsec, (double)(N), uplo,
                    (parsec_tiled_matrix_dc_t *)&dcA, random_seed);
    if(loud > 3) printf("Done\n");

    /* Getting new parsec handle of dtd type */
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    /* Allocating data arrays to be used by comm engine */
    dplasma_add2arena_tile( parsec_dtd_arenas[TILE_FULL],
                            dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, dcA.super.mb );

    /* Registering the handle with parsec context */
    parsec_enqueue( parsec, dtd_tp );

    SYNC_TIME_START();

    /* #### parsec context Starting #### */

    /* start parsec context */
    parsec_context_start( parsec );

    if( PlasmaLower == uplo ) {

        side = PlasmaRight;
        transA_p = PlasmaConjTrans;
        diag = PlasmaNonUnit;
        alpha_trsm = 1.0;
        trans = PlasmaNoTrans;
        alpha_herk = -1.0;
        beta = 1.0;
        transB = PlasmaConjTrans;
        transA_g = PlasmaNoTrans;

        total = dcA.super.mt;
        /* Testing Insert Function */
        for( k = 0; k < total; k++ ) {
            tempkm = (k == (dcA.super.mt - 1)) ? dcA.super.m - k * dcA.super.mb : dcA.super.mb;
            ldak = BLKLDD(&dcA.super, k);

            parsec_insert_task( dtd_tp, parsec_core_potrf,
                              (total - k) * (total-k) * (total - k)/*priority*/, "Potrf",
                               sizeof(int),      &uplo,              VALUE,
                               sizeof(int),      &tempkm,            VALUE,
                               PASSED_BY_REF,    TILE_OF(A, k, k), INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),      &ldak,              VALUE,
                               sizeof(int *),    &info,              SCRATCH,
                               0 );

            for( m = k+1; m < total; m++ ) {
                tempmm = m == dcA.super.mt - 1 ? dcA.super.m - m * dcA.super.mb : dcA.super.mb;
                ldam = BLKLDD(&dcA.super, m);
                parsec_insert_task( dtd_tp, parsec_core_trsm,
                                  (total - m) * (total-m) * (total - m) + 3 * ((2 * total) - k - m - 1) * (m - k)/*priority*/, "Trsm",
                                   sizeof(int),      &side,               VALUE,
                                   sizeof(int),      &uplo,               VALUE,
                                   sizeof(int),      &transA_p,           VALUE,
                                   sizeof(int),      &diag,               VALUE,
                                   sizeof(int),      &tempmm,             VALUE,
                                   sizeof(int),      &dcA.super.nb,    VALUE,
                                   sizeof(parsec_complex64_t),      &alpha_trsm,         VALUE,
                                   PASSED_BY_REF,    TILE_OF(A, k, k), INPUT | TILE_FULL,
                                   sizeof(int),      &ldak,               VALUE,
                                   PASSED_BY_REF,    TILE_OF(A, m, k), INOUT | TILE_FULL | AFFINITY,
                                   sizeof(int),      &ldam,               VALUE,
                                   0 );
            }
            parsec_dtd_data_flush( dtd_tp, TILE_OF(A, k, k) );

            for( m = k+1; m < dcA.super.nt; m++ ) {
                tempmm = m == dcA.super.mt - 1 ? dcA.super.m - m * dcA.super.mb : dcA.super.mb;
                ldam = BLKLDD(&dcA.super, m);
                parsec_insert_task( dtd_tp, parsec_core_herk,
                                  (total - m) * (total - m) * (total - m) + 3 * (m - k)/*priority*/, "Herk",
                                   sizeof(int),       &uplo,               VALUE,
                                   sizeof(int),       &trans,              VALUE,
                                   sizeof(int),       &tempmm,             VALUE,
                                   sizeof(int),       &dcA.super.mb,    VALUE,
                                   sizeof(parsec_complex64_t),       &alpha_herk,         VALUE,
                                   PASSED_BY_REF,     TILE_OF(A, m, k), INPUT | TILE_FULL,
                                   sizeof(int),       &ldam,               VALUE,
                                   sizeof(parsec_complex64_t),       &beta,               VALUE,
                                   PASSED_BY_REF,     TILE_OF(A, m, m), INOUT | TILE_FULL | AFFINITY,
                                   sizeof(int),       &ldam,               VALUE,
                                   0 );

                for( n = m+1; n < total; n++ ) {
                    ldan = BLKLDD(&dcA.super, n);
                    parsec_insert_task( dtd_tp,  parsec_core_gemm,
                                      (total - m) * (total - m) * (total - m) + 3 * ((2 * total) - m - n - 3) * (m - n) + 6 * (m - k) /*priority*/, "Gemm",
                                       sizeof(int),        &transA_g,           VALUE,
                                       sizeof(int),        &transB,             VALUE,
                                       sizeof(int),        &tempmm,             VALUE,
                                       sizeof(int),        &dcA.super.mb,    VALUE,
                                       sizeof(int),        &dcA.super.mb,    VALUE,
                                       sizeof(parsec_complex64_t),        &alpha_herk,         VALUE,
                                       PASSED_BY_REF,      TILE_OF(A, n, k), INPUT | TILE_FULL,
                                       sizeof(int),        &ldan,               VALUE,
                                       PASSED_BY_REF,      TILE_OF(A, m, k), INPUT | TILE_FULL,
                                       sizeof(int),        &ldam,               VALUE,
                                       sizeof(parsec_complex64_t),        &beta,               VALUE,
                                       PASSED_BY_REF,      TILE_OF(A, n, m), INOUT | TILE_FULL | AFFINITY,
                                       sizeof(int),        &ldan,               VALUE,
                                       0 );
                }
                parsec_dtd_data_flush( dtd_tp, TILE_OF(A, m, k) );
            }
        }
    } else {
        side = PlasmaLeft;
        transA_p = PlasmaConjTrans;
        diag = PlasmaNonUnit;
        alpha_trsm = 1.0;
        trans = PlasmaConjTrans;
        alpha_herk = -1.0;
        beta = 1.0;
        transB = PlasmaNoTrans;
        transA_g = PlasmaConjTrans;

        total = dcA.super.nt;

        for( k = 0; k < total; k++ ) {
            tempkm = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;
            ldak = BLKLDD(&dcA.super, k);
            parsec_insert_task( dtd_tp, parsec_core_potrf, 4, "Potrf",
                               sizeof(int),      &uplo,              VALUE,
                               sizeof(int),      &tempkm,            VALUE,
                               PASSED_BY_REF,    TILE_OF(A, k, k), INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),      &ldak,              VALUE,
                               sizeof(int *),    &info,              SCRATCH,
                               0 );

            for( m = k+1; m < total; m++ ) {
                tempmm = m == dcA.super.nt-1 ? dcA.super.n-m*dcA.super.nb : dcA.super.nb;
                parsec_insert_task( dtd_tp, parsec_core_trsm, 3, "Trsm",
                                   sizeof(int),      &side,               VALUE,
                                   sizeof(int),      &uplo,               VALUE,
                                   sizeof(int),      &transA_p,           VALUE,
                                   sizeof(int),      &diag,               VALUE,
                                   sizeof(int),      &dcA.super.nb,    VALUE,
                                   sizeof(int),      &tempmm,             VALUE,
                                   sizeof(parsec_complex64_t),      &alpha_trsm,         VALUE,
                                   PASSED_BY_REF,    TILE_OF(A, k, k), INPUT | TILE_FULL,
                                   sizeof(int),      &ldak,               VALUE,
                                   PASSED_BY_REF,    TILE_OF(A, k, m), INOUT | TILE_FULL | AFFINITY,
                                   sizeof(int),      &ldak,               VALUE,
                                   0 );
            }
            parsec_dtd_data_flush( dtd_tp, TILE_OF(A, k, k) );

            for( m = k+1; m < dcA.super.mt; m++ ) {
                tempmm = m == dcA.super.nt-1 ? dcA.super.n-m*dcA.super.nb : dcA.super.nb;
                ldam = BLKLDD(&dcA.super, m);
                parsec_insert_task( dtd_tp, parsec_core_herk, 2, "Herk",
                                   sizeof(int),       &uplo,               VALUE,
                                   sizeof(int),       &trans,              VALUE,
                                   sizeof(int),       &tempmm,             VALUE,
                                   sizeof(int),       &dcA.super.mb,    VALUE,
                                   sizeof(parsec_complex64_t),       &alpha_herk,         VALUE,
                                   PASSED_BY_REF,     TILE_OF(A, k, m), INPUT | TILE_FULL,
                                   sizeof(int),       &ldak,               VALUE,
                                   sizeof(parsec_complex64_t),    &beta,                  VALUE,
                                   PASSED_BY_REF,     TILE_OF(A, m, m), INOUT | TILE_FULL | AFFINITY,
                                   sizeof(int),       &ldam,               VALUE,
                                   0 );

                for( n = m+1; n < total; n++ ) {
                   ldan = BLKLDD(&dcA.super, n);
                   parsec_insert_task( dtd_tp,  parsec_core_gemm, 1, "Gemm",
                                      sizeof(int),        &transA_g,           VALUE,
                                      sizeof(int),        &transB,             VALUE,
                                      sizeof(int),        &dcA.super.mb,    VALUE,
                                      sizeof(int),        &tempmm,             VALUE,
                                      sizeof(int),        &dcA.super.mb,    VALUE,
                                      sizeof(parsec_complex64_t),        &alpha_herk,         VALUE,
                                      PASSED_BY_REF,      TILE_OF(A, k, m), INPUT | TILE_FULL,
                                      sizeof(int),        &ldak,               VALUE,
                                      PASSED_BY_REF,      TILE_OF(A, k, n), INPUT | TILE_FULL,
                                      sizeof(int),        &ldak,               VALUE,
                                      sizeof(parsec_complex64_t),        &beta,               VALUE,
                                      PASSED_BY_REF,      TILE_OF(A, m, n), INOUT | TILE_FULL | AFFINITY,
                                      sizeof(int),        &ldan,               VALUE,
                                      0 );
                }
                parsec_dtd_data_flush( dtd_tp, TILE_OF(A, k, m) );
            }
        }
    }

    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_taskpool_wait( parsec, dtd_tp );

    /* Waiting on all handle and turning everything off for this context */
    parsec_context_wait( parsec );

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
        PASTE_CODE_ALLOCATE_MATRIX(dcA0, check,
            sym_two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble,
                                       nodes, rank, MB, NB, LDA, N, 0, 0,
                                       N, N, P, uplo));
        dplasma_zplghe( parsec, (double)(N), uplo,
                        (parsec_tiled_matrix_dc_t *)&dcA0, random_seed);

        ret |= check_zpotrf( parsec, (rank == 0) ? loud : 0, uplo,
                             (parsec_tiled_matrix_dc_t *)&dcA,
                             (parsec_tiled_matrix_dc_t *)&dcA0);

        /* Check the solution */
        PASTE_CODE_ALLOCATE_MATRIX(dcB, check,
            two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, random_seed+1);

        PASTE_CODE_ALLOCATE_MATRIX(dcX, check,
            two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcB, (parsec_tiled_matrix_dc_t *)&dcX );

        dplasma_zpotrs(parsec, uplo,
                       (parsec_tiled_matrix_dc_t *)&dcA,
                       (parsec_tiled_matrix_dc_t *)&dcX );

        ret |= check_zaxmb( parsec, (rank == 0) ? loud : 0, uplo,
                            (parsec_tiled_matrix_dc_t *)&dcA0,
                            (parsec_tiled_matrix_dc_t *)&dcB,
                            (parsec_tiled_matrix_dc_t *)&dcX);

        /* Cleanup */
        parsec_data_free(dcA0.mat); dcA0.mat = NULL;
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0 );
        parsec_data_free(dcB.mat); dcB.mat = NULL;
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB );
        parsec_data_free(dcX.mat); dcX.mat = NULL;
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX );
    }

    /* Cleaning data arrays we allocated for communication */
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_FULL] );
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );

    parsec_data_free(dcA.mat); dcA.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

    cleanup_parsec(parsec, iparam);
    return ret;
}
