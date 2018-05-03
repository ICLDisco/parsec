/*
 * Copyright (c) 2015-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "dplasma/lib/dplasmatypes.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/superscalar/insert_function.h"

enum regions {
               TILE_FULL,
             };

static int check_solution( parsec_context_t *parsec, int loud,
                           PLASMA_enum transA, PLASMA_enum transB,
                           parsec_complex64_t alpha, int Am, int An, int Aseed,
                                                    int Bm, int Bn, int Bseed,
                           parsec_complex64_t beta,  int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *dcCfinal );

static int
parsec_core_gemm(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    PLASMA_enum transA;
    PLASMA_enum transB;
    int m;
    int n;
    int k;
    parsec_complex64_t alpha;
    parsec_complex64_t *A;
    int lda;
    parsec_complex64_t *B;
    int ldb;
    parsec_complex64_t beta;
    parsec_complex64_t *C;
    int ldc;

    parsec_dtd_unpack_args(this_task, &transA, &transB, &m, &n, &k, &alpha, &A,
                           &lda, &B, &ldb, &beta, &C, &ldc);

    CORE_zgemm(transA, transB, m, n, k,
               alpha,  A, lda,
                       B, ldb,
               beta,   C, ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    int Aseed = 3872;
    int Bseed = 4674;
    int Cseed = 2873;
    int tA = PlasmaNoTrans;
    int tB = PlasmaNoTrans;
    parsec_complex64_t alpha =  0.51;
    parsec_complex64_t beta  = -0.42;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    PASTE_CODE_FLOPS(FLOPS_ZGEMM, ((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K));

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* Initializing dc for dtd */
    two_dim_block_cyclic_t *__dcC = &dcC;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcC);

    /* initializing matrix structure */
    if(!check)
    {
        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, K, 0, 0,
                                   M, K, SMB, SNB, P));

        /* Initializing dc for dtd */
        two_dim_block_cyclic_t *__dcA = &dcA;
        parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);

        PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
            two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, N, 0, 0,
                                   K, N, SMB, SNB, P));

        /* Initializing dc for dtd */
        two_dim_block_cyclic_t *__dcB = &dcB;
        parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcB);

        /* Getting new parsec handle of dtd type */
        parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

        /* Default type */
        dplasma_add2arena_tile( parsec_dtd_arenas[TILE_FULL],
                                dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                                PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_complex_t, dcA.super.mb );

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed);
        if(loud > 2) printf("Done\n");

        int m, n, k;
        int ldam, ldak, ldbn, ldbk, ldcm;
        int tempmm, tempnn, tempkn;

        parsec_complex64_t zbeta;
        parsec_complex64_t zone = (parsec_complex64_t)1.0;

        parsec_context_add_taskpool( parsec, dtd_tp );

        SYNC_TIME_START();

        /* #### parsec context Starting #### */

        /* start parsec context */
        parsec_context_start(parsec);

        for( m = 0; m < dcC.super.mt; m++ ) {
            tempmm = m == dcC.super.mt-1 ? dcC.super.m-m*dcC.super.mb : dcC.super.mb;
            ldcm = BLKLDD(&dcC.super, m);
            for( n = 0; n < dcC.super.nt; n++ ) {
                tempnn = n == dcC.super.nt-1 ? dcC.super.n-n*dcC.super.nb : dcC.super.nb;
                /*
                 *  A: PlasmaNoTrans / B: PlasmaNoTrans
                 */
                if( tA == PlasmaNoTrans ) {
                    ldam = BLKLDD(&dcA.super, m);
                    if( tB == PlasmaNoTrans ) {
                        for( k = 0; k < dcA.super.nt; k++ ) {
                            tempkn = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;
                            ldbk = BLKLDD(&dcB.super, k);
                            zbeta = k == 0 ? beta : zone;

                            parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm,  0, "Gemm",
                                     sizeof(PLASMA_enum),   &tA,                           VALUE,
                                     sizeof(PLASMA_enum),   &tB,                           VALUE,
                                     sizeof(int),           &tempmm,                       VALUE,
                                     sizeof(int),           &tempnn,                       VALUE,
                                     sizeof(int),           &tempkn,                       VALUE,
                                     sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(A, m, k),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldam,                         VALUE,
                                     PASSED_BY_REF,     TILE_OF(B, k, n),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldbk,                         VALUE,
                                     sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                     sizeof(int),           &ldcm,                         VALUE,
                                               PARSEC_DTD_ARG_END );
                        }
                    }
                    /*
                     *  A: PlasmaNoTrans / B: Plasma[Conj]Trans
                     */
                    else {
                        ldbn = BLKLDD(&dcB.super, n);
                        for( k = 0; k < dcA.super.nt; k++ ) {
                            tempkn = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;
                            zbeta = k == 0 ? beta : zone;

                            parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm,  0, "Gemm",
                                     sizeof(PLASMA_enum),   &tA,                           VALUE,
                                     sizeof(PLASMA_enum),   &tB,                           VALUE,
                                     sizeof(int),           &tempmm,                       VALUE,
                                     sizeof(int),           &tempnn,                       VALUE,
                                     sizeof(int),           &tempkn,                       VALUE,
                                     sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(A, m, k),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldam,                         VALUE,
                                     PASSED_BY_REF,     TILE_OF(B, n, k),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldbn,                         VALUE,
                                     sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                     sizeof(int),           &ldcm,                         VALUE,
                                               PARSEC_DTD_ARG_END );
                        }
                    }
                }
                /*
                 *  A: Plasma[Conj]Trans / B: PlasmaNoTrans
                 */
                else {
                    if( tB == PlasmaNoTrans ) {
                        for( k = 0; k < dcA.super.mt; k++ ) {
                            ldak = BLKLDD(&dcA.super, k);
                            ldbk = BLKLDD(&dcB.super, k);
                            zbeta = k == 0 ? beta : zone;

                            parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm, 0,  "Gemm",
                                     sizeof(PLASMA_enum),   &tA,                           VALUE,
                                     sizeof(PLASMA_enum),   &tB,                           VALUE,
                                     sizeof(int),           &tempmm,                       VALUE,
                                     sizeof(int),           &tempnn,                       VALUE,
                                     sizeof(int),           &tempkn,                       VALUE,
                                     sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(A, k, m ),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldak,                         VALUE,
                                     PASSED_BY_REF,     TILE_OF(B, k, n),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldbk,                         VALUE,
                                     sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                     sizeof(int),           &ldcm,                         VALUE,
                                               PARSEC_DTD_ARG_END );
                        }
                    }
                    /*
                     *  A: Plasma[Conj]Trans / B: Plasma[Conj]Trans
                     */
                    else {
                        ldbn = BLKLDD(&dcB.super, n);
                        for( k = 0; k < dcA.super.mt; k++ ) {
                            ldak = BLKLDD(&dcA.super, k);
                            zbeta = k == 0 ? beta : zone;

                            parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm, 0,  "Gemm",
                                     sizeof(PLASMA_enum),   &tA,                           VALUE,
                                     sizeof(PLASMA_enum),   &tB,                           VALUE,
                                     sizeof(int),           &tempmm,                       VALUE,
                                     sizeof(int),           &tempnn,                       VALUE,
                                     sizeof(int),           &tempkn,                       VALUE,
                                     sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(A, k, m),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldak,                         VALUE,
                                     PASSED_BY_REF,     TILE_OF(B, n, k),     INPUT | TILE_FULL,
                                     sizeof(int),           &ldbn,                         VALUE,
                                     sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                     PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                     sizeof(int),           &ldcm,                         VALUE,
                                               PARSEC_DTD_ARG_END );
                        }
                    }
                }
            }
        }

        parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );
        parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcB );
        parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcC );

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

        /* Cleaning data arrays we allocated for communication */
        parsec_matrix_del2arena( parsec_dtd_arenas[0] );
        parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

        parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcB );
        parsec_data_free(dcB.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
    } else {
        int Am, An, Bm, Bn;
        PASTE_CODE_ALLOCATE_MATRIX(dcC2, check,
            two_dim_block_cyclic, (&dcC2, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, N, 0, 0,
                                   M, N, SMB, SNB, P));

        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC2, Cseed);

#if defined(PRECISION_z) || defined(PRECISION_c)
        for(tA=0; tA<3; tA++) {
            for(tB=0; tB<3; tB++) {
#else
        for(tA=0; tA<2; tA++) {
            for(tB=0; tB<2; tB++) {
#endif

                /* Getting new parsec handle of dtd type */
                parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

                if ( trans[tA] == PlasmaNoTrans ) {
                    Am = M; An = K;
                } else {
                    Am = K; An = M;
                }
                if ( trans[tB] == PlasmaNoTrans ) {
                    Bm = K; Bn = N;
                } else {
                    Bm = N; Bn = K;
                }

                PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
                    two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                           nodes, rank, MB, NB, LDA, LDA, 0, 0,
                                           Am, An, SMB, SNB, P));

                /* Initializing dc for dtd */
                two_dim_block_cyclic_t *__dcA = &dcA;
                parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);

                PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
                    two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                           nodes, rank, MB, NB, LDB, LDB, 0, 0,
                                           Bm, Bn, SMB, SNB, P));

                /* Initializing dc for dtd */
                two_dim_block_cyclic_t *__dcB = &dcB;
                parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcB);

                /* Allocating data arrays to be used by comm engine */
                /* Default type */
                dplasma_add2arena_tile( parsec_dtd_arenas[TILE_FULL],
                                        dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                                        PARSEC_ARENA_ALIGNMENT_SSE,
                                        parsec_datatype_double_complex_t, dcA.super.mb );

                dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
                dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed);

                if ( rank == 0 ) {
                    printf("***************************************************\n");
                    printf(" ----- TESTING ZGEMM (%s, %s) -------- \n",
                           transstr[tA], transstr[tB]);
                }

                /* matrix generation */
                if(loud) printf("Generate matrices ... ");
                dplasma_zlacpy( parsec, PlasmaUpperLower,
                                (parsec_tiled_matrix_dc_t *)&dcC2, (parsec_tiled_matrix_dc_t *)&dcC );
                if(loud) printf("Done\n");

                /* Create GEMM PaRSEC */
                if(loud) printf("Compute ... ... ");

                int m, n, k;
                int ldam, ldak, ldbn, ldbk, ldcm;
                int tempmm, tempnn, tempkn, tempkm;

                parsec_complex64_t zbeta;
                parsec_complex64_t zone = (parsec_complex64_t)1.0;

                /* Registering the handle with parsec context */
                parsec_context_add_taskpool( parsec, dtd_tp );

                SYNC_TIME_START();

                /* #### parsec context Starting #### */

                /* start parsec context */
                parsec_context_start(parsec);

                for( m = 0; m < dcC.super.mt; m++ ) {
                    tempmm = m == dcC.super.mt-1 ? dcC.super.m-m*dcC.super.mb : dcC.super.mb;
                    ldcm = BLKLDD(&dcC.super, m);

                    for( n = 0; n < dcC.super.nt; n++ ) {
                        tempnn = n == dcC.super.nt-1 ? dcC.super.n-n*dcC.super.nb : dcC.super.nb;
                        /*
                         *  A: PlasmaNoTrans / B: PlasmaNoTrans
                         */
                        if( trans[tA] == PlasmaNoTrans ) {
                            ldam = BLKLDD(&dcA.super, m);

                            if( trans[tB] == PlasmaNoTrans ) {
                                for( k = 0; k < dcA.super.nt; k++ ) {
                                    tempkn = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;
                                    ldbk = BLKLDD(&dcB.super, k);
                                    zbeta = k == 0 ? beta : zone;

                                    parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm, 0,  "Gemm",
                                             sizeof(PLASMA_enum),   &trans[tA],                    VALUE,
                                             sizeof(PLASMA_enum),   &trans[tB],                    VALUE,
                                             sizeof(int),           &tempmm,                       VALUE,
                                             sizeof(int),           &tempnn,                       VALUE,
                                             sizeof(int),           &tempkn,                       VALUE,
                                             sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(A, m, k),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldam,                         VALUE,
                                             PASSED_BY_REF,     TILE_OF(B, k, n),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldbk,                         VALUE,
                                             sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                             sizeof(int),           &ldcm,                         VALUE,
                                                       PARSEC_DTD_ARG_END );
                                }
                            }
                            /*
                             *  A: PlasmaNoTrans / B: Plasma[Conj]Trans
                             */
                            else {
                                ldbn = BLKLDD(&dcB.super, n);

                                for( k = 0; k < dcA.super.nt; k++ ) {
                                    tempkn = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;
                                    zbeta = k == 0 ? beta : zone;

                                    parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm, 0,  "Gemm",
                                             sizeof(PLASMA_enum),   &trans[tA],                    VALUE,
                                             sizeof(PLASMA_enum),   &trans[tB],                    VALUE,
                                             sizeof(int),           &tempmm,                       VALUE,
                                             sizeof(int),           &tempnn,                       VALUE,
                                             sizeof(int),           &tempkn,                       VALUE,
                                             sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(A, m, k),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldam,                         VALUE,
                                             PASSED_BY_REF,     TILE_OF(B, n, k),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldbn,                         VALUE,
                                             sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                             sizeof(int),           &ldcm,                         VALUE,
                                                       PARSEC_DTD_ARG_END );
                                }
                            }
                        }
                        /*
                         *  A: Plasma[Conj]Trans / B: PlasmaNoTrans
                         */
                        else {
                            if( trans[tB] == PlasmaNoTrans ) {
                                for( k = 0; k < dcA.super.mt; k++ ) {
                                    tempkm = k == dcA.super.mt-1 ? dcA.super.m-k*dcA.super.mb : dcA.super.mb;
                                    ldak = BLKLDD(&dcA.super, k);
                                    ldbk = BLKLDD(&dcB.super, k);
                                    zbeta = k == 0 ? beta : zone;

                                    parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm, 0,  "Gemm",
                                             sizeof(PLASMA_enum),   &trans[tA],                    VALUE,
                                             sizeof(PLASMA_enum),   &trans[tB],                    VALUE,
                                             sizeof(int),           &tempmm,                       VALUE,
                                             sizeof(int),           &tempnn,                       VALUE,
                                             sizeof(int),           &tempkm,                       VALUE,
                                             sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(A, k, m),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldak,                         VALUE,
                                             PASSED_BY_REF,     TILE_OF(B, k, n),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldbk,                         VALUE,
                                             sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                             sizeof(int),           &ldcm,                         VALUE,
                                                       PARSEC_DTD_ARG_END );
                                }
                            }
                            /*
                             *  A: Plasma[Conj]Trans / B: Plasma[Conj]Trans
                             */
                            else {
                                ldbn = BLKLDD(&dcB.super, n);

                                for( k = 0; k < dcA.super.mt; k++ ) {
                                    tempkm = k == dcA.super.mt-1 ? dcA.super.m-k*dcA.super.mb : dcA.super.mb;
                                    ldak = BLKLDD(&dcA.super, k);
                                    zbeta = k == 0 ? beta : zone;

                                    parsec_dtd_taskpool_insert_task( dtd_tp,  &parsec_core_gemm, 0,  "Gemm",
                                             sizeof(PLASMA_enum),   &trans[tA],                    VALUE,
                                             sizeof(PLASMA_enum),   &trans[tB],                    VALUE,
                                             sizeof(int),           &tempmm,                       VALUE,
                                             sizeof(int),           &tempnn,                       VALUE,
                                             sizeof(int),           &tempkm,                       VALUE,
                                             sizeof(parsec_complex64_t),           &alpha,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(A, k, m),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldak,                         VALUE,
                                             PASSED_BY_REF,     TILE_OF(B, n, k),     INPUT | TILE_FULL,
                                             sizeof(int),           &ldbn,                         VALUE,
                                             sizeof(parsec_complex64_t),           &zbeta,         VALUE,
                                             PASSED_BY_REF,     TILE_OF(C, m, n),     INOUT | TILE_FULL | AFFINITY,
                                             sizeof(int),           &ldcm,                         VALUE,
                                                       PARSEC_DTD_ARG_END );
                                }
                            }
                        }
                    }
                }

                parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );
                parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcB );
                parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcC );

                /* finishing all the tasks inserted, but not finishing the handle */
                parsec_dtd_taskpool_wait( parsec, dtd_tp );

                /* Waiting on all handle and turning everything off for this context */
                parsec_context_wait( parsec );

                if(loud) printf("Done\n");

                /* #### PaRSEC context is done #### */

                /* Cleaning up the parsec handle */
                parsec_taskpool_free( dtd_tp );

                /* Cleaning data arrays we allocated for communication */
                parsec_matrix_del2arena( parsec_dtd_arenas[0] );
                parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );

                parsec_data_free(dcA.mat);
                parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

                parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcB );

                parsec_data_free(dcB.mat);
                parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);

                /* Check the solution */
                info_solution = check_solution( parsec, (rank == 0) ? loud : 0,
                                                trans[tA], trans[tB],
                                                alpha, Am, An, Aseed,
                                                       Bm, Bn, Bseed,
                                                beta,  M,  N,  Cseed,
                                                &dcC);
                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZGEMM (%s, %s) ...... PASSED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    else {
                        printf(" ---- TESTING ZGEMM (%s, %s) ... FAILED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    printf("***************************************************\n");
                }
            }
        }
#if defined(_UNUSED_)
            }
        }
#endif
        parsec_data_free(dcC2.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC2);
    }

    /* Cleaning data arrays we allocated for communication */
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcC );

    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);

    cleanup_parsec(parsec, iparam);

    return info_solution;
}

/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( parsec_context_t *parsec, int loud,
                           PLASMA_enum transA, PLASMA_enum transB,
                           parsec_complex64_t alpha, int Am, int An, int Aseed,
                                                    int Bm, int Bn, int Bseed,
                           parsec_complex64_t beta,  int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *dcCfinal )
{
    int info_solution = 1;
    double Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int K  = ( transA == PlasmaNoTrans ) ? An : Am ;
    int MB = dcCfinal->super.mb;
    int NB = dcCfinal->super.nb;
    int LDA = Am;
    int LDB = Bm;
    int LDC = M;
    int rank  = dcCfinal->super.super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
        two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDB, Bn, 0, 0,
                               Bm, Bn, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));

    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed );
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed );
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed );

    Anorm        = dplasma_zlange( parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcA );
    Bnorm        = dplasma_zlange( parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcB );
    Cinitnorm    = dplasma_zlange( parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcC );
    Cdplasmanorm = dplasma_zlange( parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)dcCfinal );

    if ( rank == 0 ) {
        cblas_zgemm(CblasColMajor,
                    (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
                    M, N, K,
                    CBLAS_SADDR(alpha), dcA.mat, LDA,
                                        dcB.mat, LDB,
                    CBLAS_SADDR(beta),  dcC.mat, LDC );
    }

    Clapacknorm = dplasma_zlange( parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcC );

    dplasma_zgeadd( parsec, PlasmaNoTrans, -1.0, (parsec_tiled_matrix_dc_t*)dcCfinal,
                                           1.0, (parsec_tiled_matrix_dc_t*)&dcC );

    Rnorm = dplasma_zlange( parsec, PlasmaMaxNorm, (parsec_tiled_matrix_dc_t*)&dcC);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||B||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*B+b*C)||_inf = %e, ||dplasma(a*A*B+b*C)||_inf = %e, ||R||_m = %e\n",
                   Anorm, Bnorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm);
        }

        result = Rnorm / ((Anorm + Bnorm + Cinitnorm) * max(M,N) * eps);
        if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
              isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcB.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);

    return info_solution;
}
