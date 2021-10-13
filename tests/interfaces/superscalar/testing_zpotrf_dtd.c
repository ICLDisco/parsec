/*
 * Copyright (c) 2013-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "flops.h"
#include "dplasma/types.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/superscalar/insert_function.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

enum regions {
                TILE_FULL,
                TILE_BCAST
             };

int
parsec_core_potrf(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int uplo;
    int m, lda, *info;
    dplasma_complex64_t *A;

    parsec_dtd_unpack_args(this_task, &uplo, &m, &A, &lda, &info);

    int rank = this_task->taskpool->context->my_rank;
    fprintf(stderr, "core_potrf executed on rank %d\n", rank);
    CORE_zpotrf(uplo, m, A, lda, info);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_trsm(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int side, uplo, trans, diag;
    int  m, n, lda, ldc;
    dplasma_complex64_t alpha;
    dplasma_complex64_t *A, *C;

    parsec_dtd_unpack_args(this_task, &side, &uplo, &trans, &diag, &m, &n,
                           &alpha, &A, &lda, &C, &ldc);
    int rank = this_task->taskpool->context->my_rank;
    fprintf(stderr, "core_trsm executed on rank %d \n", rank);

    CORE_ztrsm(side, uplo, trans, diag,
               m, n, alpha,
               A, lda,
               C, ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_herk(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int uplo, trans;
    int m, n, lda, ldc;
    dplasma_complex64_t alpha;
    dplasma_complex64_t beta;
    dplasma_complex64_t *A;
    dplasma_complex64_t *C;

    parsec_dtd_unpack_args(this_task, &uplo, &trans, &m, &n, &alpha, &A,
                           &lda, &beta, &C, &ldc);
    //fprintf(stderr, "core_herk executed\n");

    CORE_zherk(uplo, trans, m, n,
               alpha, A, lda,
               beta,  C, ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_gemm(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int transA, transB;
    int m, n, k, lda, ldb, ldc;
    dplasma_complex64_t alpha, beta;
    dplasma_complex64_t *A;
    dplasma_complex64_t *B;
    dplasma_complex64_t *C;

    parsec_dtd_unpack_args(this_task, &transA, &transB, &m, &n, &k, &alpha,
                           &A, &lda, &B, &ldb, &beta, &C, &ldc);
    int rank = this_task->taskpool->context->my_rank;
    fprintf(stderr, "core_gemm executed on rank %d\n", rank);

    CORE_zgemm(transA, transB,
               m, n, k,
               alpha, A, lda,
                      B, ldb,
               beta,  C, ldc);

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int uplo = dplasmaUpper;
    int info = 0;
    int ret = 0;

    //sleep(30);

    int m, n, k, total; /* loop counter */
    /* Parameters passed on to Insert_task() */
    int tempkm, tempmm, ldak, ldam, side, transA_p, transA_g, diag, trans, transB, ldan;
    dplasma_complex64_t alpha_trsm, alpha_herk, beta;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
    iparam[IPARAM_NGPUS] = DPLASMA_ERR_NOT_SUPPORTED;

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));

    /* initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    LDB = dplasma_imax( LDB, N );
    KP = 1;
    KQ = 1;

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        sym_two_dim_block_cyclic, (&dcA, matrix_ComplexDouble,
                                   rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, nodes/P, uplo));
    
    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
        sym_two_dim_block_cyclic, (&dcB, matrix_Integer,
                                   rank, 15, 15, 15*N/NB, 15*N/NB, 0, 0,
                                   15*N/NB, 15*N/NB, P, nodes/P, uplo));

    /* Initializing dc for dtd */
    sym_two_dim_block_cyclic_t *__dcA = &dcA;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);
    
    sym_two_dim_block_cyclic_t *__dcB = &dcB;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcB);

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( parsec, (double)(N), uplo,
                    (parsec_tiled_matrix_dc_t *)&dcA, random_seed);
    if(loud > 3) printf("Done\n");

    /* Getting new parsec handle of dtd type */
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    /* Allocating data arrays to be used by comm engine */
    dplasma_add2arena_tile( &parsec_dtd_arenas_datatypes[TILE_FULL],
                            dcA.super.mb*dcA.super.nb*sizeof(dplasma_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, dcA.super.mb );
    
    dplasma_add2arena_tile( &parsec_dtd_arenas_datatypes[TILE_BCAST],
                            dcB.super.mb*dcB.super.nb*sizeof(int),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_int32_t, dcB.super.mb );

    /* Registering the handle with parsec context */
    parsec_context_add_taskpool( parsec, dtd_tp );

	//sleep(40);
    SYNC_TIME_START();

    /* #### parsec context Starting #### */

    /* start parsec context */
    parsec_context_start( parsec );

    if( dplasmaLower == uplo ) {

        side = dplasmaRight;
        transA_p = dplasmaConjTrans;
        diag = dplasmaNonUnit;
        alpha_trsm = 1.0;
        trans = dplasmaNoTrans;
        alpha_herk = -1.0;
        beta = 1.0;
        transB = dplasmaConjTrans;
        transA_g = dplasmaNoTrans;

        total = dcA.super.mt;
        /* Testing Insert Function */
        for( k = 0; k < total; k++ ) {
            tempkm = (k == (dcA.super.mt - 1)) ? dcA.super.m - k * dcA.super.mb : dcA.super.mb;
            ldak = BLKLDD(&dcA.super, k);

            parsec_dtd_taskpool_insert_task( dtd_tp, parsec_core_potrf,
                              (total - k) * (total-k) * (total - k)/*priority*/, "Potrf",
                               sizeof(int),      &uplo,              PARSEC_VALUE,
                               sizeof(int),      &tempkm,            PARSEC_VALUE,
                               PASSED_BY_REF,    PARSEC_DTD_TILE_OF(A, k, k), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                               sizeof(int),      &ldak,              PARSEC_VALUE,
                               sizeof(int *),    &info,              PARSEC_SCRATCH,
                               PARSEC_DTD_ARG_END );

            for( m = k+1; m < total; m++ ) {
                tempmm = m == dcA.super.mt - 1 ? dcA.super.m - m * dcA.super.mb : dcA.super.mb;
                ldam = BLKLDD(&dcA.super, m);
                parsec_dtd_taskpool_insert_task( dtd_tp, parsec_core_trsm,
                                  (total - m) * (total-m) * (total - m) + 3 * ((2 * total) - k - m - 1) * (m - k)/*priority*/, "Trsm",
                                   sizeof(int),      &side,               PARSEC_VALUE,
                                   sizeof(int),      &uplo,               PARSEC_VALUE,
                                   sizeof(int),      &transA_p,           PARSEC_VALUE,
                                   sizeof(int),      &diag,               PARSEC_VALUE,
                                   sizeof(int),      &tempmm,             PARSEC_VALUE,
                                   sizeof(int),      &dcA.super.nb,    PARSEC_VALUE,
                                   sizeof(dplasma_complex64_t),      &alpha_trsm,         PARSEC_VALUE,
                                   PASSED_BY_REF,    PARSEC_DTD_TILE_OF(A, k, k), PARSEC_INPUT | TILE_FULL,
                                   sizeof(int),      &ldak,               PARSEC_VALUE,
                                   PASSED_BY_REF,    PARSEC_DTD_TILE_OF(A, m, k), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   sizeof(int),      &ldam,               PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END );
            }
            parsec_dtd_data_flush( dtd_tp, PARSEC_DTD_TILE_OF(A, k, k) );

            for( m = k+1; m < dcA.super.nt; m++ ) {
                tempmm = m == dcA.super.mt - 1 ? dcA.super.m - m * dcA.super.mb : dcA.super.mb;
                ldam = BLKLDD(&dcA.super, m);
                parsec_dtd_taskpool_insert_task( dtd_tp, parsec_core_herk,
                                  (total - m) * (total - m) * (total - m) + 3 * (m - k)/*priority*/, "Herk",
                                   sizeof(int),       &uplo,               PARSEC_VALUE,
                                   sizeof(int),       &trans,              PARSEC_VALUE,
                                   sizeof(int),       &tempmm,             PARSEC_VALUE,
                                   sizeof(int),       &dcA.super.mb,    PARSEC_VALUE,
                                   sizeof(dplasma_complex64_t),       &alpha_herk,         PARSEC_VALUE,
                                   PASSED_BY_REF,     PARSEC_DTD_TILE_OF(A, m, k), PARSEC_INPUT | TILE_FULL,
                                   sizeof(int),       &ldam,               PARSEC_VALUE,
                                   sizeof(dplasma_complex64_t),       &beta,               PARSEC_VALUE,
                                   PASSED_BY_REF,     PARSEC_DTD_TILE_OF(A, m, m), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                   sizeof(int),       &ldam,               PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END );

                for( n = m+1; n < total; n++ ) {
                    ldan = BLKLDD(&dcA.super, n);
                    parsec_dtd_taskpool_insert_task( dtd_tp,  parsec_core_gemm,
                                      (total - m) * (total - m) * (total - m) + 3 * ((2 * total) - m - n - 3) * (m - n) + 6 * (m - k) /*priority*/, "Gemm",
                                       sizeof(int),        &transA_g,           PARSEC_VALUE,
                                       sizeof(int),        &transB,             PARSEC_VALUE,
                                       sizeof(int),        &tempmm,             PARSEC_VALUE,
                                       sizeof(int),        &dcA.super.mb,    PARSEC_VALUE,
                                       sizeof(int),        &dcA.super.mb,    PARSEC_VALUE,
                                       sizeof(dplasma_complex64_t),        &alpha_herk,         PARSEC_VALUE,
                                       PASSED_BY_REF,      PARSEC_DTD_TILE_OF(A, n, k), PARSEC_INPUT | TILE_FULL,
                                       sizeof(int),        &ldan,               PARSEC_VALUE,
                                       PASSED_BY_REF,      PARSEC_DTD_TILE_OF(A, m, k), PARSEC_INPUT | TILE_FULL,
                                       sizeof(int),        &ldam,               PARSEC_VALUE,
                                       sizeof(dplasma_complex64_t),        &beta,               PARSEC_VALUE,
                                       PASSED_BY_REF,      PARSEC_DTD_TILE_OF(A, n, m), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                       sizeof(int),        &ldan,               PARSEC_VALUE,
                                       PARSEC_DTD_ARG_END );
                }
                parsec_dtd_data_flush( dtd_tp, PARSEC_DTD_TILE_OF(A, m, k) );
            }
        }
    } else {
        side = dplasmaLeft;
        transA_p = dplasmaConjTrans;
        diag = dplasmaNonUnit;
        alpha_trsm = 1.0;
        trans = dplasmaConjTrans;
        alpha_herk = -1.0;
        beta = 1.0;
        transB = dplasmaNoTrans;
        transA_g = dplasmaConjTrans;

        total = dcA.super.nt;

        /* Variables used for collective */
        int root, num_dest_ranks, dest_rank_idx, flag;
        int *dest_ranks = (int*)malloc((P+Q)*sizeof(int));;
		for( k = 0; k < total; k++ ) {
			tempkm = k == dcA.super.nt-1 ? dcA.super.n-k*dcA.super.nb : dcA.super.nb;
			ldak = BLKLDD(&dcA.super, k);
			if(parsec_dtd_rank_of_data(&dcA.super.super, k, k) == rank) {
				//fprintf(stderr, "Inserting and executing potrf[%d %d] in rank: %d\n", k, k, rank);
				parsec_dtd_taskpool_insert_task( dtd_tp, parsec_core_potrf,
						(total - k) * (total-k) * (total - k)/*priority*/, "Potrf",
						sizeof(int),      &uplo,              PARSEC_VALUE,
						sizeof(int),      &tempkm,            PARSEC_VALUE,
						PASSED_BY_REF,    PARSEC_DTD_TILE_OF(A, k, k), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
						sizeof(int),      &ldak,              PARSEC_VALUE,
						sizeof(int *),    &info,              PARSEC_SCRATCH,
						PARSEC_DTD_ARG_END );
			}

			/*
			 * Broadcast the diagonal tile to the current panel
			 */
			root = parsec_dtd_rank_of_data(&dcA.super.super, k, k);
			num_dest_ranks = Q -1;
			//int *dest_ranks = (int*)malloc(num_dest_ranks*sizeof(int));
			dest_rank_idx = 0;
			flag = 0;
			for(int m = k+1; m < total; m++) {
				int tile_rank = parsec_dtd_rank_of_data(&dcA.super.super, k, m);
				if(tile_rank == root) {flag = 1; continue;}
				dest_ranks[dest_rank_idx] = tile_rank;
				if(tile_rank == rank) flag = 1;
				++dest_rank_idx;
				if(dest_rank_idx == Q-1) break; /* this is to populate the destination ranks */
			}

			if( ( flag || (rank == root) ) && ( dest_rank_idx >= 1) ) {
                parsec_dtd_tile_t* dtd_tile_root = PARSEC_DTD_TILE_OF(A, k, k);
                parsec_dtd_tile_t* dtd_key_root = PARSEC_DTD_TILE_OF(B, k, k);
				//fprintf(stderr, "Broadcasting PO tile to TRSM. k %d, rank %d, root %d\n", k, rank, root);
				parsec_dtd_broadcast(
						dtd_tp, rank, root,
						dtd_tile_root, TILE_FULL,
						dtd_key_root, TILE_BCAST,
						dest_ranks, dest_rank_idx);
			}

			for( m = k+1; m < total; m++ ) {
				tempmm = m == dcA.super.nt-1 ? dcA.super.n-m*dcA.super.nb : dcA.super.nb;
				//if(  (parsec_dtd_rank_of_data(&dcA.super.super, k, m) == rank || parsec_dtd_rank_of_data(&dcA.super.super, k, k) == rank )) {
				if(  (parsec_dtd_rank_of_data(&dcA.super.super, k, m) == rank )) {
					//fprintf(stderr, "Inserting trsm[%d %d][%d %d] in rank: %d owned: %d\n", k, k, k, m, rank, parsec_dtd_rank_of_data(&dcA.super.super, k, m));
					parsec_dtd_taskpool_insert_task( dtd_tp, parsec_core_trsm,
							(total - m) * (total-m) * (total - m) + 3 * ((2 * total) - k - m - 1) * (m - k)/*priority*/, "Trsm",
							sizeof(int),      &side,               PARSEC_VALUE,
							sizeof(int),      &uplo,               PARSEC_VALUE,
							sizeof(int),      &transA_p,           PARSEC_VALUE,
							sizeof(int),      &diag,               PARSEC_VALUE,
							sizeof(int),      &dcA.super.nb,    PARSEC_VALUE,
							sizeof(int),      &tempmm,             PARSEC_VALUE,
							sizeof(dplasma_complex64_t),      &alpha_trsm,         PARSEC_VALUE,
							PASSED_BY_REF,    PARSEC_DTD_TILE_OF(A, k, k), PARSEC_INPUT | TILE_FULL,
							sizeof(int),      &ldak,               PARSEC_VALUE,
							PASSED_BY_REF,    PARSEC_DTD_TILE_OF(A, k, m), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
							sizeof(int),      &ldak,               PARSEC_VALUE,
							PARSEC_DTD_ARG_END );
				}
                
                /*
                 * Broadcast the TRSM tile to the descendant SYRK/GEMM tasks
                 */
                root = parsec_dtd_rank_of_data(&dcA.super.super, k, m);
                num_dest_ranks = P+Q -1;
                //int *dest_ranks = (int*)malloc(num_dest_ranks*sizeof(int));
                dest_rank_idx = 0;
                flag = 0;
                /* Loop over P and Q processes to gather the broadcast destinations */
                for(int i = k+1; i <= m; i++) {
                    int tile_rank = parsec_dtd_rank_of_data(&dcA.super.super, i, m);
                    if(tile_rank == root) {break;} /* we have loop over all the ranks in the column */
                    dest_ranks[dest_rank_idx] = tile_rank;
                    if(tile_rank == rank) flag = 1; /* flip the flag for the owner of the tile */
                    ++dest_rank_idx;
                }
                int diag_rank = parsec_dtd_rank_of_data(&dcA.super.super, m, m);
                for(int j = m+1; j < total; j++) {
                    int tile_rank = parsec_dtd_rank_of_data(&dcA.super.super, m, j);
                    if(tile_rank == diag_rank) {break;} /* we have loop over all the ranks in the column */
                    dest_ranks[dest_rank_idx] = tile_rank;
                    if(tile_rank == rank) flag = 1; /* flip the flag for the owner of the tile */
                    ++dest_rank_idx;
                }

                if( ( flag || (rank == root) ) && ( dest_rank_idx >= 1) ) {
                    parsec_dtd_tile_t* dtd_tile_root = PARSEC_DTD_TILE_OF(A, k, m);
                    parsec_dtd_tile_t* dtd_key_root = PARSEC_DTD_TILE_OF(B, k, m);
                    //fprintf(stderr, "Broadcasting TRSM tile to SYRK and GEMM. k %d, m %d, rank %d, root %d\n", k, m, rank, root);
                    parsec_dtd_broadcast(
                            dtd_tp, rank, root,
                            dtd_tile_root, TILE_FULL,
                            dtd_key_root, TILE_BCAST,
                            dest_ranks, dest_rank_idx);
                }
            }
            //parsec_dtd_data_flush( dtd_tp, PARSEC_DTD_TILE_OF(A, k, k) );

            for( m = k+1; m < dcA.super.mt; m++ ) {
                tempmm = m == dcA.super.nt-1 ? dcA.super.n-m*dcA.super.nb : dcA.super.nb;
                ldam = BLKLDD(&dcA.super, m);
                //if(parsec_dtd_rank_of_data(&dcA.super.super, m, m) == rank || parsec_dtd_rank_of_data(&dcA.super.super, k, m) == rank ) {
                if(parsec_dtd_rank_of_data(&dcA.super.super, m, m) == rank ) {
                    parsec_dtd_taskpool_insert_task( dtd_tp, parsec_core_herk,
                            (total - m) * (total - m) * (total - m) + 3 * (m - k)/*priority*/, "Herk",
                            sizeof(int),       &uplo,               PARSEC_VALUE,
                            sizeof(int),       &trans,              PARSEC_VALUE,
                            sizeof(int),       &tempmm,             PARSEC_VALUE,
                            sizeof(int),       &dcA.super.mb,    PARSEC_VALUE,
                            sizeof(dplasma_complex64_t),       &alpha_herk,         PARSEC_VALUE,
                            PASSED_BY_REF,     PARSEC_DTD_TILE_OF(A, k, m), PARSEC_INPUT | TILE_FULL,
                            sizeof(int),       &ldak,               PARSEC_VALUE,
                            sizeof(dplasma_complex64_t),    &beta,                  PARSEC_VALUE,
                            PASSED_BY_REF,     PARSEC_DTD_TILE_OF(A, m, m), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                            sizeof(int),       &ldam,               PARSEC_VALUE,
                            PARSEC_DTD_ARG_END );
                }
                for( n = m+1; n < total; n++ ) {
                    ldan = BLKLDD(&dcA.super, n);
                    //if(parsec_dtd_rank_of_data(&dcA.super.super, m, n) == rank || parsec_dtd_rank_of_data(&dcA.super.super, k, m) == rank || parsec_dtd_rank_of_data(&dcA.super.super, k, n) == rank) {
                    if(parsec_dtd_rank_of_data(&dcA.super.super, m, n) == rank ) {
                        parsec_dtd_taskpool_insert_task( dtd_tp,  parsec_core_gemm,
                                (total - m) * (total - m) * (total - m) + 3 * ((2 * total) - m - n - 3) * (m - n) + 6 * (m - k) /*priority*/, "Gemm",
                                sizeof(int),        &transA_g,           PARSEC_VALUE,
                                sizeof(int),        &transB,             PARSEC_VALUE,
                                sizeof(int),        &dcA.super.mb,    PARSEC_VALUE,
                                sizeof(int),        &tempmm,             PARSEC_VALUE,
                                sizeof(int),        &dcA.super.mb,    PARSEC_VALUE,
                                sizeof(dplasma_complex64_t),        &alpha_herk,         PARSEC_VALUE,
                                PASSED_BY_REF,      PARSEC_DTD_TILE_OF(A, k, m), PARSEC_INPUT | TILE_FULL,
                                sizeof(int),        &ldak,               PARSEC_VALUE,
                                PASSED_BY_REF,      PARSEC_DTD_TILE_OF(A, k, n), PARSEC_INPUT | TILE_FULL,
                                sizeof(int),        &ldak,               PARSEC_VALUE,
                                sizeof(dplasma_complex64_t),        &beta,               PARSEC_VALUE,
                                PASSED_BY_REF,      PARSEC_DTD_TILE_OF(A, m, n), PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                sizeof(int),        &ldan,               PARSEC_VALUE,
                                PARSEC_DTD_ARG_END );
                    }
                }
                parsec_dtd_data_flush( dtd_tp, PARSEC_DTD_TILE_OF(A, k, m) );
            }
		}
	}

    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_taskpool_wait( dtd_tp );

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
                                       rank, MB, NB, LDA, N, 0, 0,
                                       N, N, P, nodes/P, uplo));
        dplasma_zplghe( parsec, (double)(N), uplo,
                        (parsec_tiled_matrix_dc_t *)&dcA0, random_seed);

        ret |= check_zpotrf( parsec, (rank == 0) ? loud : 0, uplo,
                             (parsec_tiled_matrix_dc_t *)&dcA,
                             (parsec_tiled_matrix_dc_t *)&dcA0);

        /* Check the solution */
        PASTE_CODE_ALLOCATE_MATRIX(dcB, check,
            two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                   rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, P, nodes/P, KP, KQ, IP, JQ));
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, random_seed+1);

        PASTE_CODE_ALLOCATE_MATRIX(dcX, check,
            two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                                   rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, P, nodes/P, KP, KQ, IP, JQ));
        dplasma_zlacpy( parsec, dplasmaUpperLower,
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
    dplasma_matrix_del2arena( &parsec_dtd_arenas_datatypes[TILE_FULL] );
    dplasma_matrix_del2arena( &parsec_dtd_arenas_datatypes[TILE_BCAST] );
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcB );

    parsec_data_free(dcA.mat); dcA.mat = NULL;
    parsec_data_free(dcB.mat); dcB.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);

    cleanup_parsec(parsec, iparam);
    return ret;
}
