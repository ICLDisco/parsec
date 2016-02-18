/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "flops.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"

struct dague_execution_unit_s;

int
call_to_kernel_PO(struct dague_execution_unit_s *context, dague_execution_context_t *this_task)
{
    (void)context;
    PLASMA_enum *uplo;
    int *m, *lda, *info;
    dague_complex64_t *A;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &m,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_VALUE, &info
                        );

    CORE_zpotrf(*uplo, *m, A, *lda, info);

    return DAGUE_HOOK_RETURN_DONE;
}

int
call_to_kernel_TR(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
    (void)context;
    PLASMA_enum *side, *uplo, *trans, *diag;
    int  *m, *n, *lda, *ldc;
    dague_complex64_t *alpha;
    dague_complex64_t *A, *C;

    dague_dtd_unpack_args(this_task,
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
                          UNPACK_VALUE, &ldc
                        );

    CORE_ztrsm(*side, *uplo, *trans, *diag,
               *m, *n, *alpha,
               A, *lda,
               C, *ldc);

    return DAGUE_HOOK_RETURN_DONE;
}

int
call_to_kernel_HE(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
    (void)context;
    PLASMA_enum *uplo, *trans;
    int *m, *n, *lda, *ldc;
    dague_complex64_t *alpha;
    double *beta;
    dague_complex64_t *A;
    dague_complex64_t *C;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &alpha,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_VALUE, &beta,
                          UNPACK_DATA,  &C,
                          UNPACK_VALUE, &ldc
                        );

    CORE_zherk( *uplo, *trans, *m, *n,
                *alpha, A, *lda,
                *beta,  C, *ldc );

    return DAGUE_HOOK_RETURN_DONE;
}


int
call_to_kernel_GE(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
    (void)context;
    PLASMA_enum *transA, *transB;
    int *m, *n, *k, *lda, *ldb, *ldc;
    dague_complex64_t *alpha, *beta;
    dague_complex64_t *A;
    dague_complex64_t *B;
    dague_complex64_t *C;

    dague_dtd_unpack_args(this_task,
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
                          UNPACK_VALUE, &ldc
                        );

    CORE_zgemm(PlasmaNoTrans, PlasmaConjTrans,
               *m, *n, *k,
               *alpha, A, *lda,
                       B, *ldb,
               *beta,  C, *ldc);

    return DAGUE_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int info = 0;
    int ret = 0;
    int m, k, n, total; // loop counter

    /* Parameters passed on to Insert_task() */
    int tempkm, tempmm, ldak, iinfo, ldam, side, transA_p, transA_g, diag, trans, transB, ldan;
    dague_complex64_t alpha_trsm, alpha_herk, beta;

    side = PlasmaRight;
    transA_p = PlasmaConjTrans;
    diag = PlasmaNonUnit;
    alpha_trsm = 1.0;
    trans = PlasmaNoTrans;
    alpha_herk = -1.0;
    beta = 1.0;
    transB = PlasmaConjTrans;
    transB = PlasmaConjTrans;
    transA_g = PlasmaNoTrans;
    iinfo = 0;
    /* End */

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(DAGUE_HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));

    /* initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    LDB = dplasma_imax( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));


    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, random_seed);
    if(loud > 3) printf("Done\n");

    dague_dtd_init();
    SYNC_TIME_START();
    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_handle_new (dague);
    dague_handle_t* DAGUE_zpotrf_dtd = (dague_handle_t *) DAGUE_dtd_handle;

    sym_two_dim_block_cyclic_t *__ddescA = &ddescA;
    total = ddescA.super.mt;
    //SYNC_TIME_START();

    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);

#if defined(OVERLAP)
    dague_context_start(dague);
#endif

    /* Testing Insert Function */
    for(k=0;k<total;k++){
        tempkm = (k == (ddescA.super.mt - 1)) ? ddescA.super.m - k * ddescA.super.mb : ddescA.super.mb;
        ldak = BLKLDD(ddescA.super, k);
        insert_task_generic_fptr(DAGUE_dtd_handle, call_to_kernel_PO, "Potrf",
                                 sizeof(int),      &uplo,              VALUE,
                                 sizeof(int),      &tempkm,            VALUE,
                                 PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, A, k, k), INOUT | REGION_FULL,
                                 sizeof(int),      &ldak,              VALUE,
                                 sizeof(int),      &iinfo,             VALUE,
                                 0);
        for(m=k+1;m<total;m++){
            tempmm = m == ddescA.super.mt - 1 ? ddescA.super.m - m * ddescA.super.mb : ddescA.super.mb;
            ldam = BLKLDD(ddescA.super, m);
            insert_task_generic_fptr(DAGUE_dtd_handle, &call_to_kernel_TR, "Trsm",
                                     sizeof(int),      &side,               VALUE,
                                     sizeof(int),      &uplo,               VALUE,
                                     sizeof(int),      &transA_p,           VALUE,
                                     sizeof(int),      &diag,               VALUE,
                                     sizeof(int),      &tempmm,             VALUE,
                                     sizeof(int),      &ddescA.super.nb,    VALUE,
                                     sizeof(dague_complex64_t),      &alpha_trsm,         VALUE,
                                     PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, A, k, k), INPUT | REGION_FULL,
                                     sizeof(int),      &ldak,               VALUE,
                                     PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, A, m, k), INOUT | REGION_FULL,
                                     sizeof(int),      &ldam,               VALUE,
                                     0);
        }
        for(m=k+1;m<total;m++){
            tempmm = m == ddescA.super.mt - 1 ? ddescA.super.m - m * ddescA.super.mb : ddescA.super.mb;
            ldam = BLKLDD(ddescA.super, m);
            insert_task_generic_fptr(DAGUE_dtd_handle, &call_to_kernel_HE, "Herk",
                                    sizeof(int),       &uplo,               VALUE,
                                    sizeof(int),       &trans,              VALUE,
                                    sizeof(int),       &tempmm,             VALUE,
                                    sizeof(int),       &ddescA.super.mb,    VALUE,
                                    sizeof(dague_complex64_t),       &alpha_herk,         VALUE,
                                    PASSED_BY_REF,     TILE_OF(DAGUE_dtd_handle, A, m, k), INPUT | REGION_FULL,
                                    sizeof(int),       &ldam,               VALUE,
                                    sizeof(double),       &beta,               VALUE,
                                    PASSED_BY_REF,     TILE_OF(DAGUE_dtd_handle, A, m, m), INOUT | REGION_FULL,
                                    sizeof(int),       &ldam,               VALUE,
                                    0);
            for(n=k+1;n<m;n++){
                   ldan = BLKLDD(ddescA.super, n);
                   insert_task_generic_fptr(DAGUE_dtd_handle,  &call_to_kernel_GE, "Gemm",
                                           sizeof(int),        &transA_g,           VALUE,
                                           sizeof(int),        &transB,             VALUE,
                                           sizeof(int),        &tempmm,             VALUE,
                                           sizeof(int),        &ddescA.super.mb,    VALUE,
                                           sizeof(int),        &ddescA.super.mb,    VALUE,
                                           sizeof(dague_complex64_t),        &alpha_herk,         VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, m, k), INPUT | REGION_FULL,
                                           sizeof(int),        &ldam,               VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, n, k), INPUT | REGION_FULL,
                                           sizeof(int),        &ldan,               VALUE,
                                           sizeof(dague_complex64_t),        &beta,               VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, m, n), INOUT | REGION_FULL,
                                           sizeof(int),        &ldam,               VALUE,
                                           0);
            }
        }
    }

    dague_dtd_handle_wait( dague, DAGUE_dtd_handle );
    dague_dtd_context_wait_on_handle( dague, DAGUE_dtd_handle );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(DAGUE_zpotrf_dtd);

    SYNC_TIME_PRINT(rank, ("\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

    dague_dtd_fini();

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
        dplasma_zplghe( dague, (double)(N), uplo,
                        (tiled_matrix_desc_t *)&ddescA0, random_seed);

        ret |= check_zpotrf( dague, (rank == 0) ? loud : 0, uplo,
                             (tiled_matrix_desc_t *)&ddescA,
                             (tiled_matrix_desc_t *)&ddescA0);

        /* Check the solution */
        PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, random_seed+1);

        PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
            two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );

        dplasma_zpotrs(dague, uplo,
                       (tiled_matrix_desc_t *)&ddescA,
                       (tiled_matrix_desc_t *)&ddescX );

        ret |= check_zaxmb( dague, (rank == 0) ? loud : 0, uplo,
                            (tiled_matrix_desc_t *)&ddescA0,
                            (tiled_matrix_desc_t *)&ddescB,
                            (tiled_matrix_desc_t *)&ddescX);

        /* Cleanup */
        dague_data_free(ddescA0.mat); ddescA0.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0 );
        dague_data_free(ddescB.mat); ddescB.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB );
        dague_data_free(ddescX.mat); ddescX.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX );
    }

    dague_data_free(ddescA.mat); ddescA.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);

    cleanup_dague(dague, iparam);
    return ret;
}
