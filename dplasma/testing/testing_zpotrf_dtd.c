/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
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
#if defined(HAVE_CUDA)
#include "dplasma/cores/cuda_zgemm.h"
#endif
#include "dague/interfaces/superscalar/insert_function_internal.h"

int
call_to_kernel_PO(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    PLASMA_enum *uplo;
    int *tempkm, *ldak, *iinfo;
    dague_data_copy_t *data;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &tempkm,
                          UNPACK_DATA,  &data,
                          UNPACK_VALUE, &ldak,
                          UNPACK_VALUE, &iinfo 
                        );

    void *TT = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)data); 

    CORE_zpotrf(*uplo, *tempkm, TT, *ldak, iinfo);
    
    return DAGUE_HOOK_RETURN_DONE;
}

int
call_to_kernel_TR(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    PLASMA_enum *side, *uplo, *trans, *diag;
    int  *tempmm, *nb, *ldak, *ldam, *alpha;
    dague_data_copy_t *gC;
    dague_data_copy_t *gT;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &side,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &diag,
                          UNPACK_VALUE, &tempmm,
                          UNPACK_VALUE, &nb,
                          UNPACK_VALUE, &alpha,
                          UNPACK_DATA,  &gT,
                          UNPACK_VALUE, &ldak,
                          UNPACK_DATA,  &gC,
                          UNPACK_VALUE, &ldam 
                        );

    void *T = DAGUE_DATA_COPY_GET_PTR(gT);
    void *C = DAGUE_DATA_COPY_GET_PTR(gC);
    (void) T;
    (void) C;

    CORE_ztrsm(*side, *uplo, *trans, *diag,
           *tempmm, *nb, (dague_complex64_t) 1.0, T, *ldak,
           C, *ldam);

    return DAGUE_HOOK_RETURN_DONE;
}

int
call_to_kernel_HE(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    PLASMA_enum *uplo, *trans;
    int *mb, *ldam, *tempmm, *alpha, *beta;
    dague_data_copy_t *gA;
    dague_data_copy_t *gT;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &uplo,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &tempmm,
                          UNPACK_VALUE, &mb,
                          UNPACK_VALUE, &alpha,
                          UNPACK_DATA,  &gA,
                          UNPACK_VALUE, &ldam,
                          UNPACK_VALUE, &beta,
                          UNPACK_DATA,  &gT,
                          UNPACK_VALUE, &ldam 
                        );

    void *A = DAGUE_DATA_COPY_GET_PTR(gA);
    void *T = DAGUE_DATA_COPY_GET_PTR(gT);
    (void)A;
    (void)T;

    CORE_zherk(*uplo, *trans, *tempmm, *mb, (double) -1.0, A, *ldam,
           (double) 1.0, T, *ldam);

    return DAGUE_HOOK_RETURN_DONE;
}


int
call_to_kernel_GE(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    PLASMA_enum *transA, *transB;
    int *tempmm, *mb, *ldam, *ldan;
    double *alpha, *beta;
    dague_data_copy_t *gA;
    dague_data_copy_t *gB;
    dague_data_copy_t *gC;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &transA,
                          UNPACK_VALUE, &transB,
                          UNPACK_VALUE, &tempmm,
                          UNPACK_VALUE, &mb,
                          UNPACK_VALUE, &alpha,
                          UNPACK_DATA,  &gA,
                          UNPACK_VALUE, &ldam,
                          UNPACK_DATA,  &gB,
                          UNPACK_VALUE, &ldan,
                          UNPACK_VALUE, &beta,
                          UNPACK_DATA,  &gC,
                          UNPACK_VALUE, &ldam
                        );

    void *A = DAGUE_DATA_COPY_GET_PTR(gA);
    void *B = DAGUE_DATA_COPY_GET_PTR(gB);
    void *C = DAGUE_DATA_COPY_GET_PTR(gC);

    CORE_zgemm(PlasmaNoTrans, PlasmaConjTrans,
           *tempmm, *mb, *mb, (dague_complex64_t) - 1.0, A, *ldam,
           B, *ldan,
           (dague_complex64_t) 1.0, C, *ldam);

    return DAGUE_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int info = 0;
    int ret = 0;
    int m, k, n, total , task_flags = 0; // loop counter

    /* Parameters passed on to Insert_task() */
    int tempkm, tempmm, ldak, iinfo, ldam, side, transA_p, transA_g, diag, trans, transB, ldan;
    double alpha_trsm, alpha_herk, beta;

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
#if defined(HAVE_CUDA)
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

    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_new (dague, 4, 1, &info); /* 4 = task_class_count, 1 = arena_count */
    dague_handle_t* DAGUE_zpotrf_dtd = (dague_handle_t *) DAGUE_dtd_handle;

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, random_seed);
    if(loud > 3) printf("Done\n");

    sym_two_dim_block_cyclic_t *__ddescA = &ddescA;
    total = ddescA.super.mt;
    SYNC_TIME_START();

    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);  
    //dague_context_start(dague);

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
                                     sizeof(int),      &alpha_trsm,         VALUE,
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
                                    sizeof(int),       &alpha_herk,         VALUE,
                                    PASSED_BY_REF,     TILE_OF(DAGUE_dtd_handle, A, m, k), INPUT | REGION_FULL,
                                    sizeof(int),       &ldam,               VALUE,
                                    sizeof(int),       &beta,               VALUE,
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
                                           sizeof(double),        &alpha_herk,         VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, m, k), INPUT | REGION_FULL,
                                           sizeof(int),        &ldam,               VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, n, k), INPUT | REGION_FULL,
                                           sizeof(int),        &ldan,               VALUE,
                                           sizeof(double),        &beta,               VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, m, n), INOUT | REGION_FULL,
                                           sizeof(int),        &ldam,               VALUE,
                                           0);
            }
        }
    }

    /*SYNC_TIME_START();
    TIME_START();*/

    /*dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle); */
    increment_task_counter(DAGUE_dtd_handle); 
    dague_context_wait(dague);


    if( loud > 3 )
        TIME_PRINT(rank, ("\t%d tasks computed,\t%f task/s rate\n",
                          nb_local_tasks,
                          nb_local_tasks/time_elapsed));
    SYNC_TIME_PRINT(rank, ("\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

    DAGUE_INTERNAL_HANDLE_DESTRUCT(DAGUE_zpotrf_dtd);
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
