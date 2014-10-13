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
#include "dague/interfaces/superscalar/insert_function.h"

int
call_to_kernel_PO(dague_execution_unit_t * context,
                  dague_execution_context_t * this_task)
{
    const dague_dtd_handle_t *__dague_handle = (dague_dtd_handle_t *) this_task->dague_handle;
    int* INFO = __dague_handle->INFO; /* zpotrf specific; should be removed */
    /* int task_id = this_task->locals[0].value;
    dtd_task_t * current_task = find_task(__dague_handle->task_h_table, task_id, __dague_handle->task_h_size)->task; */
    dtd_task_t *current_task = (dtd_task_t *)this_task;
    task_param_t *current_param = current_task->param_list;
    PLASMA_enum uplo;
    int tempkm, ldak, *iinfo;
    dague_data_copy_t *gT;

    /* Unpacking of parameters */
    uplo = *((int *)current_param->pointer_to_tile);
    current_param = current_param->next;
    tempkm = *((int *)current_param->pointer_to_tile);
    current_param = current_param->next;
    gT = ((dtd_tile_t *) (current_param->pointer_to_tile))->data_copy;
    current_param = current_param->next;
    ldak = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    iinfo = (int *) current_param->pointer_to_tile;
    /* End of upacking */

    void *T = DAGUE_DATA_COPY_GET_PTR(gT);
    (void) T;

    CORE_zpotrf(uplo, tempkm, T, ldak, iinfo);
    if (*iinfo != 0 && *INFO == 0)
        *INFO = 23 + *iinfo;   /* there was k * descA.mb + iinfo; k and descA.mb is not accesible here */

    return DAGUE_HOOK_RETURN_DONE;
}

int
call_to_kernel_TR(dague_execution_unit_t * context,
                  dague_execution_context_t * this_task)
{
    const dague_dtd_handle_t *__dague_handle = (dague_dtd_handle_t *) this_task->dague_handle;
    /* int task_id = this_task->locals[0].value;
    dtd_task_t * current_task = find_task(__dague_handle->task_h_table, task_id, __dague_handle->task_h_size)->task; */
    dtd_task_t *current_task = (dtd_task_t *)this_task;
    task_param_t *current_param = current_task->param_list;
    int i, tempmm, nb, ldak, ldam;
    dague_data_copy_t *gC;
    dague_data_copy_t *gT;

    /* Unpacking of parameters */
    for(i=0;i<4;i++){
        current_param = current_param->next;
    }
    tempmm = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    nb = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    current_param = current_param->next;
    gT = ((dtd_tile_t *) (current_param->pointer_to_tile))->data_copy;
    current_param = current_param->next;
    ldak = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    gC = ((dtd_tile_t *) (current_param->pointer_to_tile))->data_copy;
    current_param = current_param->next;
    ldam = *((int *) current_param->pointer_to_tile);
    /* End of upacking */

    void *T = DAGUE_DATA_COPY_GET_PTR(gT);
    void *C = DAGUE_DATA_COPY_GET_PTR(gC);
    (void) T;
    (void) C;

    CORE_ztrsm(PlasmaRight, PlasmaLower, PlasmaConjTrans, PlasmaNonUnit,
           tempmm, nb, (dague_complex64_t) 1.0, T, ldak,
           C, ldam);

    return DAGUE_HOOK_RETURN_DONE;
}

int
call_to_kernel_HE(dague_execution_unit_t * context,
                  dague_execution_context_t * this_task)
{
    const dague_dtd_handle_t *__dague_handle = (dague_dtd_handle_t *) this_task->dague_handle;
    /* int task_id = this_task->locals[0].value;
    dtd_task_t * current_task = find_task(__dague_handle->task_h_table, task_id, __dague_handle->task_h_size)->task; */
    dtd_task_t *current_task = (dtd_task_t *)this_task;
    task_param_t *current_param = current_task->param_list;
    int i, mb, ldam, tempmm;
    dague_data_copy_t *gT;
    dague_data_copy_t *gA;

    /* Unpacking of parameters */
    for(i=0;i<2;i++){
        current_param = current_param->next;
    }
    tempmm = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    mb = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    current_param = current_param->next;
    gA = ((dtd_tile_t *) (current_param->pointer_to_tile))->data_copy;
    current_param = current_param->next;
    ldam = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    current_param = current_param->next;
    gT = ((dtd_tile_t *) (current_param->pointer_to_tile))->data_copy;
    /* End of upacking */

    void *T = DAGUE_DATA_COPY_GET_PTR(gT);
    void *A = DAGUE_DATA_COPY_GET_PTR(gA);
    (void)A;
    (void)T;

    CORE_zherk(PlasmaLower, PlasmaNoTrans, tempmm, mb, (double) -1.0, A, ldam,
           (double) 1.0, T, ldam);

    return DAGUE_HOOK_RETURN_DONE;
}


int
call_to_kernel_GE(dague_execution_unit_t * context,
                  dague_execution_context_t * this_task)
{
    const dague_dtd_handle_t *__dague_handle = (dague_dtd_handle_t *) this_task->dague_handle;
    /* int task_id = this_task->locals[0].value;
    dtd_task_t * current_task = find_task(__dague_handle->task_h_table, task_id, __dague_handle->task_h_size)->task; */
    dtd_task_t *current_task = (dtd_task_t *)this_task;
    task_param_t *current_param = current_task->param_list;
    int i, tempmm, mb, ldam, ldan;
    dague_data_copy_t *gA;
    dague_data_copy_t *gB;
    dague_data_copy_t *gC;

    /* Unpacking of parameters */
    for(i=0;i<2;i++){
        current_param = current_param->next;
    }
    tempmm = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    mb = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    current_param = current_param->next;
    gA = ((dtd_tile_t *) (current_param->pointer_to_tile))->data_copy;
    current_param = current_param->next;
    ldam = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    gB = ((dtd_tile_t*) (current_param->pointer_to_tile))->data_copy;
    current_param = current_param->next;
    ldan = *((int *) current_param->pointer_to_tile);
    current_param = current_param->next;
    current_param = current_param->next;
    gC = ((dtd_tile_t *) (current_param->pointer_to_tile))->data_copy;
    /* End of upacking */

    void *A = DAGUE_DATA_COPY_GET_PTR(gA);
    void *B = DAGUE_DATA_COPY_GET_PTR(gB);
    void *C = DAGUE_DATA_COPY_GET_PTR(gC);
    (void)A;
    (void)B;
    (void)C;

    CORE_zgemm(PlasmaNoTrans, PlasmaConjTrans,
           tempmm, mb, mb, (dague_complex64_t) - 1.0, A, ldam,
           B, ldan,
           (dague_complex64_t) 1.0, C, ldam);

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
    LDA = max( LDA, N );
    LDB = max( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

        dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_new (4, 1, &info); /* 4 = task_class_count, 1 = arena_count */
    dague_handle_t* DAGUE_zpotrf_dtd = (dague_handle_t *) DAGUE_dtd_handle;

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, random_seed);
    if(loud > 3) printf("Done\n");

    /* load the GPU kernel */
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        if(loud > 3) printf("+++ Load GPU kernel ... ");
        dague_gpu_data_register(dague,
                                (dague_ddesc_t*)&ddescA,
                                MT*NT, MB*NB*sizeof(dague_complex64_t) );
        if(loud > 3) printf("Done\n");
    }
#endif

    total = ddescA.super.mt;
    SYNC_TIME_START();

    /* Testing Insert Function */
    for(k=0;k<total;k++){
        tempkm = (k == (ddescA.super.mt - 1)) ? ddescA.super.m - k * ddescA.super.mb : ddescA.super.mb;
        ldak = BLKLDD(ddescA.super, k);
        insert_task_generic_fptr(DAGUE_dtd_handle, call_to_kernel_PO, "Potrf",
                                 sizeof(int),      &uplo,              VALUE,
                                 sizeof(int),      &tempkm,            VALUE,
                                 PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, A, k, k), INOUT, DEFAULT,
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
                                     PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, A, k, k), INPUT, DEFAULT,
                                     sizeof(int),      &ldak,               VALUE,
                                     PASSED_BY_REF,    TILE_OF(DAGUE_dtd_handle, A, m, k), INOUT, DEFAULT,
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
                                    PASSED_BY_REF,     TILE_OF(DAGUE_dtd_handle, A, m, k), INPUT, DEFAULT,
                                    sizeof(int),       &ldam,               VALUE,
                                    sizeof(int),       &beta,               VALUE,
                                    PASSED_BY_REF,     TILE_OF(DAGUE_dtd_handle, A, m, m), INOUT, DEFAULT,
                                    sizeof(int),       &ldam,               VALUE,
                                    0);
            for(n=k+1;n<m;n++){
                   ldan = BLKLDD(ddescA.super, n);
                   insert_task_generic_fptr(DAGUE_dtd_handle,  &call_to_kernel_GE, "Gemm",
                                           sizeof(int),        &transA_g,           VALUE,
                                           sizeof(int),        &transB,             VALUE,
                                           sizeof(int),        &tempmm,             VALUE,
                                           sizeof(int),        &ddescA.super.mb,    VALUE,
                                           sizeof(int),        &alpha_herk,         VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, m, k), INPUT, DEFAULT,
                                           sizeof(int),        &ldam,               VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, n, k), INPUT, DEFAULT,
                                           sizeof(int),        &ldan,               VALUE,
                                           sizeof(int),        &beta,               VALUE,
                                           PASSED_BY_REF,      TILE_OF(DAGUE_dtd_handle, A, m, n), INOUT, DEFAULT,
                                           sizeof(int),        &ldam,               VALUE,
                                           0);
            }
        }
    }

    SYNC_TIME_START();
    TIME_START();

    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);
    dague_progress(dague);


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
