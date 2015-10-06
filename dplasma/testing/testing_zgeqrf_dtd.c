/*
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dague/interfaces/superscalar/insert_function_internal.h"



int call_to_kernel_GE_QRT(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
    int *m;
    int *n;
    int *ib;
    dague_data_copy_t *A;
    int *lda;
    dague_data_copy_t *T;
    int *ldt;
    dague_complex64_t *TAU;
    dague_complex64_t *WORK;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &T,
                          UNPACK_VALUE, &ldt,
                          UNPACK_SCRATCH, &TAU,
                          UNPACK_SCRATCH, &WORK
                        );


    void *AA = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)A);
    void *TT = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)T);


    CORE_zgeqrt(*m, *n, *ib, AA, *lda, TT, *ldt, TAU, WORK);

    return 0;
}

int
call_to_kernel_UN_MQR(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    PLASMA_enum *side;
    PLASMA_enum *trans;
    int *m;
    int *n;
    int *k;
    int *ib;
    dague_data_copy_t *gA;
    int *lda;
    dague_data_copy_t *gT;
    int *ldt;
    dague_data_copy_t *gC;
    int *ldc;
    dague_complex64_t *WORK;
    int *ldwork;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &side,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &k,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &gA,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &gT,
                          UNPACK_VALUE, &ldt,
                          UNPACK_DATA,  &gC,
                          UNPACK_VALUE, &ldc,
                          UNPACK_SCRATCH, &WORK,
                          UNPACK_VALUE, &ldwork
                        );


    void *A = DAGUE_DATA_COPY_GET_PTR(gA);
    void *T = DAGUE_DATA_COPY_GET_PTR(gT);
    void *C = DAGUE_DATA_COPY_GET_PTR(gC);

    CORE_zunmqr(*side, *trans, *m, *n, *k, *ib,
                A, *lda, T, *ldt, C, *ldc, WORK, *ldwork);

    return 0;
}


int
call_to_kernel_TS_QRT(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    int *m;
    int *n;
    int *ib;
    dague_data_copy_t *gA1;
    int *lda1;
    dague_data_copy_t *gA2;
    int *lda2;
    dague_data_copy_t *gT;
    int *ldt;
    dague_complex64_t *TAU;
    dague_complex64_t *WORK;

    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &gA1,
                          UNPACK_VALUE, &lda1,
                          UNPACK_DATA,  &gA2,
                          UNPACK_VALUE, &lda2,
                          UNPACK_DATA,  &gT,
                          UNPACK_VALUE, &ldt,
                          UNPACK_SCRATCH, &TAU,
                          UNPACK_SCRATCH, &WORK
                        );


    void *A1 = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)gA1);
    void *A2 = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)gA2);
    void *T = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)gT);

    CORE_ztsqrt(*m, *n, *ib, A1, *lda1, A2, *lda2, T, *ldt, TAU, WORK);

    return 0;
}

int
call_to_kernel_TS_MQR(dague_execution_unit_t *context, dague_execution_context_t * this_task)
{
    PLASMA_enum *side;
    PLASMA_enum *trans;
    int *m1;
    int *n1;
    int *m2;
    int *n2;
    int *k;
    int *ib;
    dague_data_copy_t *gA1;
    int *lda1;
    dague_data_copy_t *gA2;
    int *lda2;
    dague_data_copy_t *gV;
    int *ldv;
    dague_data_copy_t *gT;
    int *ldt;
    dague_complex64_t *WORK;
    int *ldwork;


    dague_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &side,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &m1,
                          UNPACK_VALUE, &n1,
                          UNPACK_VALUE, &m2,
                          UNPACK_VALUE, &n2,
                          UNPACK_VALUE, &k,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &gA1,
                          UNPACK_VALUE, &lda1,
                          UNPACK_DATA,  &gA2,
                          UNPACK_VALUE, &lda2,
                          UNPACK_DATA,  &gV,
                          UNPACK_VALUE, &ldv,
                          UNPACK_DATA,  &gT,
                          UNPACK_VALUE, &ldt,
                          UNPACK_SCRATCH, &WORK,
                          UNPACK_VALUE, &ldwork
                        );

    void *A1 = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)gA1);
    void *A2 = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)gA2);
    void *V = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)gV);
    void *T = DAGUE_DATA_COPY_GET_PTR((dague_data_copy_t *)gT);

    CORE_ztsmqr(*side, *trans, *m1, *n1, *m2, *n2, *k, *ib,
                A1, *lda1, A2, *lda2, V, *ldv, T, *ldt, WORK, *ldwork);

    return 0;
}


static int check_orthogonality(dague_context_t *dague, int loud,
                               tiled_matrix_desc_t *Q);
static int check_factorization(dague_context_t *dague, int loud,
                               tiled_matrix_desc_t *Aorig,
                               tiled_matrix_desc_t *A,
                               tiled_matrix_desc_t *Q);
static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 32, 200, 200);
    iparam[IPARAM_SMB] = 4;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGEQRF, ((DagDouble_t)M, (DagDouble_t)N));

    LDA = max(M, LDA);
    LDB = max(M, LDB);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, check,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* Check the solution */
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               M, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               M, NRHS, SMB, SNB, P));

    dague_dtd_init();


    int info = 0;

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    if( check )
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescT);
    if(loud > 3) printf("Done\n");

    two_dim_block_cyclic_t *__ddescA = &ddescA;
    two_dim_block_cyclic_t *__ddescT = &ddescT;


    int k, m, n;
    int ldak, ldam;
    int tempkm, tempkn, tempnn, tempmm;
    int ib = ddescT.super.mb;
    int minMNT = min(ddescA.super.mt, ddescA.super.nt);
    int side = PlasmaLeft;
    int trans = PlasmaConjTrans;

    SYNC_TIME_START();
    dague_dtd_handle_t* DAGUE_dtd_handle = dague_dtd_new (dague, 4); /* 4 = task_class_count, 1 = arena_count */
    dague_handle_t* DAGUE_zgeqrf_dtd = (dague_handle_t *) DAGUE_dtd_handle;
    dague_enqueue(dague, (dague_handle_t*) DAGUE_dtd_handle);
#if defined (OVERLAP)
    dague_context_start(dague);
#endif

    /* Testing Insert Function */
    for (k = 0; k < minMNT; k++) {
        tempkm = k == ddescA.super.mt-1 ? ddescA.super.m-(k*ddescA.super.mb) : ddescA.super.mb;
        tempkn = k == ddescA.super.nt-1 ? ddescA.super.n-(k*ddescA.super.nb) : ddescA.super.nb;
        ldak = BLKLDD(ddescA.super, k);

        //printf("K: %d\n",k);

        insert_task_generic_fptr(DAGUE_dtd_handle,      call_to_kernel_GE_QRT,            "geqrt",
                             sizeof(int),           &tempkm,                           VALUE,
                             sizeof(int),           &tempkn,                           VALUE,
                             sizeof(int),           &ib,                               VALUE,
                             PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, k, k),     INOUT | REGION_FULL,
                             sizeof(int),           &ldak,                             VALUE,
                             PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, T, k, k),     OUTPUT | REGION_FULL,
                             sizeof(int),           &ddescT.super.mb,                  VALUE,
                             sizeof(dague_complex64_t)*ddescT.super.nb,       NULL,    SCRATCH,
                             sizeof(dague_complex64_t)*ib*ddescT.super.nb,    NULL,    SCRATCH,
                             0);

        for (n = k+1; n < ddescA.super.nt; n++) {
            tempnn = n == ddescA.super.nt-1 ? ddescA.super.n-(n*ddescA.super.nb) : ddescA.super.nb;
            //printf("N: %d\n",n);

            insert_task_generic_fptr(DAGUE_dtd_handle,      call_to_kernel_UN_MQR,               "unmqr",
                                 sizeof(PLASMA_enum),   &side,                              VALUE,
                                 sizeof(PLASMA_enum),   &trans,                             VALUE,
                                 sizeof(int),           &tempkm,                            VALUE,
                                 sizeof(int),           &tempnn,                            VALUE,
                                 sizeof(int),           &tempkm,                            VALUE,
                                 sizeof(int),           &ib,                                VALUE,
                                 PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, k, k),      INPUT | REGION_U,
                                 sizeof(int),           &ldak,                              VALUE,
                                 PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, T, k, k),      INPUT | REGION_FULL,
                                 sizeof(int),           &ddescT.super.mb,                   VALUE,
                                 PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, k, n),      INOUT | REGION_FULL,
                                 sizeof(int),           &ldak,                              VALUE,
                                 sizeof(dague_complex64_t)*ib*ddescT.super.nb,   NULL,      SCRATCH,
                                 sizeof(int),           &ddescT.super.nb,                   VALUE,
                                 0);
        }
        for (m = k+1; m < ddescA.super.mt; m++) {
            tempmm = m == ddescA.super.mt-1 ? ddescA.super.m-(m*ddescA.super.mb) : ddescA.super.mb;
            ldam = BLKLDD(ddescA.super, m);
                //printf("M: %d\n",m);

            insert_task_generic_fptr(DAGUE_dtd_handle,      call_to_kernel_TS_QRT,             "tsqrt",
                                 sizeof(PLASMA_enum),   &tempmm,                            VALUE,
                                 sizeof(int),           &tempkn,                            VALUE,
                                 sizeof(int),           &ib,                                VALUE,
                                 PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, k, k),     INOUT | REGION_L,
                                 sizeof(int),           &ldak,                              VALUE,
                                 PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, m, k),     INOUT | REGION_FULL,
                                 sizeof(int),           &ldam,                              VALUE,
                                 PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, T, m, k),     OUTPUT | REGION_FULL,
                                 sizeof(int),           &ddescT.super.mb,                   VALUE,
                                 sizeof(dague_complex64_t)*ddescT.super.nb,       NULL,     SCRATCH,
                                 sizeof(dague_complex64_t)*ib*ddescT.super.nb,    NULL,     SCRATCH,
                                 0);

            for (n = k+1; n < ddescA.super.nt; n++) {
                tempnn = n == ddescA.super.nt-1 ? ddescA.super.n-(n*ddescA.super.nb) : ddescA.super.nb;
                int ldwork = PlasmaLeft == PlasmaLeft ? ib : ddescT.super.nb;
                //printf("N: %d\n",n);

                insert_task_generic_fptr(DAGUE_dtd_handle,      call_to_kernel_TS_MQR,               "tsmqr",
                                     sizeof(PLASMA_enum),   &side,                             VALUE,
                                     sizeof(PLASMA_enum),   &trans,                            VALUE,
                                     sizeof(int),           &ddescA.super.mb,                  VALUE,
                                     sizeof(int),           &tempnn,                           VALUE,
                                     sizeof(int),           &tempmm,                           VALUE,
                                     sizeof(int),           &tempnn,                           VALUE,
                                     sizeof(int),           &ddescA.super.nb,                  VALUE,
                                     sizeof(int),           &ib,                               VALUE,
                                     PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, k, n),     INOUT | REGION_FULL,
                                     sizeof(int),           &ldak,                             VALUE,
                                     PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, m, n),     INOUT | REGION_FULL | LOCALITY,
                                     sizeof(int),           &ldam,                             VALUE,
                                     PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, A, m, k),     INPUT | REGION_FULL,
                                     sizeof(int),           &ldam,                             VALUE,
                                     PASSED_BY_REF,         TILE_OF(DAGUE_dtd_handle, T, m, k),     INPUT | REGION_FULL,
                                     sizeof(int),           &ddescT.super.mb,                  VALUE,
                                     sizeof(dague_complex64_t)*ib*ddescT.super.nb,    NULL,    SCRATCH,
                                     sizeof(int),           &ldwork,                           VALUE,
                                     0);

            }
        }
    }


    increment_task_counter(DAGUE_dtd_handle);
    dague_context_wait(dague);

    #if 0
    /* Create DAGuE */
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgeqrf,
                              ((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescT));

    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(dague, zgeqrf);
    dplasma_zgeqrf_Destruct( DAGUE_zgeqrf );
    #endif

    SYNC_TIME_PRINT(rank, ("\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

    DAGUE_INTERNAL_HANDLE_DESTRUCT(DAGUE_zgeqrf_dtd);

    dague_dtd_fini();

    if( check ) {
        if (M >= N) {
            if(loud > 2) printf("+++ Generate the Q ...");
            dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
            dplasma_zungqr( dague,
                            (tiled_matrix_desc_t *)&ddescA,
                            (tiled_matrix_desc_t *)&ddescT,
                            (tiled_matrix_desc_t *)&ddescQ);
            if(loud > 2) printf("Done\n");

            if(loud > 2) printf("+++ Solve the system ...");
            dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescX, 2354);
            dplasma_zlacpy( dague, PlasmaUpperLower,
                            (tiled_matrix_desc_t *)&ddescX,
                            (tiled_matrix_desc_t *)&ddescB );
            dplasma_zgeqrs( dague,
                            (tiled_matrix_desc_t *)&ddescA,
                            (tiled_matrix_desc_t *)&ddescT,
                            (tiled_matrix_desc_t *)&ddescX );
            if(loud > 2) printf("Done\n");

            /* Check the orthogonality, factorization and the solution */
            ret |= check_orthogonality( dague, (rank == 0) ? loud : 0,
                                        (tiled_matrix_desc_t *)&ddescQ);
            ret |= check_factorization( dague, (rank == 0) ? loud : 0,
                                        (tiled_matrix_desc_t *)&ddescA0,
                                        (tiled_matrix_desc_t *)&ddescA,
                                        (tiled_matrix_desc_t *)&ddescQ );
            ret |= check_solution( dague, (rank == 0) ? loud : 0,
                                   (tiled_matrix_desc_t *)&ddescA0,
                                   (tiled_matrix_desc_t *)&ddescB,
                                   (tiled_matrix_desc_t *)&ddescX );

        } else {
            printf("Check cannot be performed when N > M\n");
        }

        dague_data_free(ddescA0.mat);
        dague_data_free(ddescQ.mat);
        dague_data_free(ddescB.mat);
        dague_data_free(ddescX.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescQ);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX);
    }

    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescT);

    cleanup_dague(dague, iparam);

    return ret;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static int check_orthogonality(dague_context_t *dague, int loud, tiled_matrix_desc_t *Q)
{
    two_dim_block_cyclic_t *twodQ = (two_dim_block_cyclic_t *)Q;
    double normQ = 999999.0;
    double result;
    double eps = LAPACKE_dlamch_work('e');
    int info_ortho;
    int M = Q->m;
    int N = Q->n;
    int minMN = min(M, N);

    PASTE_CODE_ALLOCATE_MATRIX(Id, 1,
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble, matrix_Tile,
                               Q->super.nodes, twodQ->grid.rank,
                               Q->mb, Q->nb, minMN, minMN, 0, 0,
                               minMN, minMN, twodQ->grid.strows, twodQ->grid.stcols, twodQ->grid.rows));

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);

    /* Perform Id - Q'Q */
    if ( M >= N ) {
        dplasma_zherk( dague, PlasmaUpper, PlasmaConjTrans,
                       1.0, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    } else {
        dplasma_zherk( dague, PlasmaUpper, PlasmaNoTrans,
                       1.0, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    }

    normQ = dplasma_zlanhe(dague, PlasmaInfNorm, PlasmaUpper, (tiled_matrix_desc_t*)&Id);

    result = normQ / (minMN * eps);
    if ( loud ) {
        printf("============\n");
        printf("Checking the orthogonality of Q \n");
        printf("||Id-Q'*Q||_oo / (N*eps) = %e \n", result);
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if ( loud ) printf("-- Orthogonality is suspicious ! \n");
        info_ortho=1;
    }
    else {
        if ( loud ) printf("-- Orthogonality is CORRECT ! \n");
        info_ortho=0;
    }

    dague_data_free(Id.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&Id);
    return info_ortho;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static int
check_factorization(dague_context_t *dague, int loud,
                    tiled_matrix_desc_t *Aorig,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *Q)
{
    tiled_matrix_desc_t *subA;
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
    double Anorm, Rnorm;
    double result;
    double eps = LAPACKE_dlamch_work('e');
    int info_factorization;
    int M = A->m;
    int N = A->n;
    int minMN = min(M, N);

    PASTE_CODE_ALLOCATE_MATRIX(Residual, 1,
        two_dim_block_cyclic, (&Residual, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, M, N, 0, 0,
                               M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(R, 1,
        two_dim_block_cyclic, (&R, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    /* Copy the original A in Residual */
    dplasma_zlacpy( dague, PlasmaUpperLower, Aorig, (tiled_matrix_desc_t *)&Residual );

    /* Extract the R */
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&R);

    subA = tiled_matrix_submatrix( A, 0, 0, N, N );
    dplasma_zlacpy( dague, PlasmaUpper, subA, (tiled_matrix_desc_t *)&R );
    free(subA);

    /* Perform Residual = Aorig - Q*R */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans,
                   -1.0, Q, (tiled_matrix_desc_t *)&R,
                    1.0, (tiled_matrix_desc_t *)&Residual);

    /* Free R */
    dague_data_free(R.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&R);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&Residual);
    Anorm = dplasma_zlange(dague, PlasmaInfNorm, Aorig);

    result = Rnorm / ( Anorm * minMN * eps);

    if ( loud ) {
        printf("============\n");
        printf("Checking the QR Factorization \n");
        printf("-- ||A-QR||_oo/(||A||_oo.N.eps) = %e \n", result );
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if ( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else {
        if ( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    dague_data_free(Residual.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&Residual);
    return info_factorization;
}

static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    tiled_matrix_desc_t *subX;
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    double eps = LAPACKE_dlamch_work('e');

    subX = tiled_matrix_submatrix( ddescX, 0, 0, ddescA->n, ddescX->n );

    Anorm = dplasma_zlange(dague, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, subX);

    /* Compute A*x-b */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, 1.0, ddescA, subX, -1.0, ddescB);

    /* Compute A' * ( A*x - b ) */
    dplasma_zgemm( dague, PlasmaConjTrans, PlasmaNoTrans,
                   1.0, ddescA, ddescB, 0., subX );

    Rnorm = dplasma_zlange( dague, PlasmaInfNorm, subX );
    free(subX);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * ddescA->n * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
