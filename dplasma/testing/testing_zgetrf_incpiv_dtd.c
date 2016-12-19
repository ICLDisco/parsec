/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

int
call_to_kernel_GE_TRF_INC(parsec_execution_unit_t *context, parsec_execution_context_t * this_task)
{
    (void)context;
    int *m;
    int *n;
    int *ib;
    parsec_complex64_t *A;
    int *lda;
    int *IPIV;
    PLASMA_bool *check_info;
    int *iinfo;

    int info;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &IPIV,
                          UNPACK_VALUE, &check_info,
                          UNPACK_VALUE, &iinfo
                          );

    CORE_zgetrf_incpiv(*m, *n, *ib, A, *lda, IPIV, &info);
    if (info != 0 && check_info)
        printf("Getrf_incpiv something is wrong\n");

    return 0; /* supposed to return this PARSEC_HOOK_RETURN_DONE; */
}

int
call_to_kernel_GE_SSM(parsec_execution_unit_t *context, parsec_execution_context_t * this_task)
{
    (void)context;
    int *m;
    int *n;
    int *k;
    int *ib;
    int *IPIV;
    parsec_complex64_t *L;
    int *ldl;
    parsec_complex64_t *D;
    int *ldd;
    parsec_complex64_t *A;
    int *lda;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &k,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &IPIV,
                          UNPACK_DATA,  &L,
                          UNPACK_VALUE, &ldl,
                          UNPACK_DATA,  &D,
                          UNPACK_VALUE, &ldd,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda
                          );

    CORE_zgessm(*m, *n, *k, *ib, IPIV, D, *ldd, A, *lda);

    return 0;
}

int
call_to_kernel_TS_TRF(parsec_execution_unit_t *context, parsec_execution_context_t * this_task)
{
    (void)context;
    int *m;
    int *n;
    int *ib;
    int *nb;
    parsec_complex64_t *U;
    int *ldu;
    parsec_complex64_t *A;
    int *lda;
    parsec_complex64_t *L;
    int *ldl;
    int *IPIV;
    parsec_complex64_t *WORK;
    int *ldwork;
    PLASMA_bool *check_info;
    int *iinfo;

    int info;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &ib,
                          UNPACK_VALUE, &nb,
                          UNPACK_DATA,  &U,
                          UNPACK_VALUE, &ldu,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &L,
                          UNPACK_VALUE, &ldl,
                          UNPACK_DATA,  &IPIV,
                          UNPACK_SCRATCH, &WORK,
                          UNPACK_VALUE, &ldwork,
                          UNPACK_VALUE, &check_info,
                          UNPACK_VALUE, &iinfo
                        );

    CORE_ztstrf(*m, *n, *ib, *nb, U, *ldu, A, *lda, L, *ldl, IPIV, WORK, *ldwork, &info);

    if (info != 0 && check_info)
        printf("Gtstrf something is wrong\n");

    return 0;
}

int
call_to_kernel_SS_SSM(parsec_execution_unit_t *context, parsec_execution_context_t * this_task)
{
    (void)context;
    int *m1;
    int *n1;
    int *m2;
    int *n2;
    int *k;
    int *ib;
    parsec_complex64_t *A1;
    int *lda1;
    parsec_complex64_t *A2;
    int *lda2;
    parsec_complex64_t *L1;
    int *ldl1;
    parsec_complex64_t *L2;
    int *ldl2;
    int *IPIV;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m1,
                          UNPACK_VALUE, &n1,
                          UNPACK_VALUE, &m2,
                          UNPACK_VALUE, &n2,
                          UNPACK_VALUE, &k,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &A1,
                          UNPACK_VALUE, &lda1,
                          UNPACK_DATA,  &A2,
                          UNPACK_VALUE, &lda2,
                          UNPACK_DATA,  &L1,
                          UNPACK_VALUE, &ldl1,
                          UNPACK_DATA,  &L2,
                          UNPACK_VALUE, &ldl2,
                          UNPACK_DATA,  &IPIV
                          );

    CORE_zssssm(*m1, *n1, *m2, *n2, *k, *ib, A1, *lda1, A2, *lda2, L1, *ldl1, L2, *ldl2, IPIV);

    return 0;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

static int check_inverse( parsec_context_t *parsec, int loud,
                          tiled_matrix_desc_t *ddescA,
                          tiled_matrix_desc_t *ddescInvA,
                          tiled_matrix_desc_t *ddescI );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_SMB] = 4;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M,(DagDouble_t)N));

    LDA = max(M, LDA);

    if ( M != N && check ) {
        fprintf(stderr, "Check is impossible if M != N\n");
        check = 0;
    }

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                               two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescL, 1,
                               two_dim_block_cyclic, (&ddescL, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
                               two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                                                      nodes, rank, MB, 1, M, NT, 0, 0,
                                                      M, NT, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
                               two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    /* Random B check */
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
                               two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
                               two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    /* Inverse check */
    PASTE_CODE_ALLOCATE_MATRIX(ddescInvA, check_inv,
                               two_dim_block_cyclic, (&ddescInvA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescI, check_inv,
                               two_dim_block_cyclic, (&ddescI, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    parsec_dtd_init();

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zpltmg( parsec, matrix_init, (tiled_matrix_desc_t *)&ddescA, random_seed );
    if ( check ) {
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescA0 );
        dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescB, random_seed + 1 );
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,
                        (tiled_matrix_desc_t *)&ddescX );
    }
    if ( check_inv ) {
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescI);
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescInvA);
    }
    if(loud > 2) printf("Done\n");

    two_dim_block_cyclic_t *__ddescA    = &ddescA;
    two_dim_block_cyclic_t *__ddescL    = &ddescL;
    two_dim_block_cyclic_t *__ddescIPIV = &ddescIPIV;

    int k, m, n;
    int ldak, ldam;
    int tempkm, tempkn, tempmm, tempnn;
    int ib = ddescL.super.mb;
    int minMNT = min(ddescA.super.mt, ddescA.super.nt);
    PLASMA_bool check_info;
    int iinfo;
    int anb, nb, ldl;

    SYNC_TIME_START();

    parsec_dtd_handle_t* PARSEC_dtd_handle = parsec_dtd_handle_new (parsec);
    parsec_handle_t* PARSEC_zgetrf_inc_dtd = (parsec_handle_t *) PARSEC_dtd_handle;

    parsec_enqueue(parsec, (parsec_handle_t*) PARSEC_dtd_handle);
#if defined (OVERLAP)
    parsec_context_start(parsec);
#endif

    /* Testing insert task function */
    for (k = 0; k < minMNT; k++) {
        tempkm = k == ddescA.super.mt-1 ? (ddescA.super.m)-k*(ddescA.super.mb) : ddescA.super.mb;
        tempkn = k == ddescA.super.nt-1 ? (ddescA.super.n)-k*(ddescA.super.nb) : ddescA.super.nb;
        ldak = BLKLDD((tiled_matrix_desc_t*)&ddescA, k);
        check_info = k == ddescA.super.mt-1;
        iinfo = (ddescA.super.nb)*k;

        parsec_insert_task(PARSEC_dtd_handle,      call_to_kernel_GE_TRF_INC,               "getrf_inc",
                             sizeof(int),           &tempkm,                           VALUE,
                             sizeof(int),           &tempkn,                           VALUE,
                             sizeof(int),           &ib,                               VALUE,
                             PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, k, k),     INOUT | REGION_FULL,
                             sizeof(int),           &ldak,                             VALUE,
                             PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, IPIV, k, k),  OUTPUT | REGION_FULL,
                             sizeof(PLASMA_bool),           &check_info,               VALUE,
                             sizeof(int),           &iinfo,                            VALUE,
                             0);

        for (n = k+1; n < ddescA.super.nt; n++) {
            tempnn = n == ddescA.super.nt-1 ? (ddescA.super.n)-n*(ddescA.super.nb) : ddescA.super.nb;
            ldl = ddescL.super.mb;

            parsec_insert_task(PARSEC_dtd_handle,      call_to_kernel_GE_SSM,               "gessm",
                                 sizeof(int),           &tempkm,                           VALUE,
                                 sizeof(int),           &tempnn,                           VALUE,
                                 sizeof(int),           &tempkm,                           VALUE,
                                 sizeof(int),           &ib,                               VALUE,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, IPIV, k, k),    INPUT | REGION_FULL,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, L, k, k),       INPUT | REGION_FULL,
                                 sizeof(int),           &ldl,                              VALUE,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, k, k),       INPUT | REGION_FULL,
                                 sizeof(int),           &ldak,                             VALUE,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, k, n),       INOUT | REGION_FULL,
                                 sizeof(int),           &ldak,                             VALUE,
                                 0);
        }
        for (m = k+1; m < ddescA.super.mt; m++) {
            tempmm = m == ddescA.super.mt-1 ? (ddescA.super.m)-m*(ddescA.super.mb) : ddescA.super.mb;
            ldam = BLKLDD( (tiled_matrix_desc_t*)&ddescA, m);
            nb = ddescL.super.nb;
            ldl = ddescL.super.mb;
            check_info = m == ddescA.super.mt-1;
            iinfo = (ddescA.super.nb)*k;

            parsec_insert_task(PARSEC_dtd_handle,      call_to_kernel_TS_TRF,               "tstrf",
                                 sizeof(int),           &tempmm,                           VALUE,
                                 sizeof(int),           &tempkn,                           VALUE,
                                 sizeof(int),           &ib,                               VALUE,
                                 sizeof(int),           &nb,                               VALUE,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, k, k),     INOUT | REGION_FULL,
                                 sizeof(int),           &ldak,                             VALUE,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, m, k),     INOUT | REGION_FULL,
                                 sizeof(int),           &ldam,                             VALUE,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, L, m, k),     OUTPUT | REGION_FULL,
                                 sizeof(int),           &ldl,                              VALUE,
                                 PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, IPIV, m, k),  OUTPUT | REGION_FULL,
                                 sizeof(parsec_complex64_t)*ib*nb,    NULL,                      SCRATCH,
                                 sizeof(int),           &nb,                               VALUE,
                                 sizeof(PLASMA_bool),           &check_info,               VALUE,
                                 sizeof(int),           &iinfo,                            VALUE,
                                 0);
            for (n = k+1; n < ddescA.super.nt; n++) {
                tempnn = n == ddescA.super.nt-1 ? (ddescA.super.n)-n*(ddescA.super.nb) : ddescA.super.nb;
                anb = ddescA.super.nb;
                ldl = ddescL.super.mb;

                parsec_insert_task(PARSEC_dtd_handle,      call_to_kernel_SS_SSM,               "ssssm",
                                     sizeof(int),           &anb,                               VALUE,
                                     sizeof(int),           &tempnn,                            VALUE,
                                     sizeof(int),           &tempmm,                            VALUE,
                                     sizeof(int),           &tempnn,                            VALUE,
                                     sizeof(int),           &anb,                               VALUE,
                                     sizeof(int),           &ib,                                VALUE,
                                     PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, k, n),     INOUT | REGION_FULL,
                                     sizeof(int),           &ldak,                              VALUE,
                                     PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, m, n),     INOUT | REGION_FULL,
                                     sizeof(int),           &ldam,                              VALUE,
                                     PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, L, m, k),     INPUT | REGION_FULL,
                                     sizeof(int),           &ldl,                               VALUE,
                                     PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, A, m, k),     INPUT | REGION_FULL,
                                     sizeof(int),           &ldam,                              VALUE,
                                     PASSED_BY_REF,         TILE_OF(PARSEC_dtd_handle, IPIV, m, k),  INPUT | REGION_FULL,
                                     0);
            }
        }
    }

    parsec_dtd_handle_wait( parsec, PARSEC_dtd_handle );
    parsec_dtd_context_wait_on_handle( parsec, PARSEC_dtd_handle );

    PARSEC_INTERNAL_HANDLE_DESTRUCT(PARSEC_zgetrf_inc_dtd);

    SYNC_TIME_PRINT(rank, ("\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

    parsec_dtd_fini();

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
        /*
         * First check with a right hand side
         */
        dplasma_zgetrs_incpiv( parsec, PlasmaNoTrans,
                               (tiled_matrix_desc_t *)&ddescA,
                               (tiled_matrix_desc_t *)&ddescL,
                               (tiled_matrix_desc_t *)&ddescIPIV,
                               (tiled_matrix_desc_t *)&ddescX );

        /* Check the solution */
        ret |= check_solution( parsec, (rank == 0) ? loud : 0,
                               (tiled_matrix_desc_t *)&ddescA0,
                               (tiled_matrix_desc_t *)&ddescB,
                               (tiled_matrix_desc_t *)&ddescX);

        /*
         * Second check with inverse
         */
        if ( check_inv ) {
            dplasma_zgetrs_incpiv( parsec, PlasmaNoTrans,
                                   (tiled_matrix_desc_t *)&ddescA,
                                   (tiled_matrix_desc_t *)&ddescL,
                                   (tiled_matrix_desc_t *)&ddescIPIV,
                                   (tiled_matrix_desc_t *)&ddescInvA );

            /* Check the solution */
            ret |= check_inverse(parsec, (rank == 0) ? loud : 0,
                                 (tiled_matrix_desc_t *)&ddescA0,
                                 (tiled_matrix_desc_t *)&ddescInvA,
                                 (tiled_matrix_desc_t *)&ddescI);
        }
    }

    if ( check ) {
        parsec_data_free(ddescA0.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
        parsec_data_free(ddescB.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
        parsec_data_free(ddescX.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX);
        if ( check_inv ) {
            parsec_data_free(ddescInvA.mat);
            tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescInvA);
            parsec_data_free(ddescI.mat);
            tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescI);
        }
    }

    parsec_data_free(ddescA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    parsec_data_free(ddescL.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescL);
    parsec_data_free(ddescIPIV.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescIPIV);

    cleanup_parsec(parsec, iparam);

    return ret;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int m = ddescB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescX);

    /* Compute b - A*x */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescB);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * m * eps ) ;

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

static int check_inverse( parsec_context_t *parsec, int loud,
                          tiled_matrix_desc_t *ddescA,
                          tiled_matrix_desc_t *ddescInvA,
                          tiled_matrix_desc_t *ddescI )
{
    int info_solution;
    double Anorm    = 0.0;
    double InvAnorm = 0.0;
    double Rnorm, result;
    int m = ddescA->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm    = dplasma_zlange(parsec, PlasmaInfNorm, ddescA   );
    InvAnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescInvA);

    /* Compute I - A*A^{-1} */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescInvA, 1.0, ddescI);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescI);

    result = Rnorm / ( ( Anorm * InvAnorm ) * m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||A^{-1}||_oo = %e, ||A A^{-1} - I||_oo = %e\n",
                    Anorm, InvAnorm, Rnorm );
        printf("-- ||AA^{-1}-I||_oo/((||A||_oo||A^{-1}||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
