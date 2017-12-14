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
#include "parsec/interfaces/superscalar/insert_function_internal.h"

enum regions {
               TILE_FULL,
               TILE_LOWER,
               TILE_UPPER,
               TILE_RECTANGLE,
               L_TILE_RECTANGLE,
             };

int
parsec_core_getrf_incpiv(parsec_execution_stream_t *es, parsec_task_t * this_task)
{
    (void)es;
    int *m;
    int *n;
    int *ib;
    parsec_complex64_t *A;
    int *lda;
    int *IPIV;
    PLASMA_bool *check_info;
    int *info;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &IPIV,
                          UNPACK_VALUE, &check_info,
                          UNPACK_SCRATCH, &info);

    CORE_zgetrf_incpiv(*m, *n, *ib, A, *lda, IPIV, info);
    if (*info != 0 && check_info)
        printf("Getrf_incpiv something is wrong\n");

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_gessm(parsec_execution_stream_t *es, parsec_task_t * this_task)
{
    (void)es;
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
                          UNPACK_VALUE, &lda);

    CORE_zgessm(*m, *n, *k, *ib, IPIV, D, *ldd, A, *lda);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_tstrf(parsec_execution_stream_t *es, parsec_task_t * this_task)
{
    (void)es;
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
    int *info;

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
                          UNPACK_VALUE, &info);

    CORE_ztstrf(*m, *n, *ib, *nb, U, *ldu, A, *lda, L, *ldl, IPIV, WORK, *ldwork, info);

    if (*info != 0 && check_info)
        printf("Gtstrf something is wrong\n");

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_ssssm(parsec_execution_stream_t *es, parsec_task_t * this_task)
{
    (void)es;
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
                          UNPACK_DATA,  &IPIV);

    CORE_zssssm(*m1, *n1, *m2, *n2, *k, *ib, A1, *lda1, A2, *lda2, L1, *ldl1, L2, *ldl2, IPIV);

    return PARSEC_HOOK_RETURN_DONE;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX );

static int check_inverse( parsec_context_t *parsec, int loud,
                          parsec_tiled_matrix_dc_t *dcA,
                          parsec_tiled_matrix_dc_t *dcInvA,
                          parsec_tiled_matrix_dc_t *dcI );

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
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
                               two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    /* Initializing dc for dtd */
    two_dim_block_cyclic_t *__dcA = &dcA;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);

    PASTE_CODE_ALLOCATE_MATRIX(dcL, 1,
                               two_dim_block_cyclic, (&dcL, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));

    /* Initializing dc for dtd */
    two_dim_block_cyclic_t *__dcL = &dcL;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcL);

    PASTE_CODE_ALLOCATE_MATRIX(dcIPIV, 1,
                               two_dim_block_cyclic, (&dcIPIV, matrix_Integer, matrix_Tile,
                                                      nodes, rank, MB, 1, M, NT, 0, 0,
                                                      M, NT, SMB, SNB, P));

    /* Initializing dc for dtd */
    two_dim_block_cyclic_t *__dcIPIV = &dcIPIV;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcIPIV);

    PASTE_CODE_ALLOCATE_MATRIX(dcA0, check,
                               two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    /* Random B check */
    PASTE_CODE_ALLOCATE_MATRIX(dcB, check,
                               two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcX, check,
                               two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    /* Inverse check */
    PASTE_CODE_ALLOCATE_MATRIX(dcInvA, check_inv,
                               two_dim_block_cyclic, (&dcInvA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcI, check_inv,
                               two_dim_block_cyclic, (&dcI, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zpltmg( parsec, matrix_init, (parsec_tiled_matrix_dc_t *)&dcA, random_seed );
    if ( check ) {
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcA,
                        (parsec_tiled_matrix_dc_t *)&dcA0 );
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, random_seed + 1 );
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcB,
                        (parsec_tiled_matrix_dc_t *)&dcX );
    }
    if ( check_inv ) {
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&dcI);
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&dcInvA);
    }
    if(loud > 2) printf("Done\n");

    /* Getting new parsec handle of dtd type */
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    /* Parameters passed on to Insert_task() */
    int k, m, n;
    int ldak, ldam;
    int tempkm, tempkn, tempmm, tempnn;
    int ib = dcL.super.mb;
    int minMNT = min(dcA.super.mt, dcA.super.nt);
    PLASMA_bool check_info;
    int anb, nb, ldl;

    /* Allocating data arrays to be used by comm engine */
    /* A */
    dplasma_add2arena_tile( parsec_dtd_arenas[TILE_FULL],
                            dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, dcA.super.mb );

    /* Lower part of A without diagonal part */
    dplasma_add2arena_lower( parsec_dtd_arenas[TILE_LOWER],
                             dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, dcA.super.mb, 0 );

    /* Upper part of A with diagonal part */
    dplasma_add2arena_upper( parsec_dtd_arenas[TILE_UPPER],
                             dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, dcA.super.mb, 1 );

    /* IPIV */
    dplasma_add2arena_rectangle( parsec_dtd_arenas[TILE_RECTANGLE],
                                 dcA.super.mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, dcA.super.mb, 1, -1 );

    /* L */
    dplasma_add2arena_rectangle( parsec_dtd_arenas[L_TILE_RECTANGLE],
                                 dcL.super.mb*dcL.super.nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, dcL.super.mb, dcL.super.nb, -1);

    /* Registering the handle with parsec context */
    parsec_enqueue( parsec, dtd_tp );

    SYNC_TIME_START();

    /* #### PaRSEC context starting #### */

    /* start parsec context */
    parsec_context_start( parsec );

    /* Testing insert task function */
    for( k = 0; k < minMNT; k++ ) {
        tempkm = k == dcA.super.mt-1 ? (dcA.super.m)-k*(dcA.super.mb) : dcA.super.mb;
        tempkn = k == dcA.super.nt-1 ? (dcA.super.n)-k*(dcA.super.nb) : dcA.super.nb;
        ldak = BLKLDD((parsec_tiled_matrix_dc_t*)&dcA, k);
        check_info = k == dcA.super.mt-1;

        parsec_dtd_taskpool_insert_task( dtd_tp,     parsec_core_getrf_incpiv,             0, "getrf_incpiv",
                           sizeof(int),           &tempkm,                           VALUE,
                           sizeof(int),           &tempkn,                           VALUE,
                           sizeof(int),           &ib,                               VALUE,
                           PASSED_BY_REF,         TILE_OF(A, k, k),     INOUT | TILE_FULL | AFFINITY,
                           sizeof(int),           &ldak,                             VALUE,
                           PASSED_BY_REF,         TILE_OF(IPIV, k, k),  OUTPUT | TILE_RECTANGLE,
                           sizeof(PLASMA_bool),   &check_info,                       VALUE,
                           sizeof(int *),         &info,                             SCRATCH,
                           0 );

        for( n = k+1; n < dcA.super.nt; n++ ) {
            tempnn = n == dcA.super.nt-1 ? (dcA.super.n)-n*(dcA.super.nb) : dcA.super.nb;
            ldl = dcL.super.mb;

            parsec_dtd_taskpool_insert_task( dtd_tp,      parsec_core_gessm,           0,  "gessm",
                               sizeof(int),           &tempkm,                           VALUE,
                               sizeof(int),           &tempnn,                           VALUE,
                               sizeof(int),           &tempkm,                           VALUE,
                               sizeof(int),           &ib,                               VALUE,
                               PASSED_BY_REF,         TILE_OF(IPIV, k, k),    INPUT | TILE_RECTANGLE,
                               PASSED_BY_REF,         TILE_OF(L, k, k),       INPUT | L_TILE_RECTANGLE,
                               sizeof(int),           &ldl,                              VALUE,
                               PASSED_BY_REF,         TILE_OF(A, k, k),       INPUT | TILE_LOWER,
                               sizeof(int),           &ldak,                             VALUE,
                               PASSED_BY_REF,         TILE_OF(A, k, n),       INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),           &ldak,                             VALUE,
                              0 );
        }
        parsec_dtd_data_flush( dtd_tp, TILE_OF(L, k, k) );
        parsec_dtd_data_flush( dtd_tp, TILE_OF(IPIV, k, k) );

        for( m = k+1; m < dcA.super.mt; m++ ) {
            tempmm = m == dcA.super.mt-1 ? (dcA.super.m)-m*(dcA.super.mb) : dcA.super.mb;
            ldam = BLKLDD( (parsec_tiled_matrix_dc_t*)&dcA, m);
            nb = dcL.super.nb;
            ldl = dcL.super.mb;
            check_info = m == dcA.super.mt-1;

            parsec_dtd_taskpool_insert_task( dtd_tp,      parsec_core_tstrf,              0,  "tstrf",
                               sizeof(int),           &tempmm,                           VALUE,
                               sizeof(int),           &tempkn,                           VALUE,
                               sizeof(int),           &ib,                               VALUE,
                               sizeof(int),           &nb,                               VALUE,
                               PASSED_BY_REF,         TILE_OF(A, k, k),     INOUT | TILE_UPPER,
                               sizeof(int),           &ldak,                             VALUE,
                               PASSED_BY_REF,         TILE_OF(A, m, k),     INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),           &ldam,                             VALUE,
                               PASSED_BY_REF,         TILE_OF(L, m, k),     OUTPUT | L_TILE_RECTANGLE,
                               sizeof(int),           &ldl,                              VALUE,
                               PASSED_BY_REF,         TILE_OF(IPIV, m, k),  OUTPUT | TILE_RECTANGLE,
                               sizeof(parsec_complex64_t)*ib*nb,    NULL,                SCRATCH,
                               sizeof(int),           &nb,                               VALUE,
                               sizeof(PLASMA_bool),   &check_info,                       VALUE,
                               sizeof(int *),         &info,                             SCRATCH,
                               0 );

            for( n = k+1; n < dcA.super.nt; n++ ) {
                tempnn = n == dcA.super.nt-1 ? (dcA.super.n)-n*(dcA.super.nb) : dcA.super.nb;
                anb = dcA.super.nb;
                ldl = dcL.super.mb;

                parsec_dtd_taskpool_insert_task( dtd_tp,      parsec_core_ssssm,            0,    "ssssm",
                                   sizeof(int),           &anb,                               VALUE,
                                   sizeof(int),           &tempnn,                            VALUE,
                                   sizeof(int),           &tempmm,                            VALUE,
                                   sizeof(int),           &tempnn,                            VALUE,
                                   sizeof(int),           &anb,                               VALUE,
                                   sizeof(int),           &ib,                                VALUE,
                                   PASSED_BY_REF,         TILE_OF(A, k, n),     INOUT | TILE_FULL,
                                   sizeof(int),           &ldak,                              VALUE,
                                   PASSED_BY_REF,         TILE_OF(A, m, n),     INOUT | TILE_FULL | AFFINITY,
                                   sizeof(int),           &ldam,                              VALUE,
                                   PASSED_BY_REF,         TILE_OF(L, m, k),     INPUT | L_TILE_RECTANGLE,
                                   sizeof(int),           &ldl,                               VALUE,
                                   PASSED_BY_REF,         TILE_OF(A, m, k),     INPUT | TILE_FULL,
                                   sizeof(int),           &ldam,                              VALUE,
                                   PASSED_BY_REF,         TILE_OF(IPIV, m, k),  INPUT | TILE_RECTANGLE,
                                   0 );
            }
            parsec_dtd_data_flush( dtd_tp, TILE_OF(L, m, k) );
            parsec_dtd_data_flush( dtd_tp, TILE_OF(IPIV, m, k) );
        }
        for( n = k+1; n < dcA.super.nt; n++ ) {
            parsec_dtd_data_flush( dtd_tp, TILE_OF(A, k, n));
        }
        for( m = k+1; m < dcA.super.mt; m++ ) {
            parsec_dtd_data_flush( dtd_tp, TILE_OF(A, m, k) );
        }
        parsec_dtd_data_flush( dtd_tp, TILE_OF(A, k, k) );
    }

    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcA );
    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcL );
    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)&dcIPIV );

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

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
        /*
         * First check with a right hand side
         */
        dplasma_zgetrs_incpiv( parsec, PlasmaNoTrans,
                               (parsec_tiled_matrix_dc_t *)&dcA,
                               (parsec_tiled_matrix_dc_t *)&dcL,
                               (parsec_tiled_matrix_dc_t *)&dcIPIV,
                               (parsec_tiled_matrix_dc_t *)&dcX );

        /* Check the solution */
        ret |= check_solution( parsec, (rank == 0) ? loud : 0,
                               (parsec_tiled_matrix_dc_t *)&dcA0,
                               (parsec_tiled_matrix_dc_t *)&dcB,
                               (parsec_tiled_matrix_dc_t *)&dcX);

        /*
         * Second check with inverse
         */
        if ( check_inv ) {
            dplasma_zgetrs_incpiv( parsec, PlasmaNoTrans,
                                   (parsec_tiled_matrix_dc_t *)&dcA,
                                   (parsec_tiled_matrix_dc_t *)&dcL,
                                   (parsec_tiled_matrix_dc_t *)&dcIPIV,
                                   (parsec_tiled_matrix_dc_t *)&dcInvA );

            /* Check the solution */
            ret |= check_inverse(parsec, (rank == 0) ? loud : 0,
                                 (parsec_tiled_matrix_dc_t *)&dcA0,
                                 (parsec_tiled_matrix_dc_t *)&dcInvA,
                                 (parsec_tiled_matrix_dc_t *)&dcI);
        }
    }

    if ( check ) {
        parsec_data_free(dcA0.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0);
        parsec_data_free(dcB.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
        parsec_data_free(dcX.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX);
        if ( check_inv ) {
            parsec_data_free(dcInvA.mat);
            parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcInvA);
            parsec_data_free(dcI.mat);
            parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcI);
        }
    }

    /* Cleaning data arrays we allocated for communication */
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_FULL] );
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_LOWER] );
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_UPPER] );
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_RECTANGLE] );
    parsec_matrix_del2arena( parsec_dtd_arenas[L_TILE_RECTANGLE] );

    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcL );
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcIPIV );

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcL.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcL);
    parsec_data_free(dcIPIV.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcIPIV);

    cleanup_parsec(parsec, iparam);

    return ret;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int m = dcB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, dcA);
    Bnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);
    Xnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcX);

    /* Compute b - A*x */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, dcA, dcX, 1.0, dcB);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);

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
                          parsec_tiled_matrix_dc_t *dcA,
                          parsec_tiled_matrix_dc_t *dcInvA,
                          parsec_tiled_matrix_dc_t *dcI )
{
    int info_solution;
    double Anorm    = 0.0;
    double InvAnorm = 0.0;
    double Rnorm, result;
    int m = dcA->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm    = dplasma_zlange(parsec, PlasmaInfNorm, dcA   );
    InvAnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcInvA);

    /* Compute I - A*A^{-1} */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, dcA, dcInvA, 1.0, dcI);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcI);

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
