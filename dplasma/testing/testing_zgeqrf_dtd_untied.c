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
             };

int
parsec_core_geqrt(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int *m;
    int *n;
    int *ib;
    parsec_complex64_t *A;
    int *lda;
    parsec_complex64_t *T;
    int *ldt;
    parsec_complex64_t *TAU;
    parsec_complex64_t *WORK;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &T,
                          UNPACK_VALUE, &ldt,
                          UNPACK_SCRATCH, &TAU,
                          UNPACK_SCRATCH, &WORK);

    CORE_zgeqrt(*m, *n, *ib, A, *lda, T, *ldt, TAU, WORK);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_unmqr(parsec_execution_stream_t *es, parsec_task_t * this_task)
{
    (void)es;
    PLASMA_enum *side;
    PLASMA_enum *trans;
    int *m;
    int *n;
    int *k;
    int *ib;
    parsec_complex64_t *A;
    int *lda;
    parsec_complex64_t *T;
    int *ldt;
    parsec_complex64_t *C;
    int *ldc;
    parsec_complex64_t *WORK;
    int *ldwork;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &side,
                          UNPACK_VALUE, &trans,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &k,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &A,
                          UNPACK_VALUE, &lda,
                          UNPACK_DATA,  &T,
                          UNPACK_VALUE, &ldt,
                          UNPACK_DATA,  &C,
                          UNPACK_VALUE, &ldc,
                          UNPACK_SCRATCH, &WORK,
                          UNPACK_VALUE, &ldwork);

    CORE_zunmqr(*side, *trans, *m, *n, *k, *ib,
                A, *lda, T, *ldt, C, *ldc, WORK, *ldwork);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_tsqrt(parsec_execution_stream_t *es, parsec_task_t * this_task)
{
    (void)es;
    int *m;
    int *n;
    int *ib;
    parsec_complex64_t *A1;
    int *lda1;
    parsec_complex64_t *A2;
    int *lda2;
    parsec_complex64_t *T;
    int *ldt;
    parsec_complex64_t *TAU;
    parsec_complex64_t *WORK;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &m,
                          UNPACK_VALUE, &n,
                          UNPACK_VALUE, &ib,
                          UNPACK_DATA,  &A1,
                          UNPACK_VALUE, &lda1,
                          UNPACK_DATA,  &A2,
                          UNPACK_VALUE, &lda2,
                          UNPACK_DATA,  &T,
                          UNPACK_VALUE, &ldt,
                          UNPACK_SCRATCH, &TAU,
                          UNPACK_SCRATCH, &WORK);

    CORE_ztsqrt(*m, *n, *ib, A1, *lda1, A2, *lda2, T, *ldt, TAU, WORK);

    return PARSEC_HOOK_RETURN_DONE;
}

int
parsec_core_tsmqr(parsec_execution_stream_t *es, parsec_task_t * this_task)
{
    (void)es;
    PLASMA_enum *side;
    PLASMA_enum *trans;
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
    parsec_complex64_t *V;
    int *ldv;
    parsec_complex64_t *T;
    int *ldt;
    parsec_complex64_t *WORK;
    int *ldwork;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE, &side,
                          UNPACK_VALUE, &trans,
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
                          UNPACK_DATA,  &V,
                          UNPACK_VALUE, &ldv,
                          UNPACK_DATA,  &T,
                          UNPACK_VALUE, &ldt,
                          UNPACK_SCRATCH, &WORK,
                          UNPACK_VALUE, &ldwork);

    CORE_ztsmqr(*side, *trans, *m1, *n1, *m2, *n2, *k, *ib,
                A1, *lda1, A2, *lda2, V, *ldv, T, *ldt, WORK, *ldwork);

    return PARSEC_HOOK_RETURN_DONE;
}

int
insert_task_geqrf(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    two_dim_block_cyclic_t *dcA;
    two_dim_block_cyclic_t *dcT;
    /* Parameters passed on to Insert_task() */
    int k, m, n;
    int ldak, ldam;
    int tempkm, tempkn, tempnn, tempmm;
    int side = PlasmaLeft;
    int trans = PlasmaConjTrans;

    int *total, *iteration, count = 0;

    parsec_taskpool_t *dtd_tp = (parsec_taskpool_t *)this_task->taskpool;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_VALUE,   &total,
                          UNPACK_VALUE,   &iteration,
                          UNPACK_SCRATCH, &dcA,
                          UNPACK_SCRATCH, &dcT);


    two_dim_block_cyclic_t *__dcA = dcA;
    two_dim_block_cyclic_t *__dcT = dcT;

    int ib = dcT->super.mb;

    for( k = *iteration; k < *total; k++, *iteration += 1, count++ ) {
        if( count > parsec_dtd_window_size-1000 ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        }

        tempkm = k == dcA->super.mt-1 ? dcA->super.m-(k*dcA->super.mb) : dcA->super.mb;
        tempkn = k == dcA->super.nt-1 ? dcA->super.n-(k*dcA->super.nb) : dcA->super.nb;
        ldak = BLKLDD( (parsec_tiled_matrix_dc_t*)dcA, k);

        parsec_dtd_taskpool_insert_task( dtd_tp,      parsec_core_geqrt,
                          (dcA->super.nt-k)*(dcA->super.nt-k)*(dcA->super.nt-k), "geqrt",
                           sizeof(int),           &tempkm,                           VALUE,
                           sizeof(int),           &tempkn,                           VALUE,
                           sizeof(int),           &ib,                               VALUE,
                           PASSED_BY_REF,         TILE_OF(A, k, k),     INOUT | TILE_FULL | AFFINITY,
                           sizeof(int),           &ldak,                             VALUE,
                           PASSED_BY_REF,         TILE_OF(T, k, k),     OUTPUT | TILE_RECTANGLE,
                           sizeof(int),           &dcT->super.mb,                  VALUE,
                           sizeof(parsec_complex64_t)*dcT->super.nb,       NULL,   SCRATCH,
                           sizeof(parsec_complex64_t)*ib*dcT->super.nb,    NULL,   SCRATCH,
                           0 );

        for( n = k+1; n < dcA->super.nt; n++, count++ ) {
            tempnn = n == dcA->super.nt-1 ? dcA->super.n-(n*dcA->super.nb) : dcA->super.nb;

            parsec_dtd_taskpool_insert_task( dtd_tp,      parsec_core_unmqr,          0,    "unmqr",
                               sizeof(PLASMA_enum),   &side,                              VALUE,
                               sizeof(PLASMA_enum),   &trans,                             VALUE,
                               sizeof(int),           &tempkm,                            VALUE,
                               sizeof(int),           &tempnn,                            VALUE,
                               sizeof(int),           &tempkm,                            VALUE,
                               sizeof(int),           &ib,                                VALUE,
                               PASSED_BY_REF,         TILE_OF(A, k, k),      INPUT | TILE_LOWER,
                               sizeof(int),           &ldak,                              VALUE,
                               PASSED_BY_REF,         TILE_OF(T, k, k),      INPUT | TILE_RECTANGLE,
                               sizeof(int),           &dcT->super.mb,                   VALUE,
                               PASSED_BY_REF,         TILE_OF(A, k, n),      INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),           &ldak,                              VALUE,
                               sizeof(parsec_complex64_t)*ib*dcT->super.nb,   NULL,     SCRATCH,
                               sizeof(int),           &dcT->super.nb,                   VALUE,
                               0 );
        }
        parsec_dtd_data_flush( dtd_tp, TILE_OF(T, k, k) );

        for( m = k+1; m < dcA->super.mt; m++, count++ ) {
            tempmm = m == dcA->super.mt-1 ? dcA->super.m-(m*dcA->super.mb) : dcA->super.mb;
            ldam = BLKLDD( (parsec_tiled_matrix_dc_t*)dcA, m);

            parsec_dtd_taskpool_insert_task( dtd_tp,      parsec_core_tsqrt,
                              (dcA->super.mt-k)*(dcA->super.mt-k)*(dcA->super.mt-k),  "tsqrt",
                               sizeof(PLASMA_enum),   &tempmm,                            VALUE,
                               sizeof(int),           &tempkn,                            VALUE,
                               sizeof(int),           &ib,                                VALUE,
                               PASSED_BY_REF,         TILE_OF(A, k, k),     INOUT | TILE_UPPER,
                               sizeof(int),           &ldak,                              VALUE,
                               PASSED_BY_REF,         TILE_OF(A, m, k),     INOUT | TILE_FULL | AFFINITY,
                               sizeof(int),           &ldam,                              VALUE,
                               PASSED_BY_REF,         TILE_OF(T, m, k),     OUTPUT | TILE_RECTANGLE,
                               sizeof(int),           &dcT->super.mb,                   VALUE,
                               sizeof(parsec_complex64_t)*dcT->super.nb,       NULL,    SCRATCH,
                               sizeof(parsec_complex64_t)*ib*dcT->super.nb,    NULL,    SCRATCH,
                               0 );

            for( n = k+1; n < dcA->super.nt; n++, count++ ) {
                tempnn = n == dcA->super.nt-1 ? dcA->super.n-(n*dcA->super.nb) : dcA->super.nb;
                int ldwork = PlasmaLeft == PlasmaLeft ? ib : dcT->super.nb;

                parsec_dtd_taskpool_insert_task( dtd_tp,      parsec_core_tsmqr,
                                  (dcA->super.mt-k)*(dcA->super.mt-n)*(dcA->super.mt-n),        "tsmqr",
                                   sizeof(PLASMA_enum),   &side,                             VALUE,
                                   sizeof(PLASMA_enum),   &trans,                            VALUE,
                                   sizeof(int),           &dcA->super.mb,                  VALUE,
                                   sizeof(int),           &tempnn,                           VALUE,
                                   sizeof(int),           &tempmm,                           VALUE,
                                   sizeof(int),           &tempnn,                           VALUE,
                                   sizeof(int),           &dcA->super.nb,                  VALUE,
                                   sizeof(int),           &ib,                               VALUE,
                                   PASSED_BY_REF,         TILE_OF(A, k, n),     INOUT | TILE_FULL,
                                   sizeof(int),           &ldak,                             VALUE,
                                   PASSED_BY_REF,         TILE_OF(A, m, n),     INOUT | TILE_FULL | AFFINITY,
                                   sizeof(int),           &ldam,                             VALUE,
                                   PASSED_BY_REF,         TILE_OF(A, m, k),     INPUT | TILE_FULL,
                                   sizeof(int),           &ldam,                             VALUE,
                                   PASSED_BY_REF,         TILE_OF(T, m, k),     INPUT | TILE_RECTANGLE,
                                   sizeof(int),           &dcT->super.mb,                  VALUE,
                                   sizeof(parsec_complex64_t)*ib*dcT->super.nb,    NULL,   SCRATCH,
                                   sizeof(int),           &ldwork,                           VALUE,
                                   0 );
            }
            parsec_dtd_data_flush( dtd_tp, TILE_OF(T, m, k) );
        }
        for( n = k+1; n < dcA->super.nt; n++, count++ ) {
            parsec_dtd_data_flush( dtd_tp, TILE_OF(A, k, n) );
        }
        for( m = k+1; m < dcA->super.mt; m++, count++ ) {
            parsec_dtd_data_flush( dtd_tp, TILE_OF(A, m, k) );
        }
        parsec_dtd_data_flush( dtd_tp, TILE_OF(A, k, k) );
    }

    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)dcA );
    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)dcT );

    return PARSEC_HOOK_RETURN_DONE;
}

static int check_orthogonality(parsec_context_t *parsec, int loud,
                               parsec_tiled_matrix_dc_t *Q);
static int check_factorization(parsec_context_t *parsec, int loud,
                               parsec_tiled_matrix_dc_t *Aorig,
                               parsec_tiled_matrix_dc_t *A,
                               parsec_tiled_matrix_dc_t *Q);
static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 32, 200, 200);
    iparam[IPARAM_SMB] = 4;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGEQRF, ((DagDouble_t)M, (DagDouble_t)N));

    LDA = max(M, LDA);
    LDB = max(M, LDB);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* Initializing dc for dtd */
    //two_dim_block_cyclic_t *__dcA = &dcA;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcA);

    PASTE_CODE_ALLOCATE_MATRIX(dcT, 1,
        two_dim_block_cyclic, (&dcT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));

    /* Initializing dc for dtd */
    //two_dim_block_cyclic_t *__dcT = &dcT;
    parsec_dtd_data_collection_init((parsec_data_collection_t *)&dcT);

    PASTE_CODE_ALLOCATE_MATRIX(dcA0, check,
        two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcQ, check,
        two_dim_block_cyclic, (&dcQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* Check the solution */
    PASTE_CODE_ALLOCATE_MATRIX(dcB, check,
        two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               M, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcX, check,
        two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               M, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, 3872);
    if( check )
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcA, (parsec_tiled_matrix_dc_t *)&dcA0 );
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (parsec_tiled_matrix_dc_t *)&dcT);
    if(loud > 3) printf("Done\n");

    /* Getting new parsec handle of dtd type */
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    int minMNT = min(dcA.super.mt, dcA.super.nt);

    /* Allocating data arrays to be used by comm engine */
    /* Default type */
    dplasma_add2arena_tile( parsec_dtd_arenas[TILE_FULL],
                            dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, dcA.super.mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( parsec_dtd_arenas[TILE_LOWER],
                             dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, dcA.super.mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( parsec_dtd_arenas[TILE_UPPER],
                             dcA.super.mb*dcA.super.nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, dcA.super.mb, 1 );

    dplasma_add2arena_rectangle( parsec_dtd_arenas[TILE_RECTANGLE],
                                 dcT.super.mb*dcT.super.nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, dcT.super.mb, dcT.super.nb, -1);

    /* Registering the handle with parsec context */
    parsec_enqueue(parsec, dtd_tp);

    SYNC_TIME_START();

    /* #### parsec context Starting #### */

    /* start parsec context */
    parsec_context_start(parsec);

    /* Testing Insert Function */


    int iteration = 0, total = minMNT;


    parsec_dtd_taskpool_insert_task( dtd_tp,       insert_task_geqrf, 0, "insert_task_geeqrf",
                       sizeof(int),           &total,             VALUE,
                       sizeof(int),           &iteration,         VALUE,
                       sizeof(two_dim_block_cyclic_t *), &dcA, SCRATCH,
                       sizeof(two_dim_block_cyclic_t *), &dcT, SCRATCH,
                       0 );


    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_taskpool_wait( parsec, dtd_tp );

    /* Waiting on all handle and turning everything off for this context */
    parsec_context_wait( parsec );

    SYNC_TIME_PRINT(rank, ("\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

    /* Cleaning up the parsec handle */
    parsec_taskpool_free( dtd_tp );

    if( check ) {
        if (M >= N) {
            if(loud > 2) printf("+++ Generate the Q ...");
            dplasma_zungqr( parsec,
                            (parsec_tiled_matrix_dc_t *)&dcA,
                            (parsec_tiled_matrix_dc_t *)&dcT,
                            (parsec_tiled_matrix_dc_t *)&dcQ);
            if(loud > 2) printf("Done\n");

            if(loud > 2) printf("+++ Solve the system ...");
            dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcX, 2354);
            dplasma_zlacpy( parsec, PlasmaUpperLower,
                            (parsec_tiled_matrix_dc_t *)&dcX,
                            (parsec_tiled_matrix_dc_t *)&dcB );
            dplasma_zgeqrs( parsec,
                            (parsec_tiled_matrix_dc_t *)&dcA,
                            (parsec_tiled_matrix_dc_t *)&dcT,
                            (parsec_tiled_matrix_dc_t *)&dcX );
            if(loud > 2) printf("Done\n");

            /* Check the orthogonality, factorization and the solution */
            ret |= check_orthogonality( parsec, (rank == 0) ? loud : 0,
                                        (parsec_tiled_matrix_dc_t *)&dcQ);
            ret |= check_factorization( parsec, (rank == 0) ? loud : 0,
                                        (parsec_tiled_matrix_dc_t *)&dcA0,
                                        (parsec_tiled_matrix_dc_t *)&dcA,
                                        (parsec_tiled_matrix_dc_t *)&dcQ );
            ret |= check_solution( parsec, (rank == 0) ? loud : 0,
                                   (parsec_tiled_matrix_dc_t *)&dcA0,
                                   (parsec_tiled_matrix_dc_t *)&dcB,
                                   (parsec_tiled_matrix_dc_t *)&dcX );

        } else {
            printf("Check cannot be performed when N > M\n");
        }

        parsec_data_free(dcA0.mat);
        parsec_data_free(dcQ.mat);
        parsec_data_free(dcB.mat);
        parsec_data_free(dcX.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcQ);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX);
    }

    /* Cleaning data arrays we allocated for communication */
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_FULL] );
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_LOWER] );
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_UPPER] );
    parsec_matrix_del2arena( parsec_dtd_arenas[TILE_RECTANGLE] );

    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcA );
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)&dcT );

    parsec_data_free(dcA.mat);
    parsec_data_free(dcT.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcT);

    cleanup_parsec(parsec, iparam);

    return ret;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static int check_orthogonality(parsec_context_t *parsec, int loud, parsec_tiled_matrix_dc_t *Q)
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

    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&Id);

    /* Perform Id - Q'Q */
    if ( M >= N ) {
        dplasma_zherk( parsec, PlasmaUpper, PlasmaConjTrans,
                       1.0, Q, -1.0, (parsec_tiled_matrix_dc_t*)&Id );
    } else {
        dplasma_zherk( parsec, PlasmaUpper, PlasmaNoTrans,
                       1.0, Q, -1.0, (parsec_tiled_matrix_dc_t*)&Id );
    }

    normQ = dplasma_zlanhe(parsec, PlasmaInfNorm, PlasmaUpper, (parsec_tiled_matrix_dc_t*)&Id);

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

    parsec_data_free(Id.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&Id);
    return info_ortho;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static int
check_factorization(parsec_context_t *parsec, int loud,
                    parsec_tiled_matrix_dc_t *Aorig,
                    parsec_tiled_matrix_dc_t *A,
                    parsec_tiled_matrix_dc_t *Q)
{
    parsec_tiled_matrix_dc_t *subA;
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
    dplasma_zlacpy( parsec, PlasmaUpperLower, Aorig, (parsec_tiled_matrix_dc_t *)&Residual );

    /* Extract the R */
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (parsec_tiled_matrix_dc_t *)&R);

    subA = tiled_matrix_submatrix( A, 0, 0, N, N );
    dplasma_zlacpy( parsec, PlasmaUpper, subA, (parsec_tiled_matrix_dc_t *)&R );
    free(subA);

    /* Perform Residual = Aorig - Q*R */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans,
                   -1.0, Q, (parsec_tiled_matrix_dc_t *)&R,
                    1.0, (parsec_tiled_matrix_dc_t *)&Residual);

    /* Free R */
    parsec_data_free(R.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&R);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&Residual);
    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, Aorig);

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

    parsec_data_free(Residual.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&Residual);
    return info_factorization;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX )
{
    parsec_tiled_matrix_dc_t *subX;
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    double eps = LAPACKE_dlamch_work('e');

    subX = tiled_matrix_submatrix( dcX, 0, 0, dcA->n, dcX->n );

    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, dcA);
    Bnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);
    Xnorm = dplasma_zlange(parsec, PlasmaInfNorm, subX);

    /* Compute A*x-b */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, 1.0, dcA, subX, -1.0, dcB);

    /* Compute A' * ( A*x - b ) */
    dplasma_zgemm( parsec, PlasmaConjTrans, PlasmaNoTrans,
                   1.0, dcA, dcB, 0., subX );

    Rnorm = dplasma_zlange( parsec, PlasmaInfNorm, subX );
    free(subX);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * dcA->n * eps ) ;

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
