/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_orthogonality(parsec_context_t *parsec, int loud, tiled_matrix_desc_t *Q);
static int check_factorization(parsec_context_t *parsec, int loud, tiled_matrix_desc_t *Aorig, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Q);

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info_ortho = 0, info_facto = 0;
    qr_piv_t *qrpiv;
    int rc;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 1, 1, 1);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGEQRF, ((DagDouble_t)M,(DagDouble_t)N))

    LDA = max(M, LDA);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1,
        two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble,
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1,
        two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble,
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, check,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));


#if defined(PARSEC_PROF_TRACE)
    ddescA.super.super.key = strdup("A");
    ddescTS.super.super.key = strdup("TS");
    ddescTT.super.super.key = strdup("TT");
    ddescA0.super.super.key = strdup("A0");
    ddescQ.super.super.key = strdup("Q");
#endif

    {
        int qr_a_tab[] = { 1, 4, 8 }, qr_a_i;
        int qr_p_tab[] = {   1,   2,   4,   8,   15,   16,   32,   64,  128,  256, 512,
                           768, 896, 960, 992, 1008, 1016, 1020, 1022, 1023, 1024 }, qr_p_i;
        int lowlvl_tree;
        int higlvl_tree;
        int domino;
        int qr_a;
        int qr_p;
        int tsrr;
        int test;
        int index, lastindex;
        char *filename;
        FILE *f;

        domino = iparam[IPARAM_QR_DOMINO];
        lowlvl_tree = iparam[IPARAM_LOWLVL_TREE];
        higlvl_tree = iparam[IPARAM_HIGHLVL_TREE];
        qr_a = iparam[IPARAM_QR_TS_SZE];

        asprintf(&filename, "persistent-trace-cas1-%dx%d", M, N);
        if( 0 == rank ) {
            f = fopen(filename, "r");
            if( NULL == f ) {
                lastindex = 0;
                fprintf(stderr, "starting at index %d\n", lastindex);
            } else {
                fscanf(f, "%d", &lastindex);
                fprintf(stderr, "restarting at index %d\n", lastindex);
                fclose(f);
            }
        }
        MPI_Bcast(&lastindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
        index = 0;

        for( test = 0; test > -1; test++ ) {
            tsrr = 1;
          //for( tsrr = 0; tsrr < 2; tsrr++ )
        {
            /* Square case */
/*             if ( MT==240 && MT >= 4*NT ) */
/*                 domino = 0; */
/*             else */
                domino = 1;
            //for( domino = 0; domino < 2; domino++ )
            {
                lowlvl_tree = 1;
                //for( lowlvl_tree = 0; lowlvl_tree < 4; lowlvl_tree++)
                {
                    higlvl_tree = 2;
                    //for( higlvl_tree = 0; higlvl_tree < 5; higlvl_tree++)
                    {
                        qr_p = P;
                        //for(qr_p_i = 0; qr_p_i < sizeof(qr_p_tab) / sizeof(int); qr_p_i++)
                        {
                            //qr_p = qr_p_tab[qr_p_i];
                            qr_a = 8;
                            //for(qr_a_i = 0; qr_a_i < sizeof(qr_a_tab) / sizeof(int); qr_a_i++)
                            {
                                //qr_a = qr_a_tab[qr_a_i];

/*                                 if( MT / qr_p < qr_a  ) */
/*                                     continue; */

                                qrpiv = dplasma_pivgen_init( (tiled_matrix_desc_t *)&ddescA,
                                                             lowlvl_tree, higlvl_tree,
                                                             qr_a, qr_p,
                                                             domino, tsrr );

                                index++;
                                if( index <= lastindex ) {
                                    if(rank == 0 ) printf("treel=%d treeh=%d qr_a=%d domino=%d ignored (%d/%d)\n",
                                                          lowlvl_tree, higlvl_tree, qr_a, domino,
                                                          index, lastindex);
                                    continue;
                                }

                                /* matrix generation */
                                if(loud > 2) printf("+++ Generate matrices ... ");
                                dplasma_zplrnt( parsec, (tiled_matrix_desc_t *)&ddescA, 3872+test*53);
                                if( check )
                                    dplasma_zlacpy( parsec, PlasmaUpperLower,
                                                    (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
                                dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS);
                                dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT);
                                if(loud > 2) printf("Done\n");

                                /* Create PaRSEC */
                                PASTE_CODE_ENQUEUE_KERNEL(parsec, zgeqrf_param,
                                                          (qrpiv,
                                                           (tiled_matrix_desc_t*)&ddescA,
                                                           (tiled_matrix_desc_t*)&ddescTS,
                                                           (tiled_matrix_desc_t*)&ddescTT));

                                /* lets rock! */
                                SYNC_TIME_START();
                                rc = parsec_context_start(parsec);
                                PARSEC_CHECK_ERROR(rc, "parsec_context_start");
                                TIME_START();
                                rc = parsec_context_wait(parsec);
                                PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

#if defined(PARSEC_SIM)
                                printf("zgeqrf simulation M= %d N= %d qr_a= %d qr_p= %d treel= %d treeh= %d domino= %d RR= %d : %d \n",
                                       M, N, qr_a, qr_p, lowlvl_tree, higlvl_tree, domino, tsrr,
                                       parsec->largest_simulation_date);
                                fflush(stdout);
#else
                                SYNC_TIME_PRINT(rank,
                                                ("zgeqrf computation NP= %d NC= %d P= %d IB= %d MB= %d NB= %d qr_a= %d qr_p= %d treel= %d treeh= %d domino= %d RR= %d M= %d N= %d : %f gflops\n",
                                                 iparam[IPARAM_NNODES],
                                                 iparam[IPARAM_NCORES],
                                                 iparam[IPARAM_P],
                                                 iparam[IPARAM_IB],
                                                 iparam[IPARAM_MB],
                                                 iparam[IPARAM_NB],
                                                 qr_a, qr_p, lowlvl_tree, higlvl_tree, domino, tsrr, M, N,
                                                 gflops = (flops/1e9)/(sync_time_elapsed)));
#endif
                                (void)flops;
                                (void)gflops;

                                if( rank == 0 ) {
                                    f = fopen(filename, "w");
                                    fprintf(f, "%d", index);
                                    fclose(f);
                                }

                                dplasma_zgeqrf_param_Destruct( PARSEC_zgeqrf_param );

                                if( check ) {
                                    if(loud > 2) printf("+++ Generate the Q ...");
                                    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
                                    dplasma_zungqr_param( parsec, qrpiv,
                                                          (tiled_matrix_desc_t *)&ddescA,
                                                          (tiled_matrix_desc_t *)&ddescTS,
                                                          (tiled_matrix_desc_t *)&ddescTT,
                                                          (tiled_matrix_desc_t *)&ddescQ);
                                    if(loud > 2) printf("Done\n");

                                    /* Check the orthogonality, factorization and the solution */
                                    info_ortho = check_orthogonality(parsec, (rank == 0) ? loud : 0,
                                                                     (tiled_matrix_desc_t *)&ddescQ);
                                    info_facto = check_factorization(parsec, (rank == 0) ? loud : 0,
                                                                     (tiled_matrix_desc_t *)&ddescA0,
                                                                     (tiled_matrix_desc_t *)&ddescA,
                                                                     (tiled_matrix_desc_t *)&ddescQ);

                                    parsec_data_free(ddescA0.mat);
                                    parsec_data_free(ddescQ.mat);
                                    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescA0);
                                    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescQ);
                                }
                                dplasma_pivgen_finalize( qrpiv );
                            }
                        }
                    }
                }
            }
        }
        }
    }

    cleanup_parsec(parsec, iparam);

    parsec_data_free(ddescA.mat);
    parsec_data_free(ddescTS.mat);
    parsec_data_free(ddescTT.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescA);
    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescTS);
    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescTT);

    return info_ortho || info_facto;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static int check_orthogonality(parsec_context_t *parsec, int loud, tiled_matrix_desc_t *Q)
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
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble,
                               Q->super.nodes, Q->super.cores, twodQ->grid.rank,
                               Q->mb, Q->nb, minMN, minMN, 0, 0,
                               minMN, minMN, twodQ->grid.strows, twodQ->grid.stcols, twodQ->grid.rows));

    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);

    /* Perform Id - Q'Q (could be done with Herk) */
    if ( M >= N ) {
      dplasma_zgemm( parsec, PlasmaConjTrans, PlasmaNoTrans,
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    } else {
      dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaConjTrans,
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    }

    normQ = dplasma_zlange(parsec, PlasmaMaxNorm, (tiled_matrix_desc_t*)&Id);

    result = normQ / (minMN * eps);
    if ( loud ) {
        printf("============\n");
        printf("Checking the orthogonality of Q \n");
        printf("||Id-Q'*Q||_oo / (N*eps) = %e \n", result);
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Orthogonality is suspicious ! \n");
        info_ortho=1;
    }
    else {
        if( loud ) printf("-- Orthogonality is CORRECT ! \n");
        info_ortho=0;
    }

    parsec_data_free(Id.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&Id);
    return info_ortho;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */

static int check_factorization(parsec_context_t *parsec, int loud, tiled_matrix_desc_t *Aorig, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Q)
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
    double Anorm, Rnorm;
    double result;
    double eps = LAPACKE_dlamch_work('e');
    int info_factorization;
    int M = A->m;
    int N = A->n;
    int minMN = min(M, N);

    PASTE_CODE_ALLOCATE_MATRIX(Residual, 1,
        two_dim_block_cyclic, (&Residual, matrix_ComplexDouble,
                               A->super.nodes, A->super.cores, twodA->grid.rank,
                               A->mb, A->nb, M, N, 0, 0,
                               M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(R, 1,
        two_dim_block_cyclic, (&R, matrix_ComplexDouble,
                               A->super.nodes, A->super.cores, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    /* Extract the L */
    dplasma_zlacpy( parsec, PlasmaUpperLower, Aorig, (tiled_matrix_desc_t *)&Residual );
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&R);

    /* Extract the R */
    dplasma_zlacpy( parsec, PlasmaUpper, A, (tiled_matrix_desc_t *)&R );

    /* Perform Residual = Aorig - Q*R */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans,
                   -1.0, Q, (tiled_matrix_desc_t *)&R,
                    1.0, (tiled_matrix_desc_t *)&Residual);

    /* Free R */
    parsec_data_free(R.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&R);

    Rnorm = dplasma_zlange(parsec, PlasmaMaxNorm, (tiled_matrix_desc_t*)&Residual);
    Anorm = dplasma_zlange(parsec, PlasmaMaxNorm, Aorig);

    result = Rnorm / ( Anorm * minMN * eps);

    if( loud ) {
        printf("============\n");
        printf("Checking the QR Factorization \n");
        printf("-- ||A-QR||_oo/(||A||_oo.N.eps) = %e \n", result );
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else {
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    parsec_data_free(Residual.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&Residual);
    return info_factorization;
}
