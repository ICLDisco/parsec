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

static inline void getnbluqr( int rank, int MT, int *lu_tab, int *nbqr, int *nblu )
{
    int i;
    (void)rank;

#if defined(PARSEC_HAVE_MPI)
    {
        int *lu_tab2 = (int*)malloc( MT*sizeof(int) );
        MPI_Allreduce ( lu_tab, lu_tab2, MT, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        memcpy( lu_tab, lu_tab2, MT*sizeof(int) );
        free(lu_tab2);
    }
#endif

    *nblu = 0;
    *nbqr = MT;
    for(i=0; i<*nbqr; i++) {
        *nblu += lu_tab[i];
    }
    *nbqr -= *nblu;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    dplasma_qrtree_t qrtree;
    double AnormI, Anorm1, BnormI, Bnorm1, XnormI, Xnorm1, RnormI, Rnorm1;
    int *lu_tab;
    int firsttest = 1;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 1, 1, 1);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M,(DagDouble_t)N))

    iparam[IPARAM_QR_HLVL_SZE] = P;
    iparam[IPARAM_QR_DOMINO]   = 0;
    iparam[IPARAM_QR_TSRR]     = 0;
    LDA = max(M, LDA);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                               two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1,
                               two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1,
                               two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
                               two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                                                      nodes, cores, rank, MB, 1, M, NT, 0, 0,
                                                      M, NT, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
                               two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    /* Random B check */
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
                               two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1,
                               two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    lu_tab = (int *)malloc( dplasma_imin(MT, NT)*sizeof(int) );

    /*
     * We should always use the same tree
     * Fibbonacci for high level tree in order to get small tree and goood
     * pipeline if 2 QR occurs one after the other
     * Greedy tree for the low level with size domain size of 4.
     */
    dplasma_hqr_init( &qrtree,
                      (tiled_matrix_desc_t *)&ddescA,
                      iparam[IPARAM_LOWLVL_TREE],
                      iparam[IPARAM_HIGHLVL_TREE],
                      iparam[IPARAM_QR_TS_SZE],
                      iparam[IPARAM_QR_HLVL_SZE],
                      iparam[IPARAM_QR_DOMINO],
                      iparam[IPARAM_QR_TSRR]);

    if (rank == 0 )
    {
        printf("facto;M;N;NP;NC;MB;NB;IB;qr_a;treel;treeh;domino;tsrr;P;type;criteria;alpha;nblu;nbqr;gflops;AnormI;Anorm1;BnormI;Bnorm1;XnormI;Xnorm1;RnormI;Rnorm1\n");
    }

    /* Generate one B that will be used for everything */
    dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescB, 3873 );
    BnormI = dplasma_zlange(parsec, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB);
    Bnorm1 = dplasma_zlange(parsec, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescB);

    {
        int nbqr, nblu, info = 0;
        int  criteria_i, criteria;
        int  criteria_tab[] = { RANDOM_CRITERIUM,
                                MUMPS_CRITERIUM,
                                /*HIGHAM_MOY_CRITERIUM,*/
                                HIGHAM_MAX_CRITERIUM,
                                LU_ONLY_CRITERIUM,
                                QR_ONLY_CRITERIUM };
        int    alpha_i;
        double alpha_tab[] = { 12.5, 25., 37.5, 50., 62.5, 75., 87.5 };
        double alpha;

        int type, type_i;
        int type_tab[] = { 0, /*1,*/ 2, 3, 4, /*5,*/ 7, 9, 12, 14, 18,
                           22, /* 23, */ 24, 27, 28, 29, 30, 31, 32, 34,
                           35, 36, 37, 38, 39, 40, 41, 100 };

        // MAX & MOY
        // alpha = ( alpha_tab[i] * 1.5 ) ** 2;

        // MUMPS
        // alpha = (1.3 + ((1.8 - 1.3) / 6.) * i) ** 2

        int test, nbtests;
        int index, lastindex;
        char *filename;
        FILE *f;

        asprintf( &filename, "persistent-trace-%dx%d-P%d", M, N, P );
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
#if defined(PARSEC_HAVE_MPI)
        MPI_Bcast(&lastindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        index = 0;

        for(type_i = 0; type_i < (int)(sizeof(type_tab) / sizeof(int)); type_i++)
        {
            type = type_tab[type_i];

            /* If random matrix, we test nbtests of them */
            if ( type == PlasmaMatrixRandom )
                nbtests = 5;
            else
                nbtests = 1;

            for(test=0; test<nbtests; test++)
            {
                for(criteria_i = 0; criteria_i < (int)(sizeof(criteria_tab) / sizeof(int)); criteria_i++)
                {
                    criteria = criteria_tab[criteria_i];
                    for(alpha_i = 0; alpha_i < (int)(sizeof(alpha_tab) / sizeof(double)); alpha_i++)
                    {
                        alpha = alpha_tab[ alpha_i ];

                        /* No need for multiple alpha for LU/QR only */
                        if ( ((criteria == LU_ONLY_CRITERIUM) ||
                              (criteria == QR_ONLY_CRITERIUM) ) &&
                             ( alpha_i > 0) )
                            continue;

                        /* Skip dead lock */
                        if ( (P == 16) && (criteria == RANDOM_CRITERIUM) &&
                             ( type == 18) && ( alpha_i > 4) )
                            continue;

                        if ( (P == 32) && (criteria == RANDOM_CRITERIUM) &&
                             ( type == 18) && ( alpha_i > 3) )
                            continue;

                        if ((criteria == HIGHAM_MOY_CRITERIUM) ||
                            (criteria == HIGHAM_MAX_CRITERIUM) )
                        {
                            alpha = ( alpha * 1.5 ) * ( alpha * 1.5 );
                        }

                        if (criteria == MUMPS_CRITERIUM)
                        {
                            alpha = (1.3 + ((1.8 - 1.3) / 6.) * (alpha_i%7));
                            alpha *= alpha;
                        }

                        index++;
                        if( index <= lastindex ) {
                            if(rank == 0 )
                                printf("type=%d, criteria=%d; alpha=%e, test=%d: ignored (%d/%d)\n",
                                       type, criteria, alpha, test,
                                       index, lastindex);
                            continue;
                        }

                        /* Matrix generation (Only once for all test using this matrix */
                        if(loud > 2) printf("+++ Generate matrices ... ");

                        if ( firsttest ) {
                            dplasma_zpltmg( parsec, type, (tiled_matrix_desc_t *)&ddescA0, 3872+test*53);
                            AnormI = dplasma_zlange(parsec, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA0);
                            Anorm1 = dplasma_zlange(parsec, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescA0);
                            firsttest = 0;
                        }

                        dplasma_zlacpy( parsec, PlasmaUpperLower,
                                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
                        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS);
                        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT);
                        if(loud > 2) printf("Done\n");

                        for(int i=0; i< dplasma_imin(MT, NT); i++)
                            lu_tab[i] = -1;

                        /* Create PaRSEC */
                        if(loud > 2) printf("+++ Computing getrf_qrf ... ");
                        PASTE_CODE_ENQUEUE_KERNEL(parsec, zgetrf_qrf,
                                                  (&qrtree,
                                                   (tiled_matrix_desc_t*)&ddescA,
                                                   (tiled_matrix_desc_t*)&ddescIPIV,
                                                   (tiled_matrix_desc_t*)&ddescTS,
                                                   (tiled_matrix_desc_t*)&ddescTT,
                                                   criteria, alpha, lu_tab,
                                                   &info));

                        /* lets rock! */
                        SYNC_TIME_START();
                        TIME_START();
                        parsec_context_wait(parsec);
                        SYNC_TIME_STOP();
                        gflops = (flops/1e9)/(sync_time_elapsed);
                        getnbluqr( rank, dplasma_imin(MT, NT), lu_tab, &nbqr, &nblu );
                        dplasma_zgetrf_qrf_Destruct( PARSEC_zgetrf_qrf );

                        if ( info != 0 ) {
                            /* That should not happen !!! QR is here to prevent this to happen */
                            fprintf(stderr, "-- Factorization is suspicious (info = %d) ! \n", info );
                            RnormI = -info; Rnorm1 = -info;
                            XnormI = -info; Xnorm1 = -info;
                        }
                        else {
                            /* Reinitialize B with same parameters as when we computed the norm */
                            dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescB, 3873 );
                            dplasma_zlacpy( parsec, PlasmaUpperLower,
                                            (tiled_matrix_desc_t *)&ddescB,
                                            (tiled_matrix_desc_t *)&ddescX );

                            /*
                             * First check with a right hand side
                             */
                            dplasma_ztrsmpl_qrf( parsec, &qrtree,
                                                 (tiled_matrix_desc_t *)&ddescA,
                                                 (tiled_matrix_desc_t *)&ddescIPIV,
                                                 (tiled_matrix_desc_t *)&ddescX,
                                                 (tiled_matrix_desc_t *)&ddescTS,
                                                 (tiled_matrix_desc_t *)&ddescTT,
                                                 lu_tab);
                            dplasma_ztrsm(parsec, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0,
                                          (tiled_matrix_desc_t *)&ddescA,
                                          (tiled_matrix_desc_t *)&ddescX);

                            XnormI = dplasma_zlange(parsec, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescX);
                            Xnorm1 = dplasma_zlange(parsec, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescX);

                            /* Compute b - A*x */
                            dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0,
                                           (tiled_matrix_desc_t*)&ddescA0,
                                           (tiled_matrix_desc_t*)&ddescX, 1.0,
                                           (tiled_matrix_desc_t*)&ddescB);

                            RnormI = dplasma_zlange(parsec, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB);
                            Rnorm1 = dplasma_zlange(parsec, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescB);
                        }

                        if (rank == 0)
                        {
                            printf("getrf_qrf;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%e;%d;%d;%f;%.15e;%.15e;%.15e;%.15e;%.15e;%.15e;%.15e;%.15e\n",
                                   M, N, iparam[IPARAM_NNODES], iparam[IPARAM_NCORES], iparam[IPARAM_MB], iparam[IPARAM_NB], iparam[IPARAM_IB],
                                   iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE], iparam[IPARAM_QR_DOMINO], iparam[IPARAM_QR_TSRR],
                                   iparam[IPARAM_P], type, criteria, alpha, nblu, nbqr, gflops, AnormI, Anorm1, BnormI, Bnorm1, XnormI, Xnorm1, RnormI, Rnorm1 );

                            f = fopen(filename, "w");
                            fprintf(f, "%d", index);
                            fclose(f);
                        }
                    }
                }
                firsttest = 1;
            }
        }
        free(filename);
    }

    free(lu_tab);
    parsec_data_free(ddescA0.mat);
    parsec_ddesc_destroy( (parsec_ddesc_t*)&ddescA0);
    parsec_data_free(ddescB.mat);
    parsec_ddesc_destroy( (parsec_ddesc_t*)&ddescB);
    parsec_data_free(ddescX.mat);
    parsec_ddesc_destroy( (parsec_ddesc_t*)&ddescX);

    parsec_data_free(ddescA.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescA);
    parsec_data_free(ddescTS.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescTS);
    parsec_data_free(ddescTT.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescTT);
    parsec_data_free(ddescIPIV.mat);
    parsec_ddesc_destroy((parsec_ddesc_t*)&ddescIPIV);

    dplasma_hqr_finalize( &qrtree );
    cleanup_parsec(parsec, iparam);

    return EXIT_SUCCESS;
}


