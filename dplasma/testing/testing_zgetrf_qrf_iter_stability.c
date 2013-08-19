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

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };

static inline void getnbluqr( int rank, int MT, int *lu_tab, int *nbqr, int *nblu )
{
    int i;
    (void)rank;

#if defined(HAVE_MPI)
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
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    dplasma_qrtree_t qrtree;
    int *lu_tab;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 1, 1, 1);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M,(DagDouble_t)N))

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
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
                               two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    /* Random B check */
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
                               two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
                               two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    lu_tab = (int *)malloc( dague_imin(MT, NT)*sizeof(int) );

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
                      P,/*iparam[IPARAM_QR_HLVL_SZE],*/
                      0 /*iparam[IPARAM_QR_DOMINO]*/,
                      0 /*iparam[IPARAM_QR_TSRR]  */);


    if (rank == 0 )
    {
        printf(";NP;NC;P;MB;NB;IB;qr_a;treel;treeh;domino;tsrr;criteria;alpha;type;M;N;nblu;nbqr;gflops;AnormI;Anorm1;BnormI;Bnorm1;XnormI;Xnorm1;RnormI;Rnorm1\n");
    }


    {
        double AnormI, Anorm1, BnormI, Bnorm1, XnormI, Xnorm1, RnormI, Rnorm1;
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
                           22, 23, 24, 27, 28, 29, 30, 31, 32, 34,
                           35, 36, 37, 38, 39, 40, 41, 100 };

        // MAX & MOY
        // alpha = ( alpha_tab[i] * 1.5 ) ** 2;

        // MUMPS
        // alpha = (1.3 + ((1.8 - 1.3) / 6.) * i) ** 2

        int debug = 0;
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
        MPI_Bcast(&lastindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
        index = 0;

        for(type_i = 0; type_i < (int)(sizeof(type_tab) / sizeof(int)); type_i++)
        {
            type = type_tab[type_i];
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

                    if ((criteria == HIGHAM_MOY_CRITERIUM) ||
                        (criteria == HIGHAM_MAX_CRITERIUM) )
                    {
                        alpha = ( alpha * 1.5 ) * ( alpha * 1.5 );
                    }

                    if (criteria == MUMPS_CRITERIUM)
                    {
                        alpha = (1.3 + ((1.8 - 1.3) / 6.) * ((alpha_i+1)%7));
                        alpha *= alpha;
                    }

                    /* If random matrix, we test nbtests of them */
                    if ( type == MATRIX_RANDOM )
                        nbtests = 5;
                    else
                        nbtests = 1;

                    for(test=0; test<nbtests; test++)
                    {
                        AnormI = -1.; Anorm1 = -1.;
                        BnormI = -1.; Bnorm1 = -1.;
                        RnormI = -1.; Rnorm1 = -1.;
                        XnormI = -1.; Xnorm1 = -1.;

                        index++;
                        if( index <= lastindex ) {
                            if(rank == 0 )
                                printf("type=%d, criteria=%d; alpha=%e, test=%d: ignored (%d/%d)\n",
                                       type, criteria, alpha, test,
                                       index, lastindex);
                            continue;
                        }

                        /* matrix generation */
                        if(loud > 2) printf("+++ Generate matrices ... ");
                        dplasma_zplrnt_perso( dague, (tiled_matrix_desc_t *)&ddescA, type, 3872+test*53);
                        if( check )
                            dplasma_zlacpy( dague, PlasmaUpperLower,
                                            (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
                        dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS);
                        dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT);
                        if(loud > 2) printf("Done\n");

                        for(int i=0; i< dague_imin(MT, NT); i++)
                            lu_tab[i] = -1;

                        if (loud > 2)
                        {
                            printf("START: NP=%d; NC=%d; P=%d; MB=%d; IB=%d; TS=%d; LT=%d; HT=%d; D=%d; RR=%d; "
                                   "CRITERUM=%d; alpha=%e; type=%d; M=%d\n",
                                   iparam[IPARAM_NNODES], iparam[IPARAM_NCORES],
                                   iparam[IPARAM_P], iparam[IPARAM_MB], iparam[IPARAM_IB],
                                   iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_LOWLVL_TREE],
                                   iparam[IPARAM_HIGHLVL_TREE], iparam[IPARAM_QR_DOMINO],
                                   iparam[IPARAM_QR_TSRR], criteria, alpha, type, M );
                        }

                        /* Create DAGuE */
                        if(loud > 2) printf("+++ Computing getrf_qrf ... ");
                        PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf_qrf,
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
                        dague_progress(dague);
                        SYNC_TIME_STOP();
                        gflops = (flops/1e9)/(sync_time_elapsed);
                        getnbluqr( rank, dague_imin(MT, NT), lu_tab, &nbqr, &nblu );
                        dplasma_zgetrf_qrf_Destruct( DAGUE_zgetrf_qrf );

                        if ( info != 0 ) {
                            if( rank == 0 && loud ) fprintf(stderr, "-- Factorization is suspicious (info = %d) ! \n", info );
                            AnormI = -info; Anorm1 = -info;
                            BnormI = -info; Bnorm1 = -info;
                            RnormI = -info; Rnorm1 = -info;
                            XnormI = -info; Xnorm1 = -info;
                        }
                        else if ( check ) {
                            dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, 3873+test*53 );
                            dplasma_zlacpy( dague, PlasmaUpperLower,
                                            (tiled_matrix_desc_t *)&ddescB,
                                            (tiled_matrix_desc_t *)&ddescX );

                            AnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA);
                            Anorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescA);
                            BnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB);
                            Bnorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescB);

                            /*
                             * First check with a right hand side
                             */
                            dplasma_ztrsmpl_qrf( dague, &qrtree,
                                                 (tiled_matrix_desc_t *)&ddescA,
                                                 (tiled_matrix_desc_t *)&ddescIPIV,
                                                 (tiled_matrix_desc_t *)&ddescX,
                                                 (tiled_matrix_desc_t *)&ddescTS,
                                                 (tiled_matrix_desc_t *)&ddescTT,
                                                 lu_tab);
                            dplasma_ztrsm(dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0,
                                          (tiled_matrix_desc_t *)&ddescA,
                                          (tiled_matrix_desc_t *)&ddescX);

                            XnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB);
                            Xnorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescB);

                            /* Compute b - A*x */
                            dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0,
                                           (tiled_matrix_desc_t*)&ddescA,
                                           (tiled_matrix_desc_t*)&ddescX, 1.0,
                                           (tiled_matrix_desc_t*)&ddescB);

                            RnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB);
                            Rnorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescB);
                        }

                        if (rank == 0)
                        {
                            /* printf("zgetrf_qrf computation NP= %d NC= %d P= %d MB= %d IB= %d qr_a= %d treel= %d treeh= %d domino= %d RR= %d criteria= %d alpha= %e, rndtype= %d, M= %d N= %d : %f gflops\n", */
                            /*        iparam[IPARAM_NNODES], iparam[IPARAM_NCORES], */
                            /*        iparam[IPARAM_P], iparam[IPARAM_MB], iparam[IPARAM_IB], */
                            /*        iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_LOWLVL_TREE], */
                            /*        iparam[IPARAM_HIGHLVL_TREE], iparam[IPARAM_QR_DOMINO], */
                            /*        iparam[IPARAM_QR_TSRR], criteria, alpha, type, */
                            /*        M, N, */
                            /*        gflops); */

                            printf("getrf_qrf;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%e;%d;%d;%d;%d;%d;%f;%e;%e;%e;%e;%e;%e;%e;%e\n",
                                   iparam[IPARAM_NNODES], iparam[IPARAM_NCORES],
                                   iparam[IPARAM_P], iparam[IPARAM_MB], iparam[IPARAM_NB], iparam[IPARAM_IB],
                                   iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_LOWLVL_TREE],
                                   iparam[IPARAM_HIGHLVL_TREE], iparam[IPARAM_QR_DOMINO],
                                   iparam[IPARAM_QR_TSRR], criteria, alpha, type,
                                   M, N, nblu, nbqr, gflops, AnormI, Anorm1, BnormI, Bnorm1, XnormI, Xnorm1, RnormI, Rnorm1 );

                            f = fopen(filename, "w");
                            fprintf(f, "%d", index);
                            fclose(f);
                        }
                    }
                }
            }
        }
        free(filename);
    }

    free(lu_tab);
    if (check)
    {
        dague_data_free(ddescA0.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescA0);
        dague_data_free(ddescB.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescB);
        dague_data_free(ddescX.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescX);
    }
    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescTS.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTS);
    dague_data_free(ddescTT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTT);
    dague_data_free(ddescIPIV.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIV);

    dplasma_hqr_finalize( &qrtree );
    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}


