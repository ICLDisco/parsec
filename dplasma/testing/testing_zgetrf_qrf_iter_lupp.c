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

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    double AnormI, Anorm1, BnormI, Bnorm1, XnormI, Xnorm1, RnormI, Rnorm1;
    int firsttest = 1;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 1, 1, 1);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
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

    if (rank == 0 )
    {
        printf("facto;M;N;NP;NC;MB;NB;IB;qr_a;treel;treeh;domino;tsrr;P;type;criteria;alpha;nblu;nbqr;gflops;AnormI;Anorm1;BnormI;Bnorm1;XnormI;Xnorm1;RnormI;Rnorm1\n");
    }

    /* Generate one B that will be used for everything */
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, 3873 );
    BnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB);
    Bnorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescB);

    {
        int  nbqr = 0, nblu = NT, info = 0;

        int type, type_i;
        int type_tab[] = { 0, /*1,*/ 2, 3, 4, /*5,*/ 7, 9, 12, 14, 18,
                           22, /* 23, */ 24, 27, 28, 29, 30, 31, 32, 34,
                           35, 36, 37, 38, 39, 40, 41, 100 };

        int test, nbtests;
        int index, lastindex;
        char *filename;
        FILE *f;

        asprintf( &filename, "persistent-lu-%dx%d-P%d", M, N, P );
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
#if defined(HAVE_MPI)
        MPI_Bcast(&lastindex, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        index = 0;

        for(type_i = 0; type_i < (int)(sizeof(type_tab) / sizeof(int)); type_i++)
        {
            type = type_tab[type_i];

            /* If random matrix, we test nbtests of them */
            if ( type == MATRIX_RANDOM )
                nbtests = 5;
            else
                nbtests = 1;

            for(test=0; test<2*nbtests; test++)
            {
                /* /\* Skip dead lock *\/ */
                /* if ( (P == 16) && (criteria == RANDOM_CRITERIUM) && */
                /*      ( type == 18) && ( alpha_i > 4) ) */
                /*     continue; */

                /* if ( (P == 32) && (criteria == RANDOM_CRITERIUM) && */
                /*      ( type == 18) && ( alpha_i > 3) ) */
                /*     continue; */

                index++;
                if( index <= lastindex ) {
                    if(rank == 0 )
                        printf("type=%d, criteria=%d; alpha=%e, test=%d: ignored (%d/%d)\n",
                               type, LU_ONLY_CRITERIUM, 0., test,
                               index, lastindex);
                    continue;
                }

                /* Matrix generation (Only once for all test using this matrix */
                if(loud > 2) printf("+++ Generate matrices ... ");

                if ( firsttest ) {
		    dplasma_zplrnt_perso( dague, (tiled_matrix_desc_t *)&ddescA0, type, 3872+(test/2)*53);
                    AnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA0);
                    Anorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescA0);
                    firsttest = 0;
                }

                dplasma_zlacpy( dague, PlasmaUpperLower,
                                (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
                if(loud > 2) printf("Done\n");

                /* Create DAGuE */
                if ( test%2 == 0 ) {
                    if(loud > 2) printf("+++ Computing getrf ... ");
                    PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf,
                                              ((tiled_matrix_desc_t*)&ddescA,
                                               (tiled_matrix_desc_t*)&ddescIPIV,
                                               &info));

                    /* lets rock! */
                    SYNC_TIME_START();
                    TIME_START();
                    dague_progress(dague);
                    SYNC_TIME_STOP();
                    gflops = (flops/1e9)/(sync_time_elapsed);
                    dplasma_zgetrf_Destruct( DAGUE_zgetrf );

#if defined(HAVE_MPI)
		    MPI_Bcast(&info, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
                } else {
                    if(loud > 2) printf("+++ Computing getrf ... ");
                    PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf_nopiv,
                                              ((tiled_matrix_desc_t*)&ddescA,
                                               &info));

                    /* lets rock! */
                    SYNC_TIME_START();
                    TIME_START();
                    dague_progress(dague);
                    SYNC_TIME_STOP();
                    gflops = (flops/1e9)/(sync_time_elapsed);
                    dplasma_zgetrf_nopiv_Destruct( DAGUE_zgetrf_nopiv );
                }


                if ( info != 0 ) {
                    /* That should not happen !!! QR is here to prevent this to happen */
                    fprintf(stderr, "-- Factorization is suspicious (info = %d) ! \n", info );
                    RnormI = -info; Rnorm1 = -info;
                    XnormI = -info; Xnorm1 = -info;
		    info = 0;
                }
                else {
                    /* Reinitialize B with same parameters as when we computed the norm */
                    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, 3873 );
                    dplasma_zlacpy( dague, PlasmaUpperLower,
                                    (tiled_matrix_desc_t *)&ddescB,
                                    (tiled_matrix_desc_t *)&ddescX );

                    /*
                     * First check with a right hand side
                     */
                    if ( test%2 == 0 ) {
                        dplasma_zgetrs(dague, PlasmaNoTrans,
                                       (tiled_matrix_desc_t *)&ddescA,
                                       (tiled_matrix_desc_t *)&ddescIPIV,
                                       (tiled_matrix_desc_t *)&ddescX );
                    } else {
                        dplasma_ztrsm( dague, PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,    1.0,
                                       (tiled_matrix_desc_t*)&ddescA,
                                       (tiled_matrix_desc_t*)&ddescX);
                        dplasma_ztrsm( dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0,
                                       (tiled_matrix_desc_t*)&ddescA,
                                       (tiled_matrix_desc_t*)&ddescX);
                    }

                    XnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescX);
                    Xnorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescX);

                    /* Compute b - A*x */
                    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0,
                                   (tiled_matrix_desc_t*)&ddescA0,
                                   (tiled_matrix_desc_t*)&ddescX, 1.0,
                                   (tiled_matrix_desc_t*)&ddescB);

                    RnormI = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB);
                    Rnorm1 = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&ddescB);
                }

                if (rank == 0)
                {
                    printf( "%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%e;%d;%d;%f;%.15e;%.15e;%.15e;%.15e;%.15e;%.15e;%.15e;%.15e\n",
                            (test%2) == 0 ? "getrf_pp" : "getrf_np", M, N, iparam[IPARAM_NNODES], iparam[IPARAM_NCORES], iparam[IPARAM_MB], iparam[IPARAM_NB], iparam[IPARAM_IB],
                            iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE], iparam[IPARAM_QR_DOMINO], iparam[IPARAM_QR_TSRR],
                            iparam[IPARAM_P], type, LU_ONLY_CRITERIUM, 0., nblu, nbqr, gflops, AnormI, Anorm1, BnormI, Bnorm1, XnormI, Xnorm1, RnormI, Rnorm1 );

                    f = fopen(filename, "w");
                    fprintf(f, "%d", index);
                    fclose(f);
                }
		
                if (test%2 == 1)
		  firsttest = 1;
            }
        }
        free(filename);
    }

    dague_data_free(ddescA0.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescA0);
    dague_data_free(ddescB.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescB);
    dague_data_free(ddescX.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescX);
    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescIPIV.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIV);

    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}


