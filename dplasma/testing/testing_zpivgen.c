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
    parsec_context_t* parsec;
    dplasma_qrtree_t qrtree;
    int rc, ret = 0;
    int iparam[IPARAM_SIZEOF];
    char *dot_filename;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 1, 1, 1);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    if (check) {
        tiled_matrix_desc_t *B;
        int alltreel[] = { 0, 1, 2, 3, 4 };
        int alltreeh[] = { 0, 1, 2, 3, 4 };
        int allP[]     = { 3, 5, 7, 8 };
        int allA[]     = { 1, 2, 4, 7 };
        int allM[]     = { 1, 3, 4, 10, 17, 25, 128 };
        int allN[]     = { 1, 2, 5, 13, 26, 58 };
        int nbtreel = 4;
        int nbtreeh = 5;
        int nbP     = 4;
        int nbA     = 4;
        int nbM     = 7;
        int nbN     = 6;
        int l, h, p, a, m, n, d, r;
        int done, todo;
        todo = 0;
        done = 0;

        /* HQR */
        todo += nbtreel * nbM * nbN * (2 * nbA - 1) * (1 +  2 * nbtreeh * nbP);
        /* systolic */
        todo +=  nbM * nbN * nbA * nbP;

        LDA = max(allM[ nbM-1 ], LDA);
        /* initializing matrix structure */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, allN[ nbN-1 ], 0, 0,
                                   allM[ nbM-1 ], allN[ nbN-1 ], SMB, SNB, P));

        /*
         *
         * Tests for HQR code
         *
         */
        for( l=0; l<nbtreel; l++) {
            /* No High Level */
            h = 0; d = 0; p = -1;
            for( a=0; a<nbA; a++) {
                for( m=0; m<nbM; m++) {
                    for( n=0; n<nbN; n++) {
                        for( r=0; r<2; r++) {
                            if (r==1 && a==1)
                                continue;

                            B = tiled_matrix_submatrix((tiled_matrix_desc_t*)&ddescA, 0, 0, allM[m], allN[n] );
                            dplasma_hqr_init( &qrtree, PlasmaNoTrans, B, alltreel[l], 0, allA[a], -1, 0, r );

                            rc = dplasma_qrtree_check( B, &qrtree );
                            if (rc != 0) {
                                fprintf(stderr, "-M %d -N %d --treel=%d --qr_a=%d --tsrr=%d                                   FAILED(%d)\n",
                                        allM[m], allN[n], alltreel[l], allA[a], r, rc);
                                ret |= 1;
                            }
                            dplasma_hqr_finalize( &qrtree );
                            free(B);

                            done++;
                            printf("\r%6d / %6d", done, todo);
                        }
                    }
                }
            }
            /* With High level */
            for( d=0; d<2; d++) { /* Domino */
                if (d == 1 && alltreel[l] == DPLASMA_GREEDY1P_TREE)
                    continue;
                for( h=0; h<nbtreeh; h++) {
                    for( p=0; p<nbP; p++) {
                        for( a=0; a<nbA; a++) {
                            for( m=0; m<nbM; m++) {
                                for( n=0; n<nbN; n++) {
                                    for( r=0; r<2; r++) {
                                        if (r==1 && a==1)
                                            continue;

                                        B = tiled_matrix_submatrix((tiled_matrix_desc_t*)&ddescA, 0, 0, allM[m], allN[n] );
                                        dplasma_hqr_init( &qrtree, PlasmaNoTrans, B, alltreel[l], alltreeh[h], allA[a], allP[p], d, r);

                                        rc = dplasma_qrtree_check( B, &qrtree );
                                        if (rc != 0) {
                                            fprintf(stderr, "-M %d -N %d --treel=%d --qr_a=%d --tsrr=%d --qr_p=%d --treeh=%d --domino=%d  FAILED(%d)\n",
                                                    allM[m], allN[n], alltreel[l], allA[a], r, allP[p], alltreeh[h], d, rc);
                                            ret |= 1;
                                        }

                                        dplasma_hqr_finalize( &qrtree );
                                        free(B);

                                        done++;
                                        printf("\r%6d / %6d", done, todo);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /*
         *
         * Tests for systolic code
         *
         */
        /* With High level */
        for( p=0; p<nbP; p++) {
            for( a=0; a<nbA; a++) {
                for( m=0; m<nbM; m++) {
                    for( n=0; n<nbN; n++) {
                        B = tiled_matrix_submatrix((tiled_matrix_desc_t*)&ddescA, 0, 0, allM[m], allN[n] );
                        dplasma_systolic_init( &qrtree, PlasmaNoTrans, B, allA[a], allP[p]);

                        rc = dplasma_qrtree_check( B, &qrtree );
                        if (rc != 0) {
                            fprintf(stderr, "systolic: -M %d -N %d --qr_a=%d --qr_p=%d    FAILED(%d)\n",
                                    allM[m], allN[n], allA[a], allP[p], rc);
                            ret |= 1;
                        }

                        dplasma_systolic_finalize( &qrtree );
                        free(B);

                        done++;
                        printf("\r%6d / %6d", done, todo);
                    }
                }
            }
        }

        parsec_data_free(ddescA.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);

    } else {

        LDA = max(M, LDA);
        /* initializing matrix structure */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   M, N, SMB, SNB, P));

#if defined(SYSTOLIC)
        dplasma_systolic_init( &qrtree,
                               PlasmaNoTrans, (tiled_matrix_desc_t *)&ddescA,
                               iparam[IPARAM_QR_HLVL_SZE],
                               iparam[IPARAM_QR_TS_SZE] );
#else
        dplasma_hqr_init( &qrtree,
                          PlasmaNoTrans, (tiled_matrix_desc_t*)&ddescA,
                          iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE],
                          iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_QR_HLVL_SZE],
                          iparam[IPARAM_QR_DOMINO], iparam[IPARAM_QR_TSRR] );
#endif

        asprintf(&dot_filename, "tree-%dx%d-a%d-p%d-l%d-h%d-d%d.dot",
                 M, N,
                 iparam[IPARAM_QR_TS_SZE],
                 iparam[IPARAM_QR_HLVL_SZE],
                 iparam[IPARAM_LOWLVL_TREE],
                 iparam[IPARAM_HIGHLVL_TREE],
                 iparam[IPARAM_QR_DOMINO]);

        /*dplasma_qrtree_print_dag( (tiled_matrix_desc_t*)&ddescA, &qrtree, dot_filename );*/
        ret = dplasma_qrtree_check( (tiled_matrix_desc_t*)&ddescA, &qrtree );

        /* dplasma_qrtree_print_pivot(   (tiled_matrix_desc_t*)&ddescA, &qrtree); */
        /* dplasma_qrtree_print_next_k(  (tiled_matrix_desc_t*)&ddescA, &qrtree, 0); */
        /* dplasma_qrtree_print_prev_k(  (tiled_matrix_desc_t*)&ddescA, &qrtree, 0); */
        /* dplasma_qrtree_print_nbgeqrt( (tiled_matrix_desc_t*)&ddescA, &qrtree ); */
        /* dplasma_qrtree_print_type   ( (tiled_matrix_desc_t*)&ddescA, &qrtree ); */

#if defined(SYSTOLIC)
        dplasma_systolic_finalize( &qrtree );
#else
        dplasma_hqr_finalize( &qrtree );
#endif

        free(dot_filename);

        parsec_data_free(ddescA.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    }

    cleanup_parsec(parsec, iparam);

    if ( ret == 0 )
      return EXIT_SUCCESS;
    else {
      fprintf(stderr, "ret = %d\n", ret);
      return EXIT_FAILURE;
    }
}
